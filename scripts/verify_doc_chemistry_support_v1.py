"""Replay frozen chemistry support coordinates and validation-only adaptation."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import read_source
from verify_doc_auxiliary_chemistry_v1 import BASIS, _auxiliary, _lineage
from verify_doc_daily_hydro_support_basis_v1 import _manual_base, _same, _sidecar

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.chemical_support_basis import ChemicalSupportBasis
from river_graph.models.nonlinear_chemistry_head import NonlinearChemistryHead
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import (
    SupportShapeAdapter,
    SupportShapeEpisode,
)

ROOT = Path("experiments/phase4_transfer/doc_chemistry_support_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
VARIANTS = ("legacy", "masks_aug", "chemistry_aug")
PIPELINES = {"neural": "neural_chemistry", "integrated": "neural_chemistry_integrated", "tree": "tree_chemistry"}
MODELS = tuple(f"{pipe}_{basis}" for pipe in PIPELINES.values() for basis in (*VARIANTS, "selected")) + ("point_integrated_legacy",)
COMPONENTS = ("y_pred", "base_pred", "adaptation_delta", "regional_gamma")


def _phi(auxiliary, head_state, feature_mean, feature_scale, mode):
    values = np.asarray(auxiliary, dtype=np.float64).copy()
    active = np.any(values[..., 2:] != 0, axis=-1)
    values[..., :2] = np.where(values[..., 2:] != 0, values[..., :2], 0)
    if mode == "masks":
        values[..., :2] = 0
    elif mode != "chemistry":
        raise ValueError("Unexpected chemistry representation mode")
    flat = values.reshape(-1, 4)
    output = np.empty((len(flat), 8), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(flat), 512):
            standardized = torch.as_tensor((flat[start:start+512]-feature_mean)/feature_scale, dtype=torch.float64)
            linear = torch.nn.functional.linear(standardized, head_state["phi.weight"], head_state["phi.bias"])
            output[start:start+512] = torch.nn.functional.silu(linear).numpy()
    return output.reshape(*values.shape[:2], 8), active


def _basis(state, auxiliary, source_ids, payload, legacy):
    """Compute source moments/PCA independently of ChemicalSupportBasis.fit."""
    summary, weights = payload["summary"], payload["head_state"]
    mean = np.asarray(summary["normalization"]["feature_mean"])[-4:]
    scale = np.asarray(summary["normalization"]["feature_scale"])[-4:]
    source_ids = np.sort(np.asarray(source_ids, dtype=np.int64))
    if (state["source_role"] != "source_training" or state["eigenvalue_floor"] != 1e-8
            or state["phi_batch_size"] != 512):
        raise ValueError("Chemical basis source role, variance floor or projection batches differ")
    for key, value in (("source_station_indices", source_ids), ("frozen_phi_weight", weights["phi.weight"].numpy()),
                       ("frozen_phi_bias", weights["phi.bias"].numpy()), ("auxiliary_mean", mean), ("auxiliary_scale", scale)):
        np.testing.assert_array_equal(state[key], value)
    source_phi, source_active = _phi(auxiliary[source_ids], weights, mean, scale, state["mode"])
    counts = source_active.sum(axis=1)
    station_mean = np.zeros((len(source_ids), 8))
    covariance_sum, total = np.zeros((8, 8)), np.zeros(8)
    for index in range(len(source_ids)):
        values = source_phi[index, source_active[index]]
        if len(values):
            station_mean[index] = values.mean(axis=0)
            centered = values-station_mean[index]
            covariance_sum += centered.T @ centered
            total += values.sum(axis=0)
    count = int(counts.sum())
    covariance, global_mean = covariance_sum/max(count, 1), total/max(count, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(-values, kind="stable")[:2]
    components = vectors[:, order].T.copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    eigenvalues = np.maximum(values[order], 0)
    identified = eigenvalues > 1e-8
    whitening = np.sqrt(np.maximum(eigenvalues, 1e-8))
    for key, value in (("source_active_counts", counts), ("source_station_active_means", station_mean),
                       ("source_global_mean", global_mean), ("covariance", covariance), ("components", components),
                       ("eigenvalues", eigenvalues), ("identified", identified), ("whitening_scale", whitening)):
        np.testing.assert_array_equal(state[key], value)
    expected = {"n_source_stations": len(source_ids), "n_source_active_stations": int((counts > 0).sum()),
                "n_source_active_rows": count, "identified_components": int(identified.sum())}
    if any(state[key] != value for key, value in expected.items()):
        raise ValueError("Chemical basis source population differs")
    full_phi, active = _phi(auxiliary, weights, mean, scale, state["mode"])
    basis = ((full_phi-global_mean) @ components.T)/whitening
    basis[..., ~identified] = 0
    basis[~active] = 0
    loaded = ChemicalSupportBasis.from_dict(state)
    if loaded.to_dict() != state:
        raise ValueError("Chemical support basis did not roundtrip exactly")
    _same(loaded.transform(auxiliary), basis)
    augmented = np.concatenate([np.asarray(legacy).reshape(*auxiliary.shape[:2], 2), basis], axis=-1)
    _same(loaded.augment(auxiliary, legacy)["augmented_basis"], augmented)
    # Projection is local to the supplied row and never centers a target timeline.
    first_month = loaded.transform(auxiliary[:, :1])
    np.testing.assert_allclose(first_month, basis[:, :1], rtol=1e-12, atol=1e-12)
    return basis, augmented, {"mode": state["mode"], **expected,
        "source_moments_eigenvectors_and_whitening_exact": True,
        "frozen_phi_and_source_normalizer_exact": True,
        "row_local_projection_rtol_atol": 1e-12,
        "row_local_different_batch_max_abs_difference": float(np.abs(first_month-basis[:, :1]).max()),
        "inactive_coordinates_zero": bool(np.count_nonzero(basis[~active]) == 0)}


def _fit_source_adapters(bases, representations, context, memory, labels, split, months, active, gamma_k0):
    """Build every selection episode directly from the fixed validation cells."""
    tasks = []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        query = query[active[query]]
        support = support[np.isin(support//months, np.unique(query//months))]
        if not len(query) or np.intersect1d(query, support).size:
            raise ValueError("Invalid active source-validation support/query episode")
        tasks.append((k, support, query))
    adapters, mixers = {}, {}
    for name, basis in representations.items():
        for pipeline in ("neural", "tree"):
            base = bases[pipeline]
            episodes = [SupportShapeEpisode(k, query, labels[query], base[query], support,
                labels[support], base[support], basis[query], basis[support]) for k, support, query in tasks]
            adapters[(pipeline, name)] = SupportShapeAdapter(n_months=months).fit(
                episodes, selection_role="source_validation")
        base = bases["neural"]
        episodes = [SupportAwareTransferEpisode(k, query, labels[query], support, labels[support],
            context[query], base[query], memory[query], context[support], base[support], memory[support],
            basis[query], basis[support]) for k, support, query in tasks]
        mixers[name] = SupportAwareResidualTransfer(months).fit(
            episodes, gamma_k0=gamma_k0, selection_role="source_validation")
    return adapters, mixers


def _candidate(pipeline, basis, k, support, query, bases, context, memory, labels, adapter):
    if pipeline == "integrated":
        gamma = adapter.selected_gamma(k)
        base = _manual_base(context, bases["neural"], memory, gamma)
        prediction = adapter.adapt(context[query], bases["neural"][query], memory[query], query,
            context[support], bases["neural"][support], memory[support], support, labels[support],
            basis[query], basis[support], k=k)
    else:
        base = bases[pipeline]
        gamma = 0.
        prediction = adapter.adapt(base[query], query, base[support], support, labels[support],
            query_basis=basis[query], support_basis=basis[support], k=k)
    return {"y_pred": prediction, "candidate_y_pred": prediction.copy(), "base_pred": base[query].copy(),
            "adaptation_delta": np.log1p(prediction)-np.log1p(base[query]),
            "regional_gamma": np.full(len(query), gamma), "support_count": np.full(len(query), k)}


def _panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers, labels, split, months, active, role):
    panels = {}
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    frozen = {name: SupportShapeAdapter.from_dict(old_adapters[f"{name}_{BASIS}"]) for name in ("point", "tree_prior")}
    point_mixer = SupportAwareResidualTransfer.from_dict(old_mixers[f"point_integrated_{BASIS}"])
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        missing = ~active[query]
        fallbacks = {name: _candidate(name, shapes["legacy"], k, support, query,
            {name: full[f"{name}_pred"].to_numpy()}, context, memory, labels, model) for name, model in frozen.items()}
        fallbacks["point_integrated"] = _candidate("integrated", shapes["legacy"], k, support, query,
            {"neural": full.point_pred.to_numpy()}, context, memory, labels, point_mixer)
        for pipeline, pipe in PIPELINES.items():
            fallback = "point_integrated" if pipeline == "integrated" else "point" if pipeline == "neural" else "tree_prior"
            for variant, basis in shapes.items():
                adapter = mixers[variant] if pipeline == "integrated" else adapters[(pipeline, variant)]
                panel = _candidate(pipeline, basis, k, support, query, bases, context, memory, labels, adapter)
                for column in COMPONENTS:
                    panel[column] = panel[column].copy()
                    panel[column][missing] = fallbacks[fallback][column][missing]
                panels[(f"{pipe}_{variant}", k)] = {"cell": query, **panel,
                    "aux_fallback": missing.copy(), "basis_name": np.full(len(query), variant),
                    "selected_basis": np.full(len(query), variant)}
        panels[("point_integrated_legacy", k)] = {"cell": query, **fallbacks["point_integrated"],
            "aux_fallback": np.zeros(len(query), dtype=bool), "basis_name": np.full(len(query), "legacy"),
            "selected_basis": np.full(len(query), "legacy")}
    return panels


def _summary(panels, labels, active):
    rows = []
    for (name, k), panel in sorted(panels.items()):
        errors = np.abs(panel["y_pred"]-labels[panel["cell"]])
        selected = active[panel["cell"]]
        rows.append({"model_name": name, "k": k, "n": len(errors), "n_active": int(selected.sum()),
            "mae": float(errors.mean()), "active_mae": float(errors[selected].mean())})
    return pd.DataFrame(rows)


def _selection(summary):
    choices = {}
    for pipe in PIPELINES.values():
        choices[pipe] = {}
        for k in KS:
            scores = []
            for variant in VARIANTS:
                row = summary[summary.model_name.eq(f"{pipe}_{variant}") & summary.k.eq(k)].iloc[0]
                scores.append({"basis": variant, "active_mae": float(row.active_mae), "mae": float(row.mae),
                    "n": int(row["n"]), "n_active": int(row.n_active)})
            winner = min(scores, key=lambda row: (row["active_mae"], VARIANTS.index(row["basis"])))
            choices[pipe][str(k)] = {"basis": winner["basis"], "active_mae": winner["active_mae"], "scores": scores}
    return {"version": 1, "selection_role": "source_validation", "score": "final_active_validation_mae",
        "tie_order": list(VARIANTS), "choices": choices}


def _add_selected(panels, selection):
    result = panels.copy()
    for pipe in PIPELINES.values():
        for k in KS:
            chosen = selection["choices"][pipe][str(k)]["basis"]
            parent = panels[(f"{pipe}_{chosen}", k)]
            result[(f"{pipe}_selected", k)] = {**parent,
                "basis_name": np.full(len(parent["cell"]), "selected"),
                "selected_basis": np.full(len(parent["cell"]), chosen)}
    return result


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    expected = {"experiment": "doc_chemistry_support_v1", "split_seed": split_seed, "seed": seed,
        "runtime_snapshot_hash": runtime, "pipelines": list(PIPELINES.values()),
        "basis_names": [*VARIANTS, "selected"], "k_values": list(KS), "inference_roles": ["train"],
        "selection_role": "source_validation", "basis_dimensions": {"legacy": 2, "masks_aug": 4, "chemistry_aug": 4},
        "pca_components": 2, "pca_eigenvalue_floor": 1e-8, "new_neural_fits": 0,
        "new_forest_fits": 0, "backbone_retraining": False}
    if any(config[key] != value for key, value in expected.items()) or set(config["models"]) != set(MODELS):
        raise ValueError("Chemical support scope or selection configuration differs")
    completion = verify_files(run, "complete.json", config)
    model_files = {"representations.npz", "basis_definition.json", "adapters.json", "mixers.json", "basis_selection.json"}
    if set(completion["files"]) != {"config.json", *model_files, "source_validation.csv", "reference_checks.json",
            "timing.json", "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json"}:
        raise ValueError("Incomplete chemical support package")
    prior = Path(config["prior_run"])
    for filename, key in (("complete.json", "prior_completion_hash"), ("full_grid.parquet", "prior_full_grid_hash"),
                          ("predictions.parquet", "prior_prediction_hash"), ("neural_chemistry.pt", "prior_checkpoint_hash")):
        if sha256_file(prior / filename) != config[key]:
            raise ValueError("A fixed decoder or prediction product changed")
    pc, _, dataset, split, parent_full = read_source(prior)
    verify_files(prior, "complete.json", pc)
    _lineage(pc)
    for name in ("dataset", "mask"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError("Dataset/mask file changed")
        for suffix in ("path", "hash"):
            if config[f"{name}_{suffix}"] != pc[f"{name}_{suffix}"]:
                raise ValueError("Dataset/mask identity differs from the parent")
    for key in ("basis_run", "basis_completion_hash", "q90_threshold_train", "query_cells",
                "auxiliary_paths", "auxiliary_hashes", "auxiliary_provenance_hashes",
                "auxiliary_feature_hash", "auxiliary_active_hash", "availability_audit_path"):
        if config[key] != pc[key]:
            raise ValueError("Frozen basis/chemistry/protocol input changed")
    parent_report = Path(config["prior_replay_path"])
    if sha256_file(parent_report) != config["prior_replay_hash"]:
        raise ValueError("Parent decoder replay report changed")
    report = json.loads(parent_report.read_text())
    records = [row for row in report["results"] if row["run"] == prior.name]
    if (len(records) != 1 or records[0]["status"] != "verified"
            or records[0]["completion_sha256"] != config["prior_completion_hash"]
            or records[0]["runtime_snapshot_hash"] != verify_runtime_snapshot(prior.parent.parent)):
        raise ValueError("Parent verification is not bound to the selected decoder")
    if sha256_file(run / "full_grid.parquet") != config["prior_full_grid_hash"]:
        raise ValueError("Frozen native full-grid predictions changed")
    full, queries = pd.read_parquet(run / "full_grid.parquet"), pd.read_parquet(run / "predictions.parquet")
    pd.testing.assert_frame_equal(full, parent_full, check_exact=True)
    for name, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, name, config, runtime, frame, completion, model_files)
    shape, months = tuple(dataset["y"].shape), dataset["y"].shape[1]
    np.testing.assert_array_equal(full.cell, np.arange(np.prod(shape)))
    auxiliary, active = _auxiliary(pc, dataset)
    auxiliary = auxiliary.reshape(*shape, 4)
    source_ids = np.unique(split["train"]//months)
    np.testing.assert_array_equal(config["source_station_ids"], source_ids)
    if np.intersect1d(source_ids, np.unique(np.r_[split["val"], split["test"]]//months)).size:
        raise ValueError("Source PCA stations overlap held stations")
    with np.load(prior / "source_training.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(source_ids, saved["source_station_ids"])
    with np.load(Path(pc["basis_run"]) / "representations.npz", allow_pickle=False) as saved:
        legacy = saved[BASIS].copy()
    payload = torch.load(prior / "neural_chemistry.pt", weights_only=True)
    head = NonlinearChemistryHead.from_payload(payload)
    torch.set_num_threads(config["torch_threads"])
    definitions = json.loads((run / "basis_definition.json").read_text())
    if set(definitions) != {"masks_aug", "chemistry_aug"}:
        raise ValueError("Missing chemical basis definitions")
    archive = {"legacy": legacy, "active": active, "source_station_ids": source_ids}
    shapes, basis_checks = {"legacy": legacy}, []
    for mode, name in (("masks", "masks_aug"), ("chemistry", "chemistry_aug")):
        if definitions[name]["mode"] != mode:
            raise ValueError("Chemical basis mode changed")
        chemical, augmented, check = _basis(definitions[name], auxiliary, source_ids, payload, legacy)
        # A source-only refit has no DOC inputs and ignores all non-source auxiliary values.
        hidden_target_aux = auxiliary.copy()
        hidden_target_aux[np.setdiff1d(np.arange(shape[0]), source_ids)] = np.nan
        refitted = ChemicalSupportBasis(head).fit(hidden_target_aux, source_ids, source_role="source_training", mode=mode)
        if refitted.to_dict() != definitions[name]:
            raise ValueError("Source-only PCA refit depends on held-station auxiliary values")
        shapes[name] = augmented.reshape(-1, 4)
        archive[name], archive[f"{mode}_chemical"] = shapes[name], chemical.reshape(-1, 2)
        basis_checks.append(check)
    with np.load(run / "representations.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(archive):
            raise ValueError("Unexpected representation archive contents")
        for key, value in archive.items():
            np.testing.assert_array_equal(saved[key], value)
    old_adapters, old_mixers = (json.loads((prior / name).read_text()) for name in ("adapters.json", "mixers.json"))
    gamma = old_mixers[f"neural_chemistry_integrated_{BASIS}"]["gamma_k0"]
    if config["gamma_k0"] != gamma:
        raise ValueError("The original K0 ecology mixture changed")
    bases = {"neural": full.neural_chemistry_pred.to_numpy(), "tree": full.tree_chemistry_pred.to_numpy()}
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    truth = np.asarray(dataset["y"], dtype=float).ravel()
    labels = np.full(truth.shape, np.nan)
    labels[split["val"]] = truth[split["val"]]
    adapters, mixers = _fit_source_adapters(bases, shapes, context, memory, labels, split, months, active, gamma)
    adapter_states = {f"{PIPELINES[pipe]}_{name}": model.to_dict() for (pipe, name), model in adapters.items()}
    mixer_states = {f"neural_chemistry_integrated_{name}": model.to_dict() for name, model in mixers.items()}
    if (adapter_states != json.loads((run / "adapters.json").read_text())
            or mixer_states != json.loads((run / "mixers.json").read_text())):
        raise ValueError("Independent validation-only support refits differ")
    for pipe in ("neural_chemistry", "tree_chemistry"):
        if adapter_states[f"{pipe}_legacy"] != old_adapters[f"{pipe}_{BASIS}"]:
            raise ValueError("Original direct support calibration changed")
    if mixer_states["neural_chemistry_integrated_legacy"] != old_mixers[f"neural_chemistry_integrated_{BASIS}"]:
        raise ValueError("Original integrated support calibration changed")
    validation = _panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
        labels, split, months, active, "val")
    selected = _selection(_summary(validation, labels, active))
    if selected != json.loads((run / "basis_selection.json").read_text()):
        raise ValueError("Representation choices are not the validation-only MAE/tie winners")
    validation = _add_selected(validation, selected)
    summary = _summary(validation, labels, active)
    pd.testing.assert_frame_equal(summary, pd.read_csv(run / "source_validation.csv"),
        check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    original_validation = pd.read_csv(prior / "source_validation.csv")
    for pipe in (*PIPELINES.values(), "point_integrated"):
        for k in KS:
            a = summary[summary.model_name.eq(f"{pipe}_legacy") & summary.k.eq(k)].iloc[0]
            b = original_validation[original_validation.model_name.eq(f"{pipe}_{BASIS}") & original_validation.k.eq(k)].iloc[0]
            np.testing.assert_allclose(a[["mae", "active_mae"]].to_numpy(dtype=float), b[["mae", "active_mae"]].to_numpy(dtype=float), rtol=0, atol=1e-12)
    reserved, query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if len(query) != config["query_cells"] or np.intersect1d(reserved, query).size:
        raise ValueError("Fixed target query/support cells changed")
    target_labels = np.full(truth.shape, np.nan)
    target_labels[reserved] = truth[reserved]
    panels = _add_selected(_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
        target_labels, split, months, active, "test"), selected)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy())) != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(split_seed).all()
            or not np.isfinite(queries.select_dtypes(include="number")).all().all()):
        raise ValueError("Missing query panels or invalid identities")
    query_checks = []
    descriptors = ("station", "month", "analyte", "visibility_role", "ecological_novelty", "upstream_support",
                   "ph_available", "ec_available", "aux_available", "doc_observed")
    for (name, k), panel in panels.items():
        order = np.argsort(panel["cell"])
        cells = panel["cell"][order]
        stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
        np.testing.assert_array_equal(cells, np.sort(query))
        for column, values in panel.items():
            np.testing.assert_array_equal(stored[column], values[order])
        for column in descriptors:
            np.testing.assert_array_equal(stored[column], full.iloc[cells][column])
        _same(stored.y_true, truth[cells])
        if k == 0:
            _same(stored.y_pred, stored.base_pred)
        query_checks.append({"model_name": name, "k": k, "n_query": len(stored),
            "max_abs_difference": 0., "bitwise_exact": True, "inactive_final_fallback_exact": True})
    previous = pd.read_parquet(prior / "predictions.parquet")
    references = []
    for pipe in (*PIPELINES.values(), "point_integrated"):
        for k in KS:
            a = queries[queries.model_name.eq(f"{pipe}_legacy") & queries.k.eq(k)].sort_values("cell")
            b = previous[previous.model_name.eq(f"{pipe}_{BASIS}") & previous.k.eq(k)].sort_values("cell")
            for column in ("cell", *COMPONENTS):
                np.testing.assert_array_equal(a[column], b[column])
            references.append({"model_name": f"{pipe}_legacy", "k": k, "rows": len(a), "reference_exact": True})
    for pipe in PIPELINES.values():
        for k in (0, 1):
            a = queries[queries.model_name.eq(f"{pipe}_legacy") & queries.k.eq(k)].sort_values("cell")
            for variant in ("masks_aug", "chemistry_aug", "selected"):
                b = queries[queries.model_name.eq(f"{pipe}_{variant}") & queries.k.eq(k)].sort_values("cell")
                for column in ("cell", *COMPONENTS):
                    np.testing.assert_array_equal(a[column], b[column])
                references.append({"model_name": f"{pipe}_{variant}", "k": k, "rows": len(b), "low_k_exact": True})
    if json.loads((run / "reference_checks.json").read_text()) != {"role": "test", "checks": references}:
        raise ValueError("Independent invariance records differ")
    return {"run": run.name, "status": "verified", "completion_sha256": sha256_file(run / "complete.json"),
        "runtime_snapshot_hash": runtime, "n_full_grid": len(full), "basis_checks": basis_checks,
        "query_replays": query_checks, "reference_controls": references,
        "source_validation_rows": len(summary), "direct_adapter_validation_refits": len(adapters),
        "integrated_mixer_validation_refits": len(mixers), "basis_selections_validation_replayed": 12,
        "source_only_pca_refits": 2, "no_doc_values_read_by_basis": True,
        "all_nonvalidation_doc_hidden_during_selection": True,
        "full_grid_parent_byte_identical": True, "neural_forest_backbone_refitted": False,
        "prior_replay_path": str(parent_report), "prior_replay_sha256": sha256_file(parent_report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    requested = [(s, r) for s in args.split_seeds for r in args.seeds]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Replay requires a nonempty unique run panel")
    results = []
    for split, seed in requested:
        print(f"Replaying split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime))
        gc.collect()
    output = args.output or args.root / "verification/replay_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(s, r) for s in SPLITS for r in SEEDS},
        "completed_production_runs_at_check": sum((args.root / "runs" / f"split{s}_seed{r}" / "complete.json").is_file()
            for s in SPLITS for r in SEEDS), "target_comparative_performance_analyzed": False,
        "neural_forest_backbone_refitted": False, "query_and_fallback_replay_rtol_atol": 0,
        "verifier_sha256": sha256_file(__file__), "shared_verifier_helpers_sha256": {name: sha256_file(name) for name in (
            "scripts/verify_doc_auxiliary_chemistry_v1.py", "scripts/verify_doc_daily_hydro_support_basis_v1.py")},
        "results": results}, indent=2)+"\n")
    print(f"Verified {len(results)} complete packages: {output}", flush=True)


if __name__ == "__main__":
    main()
