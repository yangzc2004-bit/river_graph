"""Transfer a few DOC observations using frozen recurrent and chemical states."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_auxiliary_chemistry_v1 import AUX_PATHS, BASIS, validation_tasks
from run_doc_daily_hydro_memory_v1 import KS
from run_doc_daily_hydro_readout_v1 import array_digest
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import read_source

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.auxiliary_chemistry_features import (
    build_auxiliary_chemistry_features,
)
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
PRIOR = Path("experiments/phase4_transfer/doc_chemistry_decoder_v1")
VARIANTS = ("legacy", "masks_aug", "chemistry_aug")
REPRESENTATIONS = (*VARIANTS, "selected")
PIPELINES = ("neural_chemistry", "neural_chemistry_integrated", "tree_chemistry")
MODELS = tuple(f"{pipe}_{basis}" for pipe in PIPELINES for basis in REPRESENTATIONS) + ("point_integrated_legacy",)
COMPONENTS = ("y_pred", "base_pred", "adaptation_delta", "regional_gamma")


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_auxiliary_chemistry_v1.py", "scripts/run_doc_daily_hydro_memory_v1.py",
                 "scripts/run_doc_daily_hydro_readout_v1.py", "scripts/run_doc_chemistry_decoder_v1.py",
                 "scripts/run_doc_chemistry_support_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution changed; preserve this version and use a fresh root")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def fit_calibrators(bases, context, memory, shapes, truth, split, months, active, gamma_k0):
    adapters, mixers = {}, {}
    for name, basis in shapes.items():
        for arm in ("neural_chemistry", "tree_chemistry"):
            episodes = [SupportShapeEpisode(k, query, truth[query], bases[arm][query], support,
                truth[support], bases[arm][support], basis[query], basis[support])
                for k, support, query in validation_tasks(split, months, active)]
            adapters[f"{arm}_{name}"] = SupportShapeAdapter(n_months=months).fit(
                episodes, selection_role="source_validation")
        episodes = [SupportAwareTransferEpisode(k, query, truth[query], support, truth[support],
            context[query], bases["neural_chemistry"][query], memory[query], context[support],
            bases["neural_chemistry"][support], memory[support], basis[query], basis[support])
            for k, support, query in validation_tasks(split, months, active)]
        mixers[f"neural_chemistry_integrated_{name}"] = SupportAwareResidualTransfer(months).fit(
            episodes, gamma_k0=gamma_k0, selection_role="source_validation")
    return adapters, mixers


def frozen_reference(pipe, full, legacy, old_adapters, old_mixers, labels, support, query, k):
    if pipe == "point_integrated":
        model = SupportAwareResidualTransfer.from_dict(old_mixers[f"point_integrated_{BASIS}"])
        context, base, memory = full.context_pred.to_numpy(), full.point_pred.to_numpy(), full.ecological_memory.to_numpy()
        predicted = model.adapt(context[query], base[query], memory[query], query,
            context[support], base[support], memory[support], support, labels[support],
            legacy[query], legacy[support], k=k)
        selected_base, gamma = model.selected_base(context[query], base[query], memory[query], k=k), model.selected_gamma(k)
    else:
        model = SupportShapeAdapter.from_dict(old_adapters[f"{pipe}_{BASIS}"])
        base = full[f"{pipe}_pred"].to_numpy()
        predicted = model.adapt(base[query], query, base[support], support, labels[support],
            query_basis=legacy[query], support_basis=legacy[support], k=k)
        selected_base, gamma = base[query], 0.
    return {"y_pred": predicted, "base_pred": selected_base,
        "adaptation_delta": np.log1p(predicted)-np.log1p(selected_base),
        "regional_gamma": np.full(len(query), gamma)}


def make_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
                labels, split, months, active, *, role):
    rows = []
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    descriptors = ("cell", "station", "month", "analyte", "visibility_role", "ecological_novelty", "upstream_support",
                   "ph_available", "ec_available", "aux_available", "doc_observed")
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        inactive = ~active[query]
        fallbacks = {pipe: frozen_reference(pipe, full, shapes["legacy"], old_adapters, old_mixers,
                     labels, support, query, k) for pipe in ("point", "point_integrated", "tree_prior")}
        for pipe in PIPELINES:
            parent = "tree_prior" if pipe == "tree_chemistry" else "point_integrated" if pipe.endswith("integrated") else "point"
            for variant, basis in shapes.items():
                name = f"{pipe}_{variant}"
                if pipe.endswith("integrated"):
                    model, base = mixers[name], bases["neural_chemistry"]
                    predicted = model.adapt(context[query], base[query], memory[query], query,
                        context[support], base[support], memory[support], support, labels[support],
                        basis[query], basis[support], k=k)
                    selected_base, gamma = model.selected_base(context[query], base[query], memory[query], k=k), model.selected_gamma(k)
                else:
                    model, base = adapters[name], bases[pipe]
                    predicted = model.adapt(base[query], query, base[support], support, labels[support],
                        query_basis=basis[query], support_basis=basis[support], k=k)
                    selected_base, gamma = base[query], 0.
                row = full.iloc[query][list(descriptors)].copy()
                row["model_name"], row["k"], row["basis_name"], row["selected_basis"] = name, k, variant, variant
                row["candidate_y_pred"], row["y_pred"], row["base_pred"] = predicted.copy(), predicted, selected_base
                row["adaptation_delta"], row["regional_gamma"] = np.log1p(predicted)-np.log1p(selected_base), gamma
                row["support_count"], row["aux_fallback"] = k, inactive
                for column in COMPONENTS:
                    row.loc[inactive, column] = fallbacks[parent][column][inactive]
                rows.append(row)
        row = full.iloc[query][list(descriptors)].copy()
        row["model_name"], row["k"], row["basis_name"], row["selected_basis"] = "point_integrated_legacy", k, "legacy", "legacy"
        for column in COMPONENTS:
            row[column] = fallbacks["point_integrated"][column]
        row["candidate_y_pred"], row["support_count"], row["aux_fallback"] = row.y_pred, k, False
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def validation_summary(frame, truth):
    rows = []
    for (model, k), group in frame.groupby(["model_name", "k"], sort=True):
        errors = np.abs(group.y_pred.to_numpy()-truth[group.cell.to_numpy()])
        active = group.aux_available.to_numpy(dtype=bool)
        rows.append({"model_name": model, "k": int(k), "n": len(group), "n_active": int(active.sum()),
            "mae": float(errors.mean()), "active_mae": float(errors[active].mean())})
    return pd.DataFrame(rows)


def choose_representations(validation):
    choices = {pipe: {} for pipe in PIPELINES}
    for pipe in PIPELINES:
        for k in KS:
            scores = []
            for variant in VARIANTS:
                row = validation[validation.model_name.eq(f"{pipe}_{variant}") & validation.k.eq(k)].iloc[0]
                scores.append({"basis": variant, **{key: float(row[key]) for key in ("active_mae", "mae")},
                    "n": int(row["n"]), "n_active": int(row.n_active)})
            winner = min(scores, key=lambda row: (row["active_mae"], VARIANTS.index(row["basis"])))
            choices[pipe][str(k)] = {"basis": winner["basis"], "active_mae": winner["active_mae"], "scores": scores}
    return {"version": 1, "selection_role": "source_validation", "score": "final_active_validation_mae",
        "tie_order": list(VARIANTS), "choices": choices}


def add_selected(frame, selection):
    copied = []
    for pipe in PIPELINES:
        for k in KS:
            chosen = selection["choices"][pipe][str(k)]["basis"]
            row = frame[frame.model_name.eq(f"{pipe}_{chosen}") & frame.k.eq(k)].copy()
            row["model_name"], row["basis_name"], row["selected_basis"] = f"{pipe}_selected", "selected", chosen
            copied.append(row)
    return pd.concat([frame, *copied], ignore_index=True)


def invariance_checks(frame, previous, *, role):
    checks = []
    for pipe in (*PIPELINES, "point_integrated"):
        old_name = f"{pipe}_{BASIS}"
        for k in KS:
            a = frame[frame.model_name.eq(f"{pipe}_legacy") & frame.k.eq(k)].sort_values("cell")
            b = previous[previous.model_name.eq(old_name) & previous.k.eq(k)].sort_values("cell")
            for column in ("cell", *COMPONENTS):
                np.testing.assert_array_equal(a[column], b[column])
            checks.append({"model_name": f"{pipe}_legacy", "k": k, "rows": len(a), "reference_exact": True})
    for pipe in PIPELINES:
        for k in (0, 1):
            ref = frame[frame.model_name.eq(f"{pipe}_legacy") & frame.k.eq(k)].sort_values("cell")
            for variant in REPRESENTATIONS[1:]:
                row = frame[frame.model_name.eq(f"{pipe}_{variant}") & frame.k.eq(k)].sort_values("cell")
                for column in ("cell", *COMPONENTS):
                    np.testing.assert_array_equal(row[column], ref[column])
                checks.append({"model_name": f"{pipe}_{variant}", "k": k, "rows": len(row), "low_k_exact": True})
    return {"role": role, "checks": checks}


def run_one(root, prior_root, partition, seed, runtime):
    started = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, _, dataset, split, full = read_source(prior)
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime or config["prior_completion_hash"] != sha256_file(prior / "complete.json"):
            raise ValueError("Cached chemical-support run differs from frozen sources")
        verify_files(run, "complete.json", config)
        return
    run.mkdir(parents=True, exist_ok=True)
    aux_data = {key: torch.load(path, weights_only=False) for key, path in AUX_PATHS.items()}
    auxiliary = build_auxiliary_chemistry_features(dataset, aux_data["ph"], aux_data["ec"])
    months = dataset["y"].shape[1]
    truth, active = np.asarray(dataset["y"], dtype=np.float64).ravel(), auxiliary["active"].ravel()
    if array_digest(auxiliary["full"].reshape(-1, 4)) != pc["auxiliary_feature_hash"]:
        raise ValueError("Changed auxiliary chemistry inputs")
    with np.load(prior / "source_training.npz", allow_pickle=False) as saved:
        source_ids = saved["source_station_ids"].copy()
    if np.intersect1d(source_ids, np.unique(np.r_[split["val"], split["test"]]//months)).size:
        raise ValueError("Chemical PCA source roles overlap held stations")
    with np.load(Path(pc["basis_run"]) / "representations.npz", allow_pickle=False) as saved:
        legacy = saved[BASIS].copy()
    head = NonlinearChemistryHead.from_payload(torch.load(prior / "neural_chemistry.pt", weights_only=True))
    shapes, definitions, coordinates = {"legacy": legacy}, {}, {"legacy": legacy, "active": active, "source_station_ids": source_ids}
    for mode, name in (("masks", "masks_aug"), ("chemistry", "chemistry_aug")):
        model = ChemicalSupportBasis(head, eigenvalue_floor=1e-8).fit(auxiliary["full"], source_ids,
            source_role="source_training", mode=mode)
        chemical = model.transform(auxiliary["full"]).reshape(-1, 2)
        shapes[name] = np.concatenate((legacy, chemical), axis=1)
        coordinates[name], coordinates[f"{mode}_chemical"] = shapes[name], chemical
        definitions[name] = model.to_dict()
    np.savez_compressed(run / "representations.npz", **coordinates)
    write_json(run / "basis_definition.json", definitions)
    old_adapters = json.loads((prior / "adapters.json").read_text())
    old_mixers = json.loads((prior / "mixers.json").read_text())
    gamma_k0 = old_mixers[f"neural_chemistry_integrated_{BASIS}"]["gamma_k0"]
    config = {"experiment": "doc_chemistry_support_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "prior_full_grid_hash": sha256_file(prior / "full_grid.parquet"),
        "prior_prediction_hash": sha256_file(prior / "predictions.parquet"),
        "prior_checkpoint_hash": sha256_file(prior / "neural_chemistry.pt"),
        "prior_replay_path": str(prior_root / "verification/replay_checks.json"),
        "prior_replay_hash": sha256_file(prior_root / "verification/replay_checks.json"),
        **{key: pc[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "basis_run",
            "basis_completion_hash", "q90_threshold_train", "query_cells", "auxiliary_paths", "auxiliary_hashes",
            "auxiliary_provenance_hashes", "auxiliary_feature_hash", "auxiliary_active_hash", "availability_audit_path")},
        "source_station_ids": source_ids.tolist(), "target_analyte": "doc", "target_transform": "log1p support residual",
        "inference_roles": ["train"], "selection_role": "source_validation", "pipelines": PIPELINES,
        "basis_names": REPRESENTATIONS, "models": MODELS, "k_values": KS,
        "basis_dimensions": {"legacy": 2, "masks_aug": 4, "chemistry_aug": 4},
        "pca_components": 2, "pca_eigenvalue_floor": 1e-8,
        "pca_fit_role": "within source-station active-aux months; no DOC labels",
        "pca_projection_center": "source-global active mean; target row-local", "gamma_k0": gamma_k0,
        "representation_selection": "active source-validation MAE; ties legacy,masks_aug,chemistry_aug",
        "new_neural_fits": 0, "new_forest_fits": 0, "backbone_retraining": False,
        "torch_threads": torch.get_num_threads(), "full_grid_role": "unchanged parent components; basis in representations.npz",
        "study_role": "chemical-state support calibration on reused DOC development station partitions"}
    write_json(run / "config.json", config)
    bases = {arm: full[f"{arm}_pred"].to_numpy() for arm in ("neural_chemistry", "tree_chemistry")}
    adapters, mixers = fit_calibrators(bases, full.context_pred.to_numpy(), full.ecological_memory.to_numpy(),
        shapes, truth, split, months, active, gamma_k0)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    write_json(run / "mixers.json", {name: model.to_dict() for name, model in mixers.items()})
    validation = make_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
        truth, split, months, active, role="val")
    summary = validation_summary(validation, truth)
    selection = choose_representations(summary)
    write_json(run / "basis_selection.json", selection)
    validation = add_selected(validation, selection)
    validation_summary(validation, truth).to_csv(run / "source_validation.csv", index=False)
    old_validation = pd.read_csv(prior / "source_validation.csv")
    for pipe in (*PIPELINES, "point_integrated"):
        for k in KS:
            left = summary[summary.model_name.eq(f"{pipe}_legacy") & summary.k.eq(k)].iloc[0]
            right = old_validation[old_validation.model_name.eq(f"{pipe}_{BASIS}") & old_validation.k.eq(k)].iloc[0]
            np.testing.assert_allclose(left[["mae", "active_mae"]].to_numpy(dtype=float),
                right[["mae", "active_mae"]].to_numpy(dtype=float), rtol=0, atol=1e-12)
    print(f"{run.name}: source-validation representations selected", flush=True)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth[support]
    frame = add_selected(make_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
        labels, split, months, active, role="test"), selection)
    previous = pd.read_parquet(prior / "predictions.parquet")
    write_json(run / "reference_checks.json", invariance_checks(frame, previous, role="test"))
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth[frame.cell.to_numpy()]
    if set(frame.model_name) != set(MODELS) or not np.isfinite(frame.select_dtypes(include="number")).all().all():
        raise ValueError("Invalid chemical-support product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    (run / "full_grid.parquet").write_bytes((prior / "full_grid.parquet").read_bytes())
    files = ["representations.npz", "basis_definition.json", "adapters.json", "mixers.json", "basis_selection.json"]
    for name in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, name, config, runtime, files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-started})
    completed = ["config.json", *files, "source_validation.csv", "reference_checks.json", "timing.json",
        "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json"]
    bind_files(run, "complete.json", [run / name for name in completed], config)
    print(f"{run.name}: complete in{time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime)
            gc.collect()


if __name__ == "__main__":
    main()
