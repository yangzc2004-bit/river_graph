"""Fit bounded support-residual interpolation on source validation only."""
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
from run_doc_chemistry_support_v1 import (
    PIPELINES,
    add_selected,
    make_panels,
    validation_summary,
)
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import read_source

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.chemical_residual_kernel import (
    chemical_residual_kernel,
    source_chemical_distance_scale,
)
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_chemical_kernel_v1")
PRIOR = Path("experiments/phase4_transfer/doc_chemistry_support_v1")
KS = (0, 1, 3, 5)
MODES = ("masks", "chemistry")
ETA_VALUES = (0., .25, .5, 1.)
BANDWIDTH_VALUES = (.5, 1., 2.)
MODELS = tuple(f"{pipe}_{mode}" for pipe in PIPELINES for mode in ("parent", *MODES))


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_auxiliary_chemistry_v1.py", "scripts/run_doc_daily_hydro_memory_v1.py",
                 "scripts/run_doc_daily_hydro_readout_v1.py", "scripts/run_doc_chemistry_support_v1.py",
                 "scripts/run_doc_chemical_kernel_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution changed; preserve this pilot and use a new root")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def linear_preclip(base, basis, labels, support, query, *, months, k, alpha, ridge_strength):
    """Reproduce all-K linear correction at both support and query, preclip.

    Query labels are not an input to the calculation: ``labels`` is indexed
    only at support. The later validation scoring receives query truth.
    """
    query = np.asarray(query, dtype=np.int64)
    support = np.asarray(support, dtype=np.int64)
    if np.intersect1d(query, support).size:
        raise ValueError("Support and query must be disjoint")
    base = np.asarray(base, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    qz, sz = np.log1p(base[query]), np.log1p(base[support])
    if k == 0:
        if len(support):
            raise ValueError("K0 support must be empty")
        return qz, sz, np.empty(0, dtype=np.float64)
    sr = np.log1p(labels[support])-sz
    if not np.isfinite(sr).all():
        raise ValueError("Support labels/base must be finite")
    ridge = np.inf if ridge_strength == "infinity" else float(ridge_strength)
    for station in np.unique(query // months):
        q, s = query // months == station, support // months == station
        if s.sum() != k:
            raise ValueError("Every validation query station requires exactly K support rows")
        mean_residual = float(sr[s].mean())
        qz[q] += alpha*mean_residual
        sz[s] += alpha*mean_residual
        if k > 1 and np.isfinite(ridge):
            mean_basis = basis[support[s]].mean(0)
            centered = basis[support[s]]-mean_basis
            gram = centered.T @ centered/k + ridge*np.eye(basis.shape[1])
            coefficient = np.linalg.solve(gram, centered.T @ (sr[s]-mean_residual)/k)
            qz[q] += (basis[query[q]]-mean_basis) @ coefficient
            sz[s] += centered @ coefficient
    remainder = np.log1p(labels[support])-sz
    if not all(np.isfinite(values).all() for values in (qz, sz, remainder)):
        raise FloatingPointError("Nonfinite fixed linear support fit")
    return qz, sz, remainder


def selected_linear(pipe, representation, k, full, adapters, mixers):
    name = f"{pipe}_{representation}"
    if pipe.endswith("integrated"):
        model = mixers[name]
        base = model.selected_base(full.context_pred.to_numpy(), full.neural_chemistry_pred.to_numpy(),
                                   full.ecological_memory.to_numpy(), k=k)
        choice = model.to_dict()["selection_by_k"][str(k)]
    else:
        base = full[f"{pipe}_pred"].to_numpy()
        choice = {**adapters[name].to_dict()["selection_by_k"][str(k)], "gamma": 0.}
    return base, choice


def kernel_correction(coordinates, active, remainder, support, query, *, months, scale, bandwidth, eta):
    output = np.zeros(len(query), dtype=np.float64)
    donor_count = np.zeros(len(query), dtype=np.int64)
    for station in np.unique(query // months):
        q, s = query // months == station, support // months == station
        donor_count[q] = int(active[support[s]].sum())
        output[q] = chemical_residual_kernel(coordinates[query[q]], coordinates[support[s]],
            active[query[q]], active[support[s]], remainder[s], source_distance_scale=scale,
            bandwidth_scale=bandwidth, eta=eta)
    return output, donor_count


def apply_correction(parent_prediction, preclip_z, correction):
    """Add to preclip log predictions; exact-zero rows retain the parent."""
    result = np.asarray(parent_prediction).copy()
    changed = np.asarray(correction) != 0
    with np.errstate(over="ignore", invalid="ignore"):
        result[changed] = np.maximum(0., np.expm1(preclip_z[changed]+correction[changed]))
    if not np.isfinite(result).all():
        raise FloatingPointError("Kernel correction produced nonfinite native predictions")
    return result


def score(prediction, truth, active):
    errors = np.abs(prediction-truth)
    return {"n": len(truth), "n_active": int(active.sum()), "mae": float(errors.mean()),
            "active_mae": float(errors[active].mean())}


def run_one(root, prior_root, partition, seed, runtime):
    started = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, _, dataset, split, full = read_source(prior)
    if pc["split_seed"] != partition or pc["seed"] != seed:
        raise ValueError("Parent identities differ from the requested package")
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["prior_completion_hash"] != sha256_file(prior / "complete.json")
                or config["torch_threads"] != torch.get_num_threads()):
            raise ValueError("Cached kernel pilot settings or parent changed")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing source-validation product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    labels = np.full(np.prod(shape), np.nan)
    # This is the only DOC label extraction. The entire target and source train
    # label ranges remain NaN in every downstream calculation.
    labels[split["val"]] = np.asarray(dataset["y"]).ravel()[split["val"]]
    del dataset
    if not np.isnan(labels[split["test"]]).all():
        raise ValueError("Target labels must remain absent")
    with np.load(prior / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in ("legacy", "masks_aug", "chemistry_aug")}
        coordinates = {mode: saved[f"{mode}_chemical"].copy() for mode in MODES}
        active, source_ids = saved["active"].copy(), saved["source_station_ids"].copy()
    if active.dtype != bool or active.shape != (np.prod(shape),):
        raise ValueError("Frozen auxiliary availability must align with the full grid")
    if np.intersect1d(source_ids, np.unique(np.r_[split["val"], split["test"]]//months)).size:
        raise ValueError("Source distance fitting cannot include held stations")
    distances = {mode: source_chemical_distance_scale(coordinates[mode].reshape(*shape, 2),
                    active.reshape(shape), source_ids, source_role="source_training") for mode in MODES}
    adapter_states = json.loads((prior / "adapters.json").read_text())
    mixer_states = json.loads((prior / "mixers.json").read_text())
    adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in adapter_states.items()}
    mixers = {name: SupportAwareResidualTransfer.from_dict(state) for name, state in mixer_states.items()}
    selection = json.loads((prior / "basis_selection.json").read_text())
    decoder = Path(pc["prior_run"])
    decoder_config = json.loads((decoder / "config.json").read_text())
    verify_files(decoder, "complete.json", decoder_config)
    if sha256_file(decoder / "complete.json") != pc["prior_completion_hash"]:
        raise ValueError("Frozen decoder parent changed")
    old_adapters = json.loads((decoder / "adapters.json").read_text())
    old_mixers = json.loads((decoder / "mixers.json").read_text())
    bases = {pipe: full[f"{pipe}_pred"].to_numpy() for pipe in ("neural_chemistry", "tree_chemistry")}
    parent_panels = add_selected(make_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
        labels, split, months, active, role="val"), selection)
    summary = validation_summary(parent_panels, labels)
    expected = pd.read_csv(prior / "source_validation.csv")
    joined = summary.merge(expected, on=["model_name", "k"], suffixes=("", "_saved"), validate="one_to_one")
    if len(joined) != len(expected):
        raise ValueError("Parent validation comparison is incomplete")
    np.testing.assert_allclose(joined[["mae", "active_mae"]], joined[["mae_saved", "active_mae_saved"]], rtol=0, atol=1e-12)
    _, fixed_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    config = {"experiment": "doc_chemical_kernel_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "decoder_run": str(decoder), "decoder_completion_hash": sha256_file(decoder / "complete.json"),
        **{key: pc[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train")},
        "parent_bindings": {name: sha256_file(prior / name) for name in ("full_grid.parquet", "representations.npz",
            "adapters.json", "mixers.json", "basis_selection.json", "source_validation.csv")},
        "target_analyte": "doc", "correction_space": "log1p DOC", "pipelines": list(PIPELINES), "models": list(MODELS),
        "kernel_modes": list(MODES), "k_values": list(KS), "eta_values": list(ETA_VALUES),
        "bandwidth_values": list(BANDWIDTH_VALUES), "max_source_months_per_station": 24,
        "source_station_ids": source_ids.tolist(), "selection_role": "source_validation",
        "selection_score": "active source-validation native MAE", "validation_query_cells": len(fixed_query),
        "target_evaluation": False, "new_neural_fits": 0, "new_forest_fits": 0, "linear_calibrators_refitted": False,
        "parent_representation_policy": "retain frozen selection separately for each pipeline and K",
        "remainder_policy": "subtract complete all-K linear correction at support before clipping; center active remainder within kernel",
        "kernel_policy": "normalized RBF; source-only median distance; no Gram solve or target distance fitting",
        "torch_threads": torch.get_num_threads(), "study_role": "source-validation support interpolation development only"}
    write_json(run / "config.json", config)
    write_json(run / "distances.json", distances)
    candidates, candidate_stations, products, checks = [], [], [], []
    choices = {pipe: {mode: {} for mode in MODES} for pipe in PIPELINES}
    for pipe in PIPELINES:
        for k in KS:
            support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
            np.testing.assert_array_equal(query, fixed_query)
            parent = parent_panels[parent_panels.model_name.eq(f"{pipe}_selected") & parent_panels.k.eq(k)].sort_values("cell").copy()
            np.testing.assert_array_equal(parent.cell, query)
            representation = selection["choices"][pipe][str(k)]["basis"]
            base, linear_choice = selected_linear(pipe, representation, k, full, adapters, mixers)
            qz, sz, remainder = linear_preclip(base, shapes[representation], labels, support, query,
                months=months, k=k, alpha=linear_choice["alpha"], ridge_strength=linear_choice["ridge_strength"])
            reconstructed = np.maximum(0, np.expm1(qz))
            np.testing.assert_allclose(reconstructed, parent.candidate_y_pred.to_numpy(), rtol=0, atol=1e-12)
            final = reconstructed.copy()
            final[~active[query]] = parent.y_pred.to_numpy()[~active[query]]
            np.testing.assert_allclose(final, parent.y_pred.to_numpy(), rtol=0, atol=1e-12)
            checks.append({"pipeline": pipe, "k": k, "parent_basis": representation,
                "alpha": linear_choice["alpha"], "ridge_strength": linear_choice["ridge_strength"], "gamma": linear_choice["gamma"],
                "query_candidate_max_abs_difference": float(np.abs(reconstructed-parent.candidate_y_pred.to_numpy()).max()),
                "query_final_max_abs_difference": float(np.abs(final-parent.y_pred.to_numpy()).max()),
                "support_rows": len(support), "query_rows": len(query),
                "support_preclip_log_min": float(sz.min()) if len(sz) else None,
                "support_remainder_abs_mean": float(np.abs(remainder).mean()) if len(remainder) else 0.,
                "support_preclip_used": True, "all_k_support_moments": True})
            parent["model_name"], parent["pipeline"], parent["kernel_mode"] = f"{pipe}_parent", pipe, "parent"
            parent["parent_basis"], parent["parent_y_pred"], parent["parent_preclip_z"] = representation, parent.y_pred, qz
            parent["kernel_log_correction"], parent["eta"], parent["bandwidth_scale"] = 0., 0., 1.
            _, donor_count = kernel_correction(coordinates["chemistry"], active, remainder, support, query,
                months=months, scale=distances["chemistry"]["distance_scale"], bandwidth=1., eta=0.)
            parent["active_support_count"] = donor_count
            parent["y_true"] = labels[query]
            parent.rename(columns={"candidate_y_pred": "parent_candidate_y_pred"}, inplace=True)
            products.append(parent)
            for mode in MODES:
                scale = distances[mode]["distance_scale"]
                trials, trial_products = [], {}
                for eta in ((0.,) if k < 2 else ETA_VALUES):
                    for bandwidth in ((1.,) if k < 2 else BANDWIDTH_VALUES):
                        correction, donors = kernel_correction(coordinates[mode], active, remainder, support, query,
                            months=months, scale=scale, bandwidth=bandwidth, eta=eta)
                        item = {"pipeline": pipe, "kernel_mode": mode, "k": k, "parent_basis": representation,
                            "eta": eta, "bandwidth_scale": bandwidth, "source_distance_scale": scale,
                            "valid": True, "locked": k < 2}
                        try:
                            predicted = apply_correction(parent.y_pred.to_numpy(), qz, correction)
                            item.update(score(predicted, labels[query], active[query]))
                            trial_products[eta, bandwidth] = predicted, correction, donors
                        except FloatingPointError:
                            item.update(valid=False, n=len(query), n_active=int(active[query].sum()), mae=None, active_mae=None)
                        candidates.append(item)
                        if k >= 3:
                            for station in np.unique(query // months):
                                member = query // months == station
                                member_active = member & active[query]
                                candidate_stations.append({
                                    **{key: item[key] for key in ("pipeline", "kernel_mode", "k", "parent_basis", "eta", "bandwidth_scale", "source_distance_scale", "valid")},
                                    "station_index": int(station), "station": str(parent.loc[member, "station"].iloc[0]),
                                    "n": int(member.sum()), "n_active": int(member_active.sum()),
                                    "error_sum": float(np.abs(predicted[member]-labels[query[member]]).sum()) if item["valid"] else None,
                                    "active_error_sum": float(np.abs(predicted[member_active]-labels[query[member_active]]).sum()) if item["valid"] else None,
                                    "split_seed": partition, "seed": seed})
                        if item["valid"]:
                            trials.append(item)
                chosen = min(trials, key=lambda item: (item["active_mae"], item["eta"], item["bandwidth_scale"] != 1., item["bandwidth_scale"]))
                choices[pipe][mode][str(k)] = chosen
                prediction, correction, donors = trial_products[chosen["eta"], chosen["bandwidth_scale"]]
                row = parent.copy()
                row["model_name"], row["kernel_mode"] = f"{pipe}_{mode}", mode
                row["y_pred"], row["kernel_log_correction"], row["active_support_count"] = prediction, correction, donors
                row["eta"], row["bandwidth_scale"] = chosen["eta"], chosen["bandwidth_scale"]
                changed = correction != 0
                row.loc[changed, "adaptation_delta"] = np.log1p(prediction[changed])-np.log1p(row.base_pred.to_numpy()[changed])
                np.testing.assert_array_equal(prediction[~changed], parent.y_pred.to_numpy()[~changed])
                np.testing.assert_array_equal(prediction[~active[query]], parent.y_pred.to_numpy()[~active[query]])
                if k < 2 or chosen["eta"] == 0:
                    np.testing.assert_array_equal(prediction, parent.y_pred.to_numpy())
                products.append(row)
    product = pd.concat(products, ignore_index=True)
    product["split_seed"], product["seed"] = partition, seed
    product["evaluation_role"] = "source_validation"
    if (set(product.model_name) != set(MODELS) or not np.isin(product.cell, split["val"]).all()
            or np.intersect1d(product.cell, split["test"]).size
            or not np.isfinite(product.select_dtypes(include="number")).all().all()):
        raise ValueError("Kernel output must contain finite source-validation rows only")
    product.to_parquet(run / "validation_predictions.parquet", index=False)
    validation_summary(product, labels).to_csv(run / "source_validation.csv", index=False)
    pd.DataFrame(candidates).to_csv(run / "all_candidate_scores.csv", index=False)
    pd.DataFrame(candidate_stations).to_csv(run / "candidate_station_scores.csv", index=False)
    write_json(run / "kernel_selection.json", {"version": 1, "selection_role": "source_validation",
        "score": "active_validation_native_mae", "tie_rule": "smaller eta; bandwidth1; smaller bandwidth",
        "eta_values": list(ETA_VALUES), "bandwidth_values": list(BANDWIDTH_VALUES), "pipelines": choices})
    write_json(run / "linear_remainder_checks.json", {"parent_validation_mae_tolerance": 1e-12,
        "parent_validation_groups_reproduced": len(joined), "label_scope": "split.val only", "checks": checks})
    station_rows = []
    for (model, k, station), group in product.groupby(["model_name", "k", "station"], sort=True):
        residual = group.y_true.to_numpy()-group.y_pred.to_numpy()
        station_rows.append({"model_name": model, "k": k, "station": station, "n": len(group),
            "n_active": int(group.aux_available.sum()), "mae": float(np.abs(residual).mean()),
            "parent_mae": float(np.abs(group.y_true-group.parent_y_pred).mean()), "mean_residual": float(residual.mean()),
            "mean_abs_kernel_correction": float(group.kernel_log_correction.abs().mean()),
            "split_seed": partition, "seed": seed})
    pd.DataFrame(station_rows).to_csv(run / "station_responses.csv", index=False)
    model_files = ["distances.json", "kernel_selection.json", "linear_remainder_checks.json", "all_candidate_scores.csv",
                   "candidate_station_scores.csv"]
    bind_product(run, "validation_predictions.parquet", config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-started})
    completed = ["config.json", *model_files, "source_validation.csv", "station_responses.csv", "timing.json",
                 "validation_predictions.parquet", "validation_predictions.meta.json"]
    bind_files(run, "complete.json", [run / name for name in completed], config)
    print(f"{run.name}: source-validation kernel pilot complete in {time.monotonic()-started:.1f}s", flush=True)


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
