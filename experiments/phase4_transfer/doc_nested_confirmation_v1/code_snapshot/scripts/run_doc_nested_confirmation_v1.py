"""Refit DOC chemistry and nested support calibration on new station roles."""
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
from doc_chemistry_confirmation_chemical import (
    fit_chemistry_and_calibration,
    retained_chemical_recipe,
)
from doc_chemistry_confirmation_initial import (
    fit_initial_and_basis,
    retained_initial_recipe,
)
from doc_chemistry_confirmation_native import fit_native_and_ecology
from run_doc_auxiliary_chemistry_v1 import AUX_PATHS
from run_doc_chemistry_confirmation_v1 import (
    EXECUTION_SCRIPTS as BASE_EXECUTION_SCRIPTS,
)
from run_doc_chemistry_confirmation_v1 import freeze_split
from run_doc_chemistry_support_v1 import add_selected, make_panels
from run_doc_daily_hydro_memory_v1 import load_daily_pack
from run_doc_nested_chemistry_v1 import (
    LEGACY,
    MODELS,
    NESTED,
    REFERENCES,
    build_products,
    completed_support_predictions,
)
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import (
    bind_files,
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.nested_chemical_adapter import (
    NestedChemicalAdapter,
    NestedChemicalEpisode,
)
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_nested_confirmation_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DAILY_ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
PARTITIONS, SEEDS, KS = (342, 343, 344), (42, 43, 44), (0, 1, 3, 5)
EXECUTION_SCRIPTS = (*BASE_EXECUTION_SCRIPTS, "run_doc_nested_confirmation_v1",
                     "run_doc_nested_chemistry_v1", "run_doc_chemical_kernel_v1")


def retained_nested_recipe(partition):
    return {"ridge_values": [.1, 1., 10., 100.], "strength_values": [0., .25, .5, 1.],
        "selection_folds": 5, "fold_seed": 4100 + int(partition),
        "shared_selection_k": [3, 5], "selection_role": "source_validation",
        "selection_weights": "K equal then station equal", "source_role": "source_training",
        "chemical_dimension": 2, "scale_floor": 1e-8,
        "response": "centered log1p residual after completed frozen legacy support calibration",
        "parent_parameters_reselected_by_nested_branch": False,
        "low_k_policy": "exact complete legacy prediction at K0 and K1",
        "missing_auxiliary_policy": "exact legacy prediction; ignore unavailable chemical supports",
        "validation_fold_scope": "conditional adapter diagnostics; parent previously used all source validation"}


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in (*[f"scripts/{name}.py" for name in EXECUTION_SCRIPTS], str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("Execution source changed; retain this batch and use a new root")
        return verify_runtime_snapshot(root)
    write_json(path, snapshot)
    for name in snapshot:
        destination = root / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def label_view(dataset, split, *, role):
    """Return val-only or reserved target-support-only DOC, never target query."""
    labels = np.full(np.asarray(dataset["y"]).size, np.nan)
    if role == "val":
        cells = split["val"]
    elif role == "test":
        cells, query = support_query_cells(split, target_role="test", k=5,
                                           n_months=np.asarray(dataset["y"]).shape[1])
        if np.intersect1d(cells, query).size:
            raise ValueError("Target support/query overlap")
    else:
        raise ValueError("Nested calibration role must be val or test")
    labels[cells] = np.asarray(dataset["y"]).ravel()[cells]
    return labels


def fresh_state(run):
    """Read only this run's newly fitted stage outputs, never an old parent."""
    directory = Path(run) / "chemical"
    with np.load(directory / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in ("legacy", "masks_aug", "chemistry_aug")}
        coordinates = {mode: saved[f"{mode}_chemical"].copy() for mode in NESTED}
        active, source_ids = saved["active"].copy(), saved["source_station_ids"].copy()
    states = {name: json.loads((directory / f"{name}.json").read_text()) for name in (
        "adapters", "mixers", "legacy_adapters", "legacy_mixers", "basis_selection")}
    return shapes, coordinates, active, source_ids, states


def reference_validation(full, labels, split, months, shapes, active, states):
    adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in states["adapters"].items()}
    mixers = {name: SupportAwareResidualTransfer.from_dict(state) for name, state in states["mixers"].items()}
    bases = {name: full[f"{name}_pred"].to_numpy() for name in ("neural_chemistry", "tree_chemistry")}
    frame = add_selected(make_panels(full, bases, shapes, adapters, mixers,
        states["legacy_adapters"], states["legacy_mixers"], labels, split, months, active, role="val"),
        states["basis_selection"])
    return frame[frame.model_name.isin(REFERENCES)].copy()


def role_episodes(full, references, labels, split, months, legacy, states, coordinates, active, *, role):
    """Construct episodes; test query DOC is deliberately replaced by NaN."""
    result = {}
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        parent = references[references.model_name.eq(LEGACY) & references.k.eq(k)].sort_values("cell")
        np.testing.assert_array_equal(parent.cell.to_numpy(), query)
        candidate, support_prediction = completed_support_predictions(full, legacy, states,
            labels, support, query, months=months, k=k)
        np.testing.assert_allclose(candidate[active[query]], parent.y_pred.to_numpy()[active[query]],
                                   rtol=0, atol=1e-12)
        query_values = labels[query] if role == "val" else np.full(len(query), np.nan)
        result[k] = NestedChemicalEpisode(k, query, query_values, parent.y_pred.to_numpy(), support,
            labels[support], support_prediction, coordinates[query], coordinates[support],
            active[query], active[support])
    return result


def complete_frame(groups):
    frame = pd.concat(groups, ignore_index=True)
    for field in ("chemical_delta", "chemical_support_count", "conditional_fold",
                  "chemical_coefficient_0", "chemical_coefficient_1"):
        frame[field] = frame[field].fillna(-1 if field == "conditional_fold" else 0)
    for field, parent in (("legacy_y_pred", "y_pred"), ("legacy_adaptation_delta", "adaptation_delta"),
                          ("legacy_candidate_y_pred", "candidate_y_pred")):
        frame[field] = frame[field].fillna(frame[parent])
    return frame


def fit_nested_stage(run, dataset, split, partition, result):
    """Freeze fresh validation choices before opening reserved target support."""
    full = result["full_grid"]
    months = np.asarray(dataset["y"]).shape[1]
    shapes, coordinates, active, source_ids, states = fresh_state(run)
    if np.intersect1d(source_ids, np.unique(np.r_[split["val"], split["test"]] // months)).size:
        raise ValueError("Source coordinate normalization overlaps held stations")
    labels = label_view(dataset, split, role="val")
    references = reference_validation(full, labels, split, months, shapes, active, states)
    validation, cv, models = [references.copy()], [references.copy()], {}
    for mode, name in NESTED.items():
        episodes = role_episodes(full, references, labels, split, months, shapes["legacy"], states,
                                 coordinates[mode], active, role="val")
        source_grid = coordinates[mode].reshape(-1, months, 2)[source_ids]
        source_active = active.reshape(-1, months)[source_ids]
        model = NestedChemicalAdapter(months, fold_seed=4100 + partition).fit_coordinates(
            source_grid, source_active, source_role="source_training").fit(
            [episodes[3], episodes[5]], selection_role="source_validation")
        write_json(Path(run) / f"nested_{mode}.json", model.to_dict())
        validation.append(build_products(model, episodes, references, name, cross_validation=False))
        cv.append(build_products(model, episodes, references, name, cross_validation=True))
        models[mode] = model
        print(f"{Path(run).name}/{mode}: {model.selection_}", flush=True)
    # All fitting and source-validation selection is now complete. The target
    # branch gets five reserved cells per target station, and no query truth.
    target_labels = label_view(dataset, split, role="test")
    target_references = result["predictions"][result["predictions"].model_name.isin(REFERENCES)].copy()
    target = [target_references.copy()]
    for mode, name in NESTED.items():
        episodes = role_episodes(full, target_references, target_labels, split, months, shapes["legacy"],
                                 states, coordinates[mode], active, role="test")
        target.append(build_products(models[mode], episodes, target_references, name, cross_validation=False))
    return complete_frame(target), complete_frame(validation), complete_frame(cv)


def validate_products(full, frame, dataset, split):
    """Check aligned six curves and fixed queries before final truth export."""
    n, months = tuple(dataset["y_mask"].shape)
    np.testing.assert_array_equal(full.cell.to_numpy(), np.arange(n * months))
    if "y_true" in full or "y_true" in frame:
        raise ValueError("Model helpers must return products without query truth")
    if (set(frame.model_name) != set(MODELS) or set(frame.k) != set(KS)
            or frame.duplicated(["model_name", "k", "cell"]).any()):
        raise ValueError("Incomplete or repeated fixed nested model/K experiment")
    _, query = support_query_cells(split, target_role="test", k=0, n_months=months)
    for (model, k), panel in frame.groupby(["model_name", "k"], sort=False):
        np.testing.assert_array_equal(np.sort(panel.cell.to_numpy()), query, err_msg=f"{model},K={k}")
    for table in (full, frame):
        cells = table.cell.to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(table.station.astype(str), np.asarray(dataset["site_no"], str)[cells // months])
        np.testing.assert_array_equal(table.month.astype(str), np.asarray(dataset["months"], str)[cells % months])
        if not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
            raise FloatingPointError("Nonfinite nested confirmation prediction/components")
    return query


def validation_summary(frame, labels):
    rows = []
    for (model, k), panel in frame.groupby(["model_name", "k"], sort=True):
        error = np.abs(panel.y_pred.to_numpy() - labels[panel.cell.to_numpy()])
        active = panel.aux_available.to_numpy(dtype=bool)
        rows.append({"model_name": model, "k": int(k), "n": len(panel), "n_active": int(active.sum()),
                     "mae": float(error.mean()), "active_mae": float(error[active].mean()) if active.any() else None})
    return pd.DataFrame(rows)


def run_one(root, dataset, partition, seed, runtime, daily_pack, args):
    started = time.monotonic()
    split, protocol, mask_path = freeze_split(root, dataset, partition)
    run = root / "runs" / f"split{partition}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    daily, daily_metadata, daily_binding = daily_pack
    _, query = support_query_cells(split, target_role="test", k=0, n_months=np.asarray(dataset["y"]).shape[1])
    train_y = np.asarray(dataset["y"]).ravel()[split["train"]]
    config = {"experiment": "doc_nested_confirmation_v1", "split_seed": partition, "seed": seed,
        "runtime_snapshot_hash": runtime, "dataset_path": str(DATASET), "dataset_hash": sha256_file(DATASET),
        "mask_path": str(mask_path), "mask_hash": sha256_file(mask_path),
        "protocol_path": str(mask_path.with_suffix(".json")), "protocol_hash": sha256_file(mask_path.with_suffix(".json")),
        "study_plan_hash": sha256_file(ROOT / "study_plan.md"), "smoke": bool(args.smoke),
        "n_jobs": args.n_jobs, "torch_threads": args.torch_threads,
        "initial_recipe": retained_initial_recipe(smoke=args.smoke),
        "native_recipe": {"profile_initializer": "interaction_tuned", "initializer_epochs": 1 if args.smoke else 30,
            "off_epochs": 1 if args.smoke else 120, "patience": 5, "extra_dim": 38,
            "interaction_indices": [0, 2, 4, 28, 30, 31, 32], "tail_weight": 2,
            "encoder_mode": "last_self_ecology", "encoder_learning_rate": 1e-5},
        "chemical_recipe": retained_chemical_recipe(smoke=args.smoke),
        "nested_recipe": retained_nested_recipe(partition), "target_analyte": "doc", "target_transform": "log1p",
        "models": list(MODELS), "k_values": list(KS), "query_cells": len(query),
        "q90_threshold_train": float(np.quantile(train_y, .9)), "source_station_ids": protocol["station_roles"]["train"],
        "station_roles": protocol["station_roles"],
        "auxiliary_paths": {name: str(path) for name, path in AUX_PATHS.items()},
        "auxiliary_hashes": {name: sha256_file(path) for name, path in AUX_PATHS.items()},
        "auxiliary_provenance_hashes": {name: sha256_file(path.with_suffix(".provenance.json")) for name, path in AUX_PATHS.items()},
        **daily_binding, "inference_roles": ["train"], "selection_role": "source_validation",
        "study_role": "fresh random-station nested calibration confirmation on the same ST357 cohort",
        "adaptation_mode": "retrospective station support calibration; support may postdate query",
        "source_neural_is_oof": False, "source_forest_is_oof": True,
        "spatial_message_policy": "retained empty-edge self path", "historical_fitted_models_reused": False,
        "full_grid_role": "base components; six station support curves exported on fixed observed queries"}
    path = run / "config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        timestamp = previous.pop("started_at")
        if previous != config:
            raise ValueError(f"Run configuration changed: {run}")
        config["started_at"] = timestamp
    else:
        config["started_at"] = datetime.now(timezone.utc).isoformat()
        write_json(path, config)
    if (run / "complete.json").exists():
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified complete; no refit", flush=True)
        return

    def progress(row):
        row = {"run": run.name, "updated_at": datetime.now(timezone.utc).isoformat(), **row}
        write_json(root / "progress.json", row)
        print(json.dumps(row, allow_nan=False), flush=True)

    progress({"stage": "initial_and_support_basis", "status": "training fresh source models"})
    initial = fit_initial_and_basis(run, dataset, split, seed, n_jobs=args.n_jobs,
                                   torch_threads=args.torch_threads, smoke=args.smoke)
    progress({"stage": "native_and_ecology"})
    native = fit_native_and_ecology(run, dataset, split, seed, initial, daily=daily,
        daily_metadata=daily_metadata, smoke=args.smoke, progress=progress, runtime_snapshot_hash=runtime)
    progress({"stage": "chemical_heads_and_legacy_support"})
    result = fit_chemistry_and_calibration(run, dataset, split, seed, initial, native, smoke=args.smoke)
    progress({"stage": "nested_chemical_support"})
    frame, validation, cv = fit_nested_stage(run, dataset, split, partition, result)
    full = result["full_grid"].copy()
    validate_products(full, frame, dataset, split)
    labels = label_view(dataset, split, role="val")
    for table in (full, frame, validation, cv):
        table["split_seed"], table["seed"] = partition, seed
    # This final evaluation/export layer is the first extraction of target
    # query truth. All source-only choices and support predictions are fixed.
    frame["y_true"] = np.asarray(dataset["y"]).ravel()[frame.cell.to_numpy(dtype=int)]
    validation["y_true"] = labels[validation.cell.to_numpy(dtype=int)]
    cv["y_true"] = labels[cv.cell.to_numpy(dtype=int)]
    full.to_parquet(run / "full_grid.parquet", index=False)
    frame.to_parquet(run / "predictions.parquet", index=False)
    validation.to_parquet(run / "validation.parquet", index=False)
    cv.to_parquet(run / "cv_validation.parquet", index=False)
    validation_summary(validation, labels).to_csv(run / "source_validation.csv", index=False)
    write_json(run / "stage_summary.json", {"chemical": result["metadata"],
        "nested": retained_nested_recipe(partition), "fresh_refit": True})
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic() - started})
    directories = ("initial", "source_oof", "basis", "native", "chemical")
    model_files = sorted(str(item.relative_to(run)) for directory in directories
                         for item in (run / directory).rglob("*") if item.is_file())
    model_files += ["nested_chemistry.json", "nested_masks.json"]
    products = ("full_grid.parquet", "predictions.parquet", "validation.parquet", "cv_validation.parquet")
    for name in products:
        bind_product(run, name, config, runtime, model_files)
    final_files = ["config.json", "source_validation.csv", "stage_summary.json", "timing.json", *model_files]
    for name in products:
        final_files.extend([name, str(Path(name).with_suffix(".meta.json"))])
    bind_files(run, "complete.json", [run / name for name in final_files], config)
    progress({"stage": "complete", "elapsed_seconds": time.monotonic() - started,
              "prediction_rows": len(frame), "grid_rows": len(full)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(PARTITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.root == ROOT:
        args.root = ROOT / "_smoke"
    if args.smoke:
        args.split_seeds, args.seeds = [PARTITIONS[0]], [SEEDS[0]]
    if not set(args.split_seeds).issubset(PARTITIONS) or not set(args.seeds).issubset(SEEDS):
        raise ValueError("Nested confirmation uses only the fresh written partitions/training seeds")
    if args.n_jobs < 1 or args.torch_threads < 1:
        raise ValueError("Worker counts must be positive")
    torch.set_num_threads(args.torch_threads)
    args.root.mkdir(parents=True, exist_ok=True)
    runtime = freeze_runtime(args.root)
    dataset = torch.load(DATASET, weights_only=False)
    daily_pack = load_daily_pack(DAILY_ROOT, sha256_file(DATASET), tuple(dataset["y"].shape))
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, dataset, partition, seed, runtime, daily_pack, args)
            gc.collect()
    write_json(args.root / "batch_complete.json", {"runtime_snapshot_hash": runtime,
        "split_seeds": args.split_seeds, "seeds": args.seeds, "smoke": args.smoke,
        "runs": {f"split{partition}_seed{seed}": sha256_file(args.root / "runs" / f"split{partition}_seed{seed}" / "complete.json")
                 for partition in args.split_seeds for seed in args.seeds}})


if __name__ == "__main__":
    main()
