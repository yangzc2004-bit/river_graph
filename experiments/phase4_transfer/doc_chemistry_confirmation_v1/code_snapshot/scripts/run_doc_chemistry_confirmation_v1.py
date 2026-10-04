"""Refit the accepted DOC chemistry recipe on fresh station-role partitions."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from doc_chemistry_confirmation_chemical import MODELS, fit_chemistry_and_calibration
from doc_chemistry_confirmation_initial import (
    fit_initial_and_basis,
    retained_initial_recipe,
)
from doc_chemistry_confirmation_native import fit_native_and_ecology
from run_doc_auxiliary_chemistry_v1 import AUX_PATHS
from run_doc_daily_hydro_memory_v1 import load_daily_pack
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import (
    bind_files,
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import (
    build_unified_spatial_split,
    support_query_cells,
)

ROOT = Path("experiments/phase4_transfer/doc_chemistry_confirmation_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DAILY_ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
PARTITIONS = (242, 243, 244)
SEEDS = (42, 43, 44)
KS = (0, 1, 3, 5)
EXECUTION_SCRIPTS = (
    "run_ladder", "run_doc_chemistry_confirmation_v1",
    "doc_chemistry_confirmation_initial", "doc_chemistry_confirmation_native",
    "doc_chemistry_confirmation_chemical", "run_unified_doc_spatial",
    "run_unified_doc_spatial_v2", "run_doc_tail_residual_v1",
    "run_doc_auxiliary_chemistry_v1", "run_doc_chemistry_support_v1",
    "run_doc_daily_hydro_memory_v1", "run_doc_daily_hydro_readout_v1",
    "run_doc_daily_hydro_support_basis_v1", "run_doc_distribution_head_v1",
    "run_doc_ecological_transfer_v1", "run_doc_ecological_transfer_v2",
)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in (*[f"scripts/{name}.py" for name in EXECUTION_SCRIPTS], str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("Execution source changed; preserve this batch and use a new root")
        return verify_runtime_snapshot(root)
    write_json(path, snapshot)
    for name in snapshot:
        destination = root / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def freeze_split(root, dataset, partition):
    split, protocol = build_unified_spatial_split(np.asarray(dataset["y_mask"]), seed=partition)
    path = root / "masks" / f"split{partition}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_suffix(".json")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(split):
                raise ValueError("Cached partition role schema changed")
            for role, cells in split.items():
                np.testing.assert_array_equal(saved[role], cells)
        if json.loads(metadata.read_text()) != protocol:
            raise ValueError("Cached station protocol changed")
    else:
        np.savez_compressed(path, **split)
        write_json(metadata, protocol)
    return split, protocol, path


def validate_products(full, frame, dataset, split):
    """Validate all fixed query panels after prediction choices have been made."""
    shape = tuple(dataset["y_mask"].shape)
    n, months = shape
    np.testing.assert_array_equal(full.cell.to_numpy(), np.arange(n * months))
    if "y_true" in full or "y_true" in frame:
        raise ValueError("Training helpers must return products without query truth")
    if set(frame.model_name) != set(MODELS) or set(frame.k) != set(KS):
        raise ValueError("Incomplete fixed model/K experiment")
    _, query = support_query_cells(split, target_role="test", k=0, n_months=months)
    for (model, k), panel in frame.groupby(["model_name", "k"], sort=False):
        np.testing.assert_array_equal(np.sort(panel.cell.to_numpy()), np.sort(query), err_msg=f"{model}, K={k}")
    for table in (full, frame):
        cells = table.cell.to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(table.station.astype(str), np.asarray(dataset["site_no"], str)[cells // months])
        np.testing.assert_array_equal(table.month.astype(str), np.asarray(dataset["months"], str)[cells % months])
        if not np.isfinite(table.select_dtypes(include="number").to_numpy()).all():
            raise FloatingPointError("Nonfinite confirmation prediction/components")
    return query


def run_one(root, dataset, partition, seed, runtime, daily_pack, args):
    started = time.monotonic()
    split, protocol, mask_path = freeze_split(root, dataset, partition)
    run = root / "runs" / f"split{partition}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    daily, daily_metadata, daily_binding = daily_pack
    _, query = support_query_cells(split, target_role="test", k=0, n_months=dataset["y"].shape[1])
    train_y = np.asarray(dataset["y"]).ravel()[split["train"]]
    config = {
        "experiment": "doc_chemistry_confirmation_v1", "split_seed": partition, "seed": seed,
        "runtime_snapshot_hash": runtime, "dataset_path": str(DATASET), "dataset_hash": sha256_file(DATASET),
        "mask_path": str(mask_path), "mask_hash": sha256_file(mask_path),
        "protocol_path": str(mask_path.with_suffix(".json")), "protocol_hash": sha256_file(mask_path.with_suffix(".json")),
        "study_plan_hash": sha256_file(ROOT / "study_plan.md"),
        "smoke": bool(args.smoke), "n_jobs": args.n_jobs, "torch_threads": args.torch_threads,
        "initial_recipe": retained_initial_recipe(smoke=args.smoke),
        "native_recipe": {"profile_initializer": "interaction_tuned", "initializer_epochs": 1 if args.smoke else 30,
            "off_epochs": 1 if args.smoke else 120, "patience": 5, "extra_dim": 38,
            "interaction_indices": [0, 2, 4, 28, 30, 31, 32], "tail_weight": 2,
            "encoder_mode": "last_self_ecology", "encoder_learning_rate": 1e-5},
        "chemical_recipe": {"epochs": 1 if args.smoke else 120, "patience": 10,
            "modes": ["no_aux", "masks", "chemistry"], "head_parameters": 1111,
            "pca_components": 2, "basis_names": ["legacy", "masks_aug", "chemistry_aug"],
            "basis_selection": "source-validation active MAE, legacy/masks/chemistry tie order"},
        "target_analyte": "doc", "target_transform": "log1p", "models": list(MODELS), "k_values": list(KS),
        "query_cells": len(query), "q90_threshold_train": float(np.quantile(train_y, .9)),
        "source_station_ids": protocol["station_roles"]["train"],
        "auxiliary_paths": {name: str(path) for name, path in AUX_PATHS.items()},
        "auxiliary_hashes": {name: sha256_file(path) for name, path in AUX_PATHS.items()},
        "auxiliary_provenance_hashes": {name: sha256_file(path.with_suffix(".provenance.json")) for name, path in AUX_PATHS.items()},
        **daily_binding,
        "inference_roles": ["train"], "selection_role": "source_validation",
        "study_role": "fresh random-station role confirmation on the same ST357 cohort",
        "adaptation_mode": "retrospective station support calibration; support may postdate query",
        "source_neural_is_oof": False, "source_forest_is_oof": True,
        "spatial_message_policy": "retained empty-edge self path", "historical_fitted_models_reused": False,
        "full_grid_role": "base components; station support curves exported on fixed observed queries",
    }
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
    progress({"stage": "chemical_heads_and_station_support"})
    result = fit_chemistry_and_calibration(run, dataset, split, seed, initial, native, smoke=args.smoke)
    full, frame = result["full_grid"], result["predictions"]
    validate_products(full, frame, dataset, split)
    # Only this final evaluation layer adds hidden query truths. Helpers use
    # source/validation and explicitly reserved support labels for fitting.
    frame["split_seed"], frame["seed"] = partition, seed
    full["split_seed"], full["seed"] = partition, seed
    frame["y_true"] = np.asarray(dataset["y"]).ravel()[frame.cell.to_numpy(dtype=int)]
    full.to_parquet(run / "full_grid.parquet", index=False)
    frame.to_parquet(run / "predictions.parquet", index=False)
    result["source_validation"].to_csv(run / "source_validation.csv", index=False)
    write_json(run / "stage_summary.json", result["metadata"])
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic() - started})
    directories = ("initial", "source_oof", "basis", "native", "chemical")
    model_files = sorted(str(item.relative_to(run)) for directory in directories
                         for item in (run / directory).rglob("*") if item.is_file())
    for name in ("full_grid.parquet", "predictions.parquet"):
        bind_product(run, name, config, runtime, model_files)
    final_files = ["config.json", "full_grid.parquet", "full_grid.meta.json", "predictions.parquet",
                   "predictions.meta.json", "source_validation.csv", "stage_summary.json", "timing.json", *model_files]
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
        raise ValueError("Confirmation uses only the written fresh partitions and training seeds")
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
