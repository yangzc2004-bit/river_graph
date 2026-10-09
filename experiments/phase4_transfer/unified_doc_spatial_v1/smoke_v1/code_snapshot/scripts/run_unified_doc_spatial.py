"""Fit the unified DOC predictor on new station partitions via run_ladder."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.spatial_fewshot import K_VALUES
from river_graph.experiments.transfer import DATASETS
from river_graph.experiments.unified_spatial_protocol import (
    SPLIT_SEEDS,
    TRAINING_SEEDS,
    build_unified_spatial_split,
    support_query_cells,
)
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    network_context,
    role_visible,
    target_values,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v1")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py"):
        snapshot[name] = sha256_file(name)
    target = root / "runtime_snapshot.json"
    if target.exists() and json.loads(target.read_text()) != snapshot:
        raise ValueError("Execution code changed; preserve this batch and use a new directory")
    if not target.exists():
        write_json(target, snapshot)
        for name in snapshot:
            destination = root / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def bind_files(run, name, files, config):
    record = {"config_hash": digest(config),
              "files": {str(p.relative_to(run)): sha256_file(p) for p in files}}
    write_json(run / name, record)
    return record


def verify_files(run, name, config):
    saved = json.loads((run / name).read_text())
    if saved["config_hash"] != digest(config):
        raise ValueError(f"Changed configuration: {run}")
    for filename, expected in saved["files"].items():
        if sha256_file(run / filename) != expected:
            raise ValueError(f"Changed saved product: {run / filename}")
    return saved


def covariate_diagnostics(dataset, split):
    n, months = dataset["y"].shape
    source = np.unique(split["train"] // months)
    regime = np.asarray(dataset["regime"], dtype=np.float64)
    mean, sd = regime[source].mean(0), np.maximum(regime[source].std(0), 1e-8)
    standardized = (regime - mean) / sd
    distances = cdist(standardized, standardized[source]) / np.sqrt(regime.shape[1])
    novelty = distances.min(1)
    visible = role_visible(split, FIT_ROLES, (n, months))
    upstream = network_context(target_values(dataset, "log1p"), visible,
                               np.asarray(dataset["edge_index"]))[..., 3]
    return novelty, upstream


def export_products(run, model, dataset, split, config):
    components = model.predict_components()
    n, months = dataset["y"].shape
    cells = np.arange(n * months)
    sites, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    values = np.asarray(dataset["y"], dtype=np.float64).ravel()
    roles = np.full(n * months, "unobserved", dtype=object)
    for role, selected in split.items():
        roles[selected] = role
    novelty, upstream = covariate_diagnostics(dataset, split)
    full = pd.DataFrame({"cell": cells, "station": sites[cells // months],
                         "month": dates[cells % months], "visibility_role": roles,
                         "input_roles": "train", "analyte": "doc",
                         "split_seed": config["split_seed"], "seed": config["seed"],
                         "ecological_novelty": novelty[cells // months],
                         "upstream_support": upstream.ravel()})
    for name, value in components.items():
        full[name] = value.ravel()
    full.to_parquet(run / "full_grid.parquet", index=False)
    predictions, supports = [], []
    for k in K_VALUES:
        support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
        for arm, name in (("context", "extra_trees"), ("hybrid", "hybrid")):
            for calibrated in (False, True):
                prediction = model.predict(query, support_cells=support, support_values=values[support],
                                           k=k, arm=arm, calibrated=calibrated)
                frame = full.iloc[query].copy()
                frame["k"] = k
                frame["model_name"] = name + ("_calibrated" if calibrated else "")
                frame["y_true"], frame["y_pred"] = values[query], prediction
                frame["calibration_delta"] = np.log1p(prediction) - np.log1p(
                    components[f"{arm}_pred"].ravel()[query])
                frame["support_count"] = k if calibrated else 0
                predictions.append(frame)
        frame = full.iloc[support].copy()
        frame["k"], frame["y_true"] = k, values[support]
        supports.append(frame)
    result = pd.concat(predictions, ignore_index=True)
    if not np.isfinite(result[["y_true", "y_pred", "calibration_delta"]].to_numpy()).all():
        raise ValueError("Nonfinite unified model products")
    result.to_parquet(run / "predictions.parquet", index=False)
    pd.concat(supports, ignore_index=True).to_parquet(run / "support_predictions.parquet", index=False)
    pd.DataFrame(model.residual.trace).to_csv(run / "training_trace.csv", index=False)
    for filename in ("full_grid.parquet", "predictions.parquet", "support_predictions.parquet"):
        write_json(run / filename.replace(".parquet", ".meta.json"), {
            "config": config, "config_hash": digest(config),
            "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
            "runtime_snapshot_hash": config["runtime_snapshot_hash"],
            "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"],
                                                       config["runtime_snapshot_hash"]),
            "prediction_sha256": sha256_file(run / filename),
            "model_files": json.loads((run / "fitted_model.meta.json").read_text())["files"],
        })


def run_one(root, data, split_seed, seed, args, runtime_hash):
    split, protocol = build_unified_spatial_split(np.asarray(data["y_mask"]), seed=split_seed)
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    mask = root / "masks" / f"split{split_seed}.npz"
    mask.parent.mkdir(parents=True, exist_ok=True)
    if not mask.exists():
        np.savez_compressed(mask, **split)
        write_json(mask.with_suffix(".json"), protocol)
    else:
        with np.load(mask) as saved:
            for role in split:
                np.testing.assert_array_equal(saved[role], split[role])
    config_path = run / "config.json"
    prior = json.loads(config_path.read_text()) if config_path.exists() else {}
    config = {
        "experiment": "unified_doc_spatial_v1", "split_seed": split_seed, "seed": seed,
        "dataset_path": str(DATASETS["doc"]), "dataset_hash": sha256_file(DATASETS["doc"]),
        "mask_path": str(mask), "mask_hash": sha256_file(mask),
        "runtime_snapshot_hash": runtime_hash,
        "started_at": prior.get("started_at", datetime.now(timezone.utc).isoformat()),
        "n_estimators": args.n_estimators, "max_epochs": args.max_epochs,
        "patience": args.patience, "n_jobs": args.n_jobs, "torch_threads": args.torch_threads,
        "hidden": 64, "chunk_months": 64, "temporal": "observation_aware_gru", "lookback": 12,
        "base_variant": "local", "edge_set": "empty", "edge_direction": "upstream",
        "forest_backend": "extra_trees", "forest_min_samples_leaf": 4,
        "forest_max_features": 1.0, "dropout": 0.1, "lr": 0.001,
        "inference_roles": ["train"], "target_transform": "log1p", "target_analyte": "doc",
        "context_candidates": ["et_leaf4", "et_leaf2", "et_leaf1", "et_sqrt"],
        "fusion_selection": "source-validation MAE: context, temporal, geometric, log-affine",
        "calibration_grid": [0, 0.25, 0.5, 0.75, 1.0], "calibration": "separate alpha by arm and K",
        "support_policy": "retrospective; all five candidate labels excluded from fixed query",
        "k_values": list(K_VALUES), "query_cells": protocol["target_tasks"]["test"]["fixed_query_cells"],
        "q90_threshold_train": float(np.quantile(np.asarray(data["y"]).ravel()[split["train"]], 0.9)),
    }
    if prior and prior != config:
        raise ValueError(f"Refusing to overwrite a different experiment: {run}")
    write_json(config_path, config)
    if (run / "complete.json").exists():
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified complete", flush=True)
        return
    start = time.monotonic()

    def progress(stage, **extra):
        row = {"run": run.name, "pid": os.getpid(), "stage": stage,
               "elapsed_seconds": round(time.monotonic() - start, 2),
               "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        write_json(run / "progress.json", row)
        print(json.dumps(row), flush=True)

    if (run / "fitted_model.meta.json").exists():
        verify_files(run, "fitted_model.meta.json", config)
        model = UnifiedDOCReconstructor.load(run, data, split)
        progress("restored fitted model")
    else:
        rf = None
        if (run / "forest_cache.meta.json").exists():
            verify_files(run, "forest_cache.meta.json", config)
            rf = joblib.load(run / "forests.joblib")
        progress("fitting environmental experts and station-OOF forests" if rf is None else "forests restored")

        def save_forests(forests):
            joblib.dump(forests, run / "forests.joblib", compress=3)
            bind_files(run, "forest_cache.meta.json", [run / "forests.joblib"], config)
            progress("fitting temporal residual and validation selections")

        model = UnifiedDOCReconstructor(
            seed=seed, n_estimators=args.n_estimators, n_jobs=args.n_jobs,
            max_epochs=args.max_epochs, patience=args.patience,
            epoch_callback=lambda row: progress("temporal training", **row),
        )
        model.fit(data, split, rf=rf, forest_callback=save_forests)
        model.save(run)
        bind_files(run, "forest_cache.meta.json", [run / "forests.joblib"], config)
        bind_files(run, "fitted_model.meta.json",
                   [run / f for f in ("forests.joblib", "context.joblib", "residual.pt", "model.json", "adapter.json")],
                   config)
    progress("exporting four matched models at K=0,1,3,5", fusion=model.adapter.fusion_["name"])
    export_products(run, model, data, split, config)
    files = [p for p in run.iterdir() if p.is_file() and p.name not in
             ("complete.json", "progress.json", "forest_cache.meta.json")]
    bind_files(run, "complete.json", files, config)
    progress("complete", best_epoch=model.residual.best_epoch,
             fusion=model.adapter.fusion_["name"], alpha=model.adapter.alpha_by_arm_)
    del model
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "confirmation")
    ap.add_argument("--split-seeds", nargs="+", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        runs = list((args.out_dir / "runs").glob("*/complete.json"))
        for file in runs:
            config = json.loads((file.parent / "config.json").read_text())
            verify_files(file.parent, "complete.json", config)
        print(f"Verified {len(runs)} completed runs")
        return
    runtime_hash = freeze_runtime(args.out_dir)
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    for split_seed in args.split_seeds:
        for seed in args.seeds:
            run_one(args.out_dir, data, split_seed, seed, args, runtime_hash)


if __name__ == "__main__":
    main()
