"""Train matched support-to-query projections for the existing DOC experts."""
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
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source

from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.episodic_station_adapter import EpisodicStationProjector
from river_graph.models.episodic_station_data import fit_context_oof
from river_graph.models.regularized_station_fusion import RegularizedStationFusion
from river_graph.models.unified_doc import UnifiedDOCReconstructor
from river_graph.models.unified_doc_features import extract_basis_inputs

ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v3")
SOURCE = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation")
FUSION_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v2")
KS = (0, 1, 3, 5)
HEADS = ("constant", "gru_pca", "gru_episodic", "tree_pca", "tree_episodic")
MODELS = tuple(f"{base}_{head}" for base in ("context", "fusion") for head in HEADS)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_unified_doc_spatial_v3.py",
                 str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution code changed: use a new run directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def load_fusion(run, context, temporal, split, months):
    config = json.loads((run / "config.json").read_text())
    verify_files(run, "complete.json", config)
    model = RegularizedStationFusion.from_dict(json.loads((run / "fusion.json").read_text()))
    validation = np.full_like(context, np.nan)
    cells = split["val"]
    validation[cells] = model.predict_crossfit(context[cells], temporal[cells], cells // months)
    return model, model.predict(context, temporal), validation


def fit_projectors(features, oof_z, truth, split, context, *, seed, epochs, patience, progress=None):
    """Fit both representations with source and validation stations only."""
    months = truth.shape[1]
    source_ids = features["source_station_ids"]
    validation_ids = np.unique(split["val"] // months)
    source_mask = np.zeros(truth.size, dtype=bool)
    source_mask[split["train"]] = True
    source_mask = source_mask.reshape(truth.shape)[source_ids]
    validation_mask = np.zeros(truth.size, dtype=bool)
    validation_mask[split["val"]] = True
    validation_mask = validation_mask.reshape(truth.shape)[validation_ids]
    shapes = {"constant": np.zeros((truth.size, 2))}
    projectors = {}
    for name in ("gru", "tree"):
        full_features = features[f"full_{name}"]
        model = EpisodicStationProjector(
            n_components=16, output_dim=2, epochs=epochs, patience=patience,
            learning_rate=.01, batch_size=32, seed=seed,
        ).fit(
            features[f"source_{name}"], oof_z[source_ids], truth[source_ids], source_mask,
            full_features[validation_ids], context.reshape(truth.shape)[validation_ids],
            truth[validation_ids], validation_mask, selection_role="source_validation",
            progress=(lambda row, name=name: progress(name, row)) if progress else None,
        )
        shapes[f"{name}_pca"] = model.transform_pca(full_features).reshape(-1, 2)
        shapes[f"{name}_episodic"] = model.transform(full_features).reshape(-1, 2)
        # Check serialized training state on real station features before releasing them.
        restored = EpisodicStationProjector.from_dict(model.to_dict())
        np.testing.assert_array_equal(model.transform(full_features[:2]), restored.transform(full_features[:2]))
        projectors[name] = model
    return projectors, shapes


def run_one(root, source_root, fusion_root, split_seed, seed, runtime, *, epochs, patience, n_jobs):
    started = time.monotonic()
    source = source_root / "runs" / f"split{split_seed}_seed{seed}"
    prior = fusion_root / "runs" / source.name
    run = root / "runs" / source.name
    old_config, old_complete, dataset, split, full = read_source(source)
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["n_jobs"] != n_jobs
                or config["source_completion_hash"] != sha256_file(source / "complete.json")
                or config["fusion_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Cached run differs from fitted settings or source packages")
        verify_files(run, "complete.json", config)
        print(f"{source.name}: verified completed experiment", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "unified_doc_spatial_v3", "split_seed": split_seed, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "dataset_path": old_config["dataset_path"], "dataset_hash": old_config["dataset_hash"],
        "mask_path": old_config["mask_path"], "mask_hash": old_config["mask_hash"],
        "source_run": str(source), "source_completion_hash": sha256_file(source / "complete.json"),
        "source_model_files": old_complete["files"], "fusion_run": str(prior),
        "fusion_completion_hash": sha256_file(prior / "complete.json"),
        "target_analyte": "doc", "target_transform": "log1p", "inference_roles": ["train"],
        "models": MODELS, "k_values": KS, "q90_threshold_train": old_config["q90_threshold_train"],
        "query_cells": old_config["query_cells"], "basis_dimensions": 16, "projection_dimensions": 2,
        "epochs": epochs, "patience": patience, "learning_rate": .01, "batch_size": 32,
        "optimizer": "Adam with row-orthonormal projection retraction", "training_ridge": [1, 10],
        "training_K": [3, 5], "training_alpha": 1,
        "training_loss": "raw query MAE, equal source station, K and ridge",
        "epoch_selection": "source-validation pooled query MAE, equal K and ridge; epoch0 included",
        "final_alpha_grid": [0, .25, .5, .75, 1], "final_ridge_grid": [.1, 1, 10, "infinity"],
        "backbone_retraining": False, "context_oof_folds": 5, "n_jobs": n_jobs,
        "torch_threads": torch.get_num_threads(),
        "study_role": "development on previously evaluated station partitions",
    }
    write_json(run / "config.json", config)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months = truth.shape[1]
    context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
    fusion, combined, fusion_val = load_fusion(prior, context, temporal, split, months)
    write_json(run / "fusion.json", fusion.to_dict())
    experts = UnifiedDOCReconstructor.load(source, dataset, split)
    print(f"{source.name}: fitting held-station environmental baselines", flush=True)
    oof = fit_context_oof(experts, dataset, split, n_jobs=n_jobs,
                         progress=lambda row: print(f"{source.name}: OOF fold {row['fold']+1}/5", flush=True))
    write_json(run / "oof_folds.json", oof["fold_records"])
    np.savez_compressed(run / "source_oof.npz", pred_z=oof["pred_z"])
    features = extract_basis_inputs(experts, dataset, split, verify_head=True,
        progress=lambda row: print(f"{source.name}: {row['stage']} {row.get('fold', '')}", flush=True))
    del experts
    gc.collect()

    def progress(name, row):
        if row["epoch"] % 10 == 0:
            print(f"{source.name}: {name} {json.dumps(row, allow_nan=False)}", flush=True)

    projectors, shapes = fit_projectors(features, oof["pred_z"], truth, split, context,
        seed=seed, epochs=epochs, patience=patience, progress=progress)
    for name, model in projectors.items():
        write_json(run / f"{name}_projector.json", model.to_dict())
        pd.DataFrame(model.trace_).to_csv(run / f"{name}_trace.csv", index=False)
    del features, oof
    gc.collect()
    adapters = fit_adapters({"context": context, "fusion": fusion_val}, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: adapter.to_dict() for name, adapter in adapters.items()})
    np.savez_compressed(run / "representations.npz", **shapes)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    predictions = make_predictions(full, {"context": context, "fusion": combined}, shapes,
                                   adapters, labels, split, months)
    predictions["split_seed"], predictions["seed"] = split_seed, seed
    predictions["y_true"] = truth.ravel()[predictions.cell.to_numpy()]
    if not np.isfinite(predictions[["y_pred", "y_true", "adaptation_delta"]].to_numpy()).all():
        raise ValueError("Nonfinite prediction product")
    previous = pd.read_parquet(prior / "predictions.parquet")
    for name in ("context_constant", "fusion_constant"):
        for k in KS:
            a = previous[previous.model_name.eq(name) & previous.k.eq(k)].sort_values("cell")
            b = predictions[predictions.model_name.eq(name) & predictions.k.eq(k)].sort_values("cell")
            np.testing.assert_array_equal(a.cell, b.cell)
            np.testing.assert_array_equal(a.y_pred, b.y_pred)
    predictions.to_parquet(run / "predictions.parquet", index=False)
    state_names = ("fusion.json", "adapters.json", "gru_projector.json", "tree_projector.json",
                   "representations.npz", "source_oof.npz", "oof_folds.json")
    write_json(run / "predictions.meta.json", {
        "config": config, "config_hash": digest(config), "runtime_snapshot_hash": runtime,
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime),
        "prediction_sha256": sha256_file(run / "predictions.parquet"),
        "model_files": {name: sha256_file(run / name) for name in state_names},
        "rows": len(predictions), "selection_role": "source_validation",
    })
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-started})
    products = (*state_names, "config.json", "predictions.parquet", "predictions.meta.json",
                "gru_trace.csv", "tree_trace.csv", "timing.json")
    bind_files(run, "complete.json", [run / name for name in products], config)
    print(f"{source.name}: complete, {time.monotonic()-started:.1f}s; "
          f"best epochs GRU={projectors['gru'].best_epoch_}, tree={projectors['tree'].best_epoch_}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--fusion-root", type=Path, default=FUSION_ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for split_seed in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.source_root, args.fusion_root, split_seed, seed, runtime,
                    epochs=args.epochs, patience=args.patience, n_jobs=args.n_jobs)
            gc.collect()


if __name__ == "__main__":
    main()
