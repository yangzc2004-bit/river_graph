"""Fit regularized fusion and support-dependent residual heads on saved experts.

Run via ``run_ladder.py --experiment unified-doc-spatial-v2``. The nine v1
forest/recurrent fits stay frozen. All new choices use source validation;
outer query labels are attached only after fitting the fusion and adapters.
"""
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

from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.regularized_station_fusion import RegularizedStationFusion
from river_graph.models.support_shape_adapter import (
    StationTemporalBasis,
    SupportShapeAdapter,
    SupportShapeEpisode,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor
from river_graph.models.unified_doc_features import extract_basis_inputs

ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v2")
SOURCE = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation")
KS = (0, 1, 3, 5)
MODELS = tuple(f"{base}_{head}" for base in ("context", "fusion")
               for head in ("constant", "gru_shape", "tree_shape"))


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution code changed: keep completed outputs and use a new directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def read_source(run):
    config = json.loads((run / "config.json").read_text())
    complete = verify_files(run, "complete.json", config)
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError(f"Changed source {kind}")
    dataset = torch.load(config["dataset_path"], map_location="cpu", weights_only=False)
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role] for role in ("train", "val", "test", "context")}
    full = pd.read_parquet(run / "full_grid.parquet")
    np.testing.assert_array_equal(full.cell.to_numpy(), np.arange(np.prod(dataset["y"].shape)))
    return config, complete, dataset, split, full


def fit_fusion(context, temporal, truth, split, months):
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    fusion = RegularizedStationFusion().fit(
        context[query], temporal[query], truth[query], query // months,
        selection_role="source_validation",
    )
    # Fit support hyperparameters using held-station fusion coefficients at
    # BOTH support and query dates, rather than fitted validation predictions.
    all_validation = np.asarray(split["val"], dtype=np.int64)
    validation_prediction = np.full_like(context, np.nan)
    validation_prediction[all_validation] = fusion.predict_crossfit(
        context[all_validation], temporal[all_validation], all_validation // months,
    )
    return fusion, fusion.predict(context, temporal), validation_prediction


def fit_adapters(bases, shapes, truth, split, months):
    adapters = {}
    for base_name, base in bases.items():
        for shape_name, shape in shapes.items():
            episodes = []
            for k in KS:
                support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
                episodes.append(SupportShapeEpisode(
                    k=k, query_cells=query, query_values=truth[query],
                    query_prediction=base[query], support_cells=support,
                    support_values=truth[support], support_prediction=base[support],
                    query_basis=shape[query], support_basis=shape[support],
                ))
            kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
            adapter = SupportShapeAdapter(n_months=months, **kwargs)
            adapter.fit(episodes, selection_role="source_validation")
            adapters[f"{base_name}_{shape_name}"] = adapter
    return adapters


def make_predictions(full, bases, shapes, adapters, support_truth, split, months):
    """Only support truth is accessed while creating predictions."""
    frames = []
    for k in KS:
        support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
        for base_name, base in bases.items():
            for shape_name, shape in shapes.items():
                name = f"{base_name}_{shape_name}"
                prediction = adapters[name].adapt(
                    base[query], query, base[support], support, support_truth[support],
                    query_basis=shape[query], support_basis=shape[support], k=k,
                )
                frame = full.iloc[query][["cell", "station", "month", "analyte",
                    "visibility_role", "ecological_novelty", "upstream_support"]].copy()
                frame["model_name"], frame["k"] = name, k
                frame["y_pred"], frame["base_pred"] = prediction, base[query]
                frame["adaptation_delta"] = np.log1p(prediction) - np.log1p(base[query])
                frame["support_count"] = k
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_one(root, source_root, split_seed, seed, runtime_hash):
    started = time.monotonic()
    source = source_root / "runs" / f"split{split_seed}_seed{seed}"
    run = root / "runs" / source.name
    old_config, old_complete, dataset, split, full = read_source(source)
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime_hash
                or config["source_completion_hash"] != sha256_file(source / "complete.json")):
            raise ValueError("Cached v2 run differs from current frozen inputs")
        verify_files(run, "complete.json", config)
        print(f"{source.name}: verified existing v2 products", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "unified_doc_spatial_v2", "split_seed": split_seed, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": old_config["dataset_path"], "dataset_hash": old_config["dataset_hash"],
        "mask_path": old_config["mask_path"], "mask_hash": old_config["mask_hash"],
        "source_run": str(source), "source_completion_hash": sha256_file(source / "complete.json"),
        "source_model_files": old_complete["files"], "runtime_snapshot_hash": runtime_hash,
        "target_analyte": "doc", "target_transform": "log1p", "inference_roles": ["train"],
        "models": MODELS, "k_values": KS, "q90_threshold_train": old_config["q90_threshold_train"],
        "query_cells": old_config["query_cells"], "basis_components": 2,
        "basis": "source-only station-centered whitened PCA; held-station input views",
        "fusion": "station-blocked conditional meta-CV; identity, convex, identity-centered ridge",
        "alpha_grid": [0, .25, .5, .75, 1], "ridge_grid": [.1, 1, 10, "infinity"],
        "study_role": "development on previously evaluated station partitions",
        "expert_retraining": False, "torch_threads": torch.get_num_threads(),
    }
    write_json(run / "config.json", config)
    months = dataset["y"].shape[1]
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
    context, temporal = (full[f"{key}_pred"].to_numpy() for key in ("context", "temporal"))
    fusion, combined, fusion_val = fit_fusion(context, temporal, truth, split, months)
    write_json(run / "fusion.json", fusion.to_dict())
    print(f"{source.name}: fusion selected; extracting frozen representations", flush=True)
    model = UnifiedDOCReconstructor.load(source, dataset, split)
    features = extract_basis_inputs(model, dataset, split,
        progress=lambda message: print(f"{source.name}: {message}", flush=True), verify_head=True)
    shapes, projectors = {"constant": np.zeros((len(truth), 2))}, {}
    for name in ("gru", "tree"):
        projector = StationTemporalBasis(n_components=2).fit(
            features.pop(f"source_{name}"), source_role="source_training",
        )
        shapes[f"{name}_shape"] = projector.transform(features.pop(f"full_{name}")).reshape(-1, 2)
        projectors[name] = projector.to_dict()
    write_json(run / "bases.json", projectors)
    np.savez_compressed(run / "representations.npz", **shapes)
    del features, model
    gc.collect()
    adapters = fit_adapters({"context": context, "fusion": fusion_val}, shapes, truth, split, months)
    write_json(run / "adapters.json", {name: adapter.to_dict() for name, adapter in adapters.items()})
    # Expose only the reserved support cells to prediction; query truth is absent.
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    support_truth = np.full_like(truth, np.nan)
    support_truth[support] = truth[support]
    frame = make_predictions(full, {"context": context, "fusion": combined}, shapes,
                             adapters, support_truth, split, months)
    frame["split_seed"], frame["seed"] = split_seed, seed
    frame["y_true"] = truth[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "adaptation_delta"]].to_numpy()).all():
        raise ValueError("Nonfinite query output")
    # Matched constant ExtraTrees is unchanged, making the new shape comparison clean.
    previous = pd.read_parquet(source / "predictions.parquet")
    for k in KS:
        old = previous.loc[(previous.model_name == "extra_trees_calibrated") & (previous.k == k)]
        new = frame.loc[(frame.model_name == "context_constant") & (frame.k == k)]
        np.testing.assert_array_equal(old.sort_values("cell").cell, new.sort_values("cell").cell)
        np.testing.assert_allclose(old.sort_values("cell").y_pred, new.sort_values("cell").y_pred,
                                   rtol=1e-12, atol=1e-12)
    prediction_path = run / "predictions.parquet"
    frame.to_parquet(prediction_path, index=False)
    model_files = {name: sha256_file(run / name) for name in
                   ("fusion.json", "adapters.json", "bases.json", "representations.npz")}
    write_json(run / "predictions.meta.json", {
        "config": config, "config_hash": digest(config),
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "runtime_snapshot_hash": runtime_hash,
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime_hash),
        "prediction_sha256": sha256_file(prediction_path), "model_files": model_files,
        "rows": len(frame), "selection_role": "source_validation",
    })
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic() - started})
    files = [run / name for name in ("config.json", "fusion.json", "adapters.json", "bases.json",
        "representations.npz", "predictions.parquet", "predictions.meta.json", "timing.json")]
    bind_files(run, "complete.json", files, config)
    print(f"{source.name}: complete, {len(frame)} rows, {time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for split_seed in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.source_root, split_seed, seed, runtime)
            gc.collect()


if __name__ == "__main__":
    main()
