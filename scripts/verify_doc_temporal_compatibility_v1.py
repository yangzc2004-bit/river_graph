"""Replay every fitted temporal procedure without training or model selection."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT
from run_doc_temporal_compatibility_v1 import (
    MASKS,
    MODELS,
    ROOT,
    temporal_arrays,
    temporal_label_view,
)
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.station_adapted_hybrid import StationAdaptedHybrid
from river_graph.models.unified_doc import UnifiedDOCReconstructor


def replay(run, runtime):
    config = json.loads((run/"config.json").read_text())
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("fitted temporal runtime differs from retained execution")
    for marker in ("backbone_complete.json", "trees_complete.json", "current_complete.json", "complete.json"):
        verify_files(run, marker, config)
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError(f"temporal {kind} changed")
    original = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as archive:
        split = {role: archive[role].copy() if role in archive.files else np.empty(0, dtype=np.int64)
                 for role in ("train", "val", "test", "context")}
    dataset = temporal_label_view(original, split)
    if np.isfinite(dataset["y"].ravel()[split["test"]]).any():
        raise ValueError("test DOC remained in fitted-label view")
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], dataset["y"].shape)
    expert = UnifiedDOCReconstructor.load(run/"backbone", dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    x = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    shape = dataset["y"].shape
    context = np.maximum(0, np.expm1(expert.context_forest.predict(x))).reshape(shape)
    x47 = np.column_stack([x, daily.reshape(-1, 8)])
    ordinary = np.maximum(0, np.expm1(joblib.load(run/"matched.joblib").predict(x47))).reshape(shape)
    new_context = np.maximum(0, np.expm1(joblib.load(run/"hidden.joblib").predict(x47))).reshape(shape)
    native = {}
    with np.load(run/"oof.npz", allow_pickle=False) as saved:
        for name, base, key, checkpoint in (("current_native", context, "old", "current.pt"),
                ("upgraded_native", new_context, "new", "candidate.pt")):
            oof = saved[key]
            extra = build_regime_head_features(dataset["regime"], split["train"],
                np.maximum(0, np.expm1(oof.ravel()[split["train"]])), base,
                build_causal_flow_features(dataset)["full"], n_months=shape[1])
            inputs, _ = temporal_arrays(features, extra, daily, dataset, split, oof, base)
            model = EncoderNativeResidual.from_payload(torch.load(run/checkpoint, weights_only=False, map_location="cpu"))
            native[name] = model.predict(inputs, base)
    fusion = json.loads((run/"fusion.json").read_text())
    current = StationAdaptedHybrid.from_dict(fusion["current"]).predict_components(
        context.ravel(), native["current_native"].ravel())["hybrid"].reshape(shape)
    upgrade = StationAdaptedHybrid.from_dict(fusion["upgraded"]).predict_components(
        new_context.ravel(), native["upgraded_native"].ravel())["hybrid"].reshape(shape)
    grids = dict(zip(MODELS, (expert.predict_components()["hybrid_pred"], context, ordinary,
        new_context, native["current_native"], current, native["upgraded_native"], upgrade), strict=True))
    maxima = {}
    points = pd.read_parquet(run/"predictions.parquet")
    test_cells = np.sort(split["test"])
    with np.load(run/"components.npz", allow_pickle=False) as saved:
        for name, prediction in grids.items():
            if not np.isfinite(prediction).all() or np.any(prediction < 0):
                raise ValueError("invalid temporal prediction")
            np.testing.assert_allclose(prediction, saved[name], rtol=0, atol=1e-8)
            maxima[name] = float(np.max(np.abs(prediction-saved[name])))
            group = points[points.model_name.eq(name)].sort_values("cell")
            np.testing.assert_array_equal(group.cell, test_cells)
            np.testing.assert_allclose(group.y_pred, prediction.ravel()[test_cells], rtol=0, atol=1e-8)
            np.testing.assert_array_equal(group.y_true, np.asarray(original["y"]).ravel()[test_cells])
    result = {"run": run.name, "max_grid_replay_difference": maxima, "test_cells": len(split["test"])}
    del expert, features, model, grids, dataset, original
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    runs = [args.root/"runs"/f"{mask}_seed{seed}" for mask in MASKS for seed in (42, 43, 44)]
    if not all((run/"complete.json").is_file() for run in runs):
        raise ValueError("all six temporal packages must complete before replay")
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    records = []
    for run in runs:
        row = replay(run, runtime)
        records.append(row)
        print(json.dumps(row), flush=True)
    write_json(args.root/"verification.json", {"status": "passed", "runtime_snapshot_hash": runtime,
        "verifier_sha256": sha256_file(__file__), "fitted_procedure_replay": records})


if __name__ == "__main__":
    main()
