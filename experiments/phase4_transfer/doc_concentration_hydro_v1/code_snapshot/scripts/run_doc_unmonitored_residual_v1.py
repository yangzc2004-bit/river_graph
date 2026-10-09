"""Rebuild the current ecology/GRU residual on a station-hidden tree base."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_trees_v1 import ROOT as TREE_ROOT
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import fold_split, station_folds
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_residual_v1")


def run_one(root, partition, seed, runtime):
    start = time.monotonic()
    parent = TREE_ROOT / "runs" / f"split{partition}_seed{seed}"
    old = json.loads((parent / "config.json").read_text())
    verify_files(parent, "complete.json", old)
    run = root / "runs" / parent.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("saved residual execution changed")
        verify_files(run, "complete.json", config)
        return
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train",
              "split_seed", "seed")}, "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "doc_unmonitored_residual_v1", "parent_run": str(parent),
        "parent_completion_hash": sha256_file(parent / "complete.json"), "runtime_snapshot_hash": runtime,
        "models": ["current_model", "matched_daily_trees", "station_hidden_trees", "unmonitored_residual", "unmonitored_integrated"],
        "selection_role": "source_validation", "evaluation_role": "source_validation_only", "epochs": 30, "patience": 5,
        "main_change": "station-hidden source tree base plus its OOF residual retraining",
        "residual_backbone": "existing ecology encoder and observation GRU, retained fitted initialization",
        "readout": "existing daily hydro and regime interactions; zero initialized native head",
        "input_statistics": "retained historical raw preprocessing; new source-only readout context scaling",
        "tail_weight": 2, "tree_hyperparameters": "frozen selected station-hidden tree from the parent"}
    if (run / "config.json").exists():
        existing = json.loads((run / "config.json").read_text())
        if (existing["runtime_snapshot_hash"] != runtime or existing["parent_completion_hash"] != config["parent_completion_hash"]):
            raise ValueError("partial residual execution changed")
        config = existing
    else:
        write_json(run / "config.json", config)

    def progress(stage, row):
        entry = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-start}
        write_json(root / "progress.json", entry)
        print(json.dumps(entry), flush=True)

    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    ancestor = json.loads((Path(old["parent_run"]) / "config.json").read_text())
    monthly = Path(ancestor["parent_run"])
    monthly_config = json.loads((monthly / "config.json").read_text())
    daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], dataset["y"].shape)
    forest = joblib.load(parent / "station_hidden_trees.joblib")
    folds = station_folds(split["train"], dataset["y"].shape[1], seed)
    _, full_features = station_hidden_tree_inputs(dataset, split, folds, daily)
    shape = tuple(dataset["y"].shape)
    context = np.maximum(0, np.expm1(forest.predict(full_features))).reshape(shape)
    del full_features
    cache = run / "source_oof.npz"
    if (run / "oof_complete.json").exists():
        verify_files(run, "oof_complete.json", config)
        with np.load(cache, allow_pickle=False) as saved:
            oof = saved["pred_z"].copy()
    else:
        oof = np.full(shape, np.nan)
        records = []
        for index, stations in enumerate(folds):
            outer = fold_split(split, stations, shape[1])
            training, inference = station_hidden_tree_inputs(dataset, outer,
                station_folds(outer["train"], shape[1], seed), daily)
            model = clone(forest).set_params(n_jobs=2)
            model.fit(training, np.log1p(np.asarray(dataset["y"]).ravel()[outer["train"]]))
            held = split["train"][np.isin(split["train"]//shape[1], stations)]
            oof.ravel()[held] = model.predict(inference[held])
            records.append({"fold": index, "hidden_stations": stations.tolist(),
                            "fitted_stations": np.unique(outer["train"]//shape[1]).tolist()})
            progress("station_hidden_oof", {"fold": index, "query_cells": len(held)})
            del model, training, inference
        np.savez_compressed(cache, pred_z=oof)
        write_json(run / "oof_records.json", records)
        bind_files(run, "oof_complete.json", [cache, run / "oof_records.json"], config)
    expert = UnifiedDOCReconstructor.load(Path(monthly_config["source_run"]), dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=shape[1])
    source_inputs, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    retained = EncoderNativeResidual.from_payload(torch.load(monthly / "daily.pt", weights_only=False, map_location="cpu"))
    settings = {**retained._config(), "epochs": 30, "patience": 5}
    model = EncoderNativeResidual(retained.spatial, retained.temporal, retained.decay, **settings)
    model.fit(*arrays, tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
              progress=lambda row: progress("native_residual", row))
    val = split["val"]
    val_ids = np.unique(val//shape[1])
    correction = model.predict_delta({key: value[val_ids] for key, value in full_inputs.items()})
    native = np.maximum(0, context.ravel()[val]+model.selected_scale_*correction[
        np.searchsorted(val_ids, val//shape[1]), val % shape[1]])
    memory = EcologicalResidualTransfer().fit(np.asarray(dataset["regime"]), split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), np.asarray(dataset["y"]).ravel()[split["train"]],
        n_months=shape[1], validation_cells=val, validation_y=np.asarray(dataset["y"]).ravel()[val],
        validation_context=context.ravel()[val], validation_temporal=native, selection_role="source_validation")
    temporal_for_memory = context.copy()
    temporal_for_memory.ravel()[val] = native
    integrated = memory.predict(context, temporal_for_memory).ravel()[val]
    previous = pd.read_parquet(parent / "predictions.parquet")
    frames = [previous.copy()]
    template = previous[previous.model_name.eq("current_model")].copy()
    for name, values in (("unmonitored_residual", native), ("unmonitored_integrated", integrated)):
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, values
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
    torch.save(model.to_payload(), run / "native.pt")
    write_json(run / "native.json", model.to_dict())
    write_json(run / "memory.json", memory.to_dict())
    write_json(run / "readout_normalization.json", extra["normalization"])
    np.savez_compressed(run / "validation_inputs.npz", **{key: value[val_ids] for key, value in full_inputs.items()},
                        context=context[val_ids], cell=val)
    files = ["config.json", "native.pt", "native.json", "memory.json", "readout_normalization.json",
             "source_oof.npz", "oof_records.json", "oof_complete.json", "validation_inputs.npz"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run / name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"models": 5, "best_epoch": model.best_epoch_})
    del model, expert, forest, retained, features, source_inputs, full_inputs, dataset, arrays
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_unmonitored_residual_v1.py",
                 "scripts/run_doc_unmonitored_trees_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("saved residual execution code changed")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == "__main__":
    main()
