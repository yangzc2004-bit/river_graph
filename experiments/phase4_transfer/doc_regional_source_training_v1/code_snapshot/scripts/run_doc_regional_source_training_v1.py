"""Match existing DOC residual training to whole-region observation outages."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import fit_memory
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import fold_split
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.regional_source_views import huc4_source_folds
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_regional_source_training_v1")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
              "q90_threshold_train", "seed", "split_seed")}, "experiment": "doc_regional_source_training_v1",
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "nodes_path": str(NODES), "nodes_hash": sha256_file(NODES),
        "models": [*old["models"], "regional_trees", "regional_residual", "regional_integrated"],
        "epochs": 30, "patience": 5, "tail_weight": 2,
        "main_change": "HUC4-blocked source observation views and nested regional forest OOF",
        "tree_hyperparameters": "preceding selected strong tree, unchanged; refit on new source views",
        "encoder_initialization": "preceding native candidate initial encoder/GRU/decay; zero head",
        "selection_role": "source_validation", "evaluation_role": "source_validation_only"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if saved["runtime_snapshot_hash"] != runtime or saved["parent_completion_hash"] != config["parent_completion_hash"]:
            raise ValueError("saved regional execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return

    def progress(stage, values):
        record = {"run": run.name, "stage": stage, **values, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    for path, expected in ((config["dataset_path"], config["dataset_hash"]),
                           (config["mask_path"], config["mask_hash"]), (config["nodes_path"], config["nodes_hash"])):
        if sha256_file(path) != expected:
            raise ValueError("regional source data or role assignment changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    shape = tuple(dataset["y"].shape)
    nodes = pd.read_csv(config["nodes_path"], dtype=str).set_index("site_no")
    names = np.asarray(dataset["site_no"], str)
    if not nodes.index.is_unique or not set(names) <= set(nodes.index):
        raise ValueError("node site identity differs from the source dataset")
    huc = nodes.loc[names, "huc_cd"].tolist()
    folds = huc4_source_folds(split["train"], shape[1], huc, seed)
    tree_parent = Path(old["parent_run"])
    tree_config = json.loads((tree_parent/"config.json").read_text())
    ancestor = json.loads((Path(tree_config["parent_run"])/"config.json").read_text())
    monthly = Path(ancestor["parent_run"])
    monthly_config = json.loads((monthly/"config.json").read_text())
    daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], shape)
    frozen_forest = joblib.load(tree_parent/"station_hidden_trees.joblib")
    if (run/"tree_complete.json").exists():
        verify_files(run, "tree_complete.json", config)
        forest = joblib.load(run/"regional_trees.joblib")
        with np.load(run/"tree_inputs.npz", allow_pickle=False) as saved:
            context = saved["context"].copy()
    else:
        training, full = station_hidden_tree_inputs(dataset, split, folds, daily)
        forest = clone(frozen_forest).set_params(n_jobs=2)
        forest.fit(training, np.log1p(np.asarray(dataset["y"]).ravel()[split["train"]]))
        context = np.maximum(0., np.expm1(forest.predict(full))).reshape(shape)
        joblib.dump(forest, run/"regional_trees.joblib", compress=3)
        np.savez_compressed(run/"tree_inputs.npz", context=context, query_features=full[split["val"]])
        bind_files(run, "tree_complete.json", [run/"regional_trees.joblib", run/"tree_inputs.npz"], config)
        del training, full
        progress("regional_tree_complete", {"n_folds": len(folds)})
    if (run/"oof_complete.json").exists():
        verify_files(run, "oof_complete.json", config)
        with np.load(run/"source_oof.npz", allow_pickle=False) as saved:
            oof = saved["pred_z"].copy()
    else:
        oof = np.full(shape, np.nan)
        records = []
        for index, held_stations in enumerate(folds):
            outer = fold_split(split, held_stations, shape[1])
            inner = huc4_source_folds(outer["train"], shape[1], huc, seed)
            training, full = station_hidden_tree_inputs(dataset, outer, inner, daily)
            child = clone(forest).set_params(n_jobs=2)
            child.fit(training, np.log1p(np.asarray(dataset["y"]).ravel()[outer["train"]]))
            cells = split["train"][np.isin(split["train"]//shape[1], held_stations)]
            oof.ravel()[cells] = child.predict(full[cells])
            fitted = np.unique(outer["train"]//shape[1])
            records.append({"fold": index, "hidden_stations": held_stations.tolist(), "fitted_stations": fitted.tolist(),
                "hidden_huc4": sorted({str(huc[i]).zfill(8)[:4] for i in held_stations}),
                "fitted_huc4": sorted({str(huc[i]).zfill(8)[:4] for i in fitted}),
                "inner_folds": [group.tolist() for group in inner]})
            progress("regional_oof", {"fold": index, "held_stations": len(held_stations), "held_cells": len(cells)})
            del child, training, full
        np.savez_compressed(run/"source_oof.npz", pred_z=oof)
        write_json(run/"oof_records.json", records)
        bind_files(run, "oof_complete.json", [run/"source_oof.npz", run/"oof_records.json"], config)
    expert = UnifiedDOCReconstructor.load(Path(monthly_config["source_run"]), dataset, split)
    previous_folds = expert.rf.folds
    try:
        expert.rf.folds = folds
        features = extract_raw_temporal_inputs(expert, dataset, split)
    finally:
        expert.rf.folds = previous_folds
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0., np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=shape[1])
    _, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    ids = np.unique(split["val"]//shape[1])
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        for key in ("raw", "age", "support", "env"):
            np.testing.assert_array_equal(full_inputs[key][ids], saved[key])
    if (run/"native_complete.json").exists():
        verify_files(run, "native_complete.json", config)
        model = EncoderNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    else:
        retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
        for name in ("spatial", "temporal", "decay"):
            getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
        model = EncoderNativeResidual(retained.spatial, retained.temporal, retained.decay, **retained._config())
        model.fit(*arrays, tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                  progress=lambda row: progress("regional_native", row))
        torch.save(model.to_payload(), run/"native.pt")
        write_json(run/"native.json", model.to_dict())
        bind_files(run, "native_complete.json", [run/"native.pt", run/"native.json"], config)
    prediction = model.predict({key: value[ids] for key, value in full_inputs.items()}, context[ids])
    val = split["val"]
    native = context.copy()
    native[ids] = prediction
    memory = fit_memory(dataset, split, oof, context, native, shape[1])
    previous = pd.read_parquet(parent/"predictions.parquet")
    frames = [previous.copy()]
    template = previous[previous.model_name.eq("unmonitored_integrated")].copy()
    for name, values in (("regional_trees", context.ravel()[val]),
                         ("regional_residual", prediction[np.searchsorted(ids, val//shape[1]), val % shape[1]]),
                         ("regional_integrated", memory.predict(context, native).ravel()[val])):
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, values
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    write_json(run/"memory.json", memory.to_dict())
    write_json(run/"readout_normalization.json", extra["normalization"])
    np.savez_compressed(run/"validation_inputs.npz", **{key: value[ids] for key, value in full_inputs.items()},
                        context=context[ids], cell=val)
    files = ["config.json", "regional_trees.joblib", "tree_inputs.npz", "tree_complete.json", "source_oof.npz",
             "oof_records.json", "oof_complete.json", "native.pt", "native.json", "native_complete.json",
             "memory.json", "readout_normalization.json", "validation_inputs.npz"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"models": 8, "best_epoch": model.best_epoch_})
    del model, expert, dataset, features, full_inputs, arrays, forest, memory
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
    for name in ("scripts/run_ladder.py", "scripts/run_doc_regional_source_training_v1.py",
                 "scripts/analyze_doc_regional_source_training_v1.py", "tests/test_regional_source_views.py",
                 "scripts/run_doc_geographical_confirmation_v1.py", "scripts/run_doc_unmonitored_trees_v1.py",
                 "scripts/run_doc_source_retrieval_v1.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_unified_doc_spatial.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("retain saved regional execution after code changes")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == "__main__":
    main()
