"""Fit DOC objective comparisons on unchanged station-hidden source inputs."""
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
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.doc_reference_objectives import (
    ARMS,
    make_reference,
    predict_reference,
    reference_target,
)
from river_graph.models.kgml_local_transport import station_folds

ROOT = Path("experiments/phase4_transfer/doc_native_reference_objective_v1")


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "q90_threshold_train", "split_seed", "seed")}, "experiment": root.name,
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*old["models"], *ARMS], "target_transforms": ARMS,
        "main_change": "environmental reference target scale and L1 objective;47 existing station-hidden inputs",
        "selection_role": "source_validation", "evaluation_role": "source_validation_development",
        "fitting_role": "source_train_only", "thread_limit": 2,
        "learner_selection": "none; all three fixed references reported without blending"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if any(saved[key] != config[key] for key in ("runtime_snapshot_hash", "parent_completion_hash")):
            raise ValueError("saved reference objective source execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("reference objective source inputs changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    shape = dataset["y"].shape
    tree_parent = Path(old["parent_run"])
    tree_config = json.loads((tree_parent/"config.json").read_text())
    monthly_config = json.loads((Path(tree_config["parent_run"])/"config.json").read_text())
    monthly = Path(monthly_config["parent_run"])
    daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], shape)
    folds = station_folds(split["train"], shape[1], seed)
    training, inference = station_hidden_tree_inputs(dataset, split, folds, daily)
    reference = joblib.load(tree_parent/"station_hidden_trees.joblib")
    frames = [pd.read_parquet(parent/"predictions.parquet")]
    template = frames[0][frames[0].model_name.eq("unmonitored_integrated")].copy()
    val = split["val"]
    np.testing.assert_array_equal(template.cell, val)
    val_x = inference[val]
    if training.shape[1] != 47 or not np.isfinite(training).all() or not np.isfinite(val_x).all():
        raise ValueError("objective experiment requires the47 finite retained features")
    model_files = ["config.json"]
    for name in ARMS:
        stage = f"{name}_complete.json"
        model_file, info_file = f"{name}.joblib", f"{name}.json"
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = joblib.load(run/model_file)
        else:
            model = make_reference(name, reference, seed)
            with threadpool_limits(limits=2):
                model.fit(training, reference_target(np.asarray(dataset["y"]).ravel()[split["train"]], name))
            joblib.dump(model, run/model_file, compress=3)
            write_json(run/info_file, {"model_name": name, "target_transform": ARMS[name],
                "parameters": model.get_params(), "fitted_cell_count": len(split["train"]),
                "fitted_stations": np.unique(split["train"]//shape[1]).tolist(),
                "source_fit_cells_sha256": digest(split["train"].tolist()),
                "n_features": training.shape[1], "early_stopping": False,
                "elapsed_seconds": time.monotonic()-started})
            bind_files(run, stage, [run/model_file, run/info_file], config)
        with threadpool_limits(limits=2):
            prediction = predict_reference(model, val_x, name)
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, prediction
        frames.append(frame)
        model_files += [model_file, info_file, stage]
        record = {"run": run.name, "stage": name, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)
    np.savez_compressed(run/"validation_inputs.npz", cell=val, features=val_x)
    write_json(run/"feature_definition.json", {"n_features": 47,
        "construction": "unchanged station_hidden_tree_inputs;39 monthly/context plus8 daily hydro",
        "feature_target_transform": "log1p for every arm", "additional_features": [],
        "retained_forest_parameters": reference.get_params()})
    model_files += ["validation_inputs.npz", "feature_definition.json"]
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, model_files)
    bind_files(run, "complete.json", [run/name for name in (*model_files, "predictions.parquet", "predictions.meta.json")], config)
    write_json(root/"progress.json", {"run": run.name, "stage": "complete", "elapsed_seconds": time.monotonic()-started})
    del training, inference, dataset, model, reference, frames
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_native_reference_objective_v1.py",
        "scripts/analyze_doc_native_reference_objective_v1.py", "scripts/plot_doc_native_reference_objective_v1.py",
        "scripts/run_doc_unmonitored_trees_v1.py", "scripts/run_doc_daily_hydro_residual_v1.py",
        "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py",
        "tests/test_doc_reference_objectives.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve saved reference objective execution after code changes")
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
