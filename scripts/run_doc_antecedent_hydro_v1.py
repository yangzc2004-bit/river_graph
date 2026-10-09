"""Compare antecedent hydro-climate states on source-only new-site DOC tasks."""
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
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.antecedent_hydro_states import antecedent_hydro_states
from river_graph.models.kgml_local_transport import station_folds

ROOT = Path("experiments/phase4_transfer/doc_antecedent_hydro_v1")
ARMS = (("hydro_availability_trees", "availability_only"), ("hydro_state_trees", "full"))


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
        "models": [*old["models"], *(name for name, _ in ARMS)],
        "main_change": "causal seasonal flow, antecedent deficit, recovery and thermal history; matched availability control",
        "tree_hyperparameters": "retained source-selected300-tree reference, unchanged",
        "selection_role": "source_validation", "evaluation_role": "source_validation_development"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if any(saved[key] != config[key] for key in ("runtime_snapshot_hash", "parent_completion_hash")):
            raise ValueError("saved antecedent hydro source experiment changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("antecedent hydro source inputs changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    shape = dataset["y"].shape
    states = antecedent_hydro_states(dataset)
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
    model_files = ["config.json"]
    cache = {"cell": val}
    for name, key in ARMS:
        # These descriptors read current/past hydro inputs only, not water-quality roles.
        block = states[key].reshape(-1, 8)
        train_x = np.column_stack([training, block[split["train"]]])
        val_x = np.column_stack([inference[val], block[val]])
        model = clone(reference).set_params(n_jobs=2)
        model.fit(train_x, np.log1p(np.asarray(dataset["y"]).ravel()[split["train"]]))
        prediction = np.maximum(0., np.expm1(model.predict(val_x)))
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, prediction
        frames.append(frame)
        joblib.dump(model, run/f"{name}.joblib", compress=3)
        model_files.append(f"{name}.joblib")
        cache[name] = val_x
        record = {"run": run.name, "stage": name, "n_features": train_x.shape[1],
                  "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)
    np.savez_compressed(run/"validation_inputs.npz", **cache)
    write_json(run/"feature_names.json", {"preceding_feature_count": training.shape[1],
        "added_feature_names": states["feature_names"], "values_scale": "bounded causal hydro-climate descriptors; temperature differences in Celsius"})
    model_files += ["validation_inputs.npz", "feature_names.json"]
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, model_files)
    bind_files(run, "complete.json", [run/name for name in (*model_files, "predictions.parquet", "predictions.meta.json")], config)
    write_json(root/"progress.json", {"run": run.name, "stage": "complete", "elapsed_seconds": time.monotonic()-started})
    del training, inference, dataset, model, train_x, val_x, reference, frames
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_antecedent_hydro_v1.py",
        "scripts/analyze_doc_antecedent_hydro_v1.py",
        "scripts/run_doc_unmonitored_trees_v1.py", "scripts/run_doc_daily_hydro_residual_v1.py",
        "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py", "tests/test_antecedent_hydro_states.py",
        str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve saved antecedent hydro source execution after code changes")
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
