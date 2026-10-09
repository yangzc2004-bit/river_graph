"""Develop a tree base with station-hidden source inputs matched to K0 queries."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
    station_folds,
)

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_trees_v1")
PRIOR = Path("experiments/phase4_transfer/doc_source_retrieval_v1")


def station_hidden_tree_inputs(dataset, split, folds, daily):
    train = np.asarray(split["train"], dtype=np.int64)
    months = dataset["y"].shape[1]
    inference = np.column_stack([build_rf_features(dataset, split, FIT_ROLES,
        target_transform="log1p", include_network=True), daily.reshape(-1, 8)])
    training = np.empty((len(train), inference.shape[1]), np.float32)
    for stations in folds:
        rows = np.flatnonzero(np.isin(train//months, stations))
        view = build_rf_features(dataset, fold_split(split, stations, months), FIT_ROLES,
                                 target_transform="log1p", include_network=True)
        training[rows] = np.column_stack([view[train[rows]], daily.reshape(-1, 8)[train[rows]]])
    # Legacy context indices27:39 are target lag value/visibility/month fields.
    if not np.array_equal(training[:, [27, 28, 30, 31, 33, 34, 36, 37]],
                          np.zeros((len(training), 8))):
        raise ValueError("source training stations retain a local water-quality history")
    return training, inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_unmonitored_trees_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    snapshot_path = args.root / "runtime_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise ValueError("existing execution code changed")
    if not snapshot_path.exists():
        write_json(snapshot_path, snapshot)
        for name in snapshot:
            target = args.root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    runtime = digest(snapshot)
    for partition in args.splits:
        for seed in args.seeds:
            parent = PRIOR / "runs" / f"split{partition}_seed{seed}"
            old = json.loads((parent / "config.json").read_text())
            verify_files(parent, "complete.json", old)
            run = args.root / "runs" / parent.name
            run.mkdir(parents=True, exist_ok=True)
            if (run / "complete.json").exists():
                config = json.loads((run / "config.json").read_text())
                if config["runtime_snapshot_hash"] != runtime:
                    raise ValueError("saved tree execution changed")
                verify_files(run, "complete.json", config)
                continue
            config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                      "q90_threshold_train", "split_seed", "seed")}, "experiment": "doc_unmonitored_trees_v1",
                "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
                "parent_run": str(parent), "parent_completion_hash": sha256_file(parent / "complete.json"),
                "models": ["current_model", "matched_daily_trees", "station_hidden_trees"],
                "main_change": "every source training row uses its station-fold-hidden view",
                "selection_role": "source_validation", "evaluation_role": "source_validation_only",
                "target_information": "no local DOC/pH/conductance history; source-neighbor DOC context available",
                "forest_candidates": ["leaf4", "leaf2", "leaf1", "sqrt"], "n_estimators": 300}
            write_json(run / "config.json", config)
            dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
            with np.load(config["mask_path"], allow_pickle=False) as saved:
                split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
            dataset = strip_auxiliary_water(dataset)
            dataset["y"] = development_labels(dataset, split)
            monthly = Path(old["parent_run"])
            daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], dataset["y"].shape)
            folds = station_folds(split["train"], dataset["y"].shape[1], seed)
            training, inference = station_hidden_tree_inputs(dataset, split, folds, daily)
            truth = np.asarray(dataset["y"]).ravel()
            scores, best = [], np.inf
            for name, leaf, maximum in (("leaf4", 4, 1.), ("leaf2", 2, 1.), ("leaf1", 1, 1.), ("sqrt", 1, "sqrt")):
                forest = ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=2,
                    min_samples_leaf=leaf, max_features=maximum).fit(training, np.log1p(truth[split["train"]]))
                prediction = np.maximum(0, np.expm1(forest.predict(inference[split["val"]])))
                score = float(np.abs(prediction-truth[split["val"]]).mean())
                scores.append({"candidate": name, "validation_mae": score})
                if score < best:
                    best, selected, chosen = score, prediction, forest
            joblib.dump(chosen, run / "station_hidden_trees.joblib", compress=3)
            write_json(run / "tree_selection.json", scores)
            prior = pd.read_parquet(parent / "predictions.parquet")
            output = prior[prior.model_name.isin(config["models"])].copy()
            frame = prior[prior.model_name.eq("current_model")].copy()
            frame["model_name"], frame["y_pred"] = "station_hidden_trees", selected
            pd.concat([output, frame], ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
            files = ["config.json", "tree_selection.json", "station_hidden_trees.joblib"]
            bind_product(run, "predictions.parquet", config, runtime, files)
            bind_files(run, "complete.json", [run / name for name in
                (*files, "predictions.parquet", "predictions.meta.json")], config)
            record = {"run": run.name, "stage": "complete", "validation_mae": best}
            write_json(args.root / "progress.json", record)
            print(json.dumps(record), flush=True)
            del training, inference, dataset, forest, chosen
            gc.collect()


if __name__ == "__main__":
    main()
