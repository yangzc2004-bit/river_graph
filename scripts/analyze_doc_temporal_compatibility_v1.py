"""Recompute matched temporal refit errors without changing trained models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_temporal_compatibility_v1 import MASKS, ROOT
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    completions = sorted((args.root/"runs").glob("*/complete.json"))
    expected = {f"{mask}_seed{seed}" for mask in MASKS for seed in (42, 43, 44)}
    if {path.parent.name for path in completions} != expected:
        raise ValueError("all six fixed temporal refits must finish")
    frames, records, hashes = [], [], {}
    for completion in completions:
        run = completion.parent
        config = json.loads((run/"config.json").read_text())
        for marker in ("backbone_complete.json", "trees_complete.json", "current_complete.json", "complete.json"):
            verify_files(run, marker, config)
        dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        with np.load(config["mask_path"], allow_pickle=False) as split:
            cells = np.sort(split["test"])
        frame = pd.read_parquet(run/"predictions.parquet")
        if not frame.visibility_role.eq("test").all() or set(frame.model_name) != set(config["models"]):
            raise ValueError("invalid temporal comparison panel")
        for name, group in frame.groupby("model_name"):
            group = group.sort_values("cell")
            np.testing.assert_array_equal(group.cell, cells)
            np.testing.assert_array_equal(group.y_true, np.asarray(dataset["y"]).ravel()[cells])
            np.testing.assert_array_equal(group.station, np.asarray(dataset["site_no"], str)[cells//dataset["y"].shape[1]])
            error = group.y_pred-group.y_true
            tail = group.y_true >= config["q90_threshold_train"]
            total = np.sum((group.y_true-group.y_true.mean())**2)
            records.append({"mask": config["mask"], "seed": config["seed"], "model_name": name,
                "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
                "r2": float(1-np.sum(error**2)/total), "bias": float(error.mean()),
                "log_mae": float(np.abs(np.log1p(group.y_pred)-np.log1p(group.y_true)).mean()),
                "station_equal_mae": float(group.assign(error=np.abs(error)).groupby("station").error.mean().mean()),
                "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else np.nan,
                "q90_n": int(tail.sum()), "q90_unstable": int(tail.sum()) < 20,
                "n_cells": len(cells), "n_stations": group.station.nunique()})
        frame["split_seed"] = MASKS.index(config["mask"])+1
        frames.append(frame)
        hashes[str(completion)] = sha256_file(completion)
    panel, runs = pd.concat(frames, ignore_index=True), pd.DataFrame(records)
    summary = runs.groupby(["mask", "model_name"], as_index=False).mean(numeric_only=True).drop(columns="seed")
    effects = []
    for mask, task in panel.groupby("mask"):
        for candidate, reference in (("upgraded_fusion", "original_hybrid"),
                ("upgraded_fusion", "current_fusion"), ("upgraded_fusion", "station_hidden_trees"),
                ("upgraded_native", "current_native"), ("upgraded_native", "station_hidden_trees"),
                ("current_fusion", "context_trees")):
            pair = paired(task, candidate, reference)
            direction = pair.groupby("seed")[["candidate_error", "reference_error"]].mean()
            effects.append({"mask": mask, "candidate": candidate, "reference": reference,
                **joint_station_bootstrap(pair, draws=args.bootstrap_draws),
                "improved_seeds": int((direction.candidate_error < direction.reference_error).sum())})
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    runs.to_csv(output/"run_metrics.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    pd.DataFrame(effects).to_csv(output/"paired_effects.csv", index=False)
    write_json(output/"sources.json", {"runs": hashes, "bootstrap_draws": args.bootstrap_draws,
        "analyzer_sha256": sha256_file(__file__), "estimand": "mean individual-seed errors within each temporal mask",
        "scope": "retrospective temporal compatibility; spatial memory not used"})
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(effects).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
