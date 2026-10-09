"""Compare source-only station balancing with the matched saved DOC recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_station_balanced_residual_v1 import ROOT
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    frames, metrics, sources = [], [], {}
    expected = {f"split{split}_seed{seed}" for split in (142, 143, 144) for seed in (42, 43, 44)}
    completions = sorted((args.root/"runs").glob("*/complete.json"))
    if {path.parent.name for path in completions} != expected:
        raise ValueError("all nine fixed source packages must be complete")
    for completion in completions:
        run = completion.parent
        config = json.loads((run/"config.json").read_text())
        verify_files(run, "complete.json", config)
        panel = pd.read_parquet(run/"predictions.parquet")
        if not panel.visibility_role.eq("val").all() or set(panel.model_name) != set(config["models"]):
            raise ValueError("only complete source-validation comparisons may be analyzed")
        for name, group in panel.groupby("model_name"):
            error = group.y_pred-group.y_true
            tail = group.y_true >= config["q90_threshold_train"]
            metrics.append({"model_name": name, "split_seed": config["split_seed"], "seed": config["seed"],
                "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "station_equal_mae": float(group.assign(error=np.abs(error)).groupby("station").error.mean().mean()),
                "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else np.nan,
                "q90_bias": float(error[tail].mean()) if tail.any() else np.nan,
                "n_cells": len(group), "n_stations": group.station.nunique(), "q90_n": int(tail.sum())})
        control = paired(panel, "cell_equal_replay", "unmonitored_residual")
        np.testing.assert_allclose(control.y_pred_candidate, control.y_pred_reference, rtol=1e-10, atol=1e-10)
        frames.append(panel)
        sources[str(completion)] = sha256_file(completion)
    panel, runs = pd.concat(frames, ignore_index=True), pd.DataFrame(metrics)
    fields = ["mae", "bias", "rmse", "station_equal_mae", "q90_mae", "q90_bias"]
    parts = runs.groupby(["split_seed", "model_name"], as_index=False)[fields].mean()
    summary = parts.groupby("model_name", as_index=False)[fields].mean()
    contrasts = [("station_equal_residual", "unmonitored_residual"),
        ("station_equal_integrated", "unmonitored_integrated"),
        ("station_equal_integrated", "current_model"),
        ("station_equal_integrated", "station_hidden_trees")]
    effects = []
    for candidate, reference in contrasts:
        pair = paired(panel, candidate, reference)
        direction = pair.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
        part_direction = direction.groupby("split_seed").mean()
        effects.append({"candidate": candidate, "reference": reference,
            **joint_station_bootstrap(pair, draws=args.bootstrap_draws),
            "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
            "improved_partitions": int((part_direction.candidate_error < part_direction.reference_error).sum())})
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    runs.to_csv(output/"run_metrics.csv", index=False)
    parts.to_csv(output/"partition_metrics.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    pd.DataFrame(effects).to_csv(output/"paired_effects.csv", index=False)
    write_json(output/"sources.json", {"scope": "source-validation development only", "runs": sources,
        "cell_control_replayed": True, "bootstrap_draws": args.bootstrap_draws,
        "analyzer_sha256": sha256_file(__file__), "weighting": "seed means within partition; equal three partitions"})
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(effects).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
