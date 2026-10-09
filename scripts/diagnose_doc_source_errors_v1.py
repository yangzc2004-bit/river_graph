"""Decompose retained source-validation errors without fitting a new predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_source_retrieval_v1 import load_panel
from run_doc_unmonitored_residual_v1 import ROOT as SOURCE
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=Path("experiments/phase4_transfer/doc_composition_encoder_v1/source_error_diagnostics"))
    args = parser.parse_args()
    files = sorted((SOURCE/"runs").glob("*/complete.json"))
    if {path.parent.name for path in files} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("the nine retained source-role packages are required")
    for path in files:
        config = json.loads((path.parent/"config.json").read_text())
        verify_files(path.parent, "complete.json", config)
    panel, thresholds = load_panel(SOURCE)
    panel = panel[panel.model_name.isin(["unmonitored_integrated", "station_hidden_trees"])].copy()
    rows, station_rows = [], []
    for (partition, seed, model), group in panel.groupby(["split_seed", "seed", "model_name"]):
        group = group.copy()
        error = group.y_pred.to_numpy()-group.y_true.to_numpy()
        group["error"], group["absolute_error"] = error, np.abs(error)
        group["tail"] = group.y_true >= thresholds[(partition, seed)]
        means = group.groupby("station").error.transform("mean").to_numpy()
        mse, persistent, within = (float(np.mean(x*x)) for x in (error, means, error-means))
        np.testing.assert_allclose(mse, persistent+within, rtol=1e-12, atol=1e-12)
        tail = group["tail"].to_numpy(bool)
        station = group.groupby("station", as_index=False).agg(n_cells=("cell", "size"),
            mae=("absolute_error", "mean"), bias=("error", "mean"),
            observed_mean=("y_true", "mean"), predicted_mean=("y_pred", "mean"),
            tail_cells=("tail", "sum"))
        station["split_seed"], station["seed"], station["model_name"] = partition, seed, model
        station_rows.append(station)
        losses = group.groupby("station").absolute_error.sum().sort_values(ascending=False)
        n_high = max(1, int(np.ceil(.2*len(losses))))
        total_abs, total_squared = float(np.abs(error).sum()), float((error*error).sum())
        rows.append({"split_seed": partition, "seed": seed, "model_name": model,
            "mae": float(np.abs(error).mean()), "bias": float(error.mean()), "mse": mse,
            "persistent_station_bias_mse_share": persistent/mse,
            "within_station_error_mse_share": within/mse,
            "q90_cell_fraction": float(tail.mean()),
            "q90_absolute_error_share": float(np.abs(error[tail]).sum()/total_abs),
            "q90_squared_error_share": float((error[tail]**2).sum()/total_squared),
            "q90_bias": float(error[tail].mean()), "q90_mae": float(np.abs(error[tail]).mean()),
            "non_tail_mae": float(np.abs(error[~tail]).mean()),
            "top20pct_station_absolute_error_share": float(losses.iloc[:n_high].sum()/total_abs),
            "n_cells": len(group), "n_stations": group.station.nunique()})
    runs = pd.DataFrame(rows)
    measures = [name for name in runs if name not in ("split_seed", "seed", "model_name", "n_cells", "n_stations")]
    parts = runs.groupby(["split_seed", "model_name"], as_index=False)[measures].mean()
    summary = parts.groupby("model_name", as_index=False)[measures].mean()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("run_diagnostics", runs), ("partition_diagnostics", parts),
        ("summary", summary), ("station_errors", pd.concat(station_rows, ignore_index=True))):
        frame.to_csv(args.output/f"{name}.csv", index=False)
    write_json(args.output/"sources.json", {"analyzer_sha256": sha256_file(__file__),
        "scope": "retained source-validation errors only; no new model fitting or target evaluation",
        "weighting": "seed means within partition; equal partitions",
        "decomposition": "MSE = cell-weighted squared station mean error + within-station error variance",
        "interpretation": "ex-post descriptive error budget; station bias is not available at K0 and is not a causal attribution",
        "runs": {path.parent.name: sha256_file(path) for path in files}})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
