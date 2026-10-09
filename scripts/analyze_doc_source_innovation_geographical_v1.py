"""Analyze fixed contemporaneous-source geographical replication and K curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_geographical_confirmation_v1 import (
    additional_metrics,
    improving_region_count,
    strata_for_run,
    summaries,
)
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_geographical_confirmation_v1 import TASKS
from run_doc_source_innovation_geographical_v1 import ARMS, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json
from verify_doc_source_innovation_geographical_v1 import verify

from river_graph.experiments.provenance import sha256_file


def source_support_strata(run, panel):
    """Fixed donor-count groups; no thresholds selected from target outcomes."""
    with np.load(run/"innovation_inputs.npz", allow_pickle=False) as saved:
        count = saved["support_count"].ravel()[panel.cell.to_numpy()]
    groups = np.where(count == 0, "none", np.where(count <= 3, "1-3", "4+"))
    frame = panel.assign(group=groups, error=np.abs(panel.y_pred-panel.y_true))
    config = json.loads((run/"config.json").read_text())
    rows = []
    for (model, k, group), values in frame.groupby(["model_name", "k", "group"]):
        rows.append({"split_seed": config["split_seed"], "seed": config["seed"],
            "model_name": model, "k": k, "stratum": "current_source_support", "group": group,
            "mae": float(values.error.mean()), "n_cells": len(values)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    regions = json.loads((TASKS/"task_protocol.json").read_text())["geographical_targets"]
    paths = [args.root/"runs"/f"huc4_{region}_seed{seed}" for region in regions for seed in args.seeds]
    if any(not (path/"complete.json").exists() for path in paths):
        raise ValueError("complete every fixed region/seed package before analysis")
    write_json(args.root/"verification/replay.json", [verify(path, runtime) for path in paths])
    if args.verify_only:
        return
    frames, curves, thresholds, stratification = [], [], {}, []
    for path in paths:
        config = json.loads((path/"config.json").read_text())
        primary = pd.read_parquet(path/"predictions.parquet")
        frames.append(primary)
        curves.append(pd.read_parquet(path/"support_curves.parquet"))
        thresholds[(config["split_seed"], config["seed"])] = config["q90_threshold_train"]
        stratification.extend(strata_for_run(path, primary))
        stratification.extend(source_support_strata(path, primary))
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    primary, curves = pd.concat(frames, ignore_index=True), pd.concat(curves, ignore_index=True)
    for label, panel in (("primary", primary), ("curves", curves)):
        runs, parts, summary, station_runs = summaries(panel, thresholds)
        for name, table in (("run_metrics", runs), ("region_metrics", parts), ("summary", summary),
                            ("station_metrics", station_runs), ("q90_diagnostics", additional_metrics(panel, thresholds))):
            table.to_csv(output/f"{label}_{name}.csv", index=False)
    effects, stations = [], []
    comparisons = [(ARMS[1], reference) for reference in
                   ("unmonitored_integrated", "current_model", "station_hidden_trees", ARMS[0])]
    comparisons += [(ARMS[0], "unmonitored_residual"), (ARMS[2], "station_hidden_trees"), (ARMS[1], ARMS[2])]
    for population, panel in (("all_observed_k0", primary), ("fixed_query_curve", curves)):
        for k in sorted(panel.k.unique()):
            for candidate, reference in comparisons:
                comparison = paired(panel[panel.k.eq(k)], candidate, reference)
                for zone in ("overall", "q90"):
                    selected = comparison
                    if zone == "q90":
                        limit = np.array([thresholds[(s, t)] for s, t in zip(
                            comparison.split_seed, comparison.seed, strict=True)])
                        selected = comparison[comparison.y_true_candidate.to_numpy() >= limit]
                    record = {"candidate": candidate, "reference": reference, "population": population, "k": k, "zone": zone,
                              "n_regions_usable": selected.split_seed.nunique()}
                    if selected.split_seed.nunique() != len(regions):
                        effects.append({**record, "relative_gain_pct": np.nan, "gain_ci_low_pct": np.nan,
                            "gain_ci_high_pct": np.nan, "improving_regions": np.nan,
                            "status": "not estimable for five-region mean: empty tail"})
                        continue
                    effects.append({**record, **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                        "improving_regions": improving_region_count(selected), "status": "estimated"})
                group = comparison.groupby(["split_seed", "station"], as_index=False).agg(
                    candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"), n_cells=("cell", "nunique"))
                group["delta_mae"] = group.candidate_mae-group.reference_mae
                group["candidate"], group["reference"], group["k"], group["population"] = candidate, reference, k, population
                stations.append(group)
    pd.DataFrame(effects).to_csv(output/"paired_effects.csv", index=False)
    pd.concat(stations, ignore_index=True).to_csv(output/"station_effects.csv", index=False)
    strata = pd.DataFrame(stratification)
    strata.to_csv(output/"strata_by_run.csv", index=False)
    region_strata = strata.groupby(["split_seed", "model_name", "k", "stratum", "group"], as_index=False).mae.mean()
    region_strata.to_csv(output/"strata_by_region.csv", index=False)
    region_strata.groupby(["model_name", "k", "stratum", "group"], as_index=False).agg(
        mae=("mae", "mean"), n_regions=("split_seed", "nunique")).to_csv(output/"strata_summary.csv", index=False)
    write_json(output/"sources.json", {"bootstrap_draws": args.bootstrap_draws, "seeds": args.seeds,
        "regions": regions, "estimand": "seed mean within region, equal regions", "runtime": runtime,
        "scope": "retrospective ST357 replication of source-selected contemporaneous information",
        "analyzer_sha256": sha256_file(__file__), "completed_runs": {str(path/"complete.json"): sha256_file(path/"complete.json") for path in paths}})
    print(pd.read_csv(output/"primary_summary.csv")[["model_name", "mae", "station_equal_mae", "q90_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
