"""Decompose observed tributary integration, arrival opportunity and data support."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_form_process import aggregate_receivers, cluster_mean
from river_graph.analysis.river_observed_transport import calibrated_prediction
from river_graph.analysis.river_signal_mechanisms import (
    arrival_opportunity,
    flow_budget_records,
    mixing_records,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_signal_mechanisms_v1")
OLD = Path("experiments/phase4_transfer/doc_river_observed_transport_v1")
INPUT = OLD/"analysis/input_records.parquet"
STATES = OLD/"analysis/fitted_states.json"
ORIGIN = OLD/"analysis/connection_predictions.provenance.json"
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
CELLS = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
CONTEXT = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1/analysis/network_context.csv")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_signal_mechanisms_v1.py",
    "src/river_graph/analysis/river_signal_mechanisms.py", "src/river_graph/analysis/river_mechanisms.py",
    "src/river_graph/analysis/river_form_process.py", "src/river_graph/analysis/river_observed_transport.py")))
MIX_METRICS = ("source_rho", "branch_balance", "mixture_buffer_fraction", "amplitude_buffer_fraction",
    "asynchronous_buffer_fraction", "equal_amplitude_buffer_fraction", "balanced_equal_amplitude_buffer_fraction",
    "outlet_reference_log_sd_ratio", "outlet_mixture_log_sd_ratio", "outlet_reference_variance_ratio",
    "logscale_outlet_reference_log_sd_ratio", "logscale_outlet_mixture_log_sd_ratio", "source_drainage_coverage")
ARRIVAL_METRICS = ("arrival_gap_abs", "arrival_gap_relative", "arrival_gap_over_1pct", "arrival_gap_over_5pct",
                   "prediction_gap_abs", "prediction_gap_relative")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def measured_flows(frame, dataset):
    lookup = {str(s): i for i, s in enumerate(dataset["site_no"])}
    channel = list(dataset["feature_channels"]).index("discharge")
    x, mask = np.asarray(dataset["x"], float), np.asarray(dataset["x_mask"], bool)
    result = []
    for name in ("source_a", "source_b", "target"):
        nodes = frame[name].map(lookup)
        if nodes.isna().any():
            raise ValueError("every receiver/source must be a dataset node")
        nodes, months = nodes.to_numpy(int), frame.month_index.to_numpy(int)
        q = x[nodes, months, channel]
        valid = mask[nodes, months, channel] & np.isfinite(q) & (q > 0)
        result.append(np.where(valid, q, np.nan))
    return result


def receiver_table(frame, metrics):
    result = aggregate_receivers(frame, metrics)
    classes = frame.groupby("target", as_index=False).cluster.first()
    context = pd.read_csv(CONTEXT, dtype={"station": str, "huc4": str})
    return result.merge(classes, on="target", validate="one_to_one").merge(
        context[["station", "basin_area_km2", "mainstem_share", "route_distance_cv"]],
        left_on="target", right_on="station", validate="one_to_one").drop(columns="station")


def summarize(frame, metrics, draws):
    rows = []
    for group, f in [("all", frame), *[(f"form_{k}", frame[frame.cluster.eq(k)]) for k in (1, 2, 3)]]:
        for unit in ("component", "huc4"):
            for metric in metrics:
                row = cluster_mean(f, metric, unit, draws=draws)
                if row is not None:
                    if group != "all" and f.dropna(subset=[metric]).component.nunique() < 2:
                        row.update({"ci_low": np.nan, "ci_high": np.nan,
                            "interval_status": "not_estimable_single_monitoring_system", "valid_bootstrap_draws": 0})
                    rows.append({"population": group, **row})
    return pd.DataFrame(rows)


def analyze(frame, dataset, states, max_path, draws):
    mixing, anomalies, calendar_fits = mixing_records(frame)
    receivers = receiver_table(mixing, MIX_METRICS)
    arrival = arrival_opportunity(frame, max_path)
    state_lookup = {int(s["held_component"]): s for s in states if s["operator"] == "mean_delay"}
    for held, indices in arrival.groupby("component").groups.items():
        s, f = state_lookup[int(held)], arrival.loc[indices]
        if int(held) in s["training_components"] or s["fraction"] != 1.:
            raise ValueError("reuse the saved held-system mean-delay fit at fraction=1")
        for proxy, output in (("mean_delay_proxy", "mean_input_prediction"), ("branch_arrival_proxy", "branch_input_prediction")):
            arrival.loc[indices, output] = calibrated_prediction(f, f[proxy].to_numpy(), s)
    arrival["prediction_gap_abs"] = abs(arrival.branch_input_prediction-arrival.mean_input_prediction)
    arrival["prediction_gap_relative"] = arrival.prediction_gap_abs/(1+arrival.mean_input_prediction)
    arrival_pairs = arrival.groupby(["pair_id", "target", "huc4", "component", "cluster"], as_index=False)[list(ARRIVAL_METRICS)].mean()
    arrival_receivers = receiver_table(arrival_pairs, ARRIVAL_METRICS)
    budget = flow_budget_records(frame, *measured_flows(frame, dataset))
    coverage = []
    for name, keep in (("common_doc", np.ones(len(budget), bool)),
            ("three_measured_flows", budget.three_flows_measured), ("area_ge_80pct", budget.area_screen),
            ("area_and_monthly_flow_screen", budget.budget_screen)):
        f = budget[keep]
        coverage.append({"population": name, "n_connection_months": len(f),
            "n_unique_receiver_months": len(f.drop_duplicates(["target", "month_index"])),
            "n_connections": f.pair_id.nunique(), "n_receivers": f.target.nunique(),
            "n_components": f.component.nunique(), "n_huc4": f.huc4.nunique()})
    budget_metrics = ("apparent_departure_mg_l", "flow_closure", "area_flow_share_difference")
    rows = []
    for population, keep in (("three_measured_flows", budget.three_flows_measured),
                             ("area_and_monthly_flow_screen", budget.budget_screen)):
        f = budget[keep]
        grouped = f.groupby(["pair_id", "target", "huc4", "component", "cluster"], as_index=False)[list(budget_metrics)].mean()
        r = receiver_table(grouped, budget_metrics)
        for unit in ("component", "huc4"):
            for metric in budget_metrics:
                row = cluster_mean(r, metric, unit, draws=draws)
                if row is not None:
                    rows.append({"population": population, **row})
    tables = {"mixing_connections": mixing, "mixing_receivers": receivers,
              "mixing_summary": summarize(receivers, MIX_METRICS, draws),
              "arrival_connections": arrival_pairs, "arrival_receivers": arrival_receivers,
              "arrival_summary": summarize(arrival_receivers, ARRIVAL_METRICS, draws),
              "budget_availability": pd.DataFrame(coverage), "apparent_departure_summary": pd.DataFrame(rows)}
    influence = []
    for population, r, metrics in (("mixing", receivers, MIX_METRICS), ("arrival", arrival_receivers, ARRIVAL_METRICS)):
        for component in sorted(r.component.unique()):
            remaining = r[r.component.ne(component)]
            for metric in metrics:
                influence.append({"population": population, "omitted_component": int(component), "metric": metric,
                    "estimate": remaining[metric].mean(), "n_remaining_receivers": len(remaining)})
    tables["omitted_system_sensitivity"] = pd.DataFrame(influence)
    products = {"calendar_anomalies": anomalies, "arrival_opportunity": arrival, "monthly_flow_ledger": budget}
    return tables, products, calendar_fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    frame = pd.read_parquet(INPUT)
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    states = json.loads(STATES.read_text())
    previous = json.loads((OLD/"config.json").read_text())
    config = {"bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42,
        "aggregation": "dates within pair, pairs within receiver, receivers equal",
        "primary_bootstrap_unit": "component", "sensitivity_bootstrap_unit": "huc4",
        "calendar_adjustment": ["intercept", "month_sin", "month_cos", "centered_year"],
        "variance_scale": "native DOC population variance", "arrival_fraction": 1.,
        "max_path_km": previous["max_path_km"], "arrival_relative_denominator": "1 + same-month mixture",
        "arrival_relative_thresholds": [.01, .05], "minimum_area_coverage": .8,
        "monthly_flow_closure_range": [.8, 1.2], "previous_results_seen": True, "model_refitting": False,
        "supplementary_after_native_results": ["same-dates log1p variability", "leave-one-system summaries"]}
    # Write the fixed method before calculating response summaries.
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    tables, products, fits = analyze(frame, data, states, config["max_path_km"], args.bootstrap_draws)
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)
    for name, product in products.items():
        product.to_parquet(out/f"{name}.parquet", index=False)
    (out/"calendar_fits.json").write_text(json.dumps(fits, indent=2)+"\n")
    sources = [INPUT, STATES, ORIGIN, DATASET, CELLS, CONTEXT, OLD/"config.json", OLD/"analysis_sources.json", ROOT/"study_plan.md", *CODE]
    hashes = {str(p): sha256_file(p) for p in sources}
    for path in CODE:
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    record = {"source_hashes": hashes, "config_hash": digest(config), "bootstrap_draws": args.bootstrap_draws,
              "n_connection_months": len(frame), "n_connections": frame.pair_id.nunique(),
              "n_receivers": frame.target.nunique(), "n_components": frame.component.nunique(),
              "n_unique_receiver_months": len(frame.drop_duplicates(["target", "month_index"])),
              "previous_results_seen": True, "neural_training": False, "created_date": "2026-10-07"}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    runtime = {str(p): hashes[str(p)] for p in CODE}
    provenance = {"config": config, "config_hash": digest(config), "dataset_hash": hashes[str(DATASET)],
        "mask_hash": json.loads(ORIGIN.read_text())["mask_hash"], "source_cells_sha256": hashes[str(CELLS)],
        "runtime_sources": runtime, "runtime_snapshot_hash": digest(runtime),
        "prediction_sha256": sha256_file(out/"arrival_opportunity.parquet"),
        "input_records_sha256": hashes[str(INPUT)], "saved_states_sha256": hashes[str(STATES)],
        "prediction_columns": ["mean_input_prediction", "branch_input_prediction"],
        "prediction_grain": "connection x observed receiver month; receiver truth repeats across connections",
        "product_role": "fixed saved model input sensitivity; not a newly fitted model",
        "visibility": "current and previous source DOC permitted; outlet DOC not used to alter proxies or fitted states",
        "calibrators_refitted": False, "previous_results_seen": True}
    (out/"arrival_opportunity.provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(tables["mixing_summary"].query("population == 'all' and unit == 'component'").to_string(index=False))
    print(tables["arrival_summary"].query("population == 'all' and unit == 'component'").to_string(index=False))
    print(tables["budget_availability"].to_string(index=False))


if __name__ == "__main__":
    main()
