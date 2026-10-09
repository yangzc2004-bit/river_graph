"""Link real channel geometry to DOC peak, timing and carbon export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_multicatchment_pulses_v1 import load_cases
from scipy.stats import rankdata

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_geometry_budget import (
    observed_window_budget,
    route_water_carbon,
)
from river_graph.analysis.river_multicatchment_pulses import hourly_observations
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_geometry_budget_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_multicatchment_pulses_v1")
INVENTORY = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1/analysis/tributary_inventory.csv")
METRICS = ["concentration_peak_excess", "concentration_peak_time", "concentration_flow_peak_lag",
           "concentration_half_width", "carbon_excess_centroid", "carbon_excess_sd",
           "retained_extra_carbon_fraction"]
CONTRASTS = {
    "path_dispersion": ("actual_paths", "equal_mean_paths"),
    "arrival_alignment": ("arrival_compensated", "actual_paths"),
    "shared_spreading": ("long_shared_distributed", "short_shared_distributed"),
    "junction_null": ("long_shared_deterministic", "short_shared_deterministic"),
    "pulse_processing": ("long_shared_reactive", "long_shared_distributed"),
}


def month_stat(frame, field, draws):
    s = frame.loc[np.isfinite(frame[field])].copy()
    groups = [g[field].to_numpy() for _, g in s.groupby("month_block")]
    if not groups:
        return {"median": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_events": 0, "n_months": 0}
    rng = np.random.default_rng(42)
    medians = [np.median(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
               for _ in range(draws)]
    lo, hi = np.quantile(medians, [.025, .975]) if len(groups) >= 2 else (np.nan, np.nan)
    return {"median": float(s[field].median()), "ci_low": lo, "ci_high": hi,
            "n_events": len(s), "n_months": len(groups)}


def rank_relation(frame, x, y, draws, controls=()):
    s = frame.dropna(subset=[x, y, *controls, "month_block"])

    def estimate(f):
        if len(f) < len(controls) + 5 or min(f[x].nunique(), f[y].nunique()) < 2:
            return np.nan
        a, b = rankdata(f[x]), rankdata(f[y])
        if controls:
            design = np.column_stack([np.ones(len(f)), *[rankdata(f[k]) for k in controls]])
            if np.linalg.matrix_rank(design) < design.shape[1]:
                return np.nan
            a -= design @ np.linalg.lstsq(design, a, rcond=None)[0]
            b -= design @ np.linalg.lstsq(design, b, rcond=None)[0]
        if min(np.std(a), np.std(b)) < 1e-10:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    groups = [f for _, f in s.groupby("month_block")]
    boot = []
    if len(s) >= 20 and len(groups) >= 4:
        rng = np.random.default_rng(42)
        for _ in range(draws):
            value = estimate(pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
            if np.isfinite(value):
                boot.append(value)
    lo, hi = np.quantile(boot, [.025, .975]) if len(boot) >= .9 * draws else (np.nan, np.nan)
    return {"x": x, "y": y, "controls": ";".join(controls), "rank_correlation": estimate(s),
            "ci_low": lo, "ci_high": hi, "n_events": len(s), "n_months": len(groups),
            "valid_draws": len(boot), "status": "exploratory_interval" if np.isfinite(lo) else "small_sample_or_unidentifiable"}


def geometry_experiment(draws):
    inventory = pd.read_csv(INVENTORY, dtype={"target": str, "huc4": str})
    ordered = inventory.sort_values(["path_imbalance", "pair_id"])
    representative = ordered.pair_id.iloc[len(ordered) // 2]
    records, traces = [], []
    for row in inventory.itertuples():
        paths = np.array([row.source_a_receiver_km, row.source_b_receiver_km]) / np.sqrt(row.basin_area_km2)
        common = row.junction_receiver_km / np.sqrt(row.basin_area_km2)
        weights = np.array([row.source_a_area_km2, row.source_b_area_km2])
        weights /= weights.sum()
        low, high = .2 * paths.min(), .8 * paths.min()
        scenarios = {
            "actual_paths": {"common": common},
            "equal_mean_paths": {"common": common, "path_factor": 0.},
            "arrival_compensated": {"common": common, "aligned": True},
            "short_shared_deterministic": {"common": low},
            "long_shared_deterministic": {"common": high},
            "short_shared_distributed": {"common": low, "dispersed": True},
            "long_shared_distributed": {"common": high, "dispersed": True},
            "long_shared_reactive": {"common": high, "dispersed": True, "reaction_rate": .25},
        }
        for amplitude in (1., 3.):
            for sigma in (.15, .30):
                for name, parameters in scenarios.items():
                    metrics, trace = route_water_carbon(paths, weights, water_amplitude=amplitude,
                                                       sigma=sigma, **parameters)
                    records.append({"pair_id": row.pair_id, "target": row.target, "huc4": row.huc4,
                        "component": row.component, "cluster": row.cluster, "scenario": name,
                        "water_amplitude": amplitude, "source_sigma": sigma,
                        "measured_common_path": common, "area_proxy_share_a": weights[0],
                        "source_drainage_coverage": row.source_drainage_coverage, **metrics})
                    if row.pair_id == representative and amplitude == 3. and sigma == .15:
                        traces.append(trace.assign(pair_id=row.pair_id, scenario=name))
    frame = pd.DataFrame(records)
    summaries = []
    keys = ["pair_id", "target", "huc4", "component", "cluster", "water_amplitude", "source_sigma"]
    for comparison, (candidate, reference) in CONTRASTS.items():
        a = frame.loc[frame.scenario.eq(candidate), keys + METRICS]
        b = frame.loc[frame.scenario.eq(reference), keys + METRICS]
        pairs = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        for metric in METRICS:
            pairs[metric] = pairs[metric + "_a"] - pairs[metric + "_b"]
        receivers = pairs.groupby(keys[1:], as_index=False)[METRICS].mean()
        for (amplitude, sigma), group in receivers.groupby(["water_amplitude", "source_sigma"]):
            for metric in METRICS:
                summaries.append({"comparison": comparison, "candidate": candidate, "reference": reference,
                    "water_amplitude": amplitude, "source_sigma": sigma,
                    **cluster_mean(group, metric, "component", draws=draws)})
    return frame, pd.DataFrame(summaries), pd.concat(traces, ignore_index=True)


def field_experiment(draws):
    records, series, input_paths = [], [], []
    for flow, doc, audit, _ in load_cases():
        case, area = audit["case"], audit["area_km2"]
        flow, doc = hourly_observations(flow, doc)
        joint = flow[["flow_timestamp_utc", "q_m3_s"]].merge(
            doc[["timestamp_utc", "doc_mg_l"]], left_on="flow_timestamp_utc", right_on="timestamp_utc",
            how="outer", validate="one_to_one")
        joint["clock"] = joint.flow_timestamp_utc.combine_first(joint.timestamp_utc)
        path = PREVIOUS / f"analysis/{case.lower()}_hourly_p20_events.csv"
        input_paths.append(path)
        events = pd.read_csv(path)
        for column in events.columns:
            if column.endswith("_clock"):
                events[column] = pd.to_datetime(events[column], utc=case == "Kervidy")
        eligible = events.loc[events.timing_eligible]
        example_id = eligible.loc[eligible.width_pair_eligible, "event_id"].iloc[0]
        for row in eligible.itertuples():
            for window, end in (("flow_window", row.flow_return_clock), ("response_window", row.response_end_clock)):
                budget = observed_window_budget(joint, row.start_clock, end, row.flow_peak_clock,
                                                row.flow_return_clock, row.doc_baseline_mg_l, area)
                records.append({**row._asdict(), "window": window, "area_km2": area,
                                "month_block": row.flow_peak_clock.strftime("%Y-%m"), **budget})
            if row.event_id == example_id:
                part = joint.loc[joint.clock.between(row.start_clock, row.response_end_clock)].copy()
                part["case"], part["event_id"] = case, example_id
                part["hours_from_flow_peak"] = (part.clock - row.flow_peak_clock).dt.total_seconds() / 3600
                part["carbon_flux_g_s"] = part.q_m3_s * part.doc_mg_l
                series.append(part[["case", "event_id", "clock", "hours_from_flow_peak", "q_m3_s", "doc_mg_l", "carbon_flux_g_s"]])
    frame = pd.DataFrame(records)
    stats, associations, inventories = [], [], []
    fields = ["carbon_yield_kg_km2", "flow_weighted_doc_mg_l", "positive_extra_fraction_of_export",
              "carbon_minus_water_after_peak_share", "carbon_after_flow_return_fraction",
              "flux_peak_minus_flow_peak_hours"]
    for (case, window), group in frame.groupby(["case", "window"]):
        complete = group.loc[group.budget_complete]
        inventories.append({"case": case, "window": window, "n_timing_eligible": len(group),
            "n_complete_budget": len(complete), "n_positive_responses": int(group.lag_eligible.sum()),
            "n_positive_complete": int(complete.lag_eligible.sum()),
            "n_complete_width_pairs": int(complete.width_pair_eligible.sum()),
            "n_incomplete_budget": int((~group.budget_complete).sum())})
        for cohort, s in (("all_timing_eligible", complete), ("positive_response", complete.loc[complete.lag_eligible])):
            for field in fields:
                stats.append({"case": case, "window": window, "cohort": cohort, "metric": field,
                              **month_stat(s, field, draws)})
        if window == "response_window":
            positive = complete.loc[complete.lag_eligible]
            for x, y in [("doc_peak_excess_mg_l", "doc_width_hours"),
                         ("doc_peak_excess_mg_l", "carbon_yield_kg_km2"),
                         ("doc_width_hours", "carbon_yield_kg_km2"),
                         ("doc_peak_lag_hours", "carbon_minus_water_after_peak_share")]:
                for controls in ((), ("specific_runoff_mm", "budget_duration_hours")):
                    associations.append({"case": case, **rank_relation(positive, x, y, draws, controls)})
    return (frame, pd.DataFrame(stats), pd.DataFrame(associations), pd.DataFrame(inventories),
            pd.concat(series, ignore_index=True), input_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("At least 100 bootstrap draws required")
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    simulations, contrasts, traces = geometry_experiment(args.bootstrap_draws)
    print(f"Geometry: {len(simulations)} cases on {simulations.pair_id.nunique()} pairs", flush=True)
    budgets, summaries, associations, inventory, examples, event_paths = field_experiment(args.bootstrap_draws)
    outputs = {"geometry_scenarios": simulations, "geometry_contrasts": contrasts,
               "geometry_examples": traces, "observed_budgets": budgets, "observed_summary": summaries,
               "observed_associations": associations, "observed_inventory": inventory, "observed_examples": examples}
    for name, data in outputs.items():
        data.to_csv(out / f"{name}.csv", index=False)
    conservative = simulations.loc[~simulations.scenario.eq("long_shared_reactive")]
    checks = {"n_geometry_pairs": int(simulations.pair_id.nunique()), "n_receivers": int(simulations.target.nunique()),
        "n_connected_systems": int(simulations.component.nunique()),
        "max_conservative_carbon_fraction_error": float(abs(conservative.retained_extra_carbon_fraction - 1).max()),
        "max_water_mass_error": float(abs(simulations.output_extra_water - simulations.input_extra_water).max()),
        "max_concentration_increment": float(simulations.concentration_peak_excess.max()),
        "n_field_timing_eligible": int(len(budgets) // 2)}
    if checks["max_conservative_carbon_fraction_error"] > 1e-9 or checks["max_water_mass_error"] > 1e-9:
        raise ValueError(f"Routing budget violation: {checks}")
    (ROOT / "numerical_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    old = json.loads((PREVIOUS / "analysis_sources.json").read_text())
    inputs = [INVENTORY, ROOT / "study_plan.md", PREVIOUS / "analysis_sources.json",
              *event_paths, *map(Path, old["inputs"])]
    code = [Path(__file__), Path("src/river_graph/analysis/river_geometry_budget.py"),
            Path("scripts/analyze_doc_river_multicatchment_pulses_v1.py"),
            Path("src/river_graph/analysis/river_multicatchment_pulses.py"),
            Path("src/river_graph/analysis/river_kervidy_observations.py"),
            Path("src/river_graph/analysis/river_form_process.py")]
    (ROOT / "analysis_sources.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs}, "code": {str(p): sha256_file(p) for p in code},
        "outputs": {str(p): sha256_file(p) for p in sorted(out.iterdir())},
        "bootstrap_draws": args.bootstrap_draws, "seed": 42,
        "aggregation": "geometry: pairs averaged within receiver, connected-system bootstrap; observations: within-catchment month blocks",
        "physical_scope": "Normalized transport scenarios plus complete bounded hourly outlet budgets; not measured net river removal"}, indent=2) + "\n")
    print(inventory.to_string(index=False), flush=True)
    print(summaries.loc[summaries.window.eq("response_window") & summaries.cohort.eq("positive_response")].to_string(index=False))


if __name__ == "__main__":
    main()
