"""Compare four observed confluence arrangements on the full and joint calendars."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_event_observations import match_campaigns
from river_graph.analysis.river_flow_mixing import attach_daily_flows
from river_graph.analysis.river_joint_campaigns import (
    METRICS,
    assign_flow_states,
    campaign_metrics,
    eligible_campaigns,
    flow_reference,
    joint_calendar,
    project_campaigns,
    resample_years,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_joint_campaigns_v1")
PARENT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
FLOW = Path("experiments/phase4_transfer/doc_river_flow_mixing_v1")
RECEIVERS = ("C12", "C16", "C7", "C9")
CODE = (Path(__file__), Path("src/river_graph/analysis/river_joint_campaigns.py"),
        Path("src/river_graph/analysis/river_event_observations.py"),
        Path("src/river_graph/analysis/river_flow_mixing.py"))


def build_inputs():
    folder = PARENT / "analysis"
    chemistry = pd.read_parquet(folder / "laboratory_doc.parquet")
    flow = pd.read_parquet(folder / "daily_discharge.parquet")
    connections = pd.read_csv(folder / "monitored_confluences.csv")
    gauges = pd.read_csv(FLOW / "analysis/gauge_metadata.csv").set_index("site")
    matched = []
    for row in connections.to_dict("records"):
        sub = match_campaigns(*(chemistry.loc[chemistry.site.eq(row[role])] for role
                                in ("source_a", "source_b", "receiver")))
        for role in ("source_a", "source_b", "receiver"):
            sub[role] = row[role]
        matched.append(sub)
    campaigns = attach_daily_flows(pd.concat(matched, ignore_index=True), flow)
    eligible, ledger = eligible_campaigns(campaigns)
    connections["upstream_area_share"] = (
        connections.source_a.map(gauges.catchment_area_km2) +
        connections.source_b.map(gauges.catchment_area_km2)
    ) / connections.receiver.map(gauges.catchment_area_km2)
    return eligible, ledger, connections, flow


def summarize_population(frame, flow, *, population):
    rows, states, years, detailed, references = [], [], [], [], []
    for site in RECEIVERS:
        selected = frame.loc[frame.receiver.eq(site)].copy()
        if len(selected) < 8:
            continue
        reference = flow_reference(flow, selected)
        references.append(reference | {"population": population})
        sub = assign_flow_states(project_campaigns(selected), reference)
        detailed.append(sub.assign(population=population))
        rows.append({"receiver": site, "population": population, **campaign_metrics(sub)})
        for state, values in sub.groupby("flow_state"):
            if len(values) >= 5:
                states.append({"receiver": site, "population": population,
                               "flow_state": state, **campaign_metrics(values)})
        for year, values in sub.groupby(sub.date_local.dt.year):
            if len(values) >= 5:
                years.append({"receiver": site, "population": population,
                              "year": int(year), **campaign_metrics(values)})
    return (pd.DataFrame(rows), pd.DataFrame(states), pd.DataFrame(years),
            pd.concat(detailed, ignore_index=True), pd.DataFrame(references))


def bootstrap_population(frame, references, *, draws, seed, population):
    rng = np.random.default_rng(seed)
    years = np.sort(frame.date_local.dt.year.unique())
    estimates, state_estimates, contrasts = [], [], []
    references = references.set_index("receiver").to_dict("index")
    for draw in range(draws):
        sampled = resample_years(frame, rng.choice(years, size=len(years), replace=True))
        result = {}
        for site, sub in sampled.groupby("receiver", sort=False):
            if len(sub) < 8:
                continue
            try:
                projected = assign_flow_states(project_campaigns(sub), references[site])
                metrics = campaign_metrics(projected)
            except ValueError:
                continue
            estimates.append({"draw": draw, "receiver": site, **{k: metrics[k] for k in METRICS}})
            result[site] = metrics
            for state, group in projected.groupby("flow_state", sort=False):
                if len(group) >= 5:
                    report = campaign_metrics(group)
                    state_estimates.append({"draw": draw, "receiver": site, "flow_state": state,
                                            **{k: report[k] for k in METRICS}})
        if population == "joint_calendar":
            for a, b in itertools.combinations(RECEIVERS, 2):
                if a in result and b in result:
                    contrasts.append({"draw": draw, "receiver_a": a, "receiver_b": b,
                                      "ratio_difference_b_minus_a": result[b][METRICS[0]]-result[a][METRICS[0]]})
        if (draw + 1) % 1000 == 0:
            print(f"{population}: year bootstrap {draw+1}/{draws}", flush=True)
    return pd.DataFrame(estimates), pd.DataFrame(state_estimates), pd.DataFrame(contrasts)


def append_intervals(table, draws, keys, *, requested):
    result = table.copy()
    for name in METRICS:
        result[name+"_lo"] = np.nan
        result[name+"_hi"] = np.nan
    result["n_valid_bootstrap"] = 0
    for key, sub in draws.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        match = np.ones(len(result), dtype=bool)
        for column, value in zip(keys, key, strict=True):
            match &= result[column].eq(value).to_numpy()
        result.loc[match, "n_valid_bootstrap"] = len(sub)
        # Few usable draws or a single observed year does not yield a repeatability interval.
        if len(sub) < .9 * requested or not result.loc[match, "n_years"].ge(3).all():
            continue
        for name in METRICS:
            values = sub[name].dropna()
            if len(values) >= .9 * requested:
                result.loc[match, [name+"_lo", name+"_hi"]] = np.quantile(values, [.025, .975])
    return result


def flow_state_contrasts(table, draws, *, population, requested):
    rows = []
    for site, selected in table.groupby("receiver"):
        selected = selected.set_index("flow_state")
        if not {"low", "high"}.issubset(selected.index):
            continue
        samples = draws.loc[draws.receiver.eq(site)].pivot(
            index="draw", columns="flow_state", values="adjusted_outlet_mix_sd_ratio")
        values = (samples["high"]-samples["low"]).dropna()
        enough = len(values) >= .9*requested and selected.loc[["low", "high"], "n_years"].min() >= 3
        lo, hi = np.quantile(values, [.025, .975]) if enough else (np.nan, np.nan)
        rows.append({"population": population, "receiver": site,
            "high_minus_low_sd_ratio": selected.loc["high", METRICS[0]]-selected.loc["low", METRICS[0]],
            "lo": lo, "hi": hi, "n_valid_bootstrap": len(values),
            "n_high": int(selected.loc["high", "n_campaigns"]),
            "n_low": int(selected.loc["low", "n_campaigns"])})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    eligible, ledger, connections, flow = build_inputs()
    joint = joint_calendar(eligible, RECEIVERS)
    outputs = ROOT / "analysis"
    outputs.mkdir(parents=True, exist_ok=True)
    all_rows, all_states, all_years, details, refs, pair_rows, flow_contrasts = [], [], [], [], [], [], []
    for i, (population, frame) in enumerate((("all_calendar", eligible), ("joint_calendar", joint))):
        if frame.empty:
            continue
        table, states, years, detailed, references = summarize_population(frame, flow, population=population)
        draws, state_draws, contrasts = bootstrap_population(frame, references,
            draws=args.bootstrap_draws, seed=42+i, population=population)
        all_rows.append(append_intervals(table, draws, ["receiver"], requested=args.bootstrap_draws))
        all_states.append(append_intervals(states, state_draws, ["receiver", "flow_state"], requested=args.bootstrap_draws))
        flow_contrasts.append(flow_state_contrasts(states, state_draws, population=population, requested=args.bootstrap_draws))
        all_years.append(years)
        details.append(detailed)
        refs.append(references)
        for a, b in itertools.combinations(RECEIVERS, 2):
            if contrasts.empty:
                continue
            samples = contrasts.loc[contrasts.receiver_a.eq(a) & contrasts.receiver_b.eq(b)]
            means = table.set_index("receiver")
            if a not in means.index or b not in means.index:
                continue
            lo, hi = np.quantile(samples.ratio_difference_b_minus_a, [.025, .975])
            pair_rows.append({"receiver_a": a, "receiver_b": b, "population": population,
                "ratio_difference_b_minus_a": means.loc[b, METRICS[0]]-means.loc[a, METRICS[0]],
                "lo": lo, "hi": hi, "n_valid_bootstrap": len(samples),
                "n_years": frame.date_local.dt.year.nunique(),
                "scope": "paired calendar-year temporal contrast in one nested catchment"})
    comparison = pd.concat(all_rows, ignore_index=True).merge(connections,
        on="receiver", how="left", validate="many_to_one")
    state_table = pd.concat(all_states, ignore_index=True)
    year_table = pd.concat(all_years, ignore_index=True)
    detailed = pd.concat(details, ignore_index=True)
    omissions = []
    for (population, site), group in detailed.groupby(["population", "receiver"]):
        for omitted in sorted(group.date_local.dt.year.unique()):
            subset = group.loc[group.date_local.dt.year.ne(omitted)]
            omissions.append({"population": population, "receiver": site, "omitted_year": int(omitted),
                               **campaign_metrics(project_campaigns(subset))})
    closure = []
    for (population, site), group in detailed.groupby(["population", "receiver"]):
        selected = group.loc[group.known_upstream_flow_share.between(.8, 1.2)]
        row = {"population": population, "receiver": site, "n_campaigns": len(selected),
               "n_years": selected.date_local.dt.year.nunique(),
               "available": len(selected) >= 8}
        if len(selected) >= 8:
            # Reproject on the same screened dates at all three positions.
            row.update(campaign_metrics(project_campaigns(selected)))
        closure.append(row)
    availability = ledger.groupby("receiver").agg(n_matched_campaigns=("eligible", "size"),
        n_complete_flow=("complete_flow_campaign", "sum"), n_common_flow_date=("same_flow_calendar_day", "sum"),
        n_duplicate_date_rows=("duplicate_receiver_date", "sum"), n_eligible=("eligible", "sum")).reset_index()
    tables = {"configuration_comparison": comparison, "flow_state_comparison": state_table,
              "year_comparison": year_table, "flow_references": pd.concat(refs, ignore_index=True),
              "paired_configuration_contrasts": pd.DataFrame(pair_rows),
              "flow_state_contrasts": pd.concat(flow_contrasts, ignore_index=True),
              "leave_year_out": pd.DataFrame(omissions),
              "water_coverage_sensitivity": pd.DataFrame(closure), "availability": availability}
    for name, table in tables.items():
        table.to_csv(outputs/f"{name}.csv", index=False)
    ledger.to_parquet(outputs/"campaign_ledger.parquet", index=False)
    detailed.to_parquet(outputs/"adjusted_campaigns.parquet", index=False)
    unique_samples = pd.concat([eligible[[role, timestamp]].rename(columns={role: "site", timestamp: "timestamp"})
        for role, timestamp in (("source_a", "source_a_time_utc"), ("source_b", "source_b_time_utc"),
                                ("receiver", "receiver_time_utc"))], ignore_index=True).drop_duplicates()
    summary = {"n_configurations": len(connections), "n_matched_configuration_campaigns": len(ledger),
        "n_complete_flow_before_same_day_rule": int(ledger.complete_flow_campaign.sum()),
        "n_eligible_configuration_campaigns": len(eligible),
        "n_unique_eligible_calendar_dates": eligible.date_local.nunique(),
        "n_unique_eligible_lab_sample_identities": len(unique_samples),
        "n_unique_monitored_sites": unique_samples.site.nunique(),
        "n_joint_calendar_dates": joint.date_local.nunique(),
        "n_joint_years": int(joint.date_local.dt.year.nunique()), "bootstrap_draws": args.bootstrap_draws,
        "population": "four nested monitored arrangements within one Krycklan catchment",
        "new_prediction_model_training": False, "measured_event_peaks_or_transit_lags": False,
        "morphology_classes_reassigned": False}
    (outputs/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    sources = tuple(PARENT/"analysis"/p for p in ("laboratory_doc.parquet", "daily_discharge.parquet",
        "monitored_confluences.csv")) + (FLOW/"analysis/gauge_metadata.csv", ROOT/"study_plan.md")
    (ROOT/"analysis_sources.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in sources},
        "code_hashes": {str(p): sha256_file(p) for p in CODE},
        "output_hashes": {str(p): sha256_file(p) for p in sorted(outputs.iterdir())},
        "bootstrap_cluster": "calendar year, jointly across configurations", "seed": 42,
    }, indent=2)+"\n")
    for p in CODE:
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        target = ROOT/"code_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
