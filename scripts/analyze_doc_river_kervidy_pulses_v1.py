"""Measure all flow-selected Kervidy pulses, keeping failures and sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_doc_pulses import (
    flow_pulses,
    measure_doc_responses,
    month_bootstrap,
)
from river_graph.analysis.river_kervidy_observations import prepare_flow
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_kervidy_pulses_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1")
FLOW = Path("data/raw/river_kervidy_geometry_v1/discharge_quarter_hour.parquet")


def summarize(events, name):
    lag = events.loc[events.lag_eligible]
    widths = events.loc[events.width_pair_eligible]
    recovery = lag.loc[lag.doc_recovery_censored.eq(False)]
    info = {"setting": name, "n_flow_candidates": len(events),
        "n_isolated_flow_pulses": int(events.isolated_flow_pulse.sum()),
        "n_uncensored_dense_flow_pulses": int(events.timing_eligible.sum()),
        "n_resolved_positive_doc_pulses": len(lag), "n_width_pairs": len(widths),
        "n_observed_doc_recoveries": len(recovery), "n_recovery_censored_positive_pulses": len(lag)-len(recovery),
        "n_censored_flow_peaks": int(events.flow_peak_censored.sum()),
        "n_next_flow_envelopes_overlapping_start": int(events.next_flow_envelope_overlaps_start.sum()),
        "n_record_edge_flow_boundaries": int(events.flow_boundary_on_record_edge.sum()),
        "n_nonpositive_doc_in_timing_eligible": int((events.timing_eligible & ~events.positive_doc_response).sum()),
        "n_doc_boundary_peaks_in_timing_eligible": int((events.timing_eligible & events.doc_peak_on_response_boundary).sum())}
    for field, data in (("doc_peak_lag_hours", lag), ("doc_flow_width_ratio", widths),
                        ("doc_recovery_hours", recovery)):
        info.update({f"{field}_{k}": v for k, v in month_bootstrap(data, field).items()})
    info["fraction_doc_peak_after_flow"] = float(lag.doc_peak_lag_hours.gt(0).mean())
    info["fraction_doc_wider_than_flow"] = float(widths.doc_flow_width_ratio.gt(1).mean())
    return info


def main():
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    flow = prepare_flow(pd.read_parquet(FLOW))
    joined = pd.read_parquet(PREVIOUS / "analysis/corrected_doc_flow_matches.parquet")
    doc = joined[["timestamp_utc", "doc_mg_l"]]
    flow = flow.loc[flow.flow_timestamp_utc.between(doc.timestamp_utc.min(), doc.timestamp_utc.max())]
    rows, primary = [], None
    for relative, name in ((.20, "primary_20pct"), (.15, "sensitivity_15pct"), (.25, "sensitivity_25pct")):
        events = flow_pulses(flow, relative_prominence=relative)
        measured = measure_doc_responses(events, flow, doc)
        measured.to_csv(out / f"pulse_inventory_{name}.csv", index=False)
        rows.append(summarize(measured, name))
        if name == "primary_20pct":
            primary = measured
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "threshold_sensitivity.csv", index=False)
    years = []
    for year, frame in primary.groupby(primary.flow_peak_utc.dt.year):
        years.append({"year": year, **summarize(frame, "primary_20pct")})
    pd.DataFrame(years).to_csv(out / "year_summary.csv", index=False)
    primary.loc[primary.lag_eligible].to_csv(out / "resolved_doc_pulses.csv", index=False)
    stages = pd.DataFrame({"stage": ["Flow candidates", "Single bounded flow pulse", "Dense uncensored flow/DOC response",
                                     "Resolved positive DOC peak", "Both half-widths observed"],
        "n": [len(primary), int(primary.isolated_flow_pulse.sum()), int(primary.timing_eligible.sum()),
              int(primary.lag_eligible.sum()), int(primary.width_pair_eligible.sum())]})
    stages.to_csv(out / "event_measurement_inventory.csv", index=False)
    quantiles = []
    for field, selected in (("doc_peak_lag_hours", primary.loc[primary.lag_eligible]),
                            ("doc_flow_width_ratio", primary.loc[primary.width_pair_eligible])):
        values = selected[field].to_numpy()
        for q in (.1, .25, .5, .75, .9):
            quantiles.append({"metric": field, "quantile": q, "value": float(np.quantile(values, q)), "n_events": len(values)})
    pd.DataFrame(quantiles).to_csv(out / "observed_response_distribution.csv", index=False)
    # Same flow candidates: this checks clock matching, not another event search.
    exact = joined.loc[joined.flow_exact_match, ["timestamp_utc", "doc_mg_l"]]
    exact_measured = measure_doc_responses(flow_pulses(flow), flow, exact)
    exact_measured.to_csv(out / "pulse_inventory_exact_clock_matches.csv", index=False)
    pd.DataFrame([summarize(primary, "all_corrected_doc"),
                  summarize(exact_measured, "exact_flow_clock_only")]).to_csv(
                      out / "clock_matching_sensitivity.csv", index=False)
    examples = primary.loc[primary.width_pair_eligible].copy()
    examples["year"] = examples.flow_peak_utc.dt.year
    examples.groupby("year", sort=True).head(1).to_csv(out / "chronological_examples.csv", index=False)
    inputs = [FLOW, PREVIOUS / "analysis/corrected_doc_flow_matches.parquet", ROOT / "study_plan.md"]
    code = [Path(__file__), Path("src/river_graph/analysis/river_doc_pulses.py")]
    (ROOT / "analysis_sources.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs}, "code": {str(p): sha256_file(p) for p in code},
        "outputs": {str(p): sha256_file(p) for p in sorted(out.iterdir())},
        "bootstrap": "calendar-month blocks; 5000 draws; seed42; within one catchment"}, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(stages.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
