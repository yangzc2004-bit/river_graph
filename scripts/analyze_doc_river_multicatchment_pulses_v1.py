"""Replicate pulse measurements on two independent outlet DOC/Q archives."""

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
from river_graph.analysis.river_multicatchment_pulses import (
    hourly_observations,
    normalize,
    read_bouleau,
    read_rappbode,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_multicatchment_pulses_v1")
RAW = Path("data/raw/river_multicatchment_pulses_v1")


def load_cases():
    kflow = prepare_flow(pd.read_parquet("data/raw/river_kervidy_geometry_v1/discharge_quarter_hour.parquet"))
    kd = pd.read_parquet("experiments/phase4_transfer/doc_river_kervidy_geometry_v1/analysis/corrected_doc_flow_matches.parquet")
    frame = pd.merge(kflow[["flow_timestamp_utc", "q_m3_s"]], kd[["timestamp_utc", "doc_mg_l"]],
                     left_on="flow_timestamp_utc", right_on="timestamp_utc", how="outer")
    frame["clock"] = frame.flow_timestamp_utc.combine_first(frame.timestamp_utc)
    # Keep independent source clocks; unmatched DOC rows are not assigned flow.
    kcase = normalize(frame[["clock", "q_m3_s", "doc_mg_l"]], "Kervidy", 15, 4.8896,
        "UTC source clocks", "Raw discharge; optical DOC corrected as 0.9*raw+0.34; no temporal filtering added")
    kcase[0]["flow_at_reported_cap"] = kcase[0].flow_timestamp_utc.isin(
        kflow.loc[kflow.flow_at_reported_cap, "flow_timestamp_utc"])
    return [kcase, read_rappbode(RAW / "rappbode/RB_HF_data_2018_2023.txt"),
            read_bouleau([RAW / f"bouleau/{n}.tsv" for n in (959043, 959044)])]


def summary(events, case, resolution, relative, minutes, area):
    lag = events.loc[events.lag_eligible]
    widths = events.loc[events.width_pair_eligible]
    info = {"case": case, "resolution": resolution, "relative_prominence": relative,
        "sample_minutes": minutes, "area_km2": area, "absolute_prominence_mm_day": .35,
        "absolute_prominence_m3_s": .35*area/86.4,
        "n_flow_candidates": len(events), "n_isolated": int(events.isolated_flow_pulse.sum()),
        "n_dense_uncensored": int(events.timing_eligible.sum()), "n_positive_doc": len(lag),
        "n_width_pairs": len(widths),
        "n_nonpositive_doc": int((events.timing_eligible & ~events.positive_doc_response).sum()),
        "n_doc_boundary_peaks": int((events.timing_eligible & events.doc_peak_on_response_boundary).sum()),
        "n_recovery_censored": int(lag.doc_recovery_censored.sum()),
        "fraction_doc_later": float(lag.doc_peak_lag_hours.gt(0).mean()) if len(lag) else np.nan,
        "fraction_doc_wider": float(widths.doc_flow_width_ratio.gt(1).mean()) if len(widths) else np.nan}
    for field, subset in (("doc_peak_lag_hours", lag), ("doc_flow_width_ratio", widths)):
        info.update({f"{field}_{key}": value for key, value in month_bootstrap(subset, field).items()})
    return info


def export_events(frame, path, case, clock_basis):
    export = frame.rename(columns={name: name.removesuffix("_utc")+"_clock"
                                   for name in frame.columns if name.endswith("_utc")})
    export.insert(0, "case", case)
    export["clock_basis"] = clock_basis
    export.to_csv(path, index=False)


def main():
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summaries, audits, periods, examples, duration_rows = [], [], [], [], []
    for flow, doc, audit, conflicts in load_cases():
        audits.append(audit)
        case, area = audit["case"], audit["area_km2"]
        if len(conflicts):
            conflicts.to_csv(out / f"{case.lower()}_conflicting_fields.csv", index=False)
        for resolution in ("native", "hourly"):
            f, d = (flow, doc) if resolution == "native" else hourly_observations(flow, doc)
            minutes = audit["sample_minutes"] if resolution == "native" else 60
            f = f.loc[f.flow_timestamp_utc.between(d.timestamp_utc.min(), d.timestamp_utc.max())]
            for relative in (.20, .15, .25):
                candidates = flow_pulses(f, relative, .35*area/86.4, sample_minutes=minutes)
                measured = measure_doc_responses(candidates, f, d, sample_minutes=minutes)
                if measured.empty:
                    raise ValueError(f"No flow candidates for {case}/{resolution}: review protocol before changing it")
                summaries.append(summary(measured, case, resolution, relative, minutes, area))
                export_events(measured, out / f"{case.lower()}_{resolution}_p{int(relative*100)}_events.csv",
                              case, audit["clock_basis"])
                if relative == .20:
                    # Period summaries are descriptive; do not select best years.
                    for year, frame in measured.groupby(measured.flow_peak_utc.dt.year):
                        periods.append({"year": year, **summary(frame, case, resolution, relative, minutes, area)})
                    if resolution == "hourly":
                        # This is an event-duration diagnostic after observing the
                        # cross-catchment summaries, not a new primary endpoint.
                        measured["flow_duration_band"] = pd.cut(measured.flow_span_hours,
                            bins=[0, 24, 48, 96], labels=["3–24 h", ">24–48 h", ">48–96 h"], right=True)
                        for band, frame in measured.groupby("flow_duration_band", observed=False):
                            if len(frame):
                                duration_rows.append({"flow_duration_band": str(band),
                                                      **summary(frame, case, resolution, relative, minutes, area)})
                    selected = measured.loc[measured.width_pair_eligible].head(1)
                    if len(selected):
                        example = selected.iloc[0].to_dict()
                        example.update(case=case, resolution=resolution)
                        examples.append(example)
            print(f"Measured {case} {resolution}", flush=True)
    pd.DataFrame(audits).to_csv(out / "source_audit.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "pulse_comparison.csv", index=False)
    pd.DataFrame(periods).to_csv(out / "period_sensitivity.csv", index=False)
    pd.DataFrame(duration_rows).to_csv(out / "event_duration_diagnostic.csv", index=False)
    export = pd.DataFrame(examples).rename(columns={key: key.removesuffix("_utc")+"_clock"
        for key in examples[0] if key.endswith("_utc")})
    export.to_csv(out / "chronological_examples.csv", index=False)
    sources = [RAW / "rappbode/RB_HF_data_2018_2023.txt", RAW / "bouleau/959043.tsv", RAW / "bouleau/959044.tsv",
        Path("data/raw/river_kervidy_geometry_v1/discharge_quarter_hour.parquet"),
        Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1/analysis/corrected_doc_flow_matches.parquet"),
        ROOT / "study_plan.md", ROOT / "retrieval_manifest.json"]
    code = [Path(__file__), Path("src/river_graph/analysis/river_multicatchment_pulses.py"),
            Path("src/river_graph/analysis/river_doc_pulses.py"),
            Path("src/river_graph/analysis/river_kervidy_observations.py")]
    (ROOT / "analysis_sources.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in sources}, "code": {str(p): sha256_file(p) for p in code},
        "outputs": {str(p): sha256_file(p) for p in sorted(out.iterdir())},
        "bootstrap": "5,000 calendar-month block draws within each catchment; seed42",
        "clock_note": "Exported *_clock columns retain source timestamps; no invented UTC offset"}, indent=2)+"\n")
    table = pd.DataFrame(summaries)
    print(table.loc[table.relative_prominence.eq(.20), ["case", "resolution", "n_positive_doc", "n_width_pairs",
        "doc_peak_lag_hours_median", "doc_flow_width_ratio_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
