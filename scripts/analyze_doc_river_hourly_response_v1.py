"""Compute paired hourly flow clocks, waveform widths and optical-DOC quality."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_hourly_response import (
    optical_quality,
    peak_difference,
    shifted_flow_shape,
    waveform_profile,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_hourly_response_v1")
PARENT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
INPUT = PARENT / "analysis/optical_paired_hours.parquet"
RAW = Path("data/raw/river_event_observations_v1/turbolo/REPO_DOC_final_2.xlsx")
VARIABLES = {"flow": "Discharge (m3 s-1)", "optical_doc": "DOC (mg l-1)"}


def build():
    paired = pd.read_parquet(INPUT).sort_values("date_time_source_clock").reset_index(drop=True)
    if paired.date_time_source_clock.duplicated().any():
        raise ValueError("Paired source-clock timestamps must be unique")
    sites, segments = [], []
    annotated = paired.copy()
    for site in ("upstream", "receiver"):
        turb = annotated[f"Turbidity (FNU)_{site}"]
        annotated[f"doc_reported_retrieval_regime_{site}"] = turb.gt(600)
    for block, frame in paired.groupby("block", sort=True):
        frame = frame.reset_index(drop=True)
        times = frame.date_time_source_clock
        profiles, qualities = {}, {}
        for variable, column in VARIABLES.items():
            for site in ("upstream", "receiver"):
                values = frame[f"{column}_{site}"].to_numpy()
                p = waveform_profile(times, values)
                profiles[variable, site] = p
                q = optical_quality(p, values, frame[f"Turbidity (FNU)_{site}"]) if variable == "optical_doc" else {}
                if q:
                    qualities[site] = q
                sites.append({"block": block, "site": site, "variable": variable,
                              "author_event_ids": ";".join(sorted(frame[f"event_{site}"].unique())),
                              **p, **q})
        duration = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600
        row = {"block": block, "start": times.iloc[0], "end": times.iloc[-1],
               "n_records": len(frame), "duration_hours": duration,
               "main_coverage_subset": bool(duration >= 24),
               "upstream_author_event_ids": ";".join(sorted(frame.event_upstream.unique())),
               "receiver_author_event_ids": ";".join(sorted(frame.event_receiver.unique())),
               "multiple_author_events": bool(frame.event_upstream.nunique() > 1 or frame.event_receiver.nunique() > 1)}
        for variable in VARIABLES:
            a, b = profiles[variable, "upstream"], profiles[variable, "receiver"]
            row.update({f"{variable}_{k}": v for k, v in peak_difference(a, b).items()})
            ready = a["width_status"] == b["width_status"] == "observed_crossings"
            single = a["n_half_height_lobes"] == b["n_half_height_lobes"] == 1
            row.update({f"{variable}_both_widths_available": bool(ready),
                        f"{variable}_single_half_height_lobes": bool(single),
                        f"{variable}_upstream_width_hours": a["width_hours"],
                        f"{variable}_receiver_width_hours": b["width_hours"],
                        f"{variable}_width_difference_hours": b["width_hours"]-a["width_hours"] if ready else np.nan})
        for site in ("upstream", "receiver"):
            diff = peak_difference(profiles["flow", site], profiles["optical_doc", site])
            row.update({f"{site}_doc_minus_flow_{k}": v for k, v in diff.items()})
            row[f"{site}_doc_peak_outside_retrieval_regime"] = qualities[site]["peak_outside_reported_retrieval_regime"]
        row["doc_peak_pair_outside_retrieval_regime"] = bool(all(
            qualities[site]["peak_outside_reported_retrieval_regime"] for site in qualities))
        row["doc_optical_only_width_pair"] = bool(all(
            qualities[site]["optical_only_width_available"] for site in qualities))
        a, b = profiles["flow", "upstream"], profiles["flow", "receiver"]
        offset = int((b["peak_first"]-a["peak_first"]).total_seconds()/3600)
        row.update(shifted_flow_shape(times, frame[f"{VARIABLES['flow']}_upstream"],
                                      frame[f"{VARIABLES['flow']}_receiver"], offset))
        row["shape_anchor"] = "earliest_sampled_peak_difference; not optimized lag"
        segments.append(row)
    sites, segments = pd.DataFrame(sites), pd.DataFrame(segments)
    main = segments[segments.main_coverage_subset]
    width = main[main.flow_both_widths_available & main.flow_single_half_height_lobes]
    doc_peaks = segments[segments.doc_peak_pair_outside_retrieval_regime & segments.optical_doc_interior_peak_pair]
    quality = sites[sites.variable.eq("optical_doc")].groupby("site").agg(
        n_site_segments=("block", "size"), n_hourly_records=("n_records", "sum"),
        n_records_gt600=("n_turbidity_gt600", "sum"), n_records_le40=("n_turbidity_le40", "sum"),
        n_peaks_outside_reported_retrieval=("peak_outside_reported_retrieval_regime", "sum"),
        n_peaks_above_source_lab_doc_range=("peak_above_source_lab_doc_range", "sum"),
        n_optical_only_widths=("optical_only_width_available", "sum")).reset_index()
    summary = {
        "n_paired_hours": len(paired), "n_observation_segments": len(segments),
        "n_main_coverage_segments": len(main),
        "n_main_multiple_author_events": int(main.multiple_author_events.sum()),
        "n_main_flow_peaks_later_downstream": int((main.flow_peak_difference_lower_hours > 0).sum()),
        "main_flow_peak_difference_min_hours": float(main.flow_peak_difference_lower_hours.min()),
        "main_flow_peak_difference_max_hours": float(main.flow_peak_difference_upper_hours.max()),
        "n_main_single_lobe_flow_width_pairs": len(width),
        "main_single_lobe_flow_width_difference_median_hours": float(width.flow_width_difference_hours.median()) if len(width) else None,
        "main_shifted_flow_correlation_median": float(main.shifted_flow_correlation.median()),
        "main_shifted_flow_range_nrmse_median": float(main.range_normalized_rmse.median()),
        "n_published_doc_peak_pairs_outside_reported_retrieval_regime": int(segments.doc_peak_pair_outside_retrieval_regime.sum()),
        "n_interior_doc_peak_pairs_outside_reported_retrieval_regime": len(doc_peaks),
        "n_main_doc_peak_pairs_outside_reported_retrieval_regime": int(main.doc_peak_pair_outside_retrieval_regime.sum()),
        "n_doc_optical_only_width_pairs": int(segments.doc_optical_only_width_pair.sum()),
        "n_main_doc_optical_only_width_pairs": int(main.doc_optical_only_width_pair.sum()),
        "n_site_segment_doc_peaks_above_lab_range": int(sites.loc[sites.variable.eq("optical_doc"), "peak_above_source_lab_doc_range"].sum()),
        "n_catchments": 1, "n_nested_pairs": 1,
        "sample_clock": "published source clock; timezone unspecified",
        "optical_doc": "corrected fDOM and reported flow/rainfall retrieval above 600 FNU; row flags unavailable",
        "sampling": "author-selected windows; segments are not independent storms",
        "width_reference": "segment minimum + half observed range; dominant lobe, not baseflow separation",
        "wave_peak_clock_not_tracer_travel_time": True,
    }
    return {"segment_comparison": segments, "site_waveforms": sites, "optical_quality_summary": quality}, {"paired_hourly_quality": annotated}, summary


def main():
    tables, products, summary = build()
    output = ROOT / "analysis"
    output.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in products.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = [INPUT, RAW, RAW.with_name("readme.txt"), PARENT / "optical_sources.json",
             ROOT / "study_plan.md", Path("scripts/audit_doc_river_optical_case_v1.py"),
             Path("scripts/analyze_doc_river_hourly_response_v1.py"),
             Path("src/river_graph/analysis/river_hourly_response.py")]
    for path in paths:
        if str(path).startswith(("scripts/", "src/")):
            target = ROOT / "code_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    outputs = [output / f"{name}.csv" for name in tables] + [output / f"{name}.parquet" for name in products] + [output / "summary.json"]
    receipt = {"source_hashes": {str(p): sha256_file(p) for p in paths},
               "output_hashes": {str(p): sha256_file(p) for p in outputs},
               "source_method_url": "https://doi.org/10.1029/2022WR034397",
               "source_method_section": "2.3; >600 FNU DOC estimated from flow and preceding rainfall",
               "retrieval_threshold_fnu": 600, "new_training": False}
    (ROOT / "analysis_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
