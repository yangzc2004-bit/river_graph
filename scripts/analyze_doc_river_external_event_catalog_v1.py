"""Reproduce the public-record suitability audit for river-form event research."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_external_events import (
    canonical_doc,
    exact_sequence_audit,
    interval_coverage,
    normalize_watershed,
    year_inventory,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_external_event_catalog_v1")
RAW = Path("data/raw/river_external_event_catalog_v1")
CHEM = RAW/"timeseries/ADC_2017_2023_KupOks_WaterChem_01202026.csv"
STORMS = RAW/"storms/StormCharacteristics_2017_2023_KupOks.csv"
KERVIDY = RAW/"kervidy_spectro.txt"


def source_catalog():
    # Source methods and spatial arrangements, not a DOC-result ranking.
    rows = [
        ("Arctic outlets", "https://doi.org/10.18739/A2JH3D482", "retrieved", 2,
         "year/site calibrated UV-visible DOC estimates", "subhourly; archive grids vary",
         "separate discharge archive; public endpoint returned 403",
         "outlet-only; Upper Kuparuk location/area changed in 2022",
         "varying exact cross-year sequences need source resolution"),
        ("Kervidy-Naizin", "https://doi.org/10.57745/OFOUWE", "retrieved", 1,
         "laboratory-corrected UV-visible DOC estimates", "corrected DOC 2020-2023; distributed modal spacing 15 min",
         "no discharge in downloaded chemistry table", "one outlet, not a multi-form comparison",
         "corrected DOC starts 2020; README says 10 min after 2016 but distributed modal spacing is 15 min"),
        ("NEON", "https://www.neonscience.org/resources/learning-hub/tutorials/aquatic-data-product-integration",
         "public methods reviewed; API download requires token", np.nan,
         "grab DOC; sensor fDOM requires site-specific DOC calibration", "grab/sensor streams separate",
         "discharge product exists; not retrieved", "paired mainstem sensors do not close all tributary inputs",
         "fDOM is not DOC concentration"),
        ("Krycklan and Yli-Nuortti Dryad", "https://doi.org/10.5061/dryad.wpzgmsbp9",
         "public file description reviewed", 4, "optical DOC with laboratory calibration and author cleaning",
         "distributed file is daily aggregation of hourly data", "daily Q in described table",
         "three nested Krycklan sites plus one Finland catchment; no C7 outlet sensor",
         "daily averages cannot resolve subdaily peak width"),
    ]
    columns = ["source", "url", "access", "nominal_sites", "doc_method", "resolution",
               "flow_availability", "spatial_design", "material_issue"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["direct_multi_form_event_test_ready"] = False
    return frame


def main():
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(CHEM, low_memory=False)
    series, duplicate = canonical_doc(original)
    series.to_parquet(out/"arctic_doc_record_audit.parquet", index=False)
    duplicate.to_csv(out/"duplicate_timestamps.csv", index=False)
    inventory = year_inventory(series)
    replay = exact_sequence_audit(series)
    inventory["cross_year_sequence_issue"] = [
        bool(((replay.watershed == r.watershed) & (replay.later_year == r.year)
              & replay.requires_source_resolution).any()) for r in inventory.itertuples()]
    inventory.to_csv(out/"arctic_station_years.csv", index=False)
    replay.to_csv(out/"cross_year_sequences.csv", index=False)
    events = pd.read_csv(STORMS)
    rows = []
    for i, event in events.iterrows():
        site = normalize_watershed(event.watershed)
        start, end = pd.to_datetime([event['time.start'], event['time.end']], format="mixed")
        g = series.loc[series.watershed.eq(site)]
        result = interval_coverage(g, start, end)
        issue = bool(((replay.watershed == site) & (replay.later_year == event.year)
                      & replay.requires_source_resolution).any())
        conflict = result["n_conflicting_timestamps"] > 0
        rows.append({"event_id": i, "watershed": site, "year": event.year,
                     "start_source_clock": start, "end_source_clock": end,
                     "source_start_text": event['time.start'], "source_end_text": event['time.end'],
                     "published_duration_hours": event.duration_hrs,
                     "duration_disagreement_hours": result["elapsed_hours"]-event.duration_hrs,
                     "bare_date_boundary": ':' not in str(event['time.start']) or ':' not in str(event['time.end']),
                     **result, "cross_year_sequence_issue": issue,
                     "doc_record_candidate": result["doc_coverage_qualified"] and not issue and not conflict,
                     "continuous_q_present": False, "full_doc_flow_event_qualified": False,
                     "independent_form_test_qualified": False})
    event_table = pd.DataFrame(rows)
    event_table.to_csv(out/"author_storm_audit.csv", index=False)
    grouped = event_table.groupby(["watershed", "year"]).agg(
        n_author_storms=("event_id", "size"),
        n_doc_coverage_qualified=("doc_coverage_qualified", "sum"),
        n_doc_record_candidates=("doc_record_candidate", "sum"),
        n_with_conflicting_doc=("n_conflicting_timestamps", lambda x: int((x > 0).sum())),
        n_replay_affected=("cross_year_sequence_issue", "sum"),
        n_complete_doc_flow_events=("full_doc_flow_event_qualified", "sum"),
    ).reset_index()
    grouped.to_csv(out/"author_storm_years.csv", index=False)
    k = pd.read_csv(KERVIDY, sep=";", decimal=",")
    k["timestamp"] = pd.to_datetime(k.time, format="mixed", errors="raise", utc=True)
    rows = []
    for year, g in k.groupby(k.timestamp.dt.year):
        valid = g.loc[np.isfinite(g.DOCcor)]
        delta = valid.timestamp.sort_values().diff().dt.total_seconds()/60
        rows.append({"year": year, "n_rows": len(g), "n_corrected_doc": len(valid),
                     "first_corrected_doc_utc": valid.timestamp.min(),
                     "last_corrected_doc_utc": valid.timestamp.max(),
                     "n_duplicate_timestamps": int(g.timestamp.duplicated().sum()),
                     "doc_modal_minutes": float(delta.mode().iloc[0]) if len(valid) > 1 else np.nan,
                     "doc_min_mg_l": valid.DOCcor.min(), "doc_max_mg_l": valid.DOCcor.max()})
    pd.DataFrame(rows).to_csv(out/"kervidy_years.csv", index=False)
    source_catalog().to_csv(out/"source_suitability.csv", index=False)
    summary = {"n_arctic_source_rows": len(original), "n_arctic_canonical_timestamps": len(series),
               "n_duplicate_extra_rows": len(original)-len(series),
               "n_duplicate_timestamp_groups": len(duplicate),
               "n_conflicting_doc_timestamps": int(series.doc_conflict.sum()),
               "n_arctic_finite_unambiguous_doc": int(np.isfinite(series.doc_mg_l).sum()),
               "n_author_storms": len(event_table),
               "n_doc_coverage_qualified_storms": int(event_table.doc_coverage_qualified.sum()),
               "n_doc_record_candidate_storms": int(event_table.doc_record_candidate.sum()),
               "n_complete_doc_flow_storms": 0,
               "n_independent_classified_form_event_comparisons": 0,
               "n_kervidy_corrected_doc": int(np.isfinite(k.DOCcor).sum()),
               "kervidy_corrected_doc_first_utc": str(k.loc[k.DOCcor.notna(), "timestamp"].min()),
               "kervidy_corrected_doc_last_utc": str(k.loc[k.DOCcor.notna(), "timestamp"].max()),
               "new_model_training": False,
               "clock_alignment": "Arctic source local clocks retained; storm timezone not independently verified",
               "coverage_definition": "occupied nominal 15-minute bins and maximum gap <=1h, including boundaries",
               "classification": "original mapped forms unchanged; no classes assigned using outlet DOC"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    inputs = [CHEM, STORMS, KERVIDY, ROOT/"study_plan.md", ROOT/"audit_notes.md"]
    code = [Path(__file__), Path("src/river_graph/analysis/river_external_events.py")]
    (ROOT/"analysis_sources.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in inputs},
        "code_hashes": {str(p): sha256_file(p) for p in code},
        "output_hashes": {str(p): sha256_file(p) for p in sorted(out.iterdir())},
        "source_grain": "outlet sensor record and original author storm; not independent form replicates",
    }, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(replay.loc[replay.requires_source_resolution].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
