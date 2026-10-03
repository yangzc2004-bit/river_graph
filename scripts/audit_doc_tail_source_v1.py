"""Trace large source-validation DOC errors to unchanged local WQP records.

The audit selects only validation rows from the existing v4 diagnostic. It
does not read outer-test predictions, fit models, delete labels or fetch data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.data.aggregate import to_monthly
from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file

DEFAULT_INPUT = Path("experiments/phase4_transfer/unified_doc_spatial_v4/diagnostics/validation_queries.parquet")
DEFAULT_OUTPUT = Path("experiments/phase4_transfer/doc_tail_residual_v1/data_audit")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
MANIFEST = Path("data/processed/raw_input_manifest_graphfix_cached370.json")
PROVENANCE = Path("data/processed/mississippi_graph_graphfix_st357.provenance.json")
RAW_DIR = Path("data/raw/wqp_results")
MODEL = "fusion_gru_tuned_anchor_k5"


def selected_population(path):
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.model_name.eq(MODEL)].copy()
    if (frame.empty or not frame.visibility_role.eq("source_validation_query").all()
            or not frame.k.eq(5).all()):
        raise ValueError("only the v4 K5 updated-GRU source-validation queries are eligible")
    if frame.duplicated(["split_seed", "seed", "station", "month"]).any():
        raise ValueError("duplicate prediction identities")
    if frame.groupby(["station", "month"]).y_true.nunique().max() != 1:
        raise ValueError("a unique station-month has conflicting DOC labels")
    if not np.isfinite(frame[["y_true", "y_pred", "residual", "log_residual"]].to_numpy()).all():
        raise ValueError("source-validation metrics are nonfinite")
    frame["absolute_error"] = frame.residual.abs()
    frame["squared_error"] = frame.residual**2
    frame["log_absolute_error"] = frame.log_residual.abs()
    frame["log_squared_error"] = frame.log_residual**2
    # Equal seeds within a partition, then equal available partitions for each
    # unique cell. No extra weight is given to repeated seed predictions.
    metrics = ["y_true", "y_pred", "residual", "absolute_error", "squared_error",
               "log_absolute_error", "log_squared_error"]
    per_split = frame.groupby(["station", "month", "split_seed"])[metrics].mean().reset_index()
    unique = per_split.groupby(["station", "month"])[metrics].mean().reset_index()
    occurrences = per_split.groupby(["station", "month"]).size().rename("n_validation_partitions")
    unique = unique.merge(occurrences, on=["station", "month"], validate="one_to_one")
    return frame, unique


def population_summary(frame, unique):
    rows = []
    for threshold in (50, 100):
        selected = unique.y_true > threshold
        row = {"threshold_mg_l_strictly_greater": threshold,
               "n_unique_cells": int(selected.sum()),
               "n_unique_stations": unique.loc[selected, "station"].nunique(),
               "n_all_unique_cells": len(unique), "cell_fraction": selected.mean(),
               "max_doc_mg_l": unique.loc[selected, "y_true"].max()}
        for metric in ("absolute_error", "squared_error", "log_absolute_error", "log_squared_error"):
            row[f"unique_cell_{metric}_share"] = unique.loc[selected, metric].sum() / unique[metric].sum()
            contributions = frame[metric].where(frame.y_true > threshold, 0)
            grouped = pd.DataFrame({"part": frame.split_seed, "seed": frame.seed,
                                    "all": frame[metric], "tail": contributions})
            average = grouped.groupby(["part", "seed"])[["all", "tail"]].mean().groupby("part").mean().mean()
            row[f"equal_partition_{metric}_share"] = average["tail"] / average["all"]
        rows.append(row)
    return pd.DataFrame(rows)


def joined_values(frame, column):
    if column not in frame:
        return ""
    return " | ".join(sorted(frame[column].dropna().astype(str).unique()))


def trace_month(row, manifest, dataset, station_index, month_index):
    site, month = row.station, row.month
    path = RAW_DIR / f"{site}.csv"
    result = row._asdict()
    result.update({"raw_path": str(path), "raw_file_exists": path.exists(),
                   "classification": "ambiguous_provenance"})
    if not path.exists():
        result["classification_reason"] = "Local provider file is absent."
        return result, pd.DataFrame(), pd.DataFrame()
    result["raw_sha256"] = sha256_file(path)
    result["manifest_sha256"] = manifest.get(site, {}).get("sha256", "")
    result["raw_matches_build_manifest"] = result["raw_sha256"] == result["manifest_sha256"]
    tidy = load_station_results(path)
    target_month = pd.Timestamp(month)
    # Restrict the value audit before extraction/aggregation to this selected
    # validation month; other dates in the provider file are not analyzed.
    tidy = tidy[tidy.site_no.eq(site)
                & tidy.date.dt.to_period("M").eq(target_month.to_period("M"))]
    observed = extract_doc_obs(tidy)
    monthly = to_monthly(observed, "doc")
    matching = monthly[monthly.site_no.eq(site) & monthly.month.eq(target_month)]
    admitted = observed[observed.site_no.eq(site) & observed.date.dt.to_period("M").eq(target_month.to_period("M"))]
    candidates = tidy[tidy.site_no.eq(site) & tidy.variable.eq("organic_carbon")
                      & tidy.date.dt.to_period("M").eq(target_month.to_period("M"))].copy()
    candidates["raw_csv_record_number_with_header"] = candidates.index + 2
    candidates["included_by_doc_extractor"] = candidates.index.isin(admitted.index)
    candidates["audit_station"], candidates["audit_month"] = site, month
    candidates["raw_path"], candidates["raw_sha256"] = str(path), result["raw_sha256"]
    accepted = candidates[candidates.included_by_doc_extractor].copy()
    if len(matching) != 1 or accepted.empty:
        result["classification_reason"] = "DOC month cannot be reconstructed from the local file."
        result["n_accepted_raw_records"] = len(accepted)
        return result, accepted, candidates
    rebuilt = float(matching.doc.iloc[0])
    # Inspect the frozen tensor only at the already-selected validation cell.
    frozen = float(dataset["y"][station_index[site], month_index[month]])
    if not bool(dataset["y_mask"][station_index[site], month_index[month]]):
        raise ValueError("selected validation label is unobserved in the frozen tensor")
    mean_match = bool(np.isclose(np.float32(rebuilt), frozen, rtol=0, atol=0))
    diagnostic_match = bool(np.isclose(row.y_true, frozen, rtol=0, atol=0))
    distinct_ids = accepted.Result_MeasureIdentifier.dropna()
    duplicate_ids = bool(distinct_ids.duplicated().any())
    result.update({
        "n_accepted_raw_records": len(accepted), "n_distinct_dates": accepted.date.nunique(),
        "accepted_values_mg_l": joined_values(accepted, "value"),
        "accepted_dates": " | ".join(accepted.date.dt.strftime("%Y-%m-%d").sort_values().unique()),
        "raw_result_identifiers": joined_values(accepted, "Result_MeasureIdentifier"),
        "raw_activity_identifiers": joined_values(accepted, "Activity_ActivityIdentifier"),
        "raw_characteristics": joined_values(accepted, "characteristic"),
        "raw_user_characteristic": joined_values(accepted, "Result_CharacteristicUserSupplied"),
        "raw_usgs_pcodes": joined_values(accepted, "usgs_pcode"),
        "raw_fractions": joined_values(accepted, "fraction"),
        "raw_units": joined_values(accepted, "unit"),
        "raw_detection_conditions": joined_values(accepted, "detection_condition"),
        "raw_qualifiers": joined_values(accepted, "Result_MeasureQualifierCode"),
        "raw_statuses": joined_values(accepted, "Result_MeasureStatusIdentifier"),
        "raw_value_types": joined_values(accepted, "Result_MeasureValueType"),
        "raw_qc_comments": joined_values(accepted, "DataQuality_ResultComment"),
        "raw_activity_types": joined_values(accepted, "Activity_TypeCode"),
        "raw_media": joined_values(accepted, "Activity_Media"),
        "raw_method_names": joined_values(accepted, "ResultAnalyticalMethod_Name"),
        "n_method_missing": int(accepted.ResultAnalyticalMethod_Name.isna().sum()),
        "raw_monthly_mean_mg_l": rebuilt, "frozen_dataset_mg_l": frozen,
        "aggregation_matches_float32_dataset": mean_match,
        "diagnostic_matches_dataset": diagnostic_match,
        "duplicate_result_identifiers": duplicate_ids,
        "n_other_organic_carbon_rows_excluded": len(candidates) - len(accepted),
        "unit_conversion": "none: mg/L required", "aggregation": "arithmetic mean of all retained point results",
    })
    issues = []
    if not result["raw_matches_build_manifest"]:
        issues.append("raw file differs from the recorded build input")
    if not mean_match or not diagnostic_match:
        issues.append("monthly/raw/tensor value mismatch")
    if duplicate_ids:
        issues.append("duplicate provider result identifiers enter the monthly average")
    if issues:
        confirmed_mismatch = not diagnostic_match or (result["raw_matches_build_manifest"] and not mean_match)
        result["classification"] = "actual_defect" if confirmed_mismatch else "ambiguous_provenance"
        result["classification_reason"] = "; ".join(issues)
    else:
        suspicious = (result["raw_qualifiers"] or result["raw_detection_conditions"]
                      or result["raw_qc_comments"]
                      or not accepted.Activity_TypeCode.eq("Sample-Routine").all()
                      or not accepted.Activity_Media.eq("Water").all()
                      or accepted.Result_MeasureIdentifier.isna().any()
                      or not accepted.Result_MeasureStatusIdentifier.isin(["Accepted", "Historical"]).all()
                      or not accepted.fraction.str.lower().isin(["dissolved", "filtered field and/or lab"]).all())
        result["classification"] = "ambiguous_provenance" if suspicious else "verified_source_doc"
        result["classification_reason"] = (
            "Quality or sampling metadata needs interpretation; retain the label pending review." if suspicious else
            "Unflagged routine filtered-DOC mg/L record(s), unchanged build input, exact monthly-mean reproduction."
        )
    result["measurement_assurance"] = (
        "Source-record identity and arithmetic verified; environmental accuracy is not independently established."
    )
    return result, accepted, candidates


def report(audit, population, unique, station_mass):
    counts = audit.classification.value_counts()
    top_error = audit.absolute_error.sum() / unique.absolute_error.sum()
    top_square = audit.squared_error.sum() / unique.squared_error.sum()
    methods_missing = int(audit.n_method_missing.sum())
    lines = ["# Source audit of large DOC reconstruction errors", "",
             ("The audit uses only v4 **source-validation** queries for the K=5 updated-GRU adapter. "
              "The top 20 unique station-months are ranked by mean absolute error, averaging seeds within "
              "partition and available partitions within each cell. These are previously reused selection "
              "labels. No outer-test prediction table or outer-query label is inspected; no label is altered."), "",
             "## Findings", "",
             (f"- {counts.get('verified_source_doc', 0)}/20 months have verified source DOC records; "
              f"{counts.get('ambiguous_provenance', 0)} have ambiguous provenance/QC; "
              f"{counts.get('actual_defect', 0)} have a directly established label defect."),
             (f"- All {audit.raw_path.nunique()} examined raw files "
              f"{'match' if audit.raw_matches_build_manifest.all() else 'do not all match'} "
              "their SHA-256 values in the cached370 build-input manifest. The ST357 cohort is its documented "
              "stream-station subset."),
             (f"- Monthly arithmetic means reproduce the frozen float32 labels for "
              f"{int(audit.aggregation_matches_float32_dataset.sum())}/20 months. "
              f"{int(audit.n_accepted_raw_records.eq(1).sum())}/20 are based on one retained point sample."),
             (f"- These 20 cells contribute {100*top_error:.1f}% of absolute error and "
              f"{100*top_square:.1f}% of squared error in the unique validation-cell population."), "",
             ("`verified_source_doc` means the high value is genuinely present as a DOC measurement in the "
              "unchanged provider file and is correctly reproduced by the pipeline. It does **not** establish "
              "the physical accuracy of a decades-old laboratory measurement. Historical status alone is "
              "not a rejection flag. No arbitrary DOC cutoff or decimal-point repair is applied."), "",
             "## Trace of the 20 largest errors", "",
             "| Station | Month | Dataset DOC | Raw dates | Raw values (mg/L) | n | Classification |",
             "|---|---|---:|---|---|---:|---|"]
    for row in audit.itertuples():
        lines.append(f"| {row.station} | {row.month[:7]} | {row.y_true:g} | "
                     f"{row.accepted_dates.replace(' | ', '; ')} | {row.accepted_values_mg_l.replace(' | ', '; ')} | "
                     f"{row.n_accepted_raw_records} | {row.classification} |")
    lines += ["", "## What was checked", "",
              ("The local WQP narrow-profile files identify the constituent as Organic carbon, USGS "
               "parameter 00681, with the filtered fraction and mg/L units. The DOC extractor admits "
               "00681 or, when the parameter code is absent, an explicit dissolved/filtered fraction; "
               "it requires mg/L and no detection condition. `to_monthly` uses an arithmetic mean of "
               "retained point observations, not a time-weighted monthly concentration. No DOC value-range "
               "filter is applied. Suspended-carbon observations in the same months are recorded as "
               "excluded candidates rather than mixed into DOC."), "",
              ("Accepted records were checked for provider result/activity identifiers, sample date, "
               "fraction, units, detection/censoring condition, qualifiers, status, media, routine/QC "
               "activity, laboratory method and duplicate result identifiers. The extractor itself does "
               "not filter general qualifier/status fields; this audit inspects them explicitly."), "",
              (f"Laboratory method names are missing for {methods_missing}/{int(audit.n_accepted_raw_records.sum())} "
               "retained raw results. The largest 06438000 values are Historical routine records with no "
               "qualifier/detection flag and no method name. Confirming their environmental authenticity "
               "would require provider/laboratory documentation beyond this local archive; their scale "
               "alone is not evidence of a pipeline mistake."), "",
              "## Full validation high-value population", "",
              ("Counts below use distinct station-months. Unique-cell error shares average repeated seed/"
               "partition predictions before summing, so repeated appearances add no sample count. The "
               "CSV also retains shares under the experiment's equal-partition weighting."), "",
              "| DOC threshold | Cells / all cells | Stations | Raw absolute-error share | Raw squared-error share | Log absolute-error share |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in population.itertuples():
        lines.append(f"| >{row.threshold_mg_l_strictly_greater:g} mg/L | {row.n_unique_cells}/{row.n_all_unique_cells} | "
                     f"{row.n_unique_stations} | {100*row.unique_cell_absolute_error_share:.1f}% | "
                     f"{100*row.unique_cell_squared_error_share:.1f}% | {100*row.unique_cell_log_absolute_error_share:.1f}% |")
    first = station_mass.iloc[0]
    lines += ["", (f"Station {first.station} contributes {100*first.absolute_error_share:.1f}% of all "
                       f"unique-cell absolute error and {100*first.squared_error_share:.1f}% of squared error. "
                       "This concentration explains why an SSE-based tail diagnosis can overstate the "
                       "breadth of the problem; MAE contributions remain the relevant companion."), "",
              "## Implication", "",
              ("No data deletion is supported by this audit. Preserve the high labels and their raw "
               "identities. Treat learning to reconstruct high DOC as a model question, while separately "
               "tracking how concentrated the evaluation is in historical records. An unflagged archived "
               "measurement and a validated ecological event are different levels of evidence."), "",
              "## Reproduction and evidence", "",
              "`uv run python scripts/audit_doc_tail_source_v1.py`", "",
              ("`top20_trace.csv` contains month-level provenance and classification; `raw_doc_records.csv` "
               "contains retained raw values/identifiers and metadata; `raw_organic_carbon_candidates.csv` "
               "also includes the excluded same-month carbon observations. `validation_high_values.csv` "
               "and `validation_station_error_mass.csv` retain population counts/contributions. "
               "`sources.json` records raw/code/dataset hashes. Values come from WQP; local NWIS site/catalog "
               "files provide inventory, not an independent DOC concentration measurement."), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame, unique = selected_population(args.input)
    top = unique.sort_values(["absolute_error", "station", "month"], ascending=[False, True, True]).head(20)
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    station_index = {str(site): i for i, site in enumerate(dataset["site_no"])}
    month_index = {str(np.datetime64(month, "D")): i for i, month in enumerate(dataset["months"])}
    manifest = {row["site_no"]: row for row in json.loads(MANIFEST.read_text())["entries"]}
    traces, records, candidates = [], [], []
    for row in top.itertuples(index=False):
        trace, accepted, all_candidates = trace_month(row, manifest, dataset, station_index, month_index)
        traces.append(trace)
        records.append(accepted)
        candidates.append(all_candidates)
    audit = pd.DataFrame(traces)
    population = population_summary(frame, unique)
    mass = unique.groupby("station").agg(n_cells=("month", "size"), max_doc=("y_true", "max"),
        absolute_error=("absolute_error", "sum"), squared_error=("squared_error", "sum")).reset_index()
    mass["absolute_error_share"] = mass.absolute_error / mass.absolute_error.sum()
    mass["squared_error_share"] = mass.squared_error / mass.squared_error.sum()
    mass = mass.sort_values("squared_error_share", ascending=False)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in (("top20_trace", audit), ("raw_doc_records", pd.concat(records, ignore_index=True)),
                        ("raw_organic_carbon_candidates", pd.concat(candidates, ignore_index=True)),
                        ("validation_high_values", population), ("validation_station_error_mass", mass)):
        table.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "audit.md").write_text(report(audit, population, unique, mass))
    paths = {args.input, DATASET, MANIFEST, PROVENANCE, Path(__file__),
             Path("scripts/build_dataset.py"), Path("src/river_graph/data/wqp.py"),
             Path("src/river_graph/data/aggregate.py"), Path("src/river_graph/data/quality.py")}
    paths.update(Path(path) for path in audit.raw_path if Path(path).exists())
    (args.output / "sources.json").write_text(json.dumps({
        "scope": "source-validation only, no outer-test labels or predictions",
        "model": MODEL, "n_unique_validation_cells": len(unique),
        "top_selection": "mean absolute error, seeds averaged within partition then partitions within cell",
        "classification_definition": {
            "verified_source_doc": "source/QC identity and monthly arithmetic verified, not independent laboratory validation",
            "ambiguous_provenance": "missing, conflicting or qualified source evidence needs interpretation",
            "actual_defect": "directly established diagnostic/dataset inconsistency; no values altered",
        },
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths)],
    }, indent=2) + "\n")
    print(args.output / "audit.md")


if __name__ == "__main__":
    main()
