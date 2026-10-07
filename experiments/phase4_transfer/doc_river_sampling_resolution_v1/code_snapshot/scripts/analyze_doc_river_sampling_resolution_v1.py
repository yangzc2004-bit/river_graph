"""Sampling-date and daily-flow resolution of the fixed river DOC cohort."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_form_process import aggregate_receivers, cluster_mean
from river_graph.analysis.river_sampling_resolution import (
    DAY_CUTS,
    attach_sample_flow,
    doc_activities,
    load_station_daily_flow,
    sampling_triplets,
    station_cadence,
)
from river_graph.analysis.river_signal_mechanisms import mixing_records
from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_sampling_resolution_v1")
INPUT = Path("experiments/phase4_transfer/doc_river_observed_transport_v1/analysis/input_records.parquet")
FOOTPRINT = Path("experiments/phase4_transfer/doc_monitored_river_footprint_v1/analysis/footprints.csv")
CELLS = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DTYPES = {"site_no": str, "source_a": str, "source_b": str, "target": str, "huc4": str}
METRICS = ("source_rho", "mixture_buffer_fraction", "asynchronous_buffer_fraction", "outlet_mixture_log_sd_ratio")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_sampling_resolution_v1.py",
    "scripts/plot_doc_river_sampling_resolution_v1.py", "scripts/verify_doc_river_sampling_resolution_v1.py",
    "tests/test_river_sampling_resolution.py",
    "src/river_graph/analysis/river_sampling_resolution.py", "src/river_graph/analysis/river_signal_mechanisms.py",
    "src/river_graph/analysis/river_form_process.py", "src/river_graph/analysis/river_mechanisms.py",
    "src/river_graph/data/wqp.py", "src/river_graph/models/daily_flow_features.py")))


def read_doc(inputs):
    sites = sorted(set(inputs.source_a) | set(inputs.source_b) | set(inputs.target))
    rows, inventory = [], []
    for site in sites:
        path = Path("data/raw/wqp_results")/f"{site}.csv"
        raw = load_station_results(path)
        doc = extract_doc_obs(raw)
        meta = raw.loc[doc.index, ["Activity_ActivityIdentifier", "Activity_StartTime", "Activity_StartTimeZone"]].rename(
            columns={"Activity_ActivityIdentifier": "event_id", "Activity_StartTime": "start_time", "Activity_StartTimeZone": "time_zone"})
        rows.append(doc.join(meta))
        inventory.append({"path": str(path), "sha256": sha256_file(path), "site_no": site,
                          "recognized_water_quality_rows": len(raw), "accepted_doc_rows": len(doc)})
    raw_doc = pd.concat(rows, ignore_index=True)
    raw_doc["month"] = raw_doc.date.dt.to_period("M").dt.to_timestamp()
    monthly = raw_doc.groupby(["site_no", "month"], as_index=False).agg(doc=("doc", "mean"), n_results=("doc", "size"))
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    sites_index = {str(s): i for i, s in enumerate(data["site_no"])}
    months_index = {pd.Timestamp(t): i for i, t in enumerate(data["months"])}
    allowed = set(np.load(CELLS).tolist())
    monthly["cell"] = [sites_index[s]*len(months_index)+months_index[t] for s, t in zip(monthly.site_no, monthly.month, strict=True)]
    monthly["source_cell_permitted"] = monthly.cell.isin(allowed)
    monthly["frozen_doc"] = np.asarray(data["y"]).ravel()[monthly.cell]
    # The original monthly estimator weights result rows, not activity means.
    permitted = monthly[monthly.source_cell_permitted]
    np.testing.assert_allclose(permitted.doc, permitted.frozen_doc, atol=2e-5, rtol=1e-6)
    raw_doc = raw_doc.merge(permitted[["site_no", "month"]], on=["site_no", "month"], validate="many_to_one")
    lookup = permitted.set_index(["site_no", "month"]).doc
    for station, value in (("source_a", "doc_a_now"), ("source_b", "doc_b_now"), ("target", "y_true")):
        index = pd.MultiIndex.from_arrays([inputs[station], inputs.date])
        np.testing.assert_allclose(lookup.reindex(index), inputs[value], atol=2e-5, rtol=1e-6)
    return raw_doc, permitted, inventory


def compare_sampling(triplets, draws):
    ledgers, connections, receiver_tables, summaries = [], [], [], []
    for cut in DAY_CUTS:
        selected = triplets[triplets.sample_span_days.le(cut)]
        sizes = selected.groupby("pair_id").size()
        eligible = sizes.index[sizes.ge(24)]
        selected = selected[selected.pair_id.isin(eligible)].copy()
        ledgers.append({"max_span_days": cut, "n_selected_connection_months": int(triplets.sample_span_days.le(cut).sum()),
            "n_selected_unique_receiver_months": len(triplets[triplets.sample_span_days.le(cut)].drop_duplicates(["target", "month_index"])),
            "n_eligible_pairs_24months": len(eligible), "n_eligible_receivers": selected.target.nunique(),
            "n_eligible_systems": selected.component.nunique(), "n_eligible_connection_months": len(selected)})
        if selected.empty:
            continue
        versions = {}
        for version in ("monthly_mean", "date_selected_activity", "full_monthly_same_pairs"):
            f = (triplets[triplets.pair_id.isin(eligible)] if version == "full_monthly_same_pairs" else selected).copy()
            if version == "date_selected_activity":
                for target, source in (("doc_a_now", "sample_doc_a"), ("doc_b_now", "sample_doc_b"), ("y_true", "sample_doc_receiver")):
                    f[target] = f[source]
            c, _, _ = mixing_records(f)
            connections.append(c.assign(max_span_days=cut, version=version))
            versions[version] = aggregate_receivers(c, METRICS)
            receiver_tables.append(versions[version].assign(max_span_days=cut, version=version))
            for metric in METRICS:
                s = cluster_mean(versions[version], metric, "component", draws=draws)
                if s is not None:
                    summaries.append({"max_span_days": cut, "comparison": version, **s})
        base, selected_values = versions["monthly_mean"], versions["date_selected_activity"]
        paired = selected_values.merge(base, on=["target", "huc4", "component"], suffixes=("_selected", "_monthly"), validate="one_to_one")
        for metric in METRICS:
            paired[metric] = paired[metric+"_selected"]-paired[metric+"_monthly"]
            s = cluster_mean(paired, metric, "component", draws=draws)
            if s is not None:
                summaries.append({"max_span_days": cut, "comparison": "selected_minus_monthly_same_months", **s})
        full = versions["full_monthly_same_pairs"]
        cohort = base.merge(full, on=["target", "huc4", "component"], suffixes=("_cut", "_full"), validate="one_to_one")
        for metric in METRICS:
            cohort[metric] = cohort[metric+"_cut"]-cohort[metric+"_full"]
            s = cluster_mean(cohort, metric, "component", draws=draws)
            if s is not None:
                summaries.append({"max_span_days": cut, "comparison": "monthly_cut_minus_full_same_pairs", **s})
    return {"date_cut_ledger": pd.DataFrame(ledgers), "mixing_connections": pd.concat(connections, ignore_index=True),
            "mixing_receivers": pd.concat(receiver_tables, ignore_index=True), "signal_summary": pd.DataFrame(summaries)}


def hydro_summary(triplets):
    rows = []
    for cut in DAY_CUTS:
        f = triplets[triplets.sample_span_days.le(cut)]
        unique = []
        for role, station in (("a", "source_a"), ("b", "source_b")):
            unique.append(pd.DataFrame({"site_no": f[station], "own_date": f[f"sample_date_{role}"],
                "receiver_date": f.sample_date_receiver, "log_flow_change": f[f"flow_{role}_date_log_change"]}))
        g = pd.concat(unique).drop_duplicates(["site_no", "own_date", "receiver_date"])
        offset = g.own_date.ne(g.receiver_date)
        valid = g.log_flow_change.notna()
        measured = g.loc[offset & valid, "log_flow_change"].abs()
        rows.append({"max_span_days": cut, "n_connection_months": len(f),
            "n_three_daily_flows_measured": int(f.all_three_sample_flows_measured.sum()),
            "n_three_daily_flows_positive": int(f.all_three_sample_flows_positive.sum()),
            "n_unique_source_date_comparisons": len(g), "n_offset_source_date_comparisons": int(offset.sum()),
            "n_offset_positive_measured_comparisons": len(measured),
            "median_absolute_log_flow_change_offset": measured.median(),
            "fraction_offset_flow_ratio_ge1_5": (measured >= np.log(1.5)).mean() if len(measured) else np.nan,
            "fraction_offset_flow_ratio_ge2": (measured >= np.log(2)).mean() if len(measured) else np.nan})
    return pd.DataFrame(rows)


def event_opportunity(triplets, footprints):
    rows = []
    f = triplets.merge(footprints[["pair_id", "common_storage_fraction", "common_fraction", "independent_branch_cv"]],
                       on="pair_id", validate="many_to_one")
    for storage in (False, True):
        g = f[f.common_storage_fraction.gt(0).eq(storage)]
        rows.append({"mapped_common_storage": storage, "n_pairs": g.pair_id.nunique(),
            "n_receivers": g.target.nunique(), "n_systems": g.component.nunique(), "n_connection_months": len(g),
            "n_unique_receiver_months": len(g.drop_duplicates(["target", "month_index"])),
            "n_all_three_ge3_days": int(g.all_three_dense_month.sum()),
            "n_dense_pairs": g[g.all_three_dense_month].pair_id.nunique(),
            "n_dense_receivers": g[g.all_three_dense_month].target.nunique()})
    return pd.DataFrame(rows)


def calendar_examples(triplets, footprints, daily):
    """Choose two timing displays from geometry and coverage, never DOC values."""
    flow_dates = set(zip(daily.site_no, daily.date, strict=True))
    rows = []
    for storage in (True, False):
        candidates = footprints[footprints.common_storage_fraction.gt(0).eq(storage)].copy()
        if storage:
            chosen = candidates.sort_values(["common_storage_fraction", "pair_id"], ascending=[False, True]).iloc[0]
        else:
            ref = footprints[footprints.pair_id.eq(rows[0]["pair_id"])].iloc[0]
            candidates["geometry_distance"] = abs(np.log1p(candidates.mean_total_km)-np.log1p(ref.mean_total_km))
            chosen = candidates.sort_values(["geometry_distance", "pair_id"]).iloc[0]
        dates = triplets[triplets.pair_id.eq(chosen.pair_id)]
        windows = []
        for year, f in dates.groupby(dates.date.dt.year):
            days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            covered = sum((chosen.target, d) in flow_dates for d in days)
            windows.append((len(f), covered, -int(year), int(year)))
        year = max(windows)[-1]
        rows.append({"pair_id": chosen.pair_id, "source_a": chosen.source_a, "source_b": chosen.source_b,
            "target": chosen.target, "mapped_common_storage": storage, "common_storage_fraction": chosen.common_storage_fraction,
            "year": year, "geometry_selection": "maximum common storage / closest mean path non-storage"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    a = ROOT/"analysis"
    a.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_parquet(INPUT)
    footprints = pd.read_csv(FOOTPRINT, dtype=DTYPES)
    raw_doc, monthly, inventory = read_doc(inputs)
    events = doc_activities(raw_doc)
    triplets = sampling_triplets(inputs, events)
    cadence, intervals = station_cadence(events)
    print(f"DOC: {len(raw_doc)} result rows, {len(events)} activities; {len(triplets)} fixed connection-months", flush=True)
    daily, daily_inventory = load_station_daily_flow("data/raw/nwis_dv", set(events.site_no))
    print(f"Daily flow: {len(daily)} reconciled station-days", flush=True)
    triplets = attach_sample_flow(triplets, daily)
    tables = {**compare_sampling(triplets, args.bootstrap_draws), "station_cadence": cadence,
        "sampling_intervals": intervals, "hydro_date_summary": hydro_summary(triplets),
        "event_opportunity": event_opportunity(triplets, footprints),
        "calendar_examples": calendar_examples(triplets, footprints, daily)}
    # Keep only monthly windows used by the fixed cohort for the display product.
    months = set(zip(events.site_no, events.month, strict=True))
    daily = daily[[k in months for k in zip(daily.site_no, daily.date.dt.to_period("M").dt.to_timestamp(), strict=True)]].copy()
    for name, frame in tables.items():
        frame.to_csv(a/f"{name}.csv", index=False)
    for name, frame in (("doc_activities", events), ("monthly_reconciliation", monthly), ("sample_triplets", triplets), ("daily_flow_context", daily)):
        frame.to_parquet(a/f"{name}.parquet", index=False)
    config = {"bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42, "bootstrap_unit": "connected monitoring system",
        "aggregation": "months within pair; pairs within receiver; receivers equal", "date_cuts_days": DAY_CUTS,
        "minimum_variance_months": 24, "selection": "date span, UTC span, receiver date distance, date tuple, activity identifiers; no DOC values",
        "monthly_estimator": "mean of accepted result rows", "activity_estimator": "mean of accepted results within activity",
        "new_model_training": False, "previous_results_seen": True}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    summary = {"n_result_rows": len(raw_doc), "n_activities": len(events), "n_station_days": len(events.drop_duplicates(["site_no", "date"])),
        "n_station_months": len(monthly), "n_stations": events.site_no.nunique(), "n_connection_months": len(triplets),
        "n_unique_receiver_months": len(triplets.drop_duplicates(["target", "month_index"])),
        "n_same_calendar_day": int(triplets.sample_span_days.eq(0).sum()), "median_calendar_span_days": triplets.sample_span_days.median(),
        "n_all_timestamps_known": int(triplets.all_timestamps_known.sum()),
        "n_sources_after_receiver_date": int(triplets.source_after_receiver_date.sum()),
        "n_sources_after_receiver_utc": int(triplets.source_after_receiver_utc.fillna(False).sum()),
        "median_station_median_gap_days": cadence.median_gap_days.median(), "pooled_median_gap_days": intervals.gap_days.median(),
        "fraction_all_three_one_day": triplets.all_three_one_day.mean(), "n_all_three_dense_months": int(triplets.all_three_dense_month.sum()),
        "n_missing_event_id": int(events.event_id_missing.sum()), "n_missing_timestamp": int((~events.timestamp_known).sum()),
        "n_same_metadata_activity_groups": int(events.groupby(["site_no", "date", "start_time", "time_zone"]).size().gt(1).sum()),
        "same_day_median_utc_span_hours": triplets.loc[triplets.sample_span_days.eq(0), "sample_span_hours_utc"].median(),
        "max_monthly_reconciliation_abs_difference": float(abs(monthly.doc-monthly.frozen_doc).max())}
    summary = {k: v.item() if isinstance(v, np.generic) else v for k, v in summary.items()}
    (a/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    for path in CODE:
        dest = ROOT/"code_snapshot"/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    sources = {"source_hashes": {str(p): sha256_file(p) for p in (INPUT, FOOTPRINT, CELLS, DATASET, ROOT/"study_plan.md", ROOT/"config.json", *CODE)},
               "raw_doc_inventory": inventory, "daily_inventory": daily_inventory,
               "product_hashes": {str(p): sha256_file(p) for p in sorted(a.iterdir()) if p.is_file()}}
    (ROOT/"analysis_sources.json").write_text(json.dumps(sources, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    print(tables["date_cut_ledger"].to_string(index=False))
    print(tables["signal_summary"].query("metric == 'asynchronous_buffer_fraction'").to_string(index=False))


if __name__ == "__main__":
    main()
