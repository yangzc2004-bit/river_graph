"""Find seasonal co-sampled river forms and retain their actual path geometry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_monitored_arrivals_v1 import (
    CELLS,
    DATASET,
    MAPPING,
    MASKS,
    NODES,
    PANEL,
    VAA,
    load_context,
    read_activities,
    routed_network,
)

from river_graph.analysis.river_cosampling_geometry import (
    pair_sampling_inventory,
    sampling_clock,
    select_cosampled_sources,
    within_month_dates,
)
from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_monitored_arrivals import (
    association_table,
    covered_geometry,
    form_contrasts,
    monitored_frontier,
    receiver_components,
    same_day_activities,
)
from river_graph.analysis.river_observed_synchrony import coordination_statistics
from river_graph.analysis.river_pathway_context import (
    comparison_opportunities,
    corridor_geometry,
)
from river_graph.experiments.provenance import sha256_file

BASE = Path("experiments/phase4_transfer")
ROOT = BASE/"doc_river_cosampling_geometry_v1"
PRIOR = BASE/"doc_river_monitored_arrivals_v1"
TYPES = {"station": str, "target": str, "site_no": str, "huc4": str}
METRICS = ("source_coherence", "outlet_sync_log_sd_ratio", "peak_risk_difference",
           "outlet_mix_correlation")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_cosampling_geometry_v1.py",
    "scripts/plot_doc_river_cosampling_geometry_v1.py", "scripts/verify_doc_river_cosampling_geometry_v1.py",
    "src/river_graph/analysis/river_cosampling_geometry.py", "tests/test_river_cosampling_geometry.py",
    "scripts/analyze_doc_river_monitored_arrivals_v1.py",
    "src/river_graph/analysis/river_monitored_arrivals.py",
    "src/river_graph/analysis/river_pathway_context.py",
    "src/river_graph/analysis/river_observed_synchrony.py",
    "src/river_graph/analysis/river_sampling_resolution.py")))


def calculate(context, activities, *, draws=5000):
    _, nodes, mapping, panel, _, _, _, vaa = context
    date_sets = {s: pd.DatetimeIndex(g.date.unique()).sort_values() for s, g in activities.groupby("site_no")}
    old = pd.read_csv(PRIOR/"analysis/network_inventory.csv", dtype=TYPES).set_index("station")
    inventory, candidates, geometry, corridors, observed, clocks, pair_audit = [], [], [], [], [], [], []
    files, members, station_groups = [], {}, {}
    for count, row in enumerate(panel.itertuples()):
        paths, layout, hydro, inputs = routed_network(row, vaa)
        files.extend(inputs)
        receiving = date_sets.get(row.station, pd.DatetimeIndex([]))
        inside = nodes[nodes.comid.isin(paths.comids) & nodes.comid.ne(row.comid)]
        gauges = []
        for s, node in inside.iterrows():
            shared = receiving.intersection(date_sets.get(s, pd.DatetimeIndex([])))
            if len(shared) and mapping.loc[s, "mapping_status"] != "gross_area_mismatch":
                gauges.append({"station": s, "comid": int(node.comid), "n_common_days": len(shared),
                    "reported_area_km2": mapping.loc[s, "drainage_area_km2"], "measure": node.measure})
        g = pd.DataFrame(gauges, columns=["station", "comid", "n_common_days", "reported_area_km2", "measure"])
        g = g.sort_values(["n_common_days", "station"], ascending=[False, True]).drop_duplicates("comid")
        g = monitored_frontier(paths, hydro, g)
        pair_meta = pair_sampling_inventory(g, date_sets, receiving,
            receiver_area=mapping.loc[row.station, "drainage_area_km2"])
        allowed_pairs = pair_meta[pair_meta.official_area_consistent] if len(pair_meta) else pair_meta
        pair_audit.append(pair_meta.assign(target=row.station, cluster=int(row.cluster), huc4=row.huc4,
            receiver_mapping_status=mapping.loc[row.station, "mapping_status"],
            physical_receiver_representative=row.physical_receiver_representative))
        sources, dates, status = select_cosampled_sources(g, date_sets, receiving,
            receiver_area=mapping.loc[row.station, "drainage_area_km2"])
        if mapping.loc[row.station, "mapping_status"] == "gross_area_mismatch":
            sources, status = [], "receiver_mapping_mismatch"
        if not row.physical_receiver_representative:
            sources, status = [], "receiving_COMID_alias"
        selected = g.set_index("station").loc[sources] if sources else g.iloc[:0].set_index("station")
        covered, areas, _ = covered_geometry(paths, hydro, selected.comid.to_numpy(), layout)
        dense = within_month_dates(dates) if sources else pd.DatetimeIndex([])
        meta = {"station": row.station, "comid": int(row.comid), "cluster": int(row.cluster),
            "huc4": row.huc4, "station_name": nodes.loc[row.station, "station_nm"],
            "physical_receiver_representative": row.physical_receiver_representative,
            "basin_area_km2": float(paths.area.sum()), "status": status, **covered,
            "n_common_days": len(dates) if sources else 0,
            "n_common_year_months": dates.to_period("M").nunique() if sources else 0,
            "n_within_month_days": len(dense), "n_dense_year_months": dense.to_period("M").nunique(),
            "within_month_eligible": len(dense) >= 12 and dense.to_period("M").nunique() >= 3,
            "old_monthly_included": old.loc[row.station, "status"] == "included",
            "old_monthly_status": old.loc[row.station, "status"], "n_candidate_gauges": len(g)}
        meta.update({"n_candidate_pairs": len(pair_meta),
            "maximum_pair_common_days": int(allowed_pairs.n_common_days.max()) if len(allowed_pairs) else 0,
            "maximum_pair_dense_days": int(allowed_pairs.n_within_month_days.max()) if len(allowed_pairs) else 0})
        inventory.append(meta)
        g["target"], g["selected"] = row.station, g.station.isin(sources)
        g["source_order"] = g.station.map({s: j for j, s in enumerate(sources)}).fillna(-1).astype(int)
        g["area_weight"] = g.station.map(dict(zip(sources, areas/areas.sum() if len(areas) else [], strict=True)))
        candidates.append(g)
        if sources:
            chosen = g[g.selected].sort_values("source_order")
            geo, reaches = corridor_geometry(paths, chosen, int(row.comid), nodes.loc[row.station, "measure"],
                vaa.loc[paths.comids, "wbareatype"].fillna("").to_numpy(),
                vaa.loc[paths.comids, "wbareacomi"].fillna(0).to_numpy(np.int64), layout)
            geometry.append({**meta, **geo, "target": row.station})
            corridors.append(reaches.assign(target=row.station))
            events = same_day_activities(activities, [*sources, row.station], dates.to_period("M").to_timestamp())
            np.testing.assert_array_equal(sorted(events.date.unique()), dates.to_numpy())
            observed.append(events.assign(target=row.station))
            clocks.append(sampling_clock(events).assign(target=row.station))
            members[row.station] = set(paths.comids[paths.area > 0].tolist())
            station_groups[row.station] = [int(nodes.loc[s, "comid"]) for s in [row.station, *sources]]
        if (count+1) % 50 == 0 or count+1 == len(panel):
            print(f"Co-sampled frontier inventory {count+1}/{len(panel)}", flush=True)
    inventory, gauges = pd.DataFrame(inventory), pd.concat(candidates, ignore_index=True)
    component = receiver_components(members, station_groups)
    geometry = pd.DataFrame(geometry)
    if geometry.empty:
        raise ValueError("no seasonal co-sampled sources; retain availability audit before analysis")
    geometry["component"] = geometry.target.map(component)
    events, clock = pd.concat(observed, ignore_index=True), pd.concat(clocks, ignore_index=True)
    signals, traces, periods = [], [], []
    for row in geometry.itertuples():
        e = events[events.target.eq(row.target)]
        g = gauges[gauges.target.eq(row.target) & gauges.selected].sort_values("source_order")
        w = g.area_weight.to_numpy(float)
        for version, value in (("selected_activity", "doc"), ("daily_mean", "doc_daily_mean")):
            frame = e.pivot(index="date", columns="source_order", values=value).sort_index()
            for adjustment in ("within_month", "calendar_year", "raw"):
                dates = within_month_dates(frame.index) if adjustment == "within_month" else frame.index
                if len(dates) < 12 or dates.to_period("M").nunique() < 3:
                    continue
                f = frame.loc[dates]
                for quantile in (.75, .9):
                    point, trace = coordination_statistics(f[list(range(len(w)))], f[-1], w, dates,
                        adjustment=adjustment, quantile=quantile)
                    labels = {"target": row.target, "cluster": row.cluster, "huc4": row.huc4,
                        "component": row.component, "covered_area_fraction": row.covered_area_fraction,
                        "monitored_path_cv": row.monitored_path_cv, "monitored_common_fraction": row.monitored_common_fraction,
                        "monitored_storage_length_fraction": row.monitored_storage_length_fraction,
                        "version": version, "adjustment": adjustment}
                    signals.append({**labels, **point})
                    traces.append(trace.assign(**labels, excursion_quantile=quantile))
        for month, f in e.groupby(e.date.dt.to_period("M")):
            common = pd.DatetimeIndex(f.date.unique()).sort_values()
            gap = np.diff(common.values)/np.timedelta64(1, "D")
            periods.append({"target": row.target, "cluster": row.cluster, "year_month": str(month),
                "n_common_days": len(common), "median_gap_days": np.median(gap) if len(gap) else np.nan,
                "fraction_gaps_le7_days": np.mean(gap <= 7) if len(gap) else np.nan})
    signals = pd.DataFrame(signals)
    summaries, contrasts, associations = [], [], []
    for (version, adjustment, quantile), f in signals.groupby(["version", "adjustment", "excursion_quantile"]):
        for coverage in (0., .8):
            subset = f[f.covered_area_fraction.ge(coverage)]
            labels = {"version": version, "adjustment": adjustment, "excursion_quantile": quantile,
                "minimum_coverage": coverage}
            for name, group in (("all", subset), *((f"class_{c}", subset[subset.cluster.eq(c)]) for c in (1, 2, 3))):
                for metric in METRICS:
                    result = cluster_mean(group, metric, "component", draws=draws)
                    if result:
                        summaries.append({**labels, "group": name, **result})
            contrasts.append(form_contrasts(subset, METRICS, draws=draws).assign(**labels))
            associations.append(association_table(subset,
                ("monitored_path_cv", "monitored_common_fraction", "source_coherence"),
                ("outlet_sync_log_sd_ratio", "peak_risk_difference"), draws=draws).assign(**labels))
    pairs = comparison_opportunities(inventory)
    lookup = inventory.set_index("station")
    pairs["both_within_month_eligible"] = [bool(lookup.loc[a, "within_month_eligible"] and lookup.loc[b, "within_month_eligible"])
        for a, b in zip(pairs.elongated_station, pairs.broad_station, strict=True)]
    for form, station_column in (("elongated", "elongated_station"), ("broad", "broad_station")):
        for column in ("maximum_pair_common_days", "maximum_pair_dense_days", "n_candidate_pairs"):
            pairs[form+"_"+column] = pairs[station_column].map(lookup[column])
    pairs["both_mapping_valid"] = ~pairs.elongated_status.eq("receiver_mapping_mismatch") & ~pairs.broad_status.eq("receiver_mapping_mismatch")
    priorities = pairs[pairs.area_comparable & pairs.both_mapping_valid].copy()
    priorities["balanced_common_days"] = priorities[["elongated_maximum_pair_common_days", "broad_maximum_pair_common_days"]].min(axis=1)
    priorities["balanced_dense_days"] = priorities[["elongated_maximum_pair_dense_days", "broad_maximum_pair_dense_days"]].min(axis=1)
    priorities["total_common_days"] = priorities.elongated_maximum_pair_common_days+priorities.broad_maximum_pair_common_days
    priorities = priorities.sort_values(["balanced_common_days", "balanced_dense_days", "total_common_days", "area_ratio", "elongated_station", "broad_station"],
        ascending=[False, False, False, True, True, True]).reset_index(drop=True)
    priorities["availability_rank"] = np.arange(1, len(priorities)+1)
    tables = {"network_inventory": inventory, "candidate_gauges": gauges, "network_geometry": geometry,
        "source_corridor_reaches": pd.concat(corridors, ignore_index=True), "same_day_activities": events,
        "clock_ledger": clock, "network_signals": signals, "signal_series": pd.concat(traces, ignore_index=True),
        "sampling_months": pd.DataFrame(periods), "signal_summary": pd.DataFrame(summaries),
        "form_contrasts": pd.concat(contrasts, ignore_index=True),
        "geometry_associations": pd.concat(associations, ignore_index=True), "same_region_form_pairs": pairs,
        "candidate_pair_inventory": pd.concat(pair_audit, ignore_index=True), "form_pair_observation_priorities": priorities}
    return tables, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--from-archive", action="store_true")
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    context = load_context()
    sites = {s for s, i in context[4].items() if np.isfinite(context[5][i]).any()}
    if args.from_archive:
        old = json.loads((ROOT/"analysis_sources.json").read_text())
        for r in old["raw_sources"]:
            if sha256_file(r["path"]) != r["sha256"]:
                raise ValueError("raw archive changed")
        for name in ("doc_activities.parquet", "raw_monthly_reconciliation.csv"):
            path = out/name
            if sha256_file(path) != old["output_hashes"][str(path)]:
                raise ValueError("saved permitted archive changed")
        activities = pd.read_parquet(out/"doc_activities.parquet")
        reconciliation = pd.read_csv(out/"raw_monthly_reconciliation.csv", dtype=TYPES, parse_dates=["month"])
        raw = old["raw_sources"]
    else:
        print(f"Reading accepted source-role activities for {len(sites)} stations", flush=True)
        activities, reconciliation, raw = read_activities(context, sites)
    tables, files = calculate(context, activities, draws=args.bootstrap_draws)
    tables.update({"doc_activities": activities, "raw_monthly_reconciliation": reconciliation})
    paths = []
    for name, frame in tables.items():
        path = out/f"{name}.{'parquet' if name in ('same_day_activities', 'signal_series', 'doc_activities') else 'csv'}"
        frame.to_parquet(path, index=False) if path.suffix == ".parquet" else frame.to_csv(path, index=False)
        paths.append(path)
    inventory = tables["network_inventory"]
    included = inventory[inventory.status.eq("included")]
    primary = tables["network_signals"].query("version == 'selected_activity' and adjustment == 'within_month' and excursion_quantile == .75")
    pairs = tables["same_region_form_pairs"]
    summary = {"n_station_networks": len(inventory), "n_physical_networks": inventory.comid.nunique(),
        "n_archive_stations": len(sites), "n_accepted_activities": len(activities),
        "n_same_day_receivers": len(included), "n_new_vs_monthly_receivers": int((~included.old_monthly_included).sum()),
        "n_within_month_receivers": len(primary), "n_within_month_systems": primary.component.nunique(),
        "class_same_day_counts": included.groupby("cluster").size().to_dict(),
        "class_within_month_counts": primary.groupby("cluster").size().to_dict(),
        "n_peak_eligible_receivers": int(primary.peak_comparison_eligible.sum()),
        "n_same_day_date_sets": len(tables["clock_ledger"]),
        "n_same_region_area_comparable_observed_pairs": int((pairs.area_comparable & pairs.both_observed).sum()),
        "n_same_region_area_comparable_dense_pairs": int((pairs.area_comparable & pairs.both_within_month_eligible).sum()),
        "n_same_region_area_comparable_high_coverage_pairs": int(pairs.usable_form_pair.sum()),
        "bootstrap_draws": args.bootstrap_draws, "new_training": False}
    config = {"previous_results_seen": True, "minimum_common_dates": 12, "minimum_year_months": 3,
        "within_month_minimum_dates": 3, "primary_adjustment": "within_month",
        "routing": "original geometric mainstem", "bootstrap_draws": args.bootstrap_draws,
        "source_selection": "metadata-only largest-area disjoint pair, then greedy additions",
        "source_role": "union of source train cells from splits 142/143/144",
        "thresholds": [.75, .9], "coverage_subsets": [0., .8]}
    for name, value in (("config", config), ("analysis/summary", summary)):
        path = ROOT/f"{name}.json"
        path.write_text(json.dumps(value, indent=2, default=lambda x: x.item())+"\n")
        paths.append(path)
    for path in CODE:
        destination = ROOT/"code_snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    sources = [ROOT/"study_plan.md", DATASET, NODES, CELLS, MAPPING, PANEL, VAA,
        PRIOR/"analysis/network_inventory.csv", *(MASKS/f"split{s}.npz" for s in (142, 143, 144)), *files, *CODE]
    (ROOT/"analysis_sources.json").write_text(json.dumps({
        "source_hashes": {str(p): sha256_file(p) for p in sources}, "raw_sources": raw,
        "output_hashes": {str(p): sha256_file(p) for p in paths}, "new_model_fit": False}, indent=2)+"\n")
    print(json.dumps(summary, indent=2, default=lambda x: x.item()), flush=True)


if __name__ == "__main__":
    main()
