"""Link complete mapped river form to non-overlapping monitored DOC inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_junction_layout import (
    condition_geometric_mainstem,
    partition_mainstem,
)
from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.analysis.river_monitored_arrivals import (
    association_table,
    availability_ok,
    covered_geometry,
    form_contrasts,
    monitored_frontier,
    receiver_components,
    same_day_activities,
    select_activity_set,
    select_source_set,
    signal_statistics,
)
from river_graph.analysis.river_sampling_resolution import doc_activities
from river_graph.analysis.river_whole_storage import StoragePaths
from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_monitored_arrivals_v1")
BASE = Path("experiments/phase4_transfer")
PANEL = BASE/"doc_river_morphology_effect_v1/analysis/station_morphology_doc_panel.csv"
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
CELLS = BASE/"doc_river_mechanisms_v1/analysis/source_cells.npy"
MAPPING = CELLS.parent/"station_mapping_audit.csv"
MASKS = BASE/"unified_doc_spatial_v1/confirmation/masks"
VAA = Path("cache/nldplus_vaa.parquet")
ROUTES = Path("data/raw/river_source_placement_v1/routing")
MEMBERS = Path("data/raw/river_planform_v1/members_full")
COLS = ["comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm", "areasqkm",
        "wbareatype", "wbareacomi", "arbolatesu"]
TYPES = {"station": str, "target": str, "site_no": str, "huc4": str}
METRICS = ("source_coherence", "mixture_buffer_fraction", "asynchronous_buffer_fraction",
           "outlet_mix_correlation", "real_minus_shuffle_correlation", "outlet_mix_log_sd_ratio")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_monitored_arrivals_v1.py",
    "scripts/plot_doc_river_monitored_arrivals_v1.py", "scripts/verify_doc_river_monitored_arrivals_v1.py",
    "src/river_graph/analysis/river_monitored_arrivals.py", "tests/test_river_monitored_arrivals.py",
    "src/river_graph/analysis/river_junction_layout.py", "src/river_graph/analysis/river_whole_storage.py",
    "src/river_graph/analysis/river_form_process.py", "src/river_graph/analysis/river_mechanisms.py",
    "src/river_graph/analysis/river_sampling_resolution.py", "src/river_graph/data/wqp.py")))


def load_context():
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    nodes = pd.read_csv(NODES, dtype={"site_no": str, "huc_cd": str}).set_index("site_no", drop=False)
    np.testing.assert_array_equal(nodes.site_no, np.asarray(data["site_no"], str))
    cells = np.load(CELLS)
    role_cells = []
    for seed in (142, 143, 144):
        with np.load(MASKS/f"split{seed}.npz") as z:
            role_cells.append(z["train"])
    np.testing.assert_array_equal(cells, np.unique(np.concatenate(role_cells)))
    visible = permitted_doc(data, cells)
    dates = pd.DatetimeIndex(data["months"])
    mapping = pd.read_csv(MAPPING, dtype={"site_no": str}).set_index("site_no")
    panel = pd.read_csv(PANEL, dtype=TYPES).copy()
    lookup = {str(s): i for i, s in enumerate(data["site_no"])}
    panel["n_allowed_doc_months"] = [int(np.isfinite(visible[lookup[s]]).sum()) for s in panel.station]
    chosen = panel.sort_values(["n_allowed_doc_months", "station"], ascending=[False, True]).drop_duplicates("comid").station
    panel["physical_receiver_representative"] = panel.station.isin(chosen)
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    if len(panel) != 297 or panel.comid.nunique() != 295:
        raise ValueError("retain original network population")
    return data, nodes, mapping, panel, lookup, visible, dates, vaa


def routed_network(row, vaa, shortest=False):
    route, member = ROUTES/f"comid_{int(row.comid)}.npz", MEMBERS/f"comid_{int(row.comid)}.npz"
    with np.load(route) as z:
        cid, distance = z["comids"], z["distance_km"]
    with np.load(member) as z:
        original = z["mainstem"]
        np.testing.assert_array_equal(np.sort(cid), np.sort(z["comids"]))
    finite = np.isfinite(distance)
    f = vaa.loc[cid[finite]]
    paths = StoragePaths(f, int(row.comid), distance[finite])
    if not shortest:
        paths = condition_geometric_mainstem(paths, f.hydroseq, original, f.dnhydroseq, f.dnminorhyd)
    layout = partition_mainstem(paths, f.hydroseq.to_numpy(), f.arbolatesu.to_numpy(),
        geometric_mainstem=original, prescribed_mainstem=None if shortest else original)
    return paths, layout, f.hydroseq.to_numpy(), [route, member]


def build_frontiers(context, *, shortest=False):
    _, nodes, mapping, panel, lookup, visible, dates, vaa = context
    allowed = {s: np.isfinite(visible[i]) for s, i in lookup.items()}
    inventory, gauge_tables, selected_sets, geometries, source_files = [], [], {}, {}, []
    for count, row in enumerate(panel.itertuples()):
        paths, layout, hydro, files = routed_network(row, vaa, shortest)
        source_files.extend(files)
        receiving = allowed[row.station]
        good_map = mapping.loc[row.station, "mapping_status"] != "gross_area_mismatch"
        inside = nodes[nodes.comid.isin(paths.comids) & nodes.comid.ne(row.comid)]
        candidates = []
        for station, node in inside.iterrows():
            common = receiving & allowed[station]
            if mapping.loc[station, "mapping_status"] == "gross_area_mismatch" or not availability_ok(common, dates):
                continue
            reported = mapping.loc[station, "drainage_area_km2"]
            candidates.append({"station": station, "comid": int(node.comid),
                "n_common_receiver_months": int(common.sum()), "reported_area_km2": reported,
                "measure": node.measure})
        g = pd.DataFrame(candidates, columns=["station", "comid", "n_common_receiver_months", "reported_area_km2", "measure"])
        g = g.sort_values(["n_common_receiver_months", "station"], ascending=[False, True]).drop_duplicates("comid")
        g = monitored_frontier(paths, hydro, g)
        sources, common, status = select_source_set(g, allowed, receiving, dates,
            receiver_area=mapping.loc[row.station, "drainage_area_km2"])
        if not good_map:
            sources, status = [], "receiver_mapping_mismatch"
        if not row.physical_receiver_representative:
            sources, status = [], "receiving_COMID_alias"
        selected = g.set_index("station").loc[sources] if sources else g.iloc[:0].set_index("station")
        metrics, areas, _ = covered_geometry(paths, hydro, selected.comid.to_numpy(), layout)
        meta = {"station": row.station, "comid": int(row.comid), "cluster": int(row.cluster), "huc4": row.huc4}
        info = {**meta, **metrics, "n_allowed_receiver_months": row.n_allowed_doc_months,
            "n_candidate_gauges": len(g), "n_frontier_gauges": int(g.frontier.sum()),
            "n_common_months": int(common.sum()) if sources else 0, "status": status,
            "physical_receiver_representative": row.physical_receiver_representative,
            "basin_area_km2": float(paths.area.sum()),
            "entry_branch_mean_correlation": layout.descriptors["entry_branch_mean_correlation"],
            "entry_branch_covariance_fraction": layout.descriptors["entry_branch_covariance_fraction"],
            "effective_area_weighted_entries": layout.descriptors["effective_area_weighted_entries"]}
        inventory.append(info)
        g["target"], g["selected"] = row.station, g.station.isin(sources)
        g["source_order"] = g.station.map({s: j for j, s in enumerate(sources)}).fillna(-1).astype(int)
        g["area_weight"] = g.station.map(dict(zip(sources, areas/areas.sum() if len(areas) else [], strict=True)))
        node_distance = dict(zip(paths.comids, paths.distance, strict=True))
        node_length = dict(zip(paths.comids, paths.length, strict=True))
        receiver_measure = nodes.loc[row.station, "measure"]
        g["gauge_to_receiver_km"] = [node_distance[c]-node_length[c]/2+node_length[c]*m/100
            -node_length[row.comid]*receiver_measure/100 for c, m in zip(g.comid, g.measure, strict=True)]
        gauge_tables.append(g)
        if sources:
            selected_sets[row.station] = (sources, common, areas, info)
            geometries[row.station] = set(paths.comids[paths.area > 0].tolist())
        if (count+1) % 50 == 0 or count+1 == len(panel):
            print(f"Mapped observation frontiers {count+1}/{len(panel)} ({'shortest' if shortest else 'original trunk'})", flush=True)
    gauges = pd.concat(gauge_tables, ignore_index=True)
    groups = {s: [int(nodes.loc[t, "comid"]) for t in (s, *values[0])] for s, values in selected_sets.items()}
    components = receiver_components(geometries, groups)
    return pd.DataFrame(inventory), gauges, selected_sets, components, source_files


def read_activities(context, stations):
    _, _, _, _, lookup, visible, dates, _ = context
    rows, inventory, reconcile = [], [], []
    for station in sorted(stations):
        path = Path("data/raw/wqp_results")/f"{station}.csv"
        raw = load_station_results(path)
        accepted = extract_doc_obs(raw)
        permitted = set(dates[np.isfinite(visible[lookup[station]])].to_period("M"))
        accepted = accepted[accepted.site_no.eq(station) & accepted.date.dt.to_period("M").isin(permitted)
                            & accepted.doc.ge(0) & np.isfinite(accepted.doc)].copy()
        meta = raw.loc[accepted.index, ["Activity_ActivityIdentifier", "Activity_StartTime", "Activity_StartTimeZone"]].rename(
            columns={"Activity_ActivityIdentifier": "event_id", "Activity_StartTime": "start_time", "Activity_StartTimeZone": "time_zone"})
        f = accepted.join(meta)
        f["month"] = f.date.dt.to_period("M").dt.to_timestamp()
        monthly = f.groupby("month", as_index=False).agg(doc=("doc", "mean"), n_results=("doc", "size"))
        index = dates.get_indexer(monthly.month)
        monthly["frozen_doc"] = visible[lookup[station], index]
        np.testing.assert_allclose(monthly.doc, monthly.frozen_doc, atol=2e-5, rtol=1e-6)
        expected = dates[np.isfinite(visible[lookup[station]])]
        if set(monthly.month) != set(expected):
            raise ValueError(f"raw monthly coverage differs from allowed cells: {station}")
        reconcile.append(monthly.assign(station=station))
        rows.append(f)
        inventory.append({"station": station, "path": str(path), "sha256": sha256_file(path),
                          "accepted_permitted_results": len(f)})
    events = doc_activities(pd.concat(rows, ignore_index=True))
    return events, pd.concat(reconcile, ignore_index=True), inventory


def observation_tables(context, sets, components, activities, *, draws=5000):
    data, _, _, _, lookup, visible, dates, _ = context
    statistics, series, sampling, sources_long = [], [], [], []
    event_lookup = {k: v for k, v in activities.groupby(["site_no", "month"])}
    flow = list(data["feature_channels"]).index("discharge")
    x, xm = np.asarray(data["x"]), np.asarray(data["x_mask"])
    for target, (sources, common, area, info) in sorted(sets.items()):
        ids, slots = [lookup[s] for s in sources], np.flatnonzero(common)
        a, y, weight, time = visible[ids][:, slots].T, visible[lookup[target], slots], area/area.sum(), dates[slots]
        meta = {"target": target, "huc4": info["huc4"], "cluster": info["cluster"], "component": components[target]}
        result, trace = signal_statistics(a, y, weight, time, seed=int(info["comid"])+42)
        positive = xm[[*ids, lookup[target]]][:, slots, flow] & (x[[*ids, lookup[target]]][:, slots, flow] > 0)
        measured = positive.all(axis=0)
        ratio = x[ids][:, slots[measured], flow].sum(axis=0)/x[lookup[target], slots[measured], flow]
        result.update({"n_all_positive_flow_months": int(measured.sum()),
                       "median_monitored_receiver_flow_ratio": np.median(ratio) if len(ratio) else np.nan})
        statistics.append({**meta, **result, **{k: v for k, v in info.items() if k not in ("station", "cluster", "huc4")},
                           "sample_cut_days": -1, "version": "full_monthly"})
        series.append(trace.assign(**meta, month_index=slots))
        aligned = []
        for index, date in enumerate(time):
            chosen, span = select_activity_set([event_lookup[(s, date)] for s in (*sources, target)])
            utc = [r["timestamp_utc"] for r in chosen]
            hours = (max(utc)-min(utc)).total_seconds()/3600 if all(pd.notna(t) for t in utc) else np.nan
            aligned.append((chosen, span))
            sampling.append({**meta, "month": date, "month_index": slots[index], "sample_span_days": span,
                "sample_span_hours": hours, "n_sources": len(sources),
                "all_stations_ge3_sample_days": all(event_lookup[(s, date)].date.nunique() >= 3 for s in (*sources, target)),
                "source_after_receiver_utc": any(t > utc[-1] for t in utc[:-1]) if np.isfinite(hours) else None})
            for j, (station, r) in enumerate(zip((*sources, target), chosen, strict=True)):
                sources_long.append({**meta, "month": date, "station": station, "source_order": j if j < len(sources) else -1,
                    "sample_event": r["event_id"], "sample_date": r["date"], "sample_utc": r["timestamp_utc"],
                    "doc_activity": r["doc"], "doc_monthly": a[index, j] if j < len(sources) else y[index],
                    "area_weight": weight[j] if j < len(sources) else np.nan, "sample_span_days": span})
        for cut in (0, 1, 3, 7):
            select = np.array([span <= cut for _, span in aligned])
            if not availability_ok(select, time):
                continue
            dated = np.asarray([[r["doc"] for r in chosen] for chosen, _ in aligned])[select]
            for version, src, rec in (("monthly_same_dates", a[select], y[select]),
                                      ("selected_activities", dated[:, :-1], dated[:, -1])):
                result, _ = signal_statistics(src, rec, weight, time[select], seed=int(info["comid"])+42)
                statistics.append({**meta, **result, "sample_cut_days": cut, "version": version})
    statistics = pd.DataFrame(statistics)
    summary = []
    for (cut, version), f in statistics.groupby(["sample_cut_days", "version"]):
        for group in ("all", "class_1", "class_2", "class_3"):
            g = f if group == "all" else f[f.cluster.eq(int(group[-1]))]
            for metric in METRICS:
                value = cluster_mean(g, metric, "component", draws=draws)
                if value is not None:
                    summary.append({"sample_cut_days": cut, "version": version, "group": group, **value})
    for cut in (0, 1, 3, 7):
        f = statistics[statistics.sample_cut_days.eq(cut)]
        a = f[f.version.eq("selected_activities")]
        b = f[f.version.eq("monthly_same_dates")]
        if a.empty:
            continue
        paired = a.merge(b, on=["target", "huc4", "cluster", "component"], suffixes=("_dated", "_monthly"), validate="one_to_one")
        for metric in METRICS:
            paired[metric] = paired[metric+"_dated"]-paired[metric+"_monthly"]
            value = cluster_mean(paired, metric, "component", draws=draws)
            if value is not None:
                summary.append({"sample_cut_days": cut, "version": "dated_minus_monthly_same_dates", "group": "all", **value})
    association = association_table(statistics[statistics.version.eq("full_monthly")],
        ("entry_branch_mean_correlation", "entry_branch_covariance_fraction", "full_path_cv", "covered_path_cv", "covered_area_fraction"),
        ("source_coherence", "outlet_mix_correlation", "outlet_mix_log_sd_ratio"), draws=draws)
    sample = pd.DataFrame(sampling)
    ledger = []
    for cut in (0, 1, 3, 7, 31):
        f = sample[sample.sample_span_days.le(cut)]
        eligible = statistics[(statistics.sample_cut_days.eq(cut) & statistics.version.eq("selected_activities"))]
        ledger.append({"max_span_days": cut, "n_available_receiver_months": len(f),
            "n_available_receivers": f.target.nunique(), "n_available_components": f.component.nunique(),
            "n_eligible_receivers": len(eligible) if cut != 31 else len(sets),
            "n_eligible_components": eligible.component.nunique() if cut != 31 else len(set(components.values()))})
    contrasts = form_contrasts(statistics[statistics.version.eq("full_monthly")], METRICS, draws=draws)
    return {"receiver_signals": statistics, "signal_summary": pd.DataFrame(summary), "form_contrasts": contrasts,
        "monthly_series": pd.concat(series, ignore_index=True), "aligned_sample_sets": sample,
        "selected_activities": pd.DataFrame(sources_long), "sampling_ledger": pd.DataFrame(ledger),
        "geometry_signal_associations": association}


def weekly_case_tables(context, sets, activities, sampling):
    """Sampling-density follow-up; one physical case, never a form-level test."""
    dense = sampling[sampling.all_stations_ge3_sample_days].groupby("target").size()
    if dense.empty:
        raise ValueError("no sampling-dense system available for the recorded follow-up")
    target = str(min(dense.index, key=lambda s: (-int(dense[s]), str(s))))
    sources, common, area, info = sets[target]
    dates = context[6][np.flatnonzero(common)]
    stations = [*sources, target]
    events = same_day_activities(activities, stations, dates)
    weight = area/area.sum()
    pivot = events.pivot(index="date", columns="site_no", values="doc").reindex(columns=stations)
    daily_mean = events.pivot(index="date", columns="site_no", values="doc_daily_mean").reindex(columns=stations)
    spans = events.groupby("date").timestamp_utc.agg(["min", "max", "count"])
    spans["utc_span_hours"] = (spans["max"]-spans["min"]).dt.total_seconds()/3600
    spans.loc[spans["count"].ne(len(stations)), "utc_span_hours"] = np.nan
    monthly_n = pivot.groupby(pivot.index.to_period("M")).size()
    dense_dates = np.asarray(monthly_n.reindex(pivot.index.to_period("M")) >= 3)
    statistics, traces = [], []
    for version, frame in (("metadata_selected_activity", pivot), ("all_activities_daily_mean", daily_mean)):
        for adjustment in ("calendar_year", "within_month"):
            f = frame[dense_dates] if adjustment == "within_month" else frame
            values, trace = signal_statistics(f.iloc[:, :-1], f.iloc[:, -1], weight, f.index,
                seed=int(info["comid"])+42, adjustment=adjustment)
            values["n_joint_dates"] = values.pop("n_months")
            statistics.append({"target": target, "version": version, "adjustment": adjustment,
                "n_calendar_months": f.index.to_period("M").nunique(),
                "n_sampled_years": f.index.year.nunique(), **values})
            trace["utc_span_hours"] = spans.utc_span_hours.reindex(f.index).to_numpy()
            traces.append(trace.assign(target=target, version=version, adjustment=adjustment))
    ledger = []
    for year in np.unique(pivot.index.year):
        index = pivot.index[pivot.index.year == year]
        diffs = pd.Series(index).diff().dt.days.dropna()
        ledger.append({"target": target, "year": int(year), "n_joint_days": len(index),
            "n_joint_months": index.to_period("M").nunique(), "first_joint_day": index.min(),
            "last_joint_day": index.max(), "median_within_year_interval_days": diffs.median() if len(diffs) else np.nan,
            "median_utc_span_hours": spans.utc_span_hours.reindex(index).median()})
    nodes = context[1]
    stations_table = pd.DataFrame({"target": target, "station": stations,
        "station_name": nodes.loc[stations, "station_nm"].to_numpy(),
        "source_order": [*range(len(sources)), -1], "area_weight": [*weight, np.nan],
        "n_dense_common_months": int(dense[target]),
        "selection": "maximum number of jointly dense months, then receiver ID; no DOC values",
        "context": "lake outlet present; one receiving system, not form-level replication"})
    return {"weekly_case_selected_activities": events.assign(target=target),
        "weekly_case_series": pd.concat(traces, ignore_index=True),
        "weekly_case_signals": pd.DataFrame(statistics), "weekly_case_years": pd.DataFrame(ledger),
        "weekly_case_stations": stations_table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--shortest-routes", action="store_true")
    args = parser.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    context = load_context()
    inventory, gauges, sets, components, files = build_frontiers(context, shortest=args.shortest_routes)
    if not sets:
        raise ValueError("no observational frontier passed availability; preserve inventory and revisit sources")
    sites = set(sets) | {s for values in sets.values() for s in values[0]}
    activities, reconcile, raw = read_activities(context, sites)
    tables = observation_tables(context, sets, components, activities, draws=args.bootstrap_draws)
    tables.update(weekly_case_tables(context, sets, activities, tables["aligned_sample_sets"]))
    tables.update({"network_inventory": inventory, "candidate_gauges": gauges,
                  "doc_activities": activities, "raw_monthly_reconciliation": reconcile})
    included = inventory[inventory.status.eq("included")]
    examples = included.sort_values(["covered_area_fraction", "n_common_months", "station"],
        ascending=[False, False, True]).groupby("cluster").head(1)
    tables["representatives"] = examples.assign(selection="maximum covered area, then common dates; no DOC values")
    paths = []
    parquet = {"monthly_series", "selected_activities", "doc_activities",
               "weekly_case_selected_activities", "weekly_case_series"}
    for name, f in tables.items():
        p = out/f"{name}.{'parquet' if name in parquet else 'csv'}"
        f.to_parquet(p, index=False) if name in parquet else f.to_csv(p, index=False)
        paths.append(p)
    full = tables["receiver_signals"].query("version == 'full_monthly'")
    summary = {"n_inventory_instances": len(inventory), "n_physical_receiving_networks": inventory.comid.nunique(),
        "n_observed_receivers": len(sets), "n_systems": len(set(components.values())),
        "n_receiver_months": len(tables["monthly_series"]), "n_observed_stations": len(sites),
        "class_receivers": {str(k): int(v) for k, v in full.groupby("cluster").size().items()},
        "class_systems": {str(k): int(v) for k, v in full.groupby("cluster").component.nunique().items()},
        "n_multi_input_receivers_ge3": int(full.n_inputs.ge(3).sum()),
        "n_receivers_coverage_ge80pct": int(included.covered_area_fraction.ge(.8).sum()),
        "median_covered_area_fraction": included.covered_area_fraction.median(),
        "min_covered_area_fraction": included.covered_area_fraction.min(),
        "max_covered_area_fraction": included.covered_area_fraction.max(),
        "median_sample_span_days": tables["aligned_sample_sets"].sample_span_days.median(),
        "n_dense_common_months": int(tables["aligned_sample_sets"].all_stations_ge3_sample_days.sum()),
        "n_source_after_receiver_utc": int(tables["aligned_sample_sets"].source_after_receiver_utc.fillna(False).sum()),
        "n_station_months_reconciled": len(reconcile),
        "max_raw_monthly_difference": float(abs(reconcile.doc-reconcile.frozen_doc).max()),
        "weekly_case_target": tables["weekly_case_stations"].target.iloc[0],
        "n_weekly_case_same_day_sets": tables["weekly_case_selected_activities"].date.nunique(),
        "shuffles_per_receiver": 200, "bootstrap_draws": args.bootstrap_draws, "new_training": False,
        "scope": "source-role monthly multi-gauge observation; not measured event transit or external validation"}
    config = {"previous_results_seen": True, "minimum_months": 24, "minimum_years": 3,
        "minimum_calendar_months": 6, "sampling_cuts_days": [0, 1, 3, 7], "bootstrap_draws": args.bootstrap_draws,
        "routing": "saved shortest" if args.shortest_routes else "original geometric trunk conditioned on real edges",
        "source_selection": "disjoint eligible frontier; largest-area valid pair then greedy area-ordered additions",
        "receiver_alias": "largest allowed-month count then station ID",
        "source_alias": "largest receiver-month overlap then station ID", "shuffle_draws": 200,
        "aggregation": "unique physical receiver equal; complete overlapping-catchment systems resampled",
        "weekly_follow_up": {"case_selection": "maximum jointly dense months, then receiver ID",
            "activity_selection": "earliest known UTC timestamp per station/day then activity ID",
            "within_month_minimum_joint_days": 3,
            "scope": "one lake-influenced case, no population CI or travel-time optimization"}}
    for name, value in (("config", config), ("analysis/summary", summary)):
        p = root/f"{name}.json"
        p.write_text(json.dumps(value, indent=2, default=lambda x: x.item())+"\n")
        paths.append(p)
    for p in CODE:
        target = root/"code_snapshot"/p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    sources = [ROOT/"study_plan.md", DATASET, NODES, CELLS, MAPPING, PANEL, VAA,
               *(MASKS/f"split{s}.npz" for s in (142, 143, 144)), *files, *CODE]
    (root/"analysis_sources.json").write_text(json.dumps({
        "source_hashes": {str(p): sha256_file(p) for p in sources}, "raw_sources": raw,
        "output_hashes": {str(p): sha256_file(p) for p in paths}, "new_model_fit": False}, indent=2)+"\n")
    print(json.dumps(summary, indent=2, default=lambda x: x.item()), flush=True)
    print(tables["signal_summary"].query("group == 'all' and sample_cut_days == -1").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
