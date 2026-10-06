"""Real monitored tributary mixing, source contrast and longitudinal pathways."""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from river_graph.analysis.river_doc_structure import bootstrap_ols, huc_prefix
from river_graph.analysis.river_mechanisms import (
    bootstrap_receivers,
    branch_status,
    concentration_mix,
    correlation,
    detrend,
    drainage_order_consistent,
    gauge_mapping_status,
    mixing_summary,
    permitted_doc,
    receiver_summary,
    shared_station_components,
    simulate_networks,
)
from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file
from river_graph.models.daily_flow_features import (
    load_first_daily_discharge,
    reconcile_daily_discharge,
)
from river_graph.topology.river_structure import VAA_COLUMNS, ReachNetwork

ROOT = Path("experiments/phase4_transfer/doc_river_mechanisms_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
GEOM = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/station_classes.csv")
MASKS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks")
PATHS = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1/analysis/station_edge_paths.csv")
MEMBERS = Path("data/raw/river_planform_v1/members_full")


def primary_route(network, source, target):
    i, end = network.index(source), network.index(target)
    route, seen = [], set()
    while i not in seen:
        seen.add(i)
        route.append(i)
        if i == end:
            return route
        seq = network.downstream[i]
        pos = np.searchsorted(network.sorted_hydro, seq)
        if seq <= 0 or pos == len(network.sorted_hydro) or network.sorted_hydro[pos] != seq:
            break
        i = int(network.hydro_order[pos])
    raise ValueError("station edge has no primary downstream path")


def load_source_raw(dataset, visible, source_hashes):
    dates = pd.DatetimeIndex(dataset["months"]).to_period("M")
    rows, audits = [], []
    for i, station in enumerate(dataset["site_no"]):
        allowed = dates[np.isfinite(visible[i])]
        if not len(allowed):
            continue
        path = Path("data/raw/wqp_results")/f"{station}.csv"
        if not path.exists():
            audits.append({"station": str(station), "status": "raw_missing"})
            continue
        source_hashes[str(path)] = sha256_file(path)
        frame = extract_doc_obs(load_station_results(path))
        frame = frame[frame.site_no.eq(str(station)) & frame.date.dt.to_period("M").isin(allowed)].copy()
        negative = int(frame.doc.lt(0).sum())
        frame = frame[frame.doc.ge(0) & np.isfinite(frame.doc)].copy()
        frame["date"] = frame.date.dt.normalize()
        daily = frame.groupby(["site_no", "date"], as_index=False).doc.median()
        rows.append(daily)
        audits.append({"station": str(station), "status": "loaded", "permitted_raw_samples": len(frame),
                       "negative_samples_excluded": negative, "unique_sample_days": len(daily)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def source_components(candidates):
    labels = shared_station_components([(r.target, r.source_a, r.source_b) for r in candidates.itertuples()])
    sizes = pd.Series(list(labels.values())).value_counts()
    return {"n_shared_station_components": len(sizes), "component_station_counts": sizes.tolist()}


def pathway_analysis(dataset, visible, nodes, paths, geometry, areas, mapping, draws):
    x, xm = np.asarray(dataset["x"]), np.asarray(dataset["x_mask"], bool)
    flow = list(dataset["feature_channels"]).index("discharge")
    temp = list(dataset["feature_channels"]).index("temperature")
    dates = pd.DatetimeIndex(dataset["months"])
    lookup = {str(s): i for i, s in enumerate(dataset["site_no"])}
    ecology = pd.read_csv("data/processed/streamcat_attributes.csv").set_index("comid")
    ecology["wetland"] = ecology.pcthbwet2019ws+ecology.pctwdwet2019ws
    ecology["forest"] = ecology.pctconif2019ws+ecology.pctdecid2019ws+ecology.pctmxfst2019ws
    ecology = ecology.reindex(nodes.comid.unique())
    node = nodes.set_index("site_no")
    shape = geometry.set_index("station")
    rows = []
    for p in paths.itertuples():
        if mapping.loc[p.source, "mapping_status"] == "gross_area_mismatch" or mapping.loc[p.target, "mapping_status"] == "gross_area_mismatch":
            continue
        official_a, official_t = mapping.loc[p.source, "drainage_area_km2"], mapping.loc[p.target, "drainage_area_km2"]
        if np.isfinite(official_a+official_t) and official_a > 1.1*official_t:
            continue
        ia, it = lookup[p.source], lookup[p.target]
        common = np.isfinite(visible[ia]) & np.isfinite(visible[it])
        if common.sum() < 12:
            continue
        months = np.flatnonzero(common)
        a, b = visible[ia, months], visible[it, months]
        loga, logb = np.log1p(a), np.log1p(b)
        aa, at = areas[p.source], areas[p.target]
        ca, ct = int(node.loc[p.source, "comid"]), int(node.loc[p.target, "comid"])
        hydro = (xm[ia, months, flow] & xm[it, months, flow]
                 & xm[ia, months, temp] & xm[it, months, temp]
                 & (x[ia, months, flow] > 0) & (x[it, months, flow] > 0))
        row = {"source": p.source, "target": p.target, "huc4": huc_prefix(node.loc[p.target, "huc_cd"]),
               "cluster": shape.loc[p.target, "cluster"] if p.target in shape.index else np.nan,
               "n_common_months": len(months), "n_hydro_months": int(hydro.sum()),
               "path_length_km": p.path_length_km, "path_storage_km": p.path_storage_km,
               "storage_fraction": p.path_storage_km/p.path_length_km if p.path_length_km > 0 else 0.,
               "added_drainage_fraction": 1-aa/at, "source_area_km2": aa, "target_area_km2": at,
               "log_doc_change": np.mean(logb-loga), "native_doc_change": np.mean(b-a),
               "rho_calendar_adjusted": correlation(detrend(loga, dates[months]), detrend(logb, dates[months])),
               "downstream_cv_change": b.std()/b.mean()-a.std()/a.mean() if min(a.mean(), b.mean()) > 0 else np.nan,
               "wetland_change_pct": ecology.loc[ct, "wetland"]-ecology.loc[ca, "wetland"],
               "forest_change_pct": ecology.loc[ct, "forest"]-ecology.loc[ca, "forest"],
               "source_ecology_available": bool(np.isfinite(ecology.loc[ca, "wetland"])),
               "target_ecology_available": bool(np.isfinite(ecology.loc[ct, "wetland"])),
               "mapping_area_both_known": bool(np.isfinite(official_a+official_t)),
               "log_flow_ratio": np.nan, "temperature_change": np.nan, "rho_hydro_adjusted": np.nan}
        if hydro.sum() >= 24:
            qa, qt = x[ia, months[hydro], flow], x[it, months[hydro], flow]
            ta, tt = x[ia, months[hydro], temp], x[it, months[hydro], temp]
            ha, ht = np.column_stack([np.log1p(qa), ta]), np.column_stack([np.log1p(qt), tt])
            row.update(log_flow_ratio=float(np.mean(np.log(qt/qa))),
                       temperature_change=float(np.mean(tt-ta)),
                       rho_hydro_adjusted=correlation(detrend(loga[hydro], dates[months[hydro]], ha),
                                                      detrend(logb[hydro], dates[months[hydro]], ht)),
                       rho_calendar_common_hydro=correlation(detrend(loga[hydro], dates[months[hydro]]),
                                                            detrend(logb[hydro], dates[months[hydro]])))
        rows.append(row)
    edges = pd.DataFrame(rows)
    components = shared_station_components(list(zip(edges.source, edges.target, strict=True)))
    edges["component"] = edges.target.map(components)
    # Average at receiving stations: incoming pairs do not become replicates.
    columns = ["path_length_km", "storage_fraction", "added_drainage_fraction", "source_area_km2",
               "wetland_change_pct", "forest_change_pct", "log_flow_ratio", "temperature_change",
               "log_doc_change", "rho_calendar_adjusted", "downstream_cv_change"]
    receivers = edges.groupby(["target", "huc4"], as_index=False)[columns].mean()
    receivers["component"] = receivers.target.map(components)
    receivers["log_path_km"] = np.log1p(receivers.path_length_km)
    receivers["log_source_area"] = np.log1p(receivers.source_area_km2)
    coefficients = []
    terms = ["log_path_km", "storage_fraction", "added_drainage_fraction", "log_source_area",
             "wetland_change_pct", "forest_change_pct", "log_flow_ratio", "temperature_change"]
    for outcome in ("log_doc_change", "rho_calendar_adjusted", "downstream_cv_change"):
        s = receivers.dropna(subset=[outcome]).copy()
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        raw = imputer.fit_transform(s[terms])
        names = imputer.get_feature_names_out(terms)
        valid = raw.std(axis=0) > 1e-9
        design = np.column_stack([np.ones(len(s)), StandardScaler().fit_transform(raw[:, valid])])
        labels = ["intercept", *names[valid]]
        point, boot, diag = bootstrap_ols(design, s[outcome], draws=draws, groups=s.huc4)
        _, component_boot, _ = bootstrap_ols(design, s[outcome], draws=draws, groups=s.component)
        for j, label in enumerate(labels):
            if label == "intercept" or j in diag["dropped_columns"]:
                continue
            lo, hi = np.quantile(boot[:, j], [.025, .975])
            clo, chi = np.quantile(component_boot[:, j], [.025, .975])
            coefficients.append({"outcome": outcome, "term": label, "estimate_per_receiver_sd": point[j],
                                 "huc4_ci_low": lo, "huc4_ci_high": hi,
                                 "component_ci_low": clo, "component_ci_high": chi, "n_components": s.component.nunique(),
                                 "n_receivers": len(s), "n_huc4": s.huc4.nunique(), "condition": diag["condition"]})
    return edges, receivers, pd.DataFrame(coefficients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    paths = [DATASET, GEOM, PATHS, Path("data/processed/graph_nodes_graphfix_st357.csv"),
             Path("cache/nldplus_vaa.parquet"), Path("data/processed/streamcat_attributes.csv"), ROOT/"study_plan.md",
             ROOT/"metadata/usgs_gauge_metadata.csv", ROOT/"metadata/sources.json"]
    source_hashes = {str(p): sha256_file(p) for p in paths}
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    nodes = pd.read_csv(paths[3], dtype={"site_no": str, "huc_cd": str})
    np.testing.assert_array_equal(nodes.site_no.to_numpy(), np.asarray(dataset["site_no"], str))
    geometry = pd.read_csv(GEOM, dtype={"station": str})
    station_paths = pd.read_csv(PATHS, dtype={"source": str, "target": str})
    cells = []
    for split in (142, 143, 144):
        p = MASKS/f"split{split}.npz"
        source_hashes[str(p)] = sha256_file(p)
        with np.load(p) as mask:
            cells.extend(mask["train"])
    cells = np.unique(cells)
    np.save(out/"source_cells.npy", cells)
    visible = permitted_doc(dataset, cells)
    months = pd.DatetimeIndex(dataset["months"])
    lookup = {str(s): i for i, s in enumerate(dataset["site_no"])}
    node = nodes.set_index("site_no")
    classes = geometry.set_index("station").cluster
    vaa = pd.read_parquet(paths[4], columns=[*VAA_COLUMNS, "areasqkm"])
    area_lookup = vaa.set_index("comid").areasqkm
    network = ReachNetwork(vaa)
    memberships, areas = {}, {}
    for r in nodes.itertuples():
        p = MEMBERS/f"comid_{int(r.comid)}.npz"
        source_hashes[str(p)] = sha256_file(p)
        with np.load(p) as z:
            memberships[r.site_no] = np.unique(z["comids"])
        values = area_lookup.reindex(memberships[r.site_no]).to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("missing/invalid incremental area")
        areas[r.site_no] = values.sum()
    mapping = pd.read_csv(ROOT/"metadata/usgs_gauge_metadata.csv", dtype={"site_no": str}).set_index("site_no")
    mapping["mapped_area_km2"] = pd.Series(areas)
    mapping["mapped_reported_area_ratio"] = mapping.mapped_area_km2/mapping.drainage_area_km2
    mapping["mapping_status"] = [gauge_mapping_status(a, b) for a, b in
                                 zip(mapping.mapped_area_km2, mapping.drainage_area_km2, strict=True)]
    mapping.to_csv(out/"station_mapping_audit.csv")
    inventory = []
    for target, sub in station_paths.groupby("target"):
        for a, b in itertools.combinations(sorted(sub.source.unique()), 2):
            ca, cb, ct = (int(node.loc[s, "comid"]) for s in (a, b, target))
            status = branch_status(ca, cb, memberships[a], memberships[b], area_lookup)
            common = np.isfinite(visible[lookup[a]]) & np.isfinite(visible[lookup[b]]) & np.isfinite(visible[lookup[target]])
            row = {"pair_id": f"{a}__{b}__{target}", "source_a": a, "source_b": b, "target": target,
                   "huc4": huc_prefix(node.loc[target, "huc_cd"]), "cluster": classes.get(target, np.nan),
                   **status, "n_common_doc_months": int(common.sum()),
                   "source_drainage_coverage": (areas[a]+areas[b])/areas[target],
                   "junction_receiver_km": np.nan, "first_common_comid": np.nan,
                   "source_a_receiver_km": sub.set_index("source").loc[a, "path_length_km"],
                   "source_b_receiver_km": sub.set_index("source").loc[b, "path_length_km"]}
            if status["branch_status"] == "independent":
                ra, rb = primary_route(network, ca, ct), primary_route(network, cb, ct)
                rb_set = set(rb)
                joint = next(i for i in ra if i in rb_set)
                p = network.mainstem_path(network.comid[joint], 100., ct, node.loc[target, "measure"], max_distance=10000.)
                row.update(first_common_comid=int(network.comid[joint]), junction_receiver_km=p["path_length_km"])
            gauge = mapping.loc[[a, b, target]]
            row["mapping_area_all_known"] = gauge.mapping_status.eq("area_consistent").all()
            row["mapping_gross_mismatch"] = gauge.mapping_status.eq("gross_area_mismatch").any()
            row["reported_drainage_order_ok"] = drainage_order_consistent(*gauge.drainage_area_km2.to_numpy())
            row["eligible_monthly_nhd_candidate"] = status["branch_status"] == "independent" and common.sum() >= 12
            row["eligible_monthly"] = row["eligible_monthly_nhd_candidate"] and not row["mapping_gross_mismatch"] and row["reported_drainage_order_ok"]
            # Missing official areas are retained as unverified mapping, never
            # counted as an independently confirmed complete confluence.
            row["near_complete_confluence"] = row["source_drainage_coverage"] >= .8 and row["junction_receiver_km"] <= 20
            row["near_complete_confluence"] &= bool(row["mapping_area_all_known"] and row["eligible_monthly"])
            inventory.append(row)
    inventory = pd.DataFrame(inventory)
    components = shared_station_components([(r.target, r.source_a, r.source_b)
                                           for r in inventory[inventory.eligible_monthly].itertuples()])
    inventory["component"] = inventory.target.map(components)
    inventory.to_csv(out/"confluence_inventory.csv", index=False)
    print(f"Candidates {len(inventory)}, monthly eligible {inventory.eligible_monthly.sum()}", flush=True)
    raw, raw_audit = load_source_raw(dataset, visible, source_hashes)
    raw_audit.to_csv(out/"raw_doc_availability.csv", index=False)
    print(f"Permitted DOC sample days {len(raw)}; loading daily discharge", flush=True)
    daily, daily_inventory = load_first_daily_discharge("data/raw/nwis_dv")
    daily, qc = reconcile_daily_discharge(daily)
    for r in daily_inventory["raw_cache_files"]:
        source_hashes[r["path"]] = r["sha256"]
    (ROOT/"daily_flow_inventory.json").write_text(json.dumps({**daily_inventory, "reconciliation": qc}, indent=2)+"\n")
    raw = raw.merge(daily, on=["site_no", "date"], how="left", validate="one_to_one")
    raw.to_parquet(out/"source_sample_days.parquet", index=False)
    by_station = {s: f.set_index("date") for s, f in raw.groupby("site_no")}
    x, xm = np.asarray(dataset["x"]), np.asarray(dataset["x_mask"], bool)
    flow = list(dataset["feature_channels"]).index("discharge")
    summaries, records = [], []
    for p in inventory[inventory.eligible_monthly].itertuples():
        ids = [lookup[s] for s in (p.source_a, p.source_b, p.target)]
        common = np.isfinite(visible[ids]).all(axis=0)
        t = np.flatnonzero(common)
        monthly = pd.DataFrame({"date": months[t], "doc_a": visible[ids[0], t],
                                "doc_b": visible[ids[1], t], "doc_target": visible[ids[2], t]})
        for label, i in zip(("a", "b", "target"), ids, strict=True):
            monthly[f"q_{label}"] = np.where(xm[i, t, flow], x[i, t, flow], np.nan)
        frames = [("monthly", monthly)]
        if all(s in by_station for s in (p.source_a, p.source_b, p.target)):
            joined = pd.concat([by_station[s][["doc", "discharge_cfs"]].rename(
                columns={"doc": f"doc_{label}", "discharge_cfs": f"q_{label}"})
                for s, label in zip((p.source_a, p.source_b, p.target), ("a", "b", "target"), strict=True)], axis=1, join="inner").reset_index()
            frames.append(("same_day", joined))
        for population, frame in frames:
            for weighting in ("area", "flow"):
                f = frame.copy()
                if weighting == "flow":
                    f = f[np.isfinite(f.q_a) & np.isfinite(f.q_b) & f.q_a.gt(0) & f.q_b.gt(0)].copy()
                if len(f) < 12 or f.date.dt.to_period("M").nunique() < 12:
                    continue
                weights = (p.source_a_area_km2, p.source_b_area_km2) if weighting == "area" else (f.q_a, f.q_b)
                result = mixing_summary(f, weights=weights, dates=f.date)
                result.update(pair_id=p.pair_id, target=p.target, huc4=p.huc4, cluster=p.cluster,
                              population=population, weighting=weighting, component=p.component,
                              near_complete_confluence=p.near_complete_confluence,
                              source_drainage_coverage=p.source_drainage_coverage,
                              junction_receiver_km=p.junction_receiver_km)
                valid_flow = f.q_a.gt(0) & f.q_b.gt(0) & f.q_target.gt(0) & np.isfinite(f.q_a+f.q_b+f.q_target)
                result["flow_closure_median"] = ((f.q_a+f.q_b)/f.q_target)[valid_flow].median()
                result["n_flow_closure"] = int(valid_flow.sum())
                summaries.append(result)
                if p.near_complete_confluence:
                    summaries.append({**result, "population": population+"_near_complete"})
                f["mixture_doc"] = concentration_mix(f.doc_a, f.doc_b, *weights)
                f["pair_id"], f["target"], f["population"], f["weighting"] = p.pair_id, p.target, population, weighting
                records.append(f)
    cases = pd.DataFrame(summaries)
    cases.to_csv(out/"mixing_cases.csv", index=False)
    pd.concat(records, ignore_index=True).to_parquet(out/"mixing_records.parquet", index=False)
    receivers = receiver_summary(cases)
    receivers.to_csv(out/"mixing_receivers.csv", index=False)
    bootstrap_receivers(receivers, args.bootstrap_draws).to_csv(out/"mixing_results.csv", index=False)
    edges, edge_receivers, coefficients = pathway_analysis(dataset, visible, nodes, station_paths, geometry, areas, mapping, args.bootstrap_draws)
    edges.to_csv(out/"pathway_edges.csv", index=False)
    edge_receivers.to_csv(out/"pathway_receivers.csv", index=False)
    coefficients.to_csv(out/"pathway_associations.csv", index=False)
    sim, sim_summary = simulate_networks()
    sim.to_csv(out/"simulation_timeseries.csv", index=False)
    sim_summary.to_csv(out/"simulation_summary.csv", index=False)
    code = [Path(__file__), Path("src/river_graph/analysis/river_mechanisms.py"),
            Path("src/river_graph/analysis/river_doc_structure.py"), Path("src/river_graph/topology/river_structure.py"),
            Path("src/river_graph/data/wqp.py"), Path("src/river_graph/models/daily_flow_features.py")]
    for path in code:
        path = path.resolve().relative_to(Path.cwd())
        source_hashes[str(path)] = sha256_file(path)
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    record = {"source_hashes": source_hashes, "source_cells": len(cells),
              "raw_same_day_policy": "median dissolved uncensored mg/L within allowed source months; no date widening",
              "n_candidates": len(inventory), "n_monthly_eligible": int(inventory.eligible_monthly.sum()),
              "n_pre_screen_eligible": int(inventory.eligible_monthly_nhd_candidate.sum()),
              "mapping_audit": mapping.mapping_status.value_counts().to_dict(),
              "n_mixing_rows": len(cases), "n_pathway_edges": len(edges), "bootstrap_draws": args.bootstrap_draws,
              "shared_stations": source_components(inventory[inventory.eligible_monthly]),
              "models_retrained": False, "external_or_geographical_test_read": False,
              "simulation": "synthetic normalized timing, unfitted, equal source flow and reach budget"}
    (ROOT/"sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
