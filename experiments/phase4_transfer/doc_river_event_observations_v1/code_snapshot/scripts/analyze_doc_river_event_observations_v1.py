"""Audit real DOC cadence, mapped connections and flow-selected spring windows."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
from pathlib import Path

import networkx as nx
import pandas as pd
from shapely.geometry import shape

from river_graph.analysis.river_event_observations import (
    cadence,
    mapped_network,
    mapped_relations,
    match_campaigns,
    read_sites_csv,
    spring_windows,
    window_coverage,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
RAW = Path("data/raw/river_event_observations_v1")
OLD = Path("experiments/phase4_transfer/doc_river_sampling_resolution_v1/analysis")


def read_public(manifest):
    observations, flows, stations, profiles, polygons = {}, {}, [], [], {}
    seen = set()
    for item in manifest["objects"]:
        if item["kind"] not in ("chemistry", "flow") or item["path"] in seen:
            continue
        seen.add(item["path"])
        site = f"C{item['site']}"
        path = Path(item["path"])
        variable = "DOC" if item["kind"] == "chemistry" else "Q"
        frame = read_sites_csv(path, variable)
        frame["site"] = site
        frame["evidence_type"] = "laboratory_DOC" if variable == "DOC" else "stage_derived_daily_discharge"
        (observations if variable == "DOC" else flows)[site] = frame
        if variable != "DOC":
            continue
        metadata = json.loads(Path(item["metadata"]).read_text())
        acquisition = metadata["specificInfo"]["acquisition"]
        point = acquisition["samplingPoint"]
        stations.append({"site": site, "latitude": point["lat"], "longitude": point["lon"],
                         "sampling_point": point["label"], "object_url": item["object_url"]})
        profiles.append({"site": site, **cadence(frame)})
        for feature in metadata["coverageGeo"]["features"]:
            if feature["geometry"]["type"] in ("Polygon", "MultiPolygon"):
                polygons[site] = shape(feature["geometry"])
                break
    return observations, flows, pd.DataFrame(stations), pd.DataFrame(profiles), polygons


def build_tables(manifest):
    observations, flows, stations, profiles, polygons = read_public(manifest)
    features = json.loads((RAW / "streams.geojson").read_text())["features"]
    graph, snapping, _ = mapped_network(features, stations)
    relations = mapped_relations(graph, snapping, polygons)
    connections = []
    nearest = relations[relations.usable_mapped_connection & relations.nearest_monitored_upstream]
    for receiver, candidates in nearest.groupby("receiver"):
        sources = sorted(candidates.source)
        for a, b in itertools.combinations(sources, 2):
            # Independence means separate monitored upstream routes, not disjoint
            # metadata polygons, which can depict incremental local areas.
            source_relations = relations[
                relations.source.isin([a, b]) & relations.receiver.isin([a, b])
            ]
            if not source_relations.empty:
                continue
            overlap = polygons[a].intersection(polygons[b]).area / min(polygons[a].area, polygons[b].area)
            paths = [nx.shortest_path(graph, f"site:{s}", f"site:{receiver}", weight="length_m")
                     for s in (a, b)]
            first_common = next(n for n in paths[0] if n in set(paths[1]))
            shared = paths[0][paths[0].index(first_common):]
            common_km = sum(graph[x][y]["length_m"] for x, y in itertools.pairwise(shared)) / 1000
            path_a = float(candidates[candidates.source.eq(a)].path_km.iloc[0])
            path_b = float(candidates[candidates.source.eq(b)].path_km.iloc[0])
            connections.append({"source_a": a, "source_b": b, "receiver": receiver,
                                "source_polygon_overlap_share": overlap,
                                "path_a_km": path_a, "path_b_km": path_b,
                                "common_path_km": common_km,
                                "common_path_fraction_unweighted": 2 * common_km / (path_a + path_b),
                                "independent_path_difference_km": abs(path_a - path_b)})
    connections = pd.DataFrame(connections)
    windows, coverage = [], []
    years = sorted(set().union(*(set(f.date_local.dt.year) for f in observations.values())))
    for site, flow in sorted(flows.items()):
        if site not in observations:
            continue
        selected = spring_windows(flow, years)
        for window in selected.to_dict("records"):
            windows.append({"receiver": site, **window})
            linked = set(relations.loc[relations.usable_mapped_connection & relations.receiver.eq(site), "source"]) | {site}
            for sample_site in sorted(linked):
                coverage.append({"receiver": site, "sample_site": sample_site,
                                 "role": "receiver" if sample_site == site else "upstream",
                                 **window, **window_coverage(observations[sample_site], window["peak_date"])})
    windows, coverage = pd.DataFrame(windows), pd.DataFrame(coverage)
    triples, campaigns, variation = [], [], []
    for connection in connections.to_dict("records"):
        receiver = connection["receiver"]
        for window in windows[windows.receiver.eq(receiver)].to_dict("records"):
            sites = [connection["source_a"], connection["source_b"], receiver]
            sub = coverage[(coverage.receiver.eq(receiver)) & coverage.year.eq(window["year"])]
            per_site = sub.set_index("sample_site").loc[sites]
            dates = []
            for site in sites:
                f = observations[site]
                dates.append(set(f.loc[f.value.notna() & f.date_local.between(
                    window["start_date"], window["end_date"]), "date_local"]))
            common_dates = set.intersection(*dates)
            triples.append({**connection, **window,
                            "min_sample_days": int(per_site.n_days.min()),
                            "n_common_sample_days": len(common_dates),
                            "worst_gap_days": per_site.max_gap_with_edges_days.max(),
                            "all_three_span_response": bool(per_site.spans_response.all()),
                            "all_three_daily_or_better_sampling": bool(per_site.max_gap_with_edges_days.le(1).all()),
                            "system": "Krycklan"})
            sampled = [observations[s][observations[s].date_local.between(
                window["start_date"], window["end_date"])] for s in sites]
            matched = match_campaigns(*sampled)
            if matched.empty:
                continue
            key = {k: v for k, v in connection.items() if k in ("source_a", "source_b", "receiver")}
            key.update({"year": window["year"], "peak_date": window["peak_date"]})
            campaigns.append(matched.assign(**key))
            if per_site.spans_response.all() and len(matched) >= 5:
                cv = {col: matched[col].std(ddof=1) / matched[col].mean()
                      for col in ("doc_a", "doc_b", "doc_receiver")}
                variation.append({**key, "n_campaigns": len(matched),
                                  "source_a_cv": cv["doc_a"], "source_b_cv": cv["doc_b"],
                                  "receiver_cv": cv["doc_receiver"],
                                  "mean_upstream_cv": (cv["doc_a"] + cv["doc_b"]) / 2,
                                  "receiver_minus_mean_upstream_cv": cv["doc_receiver"] -
                                  (cv["doc_a"] + cv["doc_b"]) / 2,
                                  "source_concentration_correlation": matched.doc_a.corr(matched.doc_b),
                                  "largest_sampling_span_hours": matched.sampling_span_hours.max(),
                                  "evidence_type": "observed_campaign_variation_not_transport_lag"})
    triples = pd.DataFrame(triples)
    # Inspect the two original dense connection-months without changing their cohort.
    old = pd.read_parquet(OLD / "sample_triplets.parquet")
    dense = old.loc[old.all_three_dense_month].copy()
    activity = pd.read_parquet(OLD / "doc_activities.parquet")
    original_samples = []
    for row in dense.itertuples():
        station_col = "site_no" if "site_no" in activity else "station"
        date_col = "date" if "date" in activity else "date_local"
        for role, site in (("a", row.source_a), ("b", row.source_b), ("receiver", row.target)):
            sub = activity[(activity[station_col].eq(site)) &
                           (pd.to_datetime(activity[date_col]).dt.to_period("M") == row.date.to_period("M"))].copy()
            sub["connection_month"] = f"{row.pair_id}__{row.date:%Y-%m}"
            sub["role"] = role
            original_samples.append(sub)
    tables = {
        "station_cadence": profiles.merge(snapping, on="site", validate="one_to_one"),
        "stations": stations,
        "mapped_relations": relations,
        "monitored_confluences": connections,
        "spring_windows": windows,
        "window_sampling_coverage": coverage,
        "confluence_event_coverage": triples,
        "campaign_variation": pd.DataFrame(variation),
        "original_dense_connection_months": dense,
    }
    products = {
        "laboratory_doc": pd.concat(observations.values(), ignore_index=True),
        "daily_discharge": pd.concat(flows.values(), ignore_index=True),
        "original_dense_samples": pd.concat(original_samples, ignore_index=True),
        "matched_doc_campaigns": pd.concat(campaigns, ignore_index=True),
    }
    return tables, products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = json.loads((ROOT / "retrieval_manifest.json").read_text())
    tables, products = build_tables(manifest)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    for name, frame in products.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    cadence_table = tables["station_cadence"]
    triples = tables["confluence_event_coverage"]
    summary = {
        "n_doc_stations": len(cadence_table),
        "n_valid_laboratory_doc": int(cadence_table.n_valid.sum()),
        "n_mapped_stations": int(cadence_table.mapped.sum()),
        "median_station_gap_days": cadence_table.median_gap_days.median(),
        "median_station_spring_gap_days": cadence_table.spring_median_gap_days.median(),
        "n_usable_mapped_connections": int(tables["mapped_relations"].usable_mapped_connection.sum()),
        "n_monitored_confluence_configurations": len(tables["monitored_confluences"]),
        "n_flow_selected_spring_windows": len(tables["spring_windows"]),
        "n_confluence_windows": len(triples),
        "n_triplets_spanning_response": int(triples.all_three_span_response.sum()),
        "n_day_resolution_triplet_windows": int(triples.all_three_daily_or_better_sampling.sum()),
        "n_independent_research_catchments": 1,
        "window_selection": "March--June discharge maximum per year; minus/plus 14 days",
        "doc_curve_interpolation": False,
        "n_original_dense_connection_months": len(tables["original_dense_connection_months"]),
        "n_coeval_doc_campaigns": len(products["matched_doc_campaigns"]),
        "n_variation_comparisons": len(tables["campaign_variation"]),
        "n_receiver_lower_cv": int(tables["campaign_variation"].receiver_minus_mean_upstream_cv.lt(0).sum()),
        "n_variation_receivers": int(tables["campaign_variation"].receiver.nunique()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = [Path(x["path"]) for x in manifest["objects"]]
    sources += [Path(x["metadata"]) for x in manifest["objects"] if "metadata" in x]
    sources += [ROOT / "study_plan.md", ROOT / "retrieval_manifest.json", OLD / "sample_triplets.parquet",
                OLD / "doc_activities.parquet", Path("scripts/analyze_doc_river_event_observations_v1.py"),
                Path("src/river_graph/analysis/river_event_observations.py")]
    code = ROOT / "code_snapshot"
    for path in sources:
        if re.search(r"(^scripts/|^src/)", str(path)):
            target = code / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    outputs = [out / f"{n}.csv" for n in tables] + [out / f"{n}.parquet" for n in products] + [out / "summary.json"]
    receipt = {"source_hashes": {str(p): sha256_file(p) for p in sources},
               "output_hashes": {str(p): sha256_file(p) for p in outputs}}
    (ROOT / "analysis_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
