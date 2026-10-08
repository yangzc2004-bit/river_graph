"""Resolve non-ST357 public station leads and the current continuous DOC catalogue."""
from __future__ import annotations

import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from analyze_doc_river_monitored_arrivals_v1 import load_context, routed_network

from river_graph.analysis.river_cosampling_pairs import disjoint_pair_inventory
from river_graph.analysis.river_monitored_arrivals import monitored_frontier
from river_graph.analysis.river_morphology_closure import source_dates
from river_graph.analysis.river_sampling_resolution import doc_activities
from river_graph.data.wqp import (
    _payload_looks_complete,
    extract_doc_obs,
    load_station_results,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import (
    project_geometry,
    project_lines,
    read_cached_lines,
)

ROOT = Path("experiments/phase4_transfer/doc_river_morphology_closure_v1")
RAW = Path("data/raw/river_morphology_closure_v1/additional")
TYPES = {"station": str, "target": str, "huc4": str, "source_a": str, "source_b": str}


def download(item):
    identity = item.station_id
    url = "https://www.waterqualitydata.us/wqx3/Result/search?" + urlencode({
        "siteid": identity, "dataProfile": "narrow", "mimeType": "csv", "characteristicName": "Organic carbon"})
    path = RAW / f"{identity}.csv"
    record = {"station_id": identity, "url": url}
    if path.exists():
        return {**record, "status": "saved_download", "path": str(path), "sha256": sha256_file(path)}
    try:
        response = requests.get(url, timeout=(8, 25), headers={"Accept-Encoding": "identity"})
        record.update(http_status=response.status_code, bytes=len(response.content))
        if response.ok and _payload_looks_complete(response.content, response.headers.get("Content-Length")):
            frame = pd.read_csv(io.BytesIO(response.content), dtype=str, low_memory=False)
            if "Location_Identifier" not in frame:
                raise ValueError("unsupported result schema")
            path.write_bytes(response.content)
            record.update(status="retrieved", path=str(path), sha256=sha256_file(path), raw_rows=len(frame))
        else:
            record.update(status="provider_unavailable_or_incomplete")
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        record.update(status="access_error", error=str(exc)[:200])
    print(f"{identity}: {record['status']}", flush=True)
    return record


def continuous_catalogue():
    collection_path = Path("data/raw/river_morphology_closure_v1/discovery/usgs_collections.json")
    record = {"status": "not_available"}
    if not collection_path.exists():
        return record
    collections = json.loads(collection_path.read_text())["collections"]
    meta = next(c for c in collections if c["id"] == "time-series-metadata")
    queryables = next(link["href"] for link in meta["links"] if "queryables" in link["rel"] and "json" in link.get("type", ""))
    try:
        response = requests.get(queryables, timeout=(8, 25))
        record.update(queryables_url=queryables, queryables_http_status=response.status_code)
        response.raise_for_status()
        schema = response.json()
        path = RAW / "time_series_queryables.json"
        path.write_bytes(response.content)
        record.update(queryables_path=str(path), queryables_sha256=sha256_file(path))
        if "parameter_code" not in schema.get("properties", {}):
            record.update(status="unsupported_parameter_schema")
            return record
        base = next(link["href"] for link in meta["links"] if link["rel"] == "items" and "json" in link.get("type", ""))
        url = base.split("?")[0] + "?" + urlencode({"filter": "parameter_code = '00681'", "filter-lang": "cql2-text", "limit": 10000, "f": "json"})
        response = requests.get(url, timeout=(8, 25))
        record.update(items_url=url, items_http_status=response.status_code)
        response.raise_for_status()
        payload = response.json()
        path = RAW / "continuous_doc_metadata.json"
        path.write_bytes(response.content)
        features = payload.get("features", [])
        if any(str(f["properties"].get("parameter_code")) != "00681" for f in features):
            raise ValueError("provider did not apply the DOC parameter filter")
        sites = pd.read_csv(ROOT / "recovery/station_requests.csv", dtype={"station": str})
        targets = {"USGS-"+s for s in sites.loc[sites.receiver, "station"]}
        matched = [f["properties"] for f in features if f["properties"].get("monitoring_location_id") in targets]
        more = any(link.get("rel") == "next" for link in payload.get("links", []))
        record.update(status="retrieved", path=str(path), sha256=sha256_file(path),
            returned_doc_series=len(features), matching_target_series=len(matched),
            matched_target_series=matched, pagination_complete=not more)
    except (requests.RequestException, ValueError, StopIteration) as exc:
        record.update(status="access_or_schema_error", error=str(exc)[:250])
    return record


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "recovery/additional_retrieval.json"
    leads = pd.read_csv(ROOT / "recovery/additional_in_basin_stations.csv", dtype={"target": str, "huc4": str})
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text())["records"]
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(download, leads.drop_duplicates("station_id").sort_values("station_id").itertuples()))
        manifest_path.write_text(json.dumps({"records": records, "continuous_doc_catalogue": continuous_catalogue(),
            "selection": "all 49 in-basin river/stream leads from the one successfully returned regional organic-carbon catalogue; no DOC outcomes used",
            "other_regional_catalogues": "six timed out; those regions are not exhaustively covered"}, indent=2)+"\n")
    accepted, counts = [], []
    for record in records:
        if "path" not in record:
            counts.append({"station_id": record["station_id"], "status": record["status"], "accepted_activities": 0})
            continue
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError("additional archive changed")
        raw = load_station_results(path)
        f = extract_doc_obs(raw)
        identity = record["station_id"].removeprefix("USGS-")
        f = f[f.site_no.eq(identity) & f.doc.ge(0) & np.isfinite(f.doc)]
        meta = raw.loc[f.index, ["Activity_ActivityIdentifier", "Activity_StartTime", "Activity_StartTimeZone"]].rename(
            columns={"Activity_ActivityIdentifier": "event_id", "Activity_StartTime": "start_time", "Activity_StartTimeZone": "time_zone"})
        events = doc_activities(f.join(meta))
        accepted.append(events)
        counts.append({"station_id": record["station_id"], "status": record["status"], "accepted_activities": len(events), "accepted_dates": events.date.nunique()})
    added = pd.concat(accepted, ignore_index=True)
    added.to_parquet(ROOT / "analysis/additional_activities.parquet", index=False)
    pd.DataFrame(counts).to_csv(ROOT / "analysis/additional_activity_inventory.csv", index=False)
    core = pd.read_parquet(ROOT / "analysis/recovered_activities.parquet")
    events = pd.concat([core, added], ignore_index=True)
    dates = source_dates(events)
    _, nodes, mapping, panel, _, _, _, vaa = load_context()
    database = sqlite3.connect("file:data/raw/river_planform_v1/flowlines.sqlite?mode=ro", uri=True)
    mapped, all_pairs, receiver_rows = [], [], []
    for row in panel[panel.station.isin(leads.target)].itertuples():
        paths, _, hydro, _ = routed_network(row, vaa)
        lines = project_lines(read_cached_lines(database, paths.comids.tolist()))
        gauges = []
        receiving = dates[row.station]
        inside = nodes[nodes.comid.isin(paths.comids) & nodes.comid.ne(row.comid)]
        for station, n in inside.iterrows():
            if station in dates and mapping.loc[station, "mapping_status"] != "gross_area_mismatch":
                gauges.append({"station": station, "comid": int(n.comid),
                    "reported_area_km2": mapping.loc[station, "drainage_area_km2"],
                    "n_common": len(receiving.intersection(dates[station]))})
        from shapely import STRtree
        from shapely.geometry import Point

        ids = list(lines)
        geometries = [lines[c] for c in ids]
        tree = STRtree(geometries)
        for r in leads[leads.target.eq(row.station)].itertuples():
            identity = r.station_id.removeprefix("USGS-")
            point = project_geometry(Point(r.longitude, r.latitude))
            nearest, distances = tree.query_nearest(point, return_distance=True)
            distance = float(distances.min())
            tied = sorted(ids[int(i)] for i in nearest)
            cid = tied[0]
            # This mapping is explicitly provisional. Keep both 100 m and
            # 300 m cuts and exclude receiving-reach aliases.
            mapped.append({"target": row.station, "station_id": r.station_id, "station": identity,
                "comid": cid, "distance_to_mapped_stream_m": distance,
                "mapping_tied_reaches": len(tied), "accepted_activities": int(events.site_no.eq(identity).sum()),
                "source_scope": "coordinate-snapped public station, not verified NLDI hydrolocation"})
            if distance <= 300 and len(tied) == 1 and cid != row.comid and identity in dates:
                gauges.append({"station": identity, "comid": cid, "reported_area_km2": np.nan,
                    "n_common": len(receiving.intersection(dates[identity])), "snap_distance_m": distance})
        g = pd.DataFrame(gauges)
        if "snap_distance_m" not in g:
            g["snap_distance_m"] = 0.
        g["snap_distance_m"] = g.snap_distance_m.fillna(0.)
        for threshold in (100., 300.):
            selected = g[g.snap_distance_m.le(threshold)].sort_values(["n_common", "station"], ascending=[False, True]).drop_duplicates("comid")
            selected = monitored_frontier(paths, hydro, selected)
            pairs = disjoint_pair_inventory(selected, dates, receiving, receiver_area=mapping.loc[row.station, "drainage_area_km2"])
            valid = pairs[pairs.official_area_consistent] if len(pairs) else pairs
            all_pairs.append(pairs.assign(target=row.station, cluster=int(row.cluster), snap_threshold_m=threshold))
            receiver_rows.append({"target": row.station, "cluster": int(row.cluster), "snap_threshold_m": threshold,
                "n_pairs": len(pairs), "maximum_common_days": int(valid.n_common_days.max()) if len(valid) else 0,
                "maximum_dense_days": int(valid.n_within_month_days.max()) if len(valid) else 0,
                "same_day_eligible": bool(valid.same_day_eligible.any()) if len(valid) else False,
                "within_month_eligible": bool(valid.within_month_eligible.any()) if len(valid) else False})
    database.close()
    pd.DataFrame(mapped).to_csv(ROOT / "analysis/additional_mapping.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(ROOT / "analysis/additional_source_pairs.csv", index=False)
    pd.DataFrame(receiver_rows).to_csv(ROOT / "analysis/additional_receiver_coverage.csv", index=False)
    report = {"additional_station_leads": leads.station_id.nunique(),
        "station_downloads": sum("path" in r for r in records),
        "stations_with_accepted_doc": sum(r.get("accepted_activities", 0) > 0 for r in counts),
        "accepted_additional_activities": len(added), "provisional_snap_mapping": True,
        "receivers_with_any_eligible_pair": pd.DataFrame(receiver_rows).groupby("snap_threshold_m")["same_day_eligible"].sum().to_dict(),
        "receivers_with_dense_pair": pd.DataFrame(receiver_rows).groupby("snap_threshold_m")["within_month_eligible"].sum().to_dict()}
    (ROOT / "recovery/additional_summary.json").write_text(json.dumps(report, indent=2, default=int)+"\n")
    print(json.dumps(report, indent=2, default=int), flush=True)


if __name__ == "__main__":
    main()
