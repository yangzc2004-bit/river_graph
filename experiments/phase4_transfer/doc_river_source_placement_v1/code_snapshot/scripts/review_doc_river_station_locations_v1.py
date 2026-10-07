"""Review flagged gauge mappings with independent USGS hydrographic references."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape

from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import (
    COLUMNS,
    UpstreamNetwork,
    project_geometry,
)

ROOT = Path("experiments/phase4_transfer/doc_river_source_placement_v1")
CACHE = Path("data/raw/river_source_placement_v1/locations")
API = "https://api.water.usgs.gov/nldi/linked-data"


def fetch_json(url, path):
    if path.exists():
        value = json.loads(path.read_text())
    else:
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        value = response.json()
        if not value.get("features"):
            raise ValueError("empty USGS feature response")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, separators=(",", ":")))
        temp.replace(path)
    return value


def station_requests(row):
    identifier = f"USGS-{row.site_no}"
    urls = {
        "nwissite": f"{API}/nwissite/{identifier}?f=json",
        "hydrolocation": requests.Request("GET", f"{API}/hydrolocation",
            params={"f": "json", "coords": f"POINT({row.dec_long_va} {row.dec_lat_va})"}).prepare().url,
        "site_basin": f"{API}/nwissite/{identifier}/basin?f=json&simplified=false&splitCatchment=true",
    }
    records = []
    for kind, url in urls.items():
        path = CACHE/f"{row.site_no}_{kind}.json"
        try:
            value = fetch_json(url, path)
            records.append({"station": row.site_no, "kind": kind, "status": "ok",
                            "url": url, "file": str(path), "sha256": sha256_file(path), "value": value})
        except (requests.RequestException, ValueError) as error:
            records.append({"station": row.site_no, "kind": kind, "status": "error",
                            "url": url, "error": str(error)})
    return records


def main():
    audit_path = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/station_mapping_audit.csv")
    nodes_path = Path("data/processed/graph_nodes_graphfix_st357.csv")
    audit = pd.read_csv(audit_path, dtype={"site_no": str, "huc_cd": str})
    flagged = audit[audit.mapping_status.eq("gross_area_mismatch")].copy()
    nodes = pd.read_csv(nodes_path, dtype={"site_no": str}).set_index("site_no")
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reviewing {len(flagged)} flagged station mappings", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = [r for rows in pool.map(station_requests, flagged.itertuples()) for r in rows]
    vaa_path = Path("cache/nldplus_vaa.parquet")
    network = UpstreamNetwork(pd.read_parquet(vaa_path, columns=[*COLUMNS, "areasqkm"]))
    areas = network.frame.areasqkm.to_numpy(float)
    rows = []
    for r in flagged.itertuples():
        row = {"station": r.site_no, "station_name": r.station_nm,
               "original_comid": int(nodes.loc[r.site_no, "comid"]),
               "official_area_km2": r.drainage_area_km2, "original_area_km2": r.mapped_area_km2,
               "original_area_ratio": r.mapped_reported_area_ratio}
        cached_basin = Path("data/raw/river_planform_v1/basins")/f"comid_{row['original_comid']}.json"
        if cached_basin.exists():
            value = json.loads(cached_basin.read_text())
            basin = project_geometry(shape(value["features"][0]["geometry"]))
            row["cached_whole_basin_area_km2"] = basin.area/1e6
            row["cached_whole_basin_area_ratio"] = basin.area/1e6/r.drainage_area_km2
            row["membership_to_polygon_area_ratio"] = r.mapped_area_km2/(basin.area/1e6)
        for response in [v for v in responses if v["station"] == r.site_no and v["status"] == "ok"]:
            kind, value = response["kind"], response["value"]
            if kind == "site_basin":
                basin = project_geometry(shape(value["features"][0]["geometry"]))
                row["usgs_split_basin_area_km2"] = basin.area/1e6
                row["usgs_split_basin_ratio"] = basin.area/1e6/r.drainage_area_km2
                continue
            candidates = [f for f in value["features"] if f["properties"].get("comid")]
            if not candidates:
                continue
            f = candidates[0]
            cid = int(f["properties"]["comid"])
            indices, _ = network.membership(cid)
            row[f"{kind}_comid"] = cid
            row[f"{kind}_area_km2"] = float(areas[indices].sum())
            row[f"{kind}_area_ratio"] = float(areas[indices].sum()/r.drainage_area_km2)
            row[f"{kind}_measure"] = f["properties"].get("measure")
            point = project_geometry(shape(f["geometry"]))
            official = project_geometry(Point(r.dec_long_va, r.dec_lat_va))
            row[f"{kind}_location_distance_m"] = point.distance(official)
        alternatives = [row.get(f"{k}_comid") for k in ("nwissite", "hydrolocation")
                        if row.get(f"{k}_comid") != row["original_comid"]
                        and .5 <= row.get(f"{k}_area_ratio", np.nan) <= 2.]
        if r.site_no == "03201720":
            status = "independent_control_stream_confirmed_from_USGS_report"
        elif alternatives:
            status = "alternative_hydrographic_reference_area_consistent"
        elif .5 <= row.get("usgs_split_basin_ratio", np.nan) <= 2.:
            status = "USGS_split_basin_consistent_review_membership_or_local_position"
        elif .5 <= row.get("cached_whole_basin_area_ratio", np.nan) <= 2.:
            status = "receiving_basin_consistent_review_secondary_connections_or_membership"
        else:
            status = "unresolved_scale_position_or_drainage_discrepancy"
        row["review_status"] = status
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"station_location_review.csv", index=False)
    receipt = [{k: v for k, v in r.items() if k != "value"} for r in responses]
    (ROOT/"location_sources.json").write_text(json.dumps({"source_hashes": {str(p): sha256_file(p)
        for p in (audit_path, nodes_path, vaa_path, Path(__file__), ROOT/"study_plan.md")},
        "requests": receipt, "historical_graph_modified": False}, indent=2)+"\n")
    print(frame[["station", "original_area_ratio", "review_status"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
