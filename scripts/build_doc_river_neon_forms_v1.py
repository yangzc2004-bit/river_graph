"""Measure new NEON upstream networks without changing the ST357 geometry cache."""
from __future__ import annotations

import gzip
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely import to_wkb
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from river_graph.analysis.river_source_placement import UpstreamDistances
from river_graph.topology.river_planform import (
    COLUMNS,
    UpstreamNetwork,
    line_from_geojson,
    network_features,
    project_geometry,
    read_cached_lines,
)

ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")
RAW = Path("data/raw/river_neon_form_validation_v1")
OLD = Path("data/raw/river_planform_v1")
API = "https://api.water.usgs.gov/nldi/linked-data/comid"


def fetch(url, path):
    if path.exists():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            return json.load(stream)
    for attempt in range(3):
        response = requests.get(url, timeout=(15, 180))
        if response.status_code in (401, 403):
            response.raise_for_status()
        if response.status_code in (500, 502, 503, 504) and attempt < 2:
            time.sleep(attempt + 1)
            continue
        response.raise_for_status()
        value = response.json()
        if not value.get("features"):
            raise ValueError("empty upstream geometry")
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt") as stream:
            json.dump(value, stream)
        return value
    raise RuntimeError("public geometry retries exhausted")


def location_choice(group):
    ok = group[group.status.eq("resolved_catchment")]
    if ok.empty:
        raise ValueError("no resolved routine location")
    # A name is not evidence of an individual sample's coordinate. Require all
    # registered sensor alternatives to agree at the receiving-reach scale.
    if ok.comid.nunique() != 1 or len(ok) != len(group):
        raise ValueError("historical/current routine locations unresolved or span different reaches")
    named = ok[ok.location_kind.eq("registered_routine_named_location")]
    if named.empty:
        raise ValueError("routine named location unavailable")
    return named.iloc[0]


def main():
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    locations = pd.read_csv(out / "registered_locations.csv")
    frame = pd.read_parquet("cache/nldplus_vaa.parquet", columns=[*COLUMNS, "areasqkm"])
    network = UpstreamNetwork(frame)
    historical = sqlite3.connect(f"file:{OLD / 'flowlines.sqlite'}?mode=ro", uri=True)
    RAW.mkdir(parents=True, exist_ok=True)
    current = sqlite3.connect(RAW / "flowlines.sqlite")
    current.execute("CREATE TABLE IF NOT EXISTS lines (comid INTEGER PRIMARY KEY, wkb BLOB NOT NULL)")
    old_ids = {c for (c,) in historical.execute("SELECT comid FROM lines")}
    new_ids = {c for (c,) in current.execute("SELECT comid FROM lines")}
    records_path = out / "geometry_records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    features_path = out / "network_forms.csv"
    values = pd.read_csv(features_path).to_dict("records") if features_path.exists() else []
    completed = {v["site"] for v in values}
    st_nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str})
    st_union = set()
    for cid in st_nodes.comid.unique():
        st_union.update(network.comid[network.membership(cid)[0]].tolist())
    for site, group in locations.groupby("site", sort=True):
        if site in completed:
            continue
        record = {"site": site, "geometry_unit": "complete_upstream_network_at_receiving_reach_outlet"}
        try:
            selected = location_choice(group)
            cid = int(selected.comid)
            indices, main = network.membership(cid)
            comids = network.comid[indices]
            record.update(comid=cid, n_expected=len(comids), latitude=float(selected.latitude),
                          longitude=float(selected.longitude), registered_location=selected.location)
            p = RAW / "members" / f"{site}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, comids=comids, mainstem=network.comid[main])
            overlap = set(comids.tolist()) & st_union
            record.update(st357_upstream_overlap_reaches=len(overlap),
                          receiver_in_st357_upstream_union=cid in st_union)
            missing = set(comids.tolist()) - old_ids - new_ids
            if missing:
                geo = fetch(f"{API}/{cid}/navigation/UT/flowlines?f=json&distance=9999",
                            RAW / "flowlines" / f"{cid}.json.gz")
                rows = [(int(f["properties"]["nhdplus_comid"]), to_wkb(line_from_geojson(f)))
                        for f in geo["features"] if int(f["properties"]["nhdplus_comid"]) in missing]
                current.executemany("INSERT OR IGNORE INTO lines VALUES (?, ?)", rows)
                current.commit()
                new_ids.update(c for c, _ in rows)
            lines = read_cached_lines(historical, comids)
            lines.update(read_cached_lines(current, comids))
            record["geometry_coverage"] = len(lines) / len(comids)
            if len(lines) != len(comids):
                raise ValueError(f"incomplete upstream geometry {len(lines)}/{len(comids)}")
            basin_path = OLD / "basins" / f"comid_{cid}.json"
            if basin_path.exists():
                basin_json = json.loads(basin_path.read_text())
            else:
                basin_json = fetch(f"{API}/{cid}/basin?f=json&simplified=false&splitCatchment=false",
                                   RAW / "basins" / f"{cid}.json")
            basin = unary_union([shape(f["geometry"]) for f in basin_json["features"]])
            result = network_features(network, indices, main, lines, basin)
            row = result | {k: v for k, v in record.items() if k != "geometry_unit"}
            point = project_geometry(Point(selected.longitude, selected.latitude))
            distances = [point.distance(project_geometry(lines[cid]))]
            for loc in group.itertuples():
                distances.append(project_geometry(Point(loc.longitude, loc.latitude)).distance(project_geometry(lines[cid])))
            row["registered_point_max_channel_distance_m"] = max(distances)
            row["unique_catchment_area_km2"] = float(frame.iloc[indices].areasqkm.sum())
            row["basin_to_catchment_area_ratio"] = result["basin_area_km2"] / row["unique_catchment_area_km2"]
            if len(comids) == 1:
                # This mapped headwater has no graph edges; still inventory its
                # geometry, then exclude it under the declared five-reach rule.
                distance = np.asarray([frame.iloc[indices[0]].lengthkm/2])
            else:
                routing = UpstreamDistances(frame.iloc[indices].reset_index(drop=True))
                distance = routing.distances(cid, comids)
            area = frame.iloc[indices].areasqkm.to_numpy(float)
            valid = (area > 0) & np.isfinite(distance)
            mean = np.average(distance[valid], weights=area[valid])
            sd = np.sqrt(np.average((distance[valid] - mean)**2, weights=area[valid]))
            row.update(route_mean_km=mean, log_route_mean_scaled=np.log1p(mean/np.sqrt(row["basin_area_km2"])),
                       route_distance_cv=sd/mean if mean > 0 else 0.,
                       route_area_coverage=float(area[valid].sum()/area.sum()))
            row["form_eligibility"] = "eligible"
            if len(comids) < 5:
                row["form_eligibility"] = "small_network_fewer_than_five_reaches"
            elif max(distances) > 200:
                row["form_eligibility"] = "routine_location_more_than_200m_from_mapped_channel"
            elif not .8 <= row["basin_to_catchment_area_ratio"] <= 1.2:
                row["form_eligibility"] = "basin_area_inconsistent"
            elif row["route_area_coverage"] < .99:
                row["form_eligibility"] = "incomplete_channel_routing"
            values.append(row)
            record.update(status="measured", form_eligibility=row["form_eligibility"], n_reaches=len(comids))
        except (ValueError, KeyError, requests.RequestException) as error:
            record.update(status="unavailable", error=str(error))
        records[site] = record
        records_path.write_text(json.dumps(records, indent=2) + "\n")
        pd.DataFrame(values).to_csv(features_path, index=False)
        print(site, record.get("comid"), record["status"], record.get("form_eligibility", record.get("error")), flush=True)
    current.close()
    historical.close()


if __name__ == "__main__":
    main()
