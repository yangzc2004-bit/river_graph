"""Resolve public registered NEON routine-water sampling locations to NHD reaches."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from pyproj import Geod

ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")
RAW = Path("data/raw/river_neon_form_validation_v1")
INVENTORY = Path("experiments/phase4_transfer/doc_river_wholeform_evidence_v1/analysis/neon_site_inventory.csv")
API = "https://data.neonscience.org/api/v0/locations/"
NLDI = "https://api.water.usgs.gov/nldi/linked-data/comid/position"


def fetch_json(url, path, *, params=None):
    """Public endpoints only; stop immediately for authorization failures."""
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=(15, 60))
        if response.status_code in (401, 403):
            response.raise_for_status()
        if response.status_code in (500, 502, 503, 504) and attempt < 2:
            time.sleep(1 + attempt)
            continue
        response.raise_for_status()
        value = response.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
        return value
    raise RuntimeError("public endpoint retries exhausted")


def location(name):
    return fetch_json(API + name, RAW / "locations" / f"{name}.json")["data"]


def registered_points(site):
    root = location(site)
    water = [c for c in root.get("locationChildren", []) if c.startswith(("STREAM", "RIVER"))]
    points = []
    for parent in water:
        body = location(parent)
        candidates = [c for c in body.get("locationChildren", []) if c.startswith(("S2LOC", "BUOY"))]
        for child in candidates:
            sensor = location(child)
            points.append((sensor, "physical_S2_or_buoy"))
            for named in sensor.get("locationChildren", []):
                if named.endswith((".AOS.S2", ".AOS.buoy.c0")):
                    points.append((location(named), "registered_routine_named_location"))
    if not points:
        raise ValueError("no public S2/buoy location beneath stream/river site")
    return points


def resolve_site(site):
    rows = []
    try:
        points = registered_points(site)
    except (requests.RequestException, ValueError, KeyError) as error:
        return [{"site": site, "status": "location_metadata_error", "error": str(error)}]
    for point, kind in points:
        name = point["locationName"]
        lat, lon = point.get("locationDecimalLatitude"), point.get("locationDecimalLongitude")
        row = {"site": site, "location": name, "location_kind": kind,
               "latitude": lat, "longitude": lon, "metadata_url": API + name,
               "active_periods": json.dumps(point.get("activePeriods", [])),
               "location_description": point.get("locationDescription"),
               "coordinate_scope": "registered_location_not_individual_sample_coordinates"}
        try:
            if lat is None or lon is None:
                raise ValueError("location has no geographic coordinates")
            value = fetch_json(NLDI, RAW / "positions" / f"{name}.json",
                               params={"f": "json", "coords": f"POINT({lon} {lat})"})
            features = value.get("features", [])
            if len(features) != 1:
                raise ValueError(f"position returns {len(features)} reaches")
            f = features[0]
            p = f["properties"]
            row["comid"] = int(p.get("comid", p.get("identifier")))
            coords = f["geometry"]["coordinates"]
            if f["geometry"]["type"] == "Point":
                _, _, distance = Geod(ellps="WGS84").inv(lon, lat, coords[0], coords[1])
                row["nldi_representative_point_distance_m"] = distance
            row["status"] = "resolved_catchment"
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            row.update(status="nhd_position_error", error=str(error))
        rows.append(row)
    return rows


def main():
    inventory = pd.read_csv(INVENTORY)
    sites = sorted(inventory.loc[inventory.waterbody_scope.eq("stream_or_river"), "site"])
    rows = []
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        for site, values in zip(sites, pool.map(resolve_site, sites), strict=True):
            rows.extend(values)
            pd.DataFrame(rows).to_csv(ROOT / "analysis" / "registered_locations.csv", index=False)
            print(site, [(r.get("location"), r["status"], r.get("comid")) for r in values], flush=True)
    print(pd.DataFrame(rows).groupby("status").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
