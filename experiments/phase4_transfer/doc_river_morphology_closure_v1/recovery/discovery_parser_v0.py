"""Check official station and continuous-series catalogues beyond mapped gauges."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from shapely.geometry import Point, shape

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_morphology_closure_v1")
RAW = Path("data/raw/river_morphology_closure_v1/discovery")


def get_record(url, name):
    path = RAW / name
    record = {"url": url, "attempted_at": datetime.now(timezone.utc).isoformat()}
    if path.exists():
        return {**record, "status": "saved_download", "path": str(path), "sha256": sha256_file(path)}
    try:
        response = requests.get(url, timeout=(8, 25))
        record.update(http_status=response.status_code, bytes=len(response.content))
        if response.ok:
            path.write_bytes(response.content)
            record.update(status="retrieved", path=str(path), sha256=sha256_file(path))
        else:
            record.update(status="provider_unavailable", error=response.text[:150])
    except requests.RequestException as exc:
        record.update(status="access_error", error=str(exc)[:300])
    return record


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    out = ROOT / "recovery/catalog_discovery.json"
    if out.exists():
        print("Preserve recorded catalogue discovery")
        return
    receivers = pd.read_csv(ROOT / "recovery/receivers.csv", dtype={"station": str, "huc4": str})
    regions = sorted(receivers.huc4.unique())
    requests_fixed = [("https://www.waterqualitydata.us/wqx3/Station/search?" + urlencode({
        "huc": huc, "characteristicName": "Organic carbon", "mimeType": "csv"}), f"stations_{huc}.csv") for huc in regions]
    requests_fixed.append(("https://api.waterdata.usgs.gov/ogcapi/v0/collections?f=json", "usgs_collections.json"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda r: get_record(*r), requests_fixed))
    candidates, schemas = [], []
    known = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str})
    known_ids = {"USGS-"+s for s in known.site_no}
    for huc, record in zip(regions, records[:len(regions)], strict=True):
        if "path" not in record:
            continue
        f = pd.read_csv(record["path"], dtype=str, low_memory=False)
        schemas.append({"huc4": huc, "columns": list(f.columns), "rows": len(f)})
        key = "Location_Identifier"
        lat, lon = "Location_LatitudeMeasure", "Location_LongitudeMeasure"
        if not {key, lat, lon}.issubset(f):
            continue
        f[lat], f[lon] = pd.to_numeric(f[lat], errors="coerce"), pd.to_numeric(f[lon], errors="coerce")
        typ = "Location_Type"
        if typ in f:
            f = f[f[typ].str.contains("stream|river", case=False, na=False)]
        f = f.dropna(subset=[key, lat, lon]).drop_duplicates(key)
        for receiver in receivers[receivers.huc4.eq(huc)].itertuples():
            basin_path = Path(f"data/raw/river_planform_v1/basins/comid_{int(receiver.comid)}.json")
            if not basin_path.exists():
                continue
            geo = json.loads(basin_path.read_text())
            polygon = shape(geo["features"][0]["geometry"] if "features" in geo else geo["geometry"])
            for station in f.to_dict("records"):
                if station[key] in known_ids:
                    continue
                if polygon.covers(Point(station[lon], station[lat])):
                    candidates.append({"target": receiver.station, "huc4": huc, "station_id": station[key],
                        "latitude": station[lat], "longitude": station[lon], "station_type": station.get(typ),
                        "station_name": station.get("Location_Name"), "source": record["path"]})
    frame = pd.DataFrame(candidates, columns=["target", "huc4", "station_id", "latitude", "longitude", "station_type", "station_name", "source"])
    frame.to_csv(ROOT / "recovery/additional_in_basin_stations.csv", index=False)
    api = records[-1]
    if "path" in api:
        catalogue = json.loads(Path(api["path"]).read_text())
        api["collection_ids"] = [c["id"] for c in catalogue.get("collections", [])]
    payload = {"records": records, "station_catalog_schemas": schemas,
        "additional_in_basin_station_rows": len(frame), "additional_unique_stations": frame.station_id.nunique(),
        "interpretation": "Station locations inside saved basin polygons are leads; hydrological routing and co-sampling remain to be established."}
    out.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
