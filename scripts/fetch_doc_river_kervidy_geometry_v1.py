"""Acquire the catalog-linked Kervidy network, catchment and flow companion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd
import requests

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1")
RAW = Path("data/raw/river_kervidy_geometry_v1")
SERVICE = "https://sensorthings.umrsas.inrae.fr/agrhys/v1.1/"
OBS = SERVICE + "Datastreams(2)/Observations"
START, END = "2020-10-14T00:00:00Z", "2023-09-06T23:59:59Z"


def get_json(session, url, path):
    if path.exists():
        return json.loads(path.read_text())
    response = session.get(url, timeout=(15, 60))
    response.raise_for_status()
    data = response.json()
    path.write_bytes(response.content)
    return data


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    pages = RAW / "flow_months"
    pages.mkdir(exist_ok=True)
    manifest_path = ROOT / "retrieval_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Retrieval is recorded; reuse the existing objects")
    session = requests.Session()
    records = []
    for name, url in (
        ("flow_stream", SERVICE + "Datastreams(2)"),
        ("outlet_locations", SERVICE + "Things(1)/Locations"),
        ("flow_sensor", SERVICE + "Datastreams(2)/Sensor"),
    ):
        path = RAW / f"{name}.json"
        data = get_json(session, url, path)
        records.append({"name": name, "url": url, "path": str(path), "sha256": sha256_file(path)})
        if name == "flow_stream" and data["unitOfMeasurement"]["symbol"] != "dm³/s":
            raise ValueError("Unexpected discharge unit; inspect before conversion")
    method_url = "https://sensorthings.umrsas.inrae.fr/public/sensors/metadonneeshQ20250326.pdf"
    method_path = RAW / "flow_method.pdf"
    if not method_path.exists():
        response = session.get(method_url, timeout=(15, 60))
        response.raise_for_status()
        method_path.write_bytes(response.content)
    if not method_path.read_bytes().startswith(b"%PDF"):
        raise ValueError("Expected the catalog-linked method PDF")
    records.append({"name": "flow_method", "url": method_url, "path": str(method_path),
                    "sha256": sha256_file(method_path)})
    for name, layer in (("river_network", "ore:nzn_rh"),
                        ("catchment_boundary", "ore:kervidy_bv_mnt20geob")):
        url = "https://geosas.fr/geoserver/wfs?" + urlencode({
            "service": "WFS", "request": "GetFeature", "version": "2.0.0",
            "typeName": layer, "outputFormat": "SHAPE-ZIP"}, quote_via=quote)
        path = RAW / f"{name}.zip"
        if not path.exists():
            response = session.get(url, timeout=(15, 60))
            response.raise_for_status()
            path.write_bytes(response.content)
        if not path.read_bytes().startswith(b"PK"):
            raise ValueError("Expected the public shapefile ZIP, not a service error")
        records.append({"name": name, "layer": layer, "url": url, "path": str(path),
                        "sha256": sha256_file(path), "bytes": path.stat().st_size})
    # The service's parser requires %20 spaces and supports minute equality,
    # while modulo expressions are not implemented. Do not rely on @iot.count.
    # A three-year request returned 10,000 rows and an empty next page despite
    # records existing later. Use non-overlapping calendar months, whose maximum
    # possible quarter-hour count is below the page size. Keep that attempt raw.
    boundaries = [pd.Timestamp(START)] + list(pd.date_range(
        "2020-11-01", "2023-09-01", freq="MS", tz="UTC")) + [pd.Timestamp(END)+pd.Timedelta(seconds=1)]
    rows, page_records = [], []
    for start, end in pairwise(boundaries):
        lo, hi = start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")
        condition = (f"phenomenonTime ge {lo} and phenomenonTime lt {hi} and "
            "(minute(phenomenonTime) eq 0 or minute(phenomenonTime) eq 15 or "
            "minute(phenomenonTime) eq 30 or minute(phenomenonTime) eq 45)")
        url = OBS + "?" + urlencode({"$filter": condition, "$top": 10000,
            "$orderby": "phenomenonTime asc", "$select": "phenomenonTime,result,resultQuality"},
            quote_via=quote, safe="$(),:")
        path = pages / f"month_{start:%Y_%m}.json"
        data = get_json(session, url, path)
        values = data["value"]
        if values:
            t = pd.to_datetime([v["phenomenonTime"] for v in values], utc=True)
            if not ((t >= start) & (t < end)).all():
                raise ValueError("Public time filter was not applied")
            if not ((t.minute.isin([0, 15, 30, 45])) & (t.second == 0)).all():
                raise ValueError("Public quarter-hour filter was not applied")
            if not t.is_monotonic_increasing:
                raise ValueError("Flow page is not chronologically ordered")
        rows.extend(values)
        expected = len(pd.date_range(start, end, freq="15min", inclusive="left"))
        if len(values) > expected or len(values) == 10000:
            raise ValueError("Month is not bounded to one observation per quarter hour")
        if len(rows) > 110000:
            raise ValueError("Unexpected number of quarter-hour records in bounded interval")
        page_records.append({"month": f"{start:%Y-%m}", "url": url, "path": str(path),
                             "n_rows": len(values), "expected_quarter_hours": expected,
                             "sha256": sha256_file(path)})
        print(f"Flow month {start:%Y-%m}: {len(values)} records; cumulative {len(rows)}", flush=True)
    frame = pd.DataFrame({"timestamp_utc": [r["phenomenonTime"] for r in rows],
                          "q_dm3_s": [r["result"] for r in rows],
                          "result_quality_json": [json.dumps(r.get("resultQuality"), sort_keys=True)
                                                  for r in rows]})
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    if frame.timestamp_utc.duplicated().any() or not frame.timestamp_utc.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered flow timestamps across pages")
    flow_path = RAW / "discharge_quarter_hour.parquet"
    frame.to_parquet(flow_path, index=False)
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "objects": records,
        "flow_months": page_records, "n_flow_records": len(frame),
        "flow_path": str(flow_path), "flow_sha256": sha256_file(flow_path),
        "time_scope_utc": [START, END], "flow_unit": "dm³/s", "m3_s_factor": .001,
        "source_grid": "public minute-grid rating-curve stream, requested at exact quarter hours",
        "count_semantics": "service count is not used for total or filter validation",
        "retrieval_method": "non-overlapping UTC months with no pagination needed; each scope has fewer than 3,000 quarter hours",
        "map_credit": "Source : UMR 1069 SAS INRA - Agrocampus Ouest",
        "flow_credit": "Fovet et al. (2018), doi:10.2136/vzj2018.04.0066",
        "discovery": ["https://catalogue.theia.data-terra.org/meta/TheiaOZCAR.AGRH_DAT_ore_AgrHys_Naizin",
                      "https://geosas.fr/geonetwork/srv/api/records/f5ceee60-c8db-4a96-a4f4-a7c922b43c8f",
                      "https://geosas.fr/geonetwork/srv/api/records/82abb5f4-0ddc-4406-90ec-57b3fb36daf2",
                      "https://geosas.fr/geoserver/wfs?service=WFS&request=GetCapabilities&version=2.0.0"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("Kervidy geometry and flow retrieval recorded", flush=True)


if __name__ == "__main__":
    main()
