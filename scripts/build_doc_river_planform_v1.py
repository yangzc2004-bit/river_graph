"""Acquire actual upstream channels/basins and measure ST357 whole-network form."""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely import to_wkb
from shapely.geometry import shape
from shapely.ops import unary_union

from river_graph.topology.river_planform import (
    COLUMNS,
    UpstreamNetwork,
    line_from_geojson,
    network_features,
    read_cached_lines,
)

ROOT = Path("experiments/phase4_transfer/doc_river_planform_typology_v1")
CACHE = Path("data/raw/river_planform_v1")
API = "https://api.water.usgs.gov/nldi/linked-data/comid"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch(url, path):
    if path.exists():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            return json.load(stream)
    failures = []
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 180))
            response.raise_for_status()
            value = response.json()
            if not value.get("features"):
                raise ValueError("empty FeatureCollection")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(temp, "wt") as stream:
                json.dump(value, stream, separators=(",", ":"))
            temp.replace(path)
            return value
        except (requests.RequestException, ValueError) as error:
            failures.append(str(error))
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError("; ".join(failures))


def ingest(connection, features):
    rows = []
    for feature in features:
        cid = int(feature["properties"]["nhdplus_comid"])
        rows.append((cid, to_wkb(line_from_geojson(feature))))
    connection.executemany("INSERT OR IGNORE INTO lines VALUES (?, ?)", rows)
    connection.commit()


def task(row, network, available):
    cid = int(row.comid)
    indices, main = network.membership(cid)
    comids = network.comid[indices]
    membership_path = CACHE / "members_full" / f"comid_{cid}.npz"
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(membership_path, comids=comids, mainstem=network.comid[main])
    flow_url = f"{API}/{cid}/navigation/UT/flowlines?f=json&distance=9999"
    basin_url = f"{API}/{cid}/basin?f=json&simplified=false&splitCatchment=false"
    result = {"station": str(row.site_no), "comid": cid, "expected_reaches": len(comids),
              "connectivity": "primary_and_secondary_downstream_links",
              "unit": "receiving_reach_outlet", "flowline_url": flow_url, "basin_url": basin_url}
    features = []
    try:
        if any(int(v) not in available for v in comids):
            geojson = fetch(flow_url, CACHE / "downloads" / f"ut_comid_{cid}.json.gz")
            features = geojson["features"]
            returned = {int(f["properties"]["nhdplus_comid"]) for f in features}
            result.update(returned_reaches=len(returned), returned_extra_reaches=len(returned-set(map(int, comids))))
        basin = fetch(basin_url, CACHE / "basins" / f"comid_{cid}.json")
        result["status"] = "downloaded"
        return result, indices, main, features, basin
    except Exception as error:  # noqa: BLE001 -- retain station-specific acquisition failures
        result.update(status="acquisition_error", error=str(error))
        return result, indices, main, features, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prefetch-largest", action="store_true", help="Acquire largest remaining basin alone, then reuse its channels for nested basins")
    parser.add_argument("--retry-errors", action="store_true", help="Retry recorded acquisition/measurement failures")
    args = parser.parse_args()
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CACHE / "flowlines.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS lines (comid INTEGER PRIMARY KEY, wkb BLOB NOT NULL)")
    if connection.execute("SELECT COUNT(*) FROM lines").fetchone()[0] == 0:
        for path in sorted(Path("data/raw/nldi/flowlines").glob("*.json")):
            value = json.loads(path.read_text())
            if isinstance(value, dict) and value.get("features"):
                ingest(connection, value["features"])
    available = {cid for (cid,) in connection.execute("SELECT comid FROM lines")}
    network = UpstreamNetwork(pd.read_parquet("cache/nldplus_vaa.parquet", columns=list(COLUMNS)))
    nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str, "huc_cd": str})
    nodes["upstream_area"] = [network.frame.iloc[network.index(v)].totdasqkm for v in nodes.comid]
    nodes = nodes.sort_values(["upstream_area", "site_no"])
    records_path = ROOT / "analysis" / "acquisition_records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    feature_path = ROOT / "analysis" / "station_planform.csv"
    values = pd.read_csv(feature_path, dtype={"station": str}).to_dict("records") if feature_path.exists() else []
    completed = {r["station"] for r in values}
    pending = [row for row in nodes.itertuples() if str(row.site_no) not in completed
               and (args.retry_errors or str(row.site_no) not in records)]
    if args.limit:
        pending = pending[:args.limit]
    batches = []
    if args.prefetch_largest and pending:
        batches.append([pending.pop()])
    batches.extend(pending[start:start+args.workers] for start in range(0, len(pending), args.workers))
    started = time.time()
    print(f"ST357: {len(completed)} measured; {len(available)} cached reaches; {len(pending)} pending", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Two requests in flight, then update the shared geometry cache before
        # creating the next batch; nested basins reuse previous downloads.
        for batch in batches:
            futures = [executor.submit(task, row, network, available.copy()) for row in batch]
            for future in futures:
                record, indices, mainstem, features, basin_json = future.result()
                if features:
                    ingest(connection, features)
                    available.update(int(f["properties"]["nhdplus_comid"]) for f in features)
                if basin_json is not None:
                    comids = network.comid[indices]
                    lines = read_cached_lines(connection, comids)
                    record.update(geometry_reaches=len(lines), geometry_coverage=len(lines)/len(comids))
                    if len(lines) != len(comids):
                        record["status"] = "incomplete_geometry"
                        record["missing_comids"] = [int(c) for c in comids if int(c) not in lines][:50]
                    else:
                        try:
                            basin = unary_union([shape(f["geometry"]) for f in basin_json["features"]])
                            value = network_features(network, indices, mainstem, lines, basin)
                            value.update(station=record["station"], comid=record["comid"],
                                         network_definition="primary_and_secondary_downstream_links",
                                         huc_cd=str(nodes.loc[nodes.site_no.eq(record["station"]), "huc_cd"].iloc[0]),
                                         vaa_area_km2=float(network.frame.iloc[network.index(record["comid"])].totdasqkm))
                            values.append(value)
                            record["status"] = "complete"
                        except Exception as error:  # noqa: BLE001 -- retain malformed geometry diagnostics
                            record.update(status="measurement_error", error=str(error))
                records[record["station"]] = record
                save_json(records_path, records)
                temporary = feature_path.with_suffix(".csv.tmp")
                pd.DataFrame(values).to_csv(temporary, index=False)
                temporary.replace(feature_path)
                counts = pd.Series([r["status"] for r in records.values()]).value_counts().to_dict()
                save_json(ROOT / "progress.json", {"n_stations": len(nodes), "counts": counts,
                    "cached_reaches": len(available), "last_station": record["station"],
                    "elapsed_seconds_this_session": round(time.time()-started, 1),
                    "full_cohort_complete": counts.get("complete", 0) == len(nodes)})
                print(f"{record['station']} {record['comid']} {record['status']} expected={record['expected_reaches']} measured={len(values)}/357", flush=True)
    connection.close()


if __name__ == "__main__":
    main()
