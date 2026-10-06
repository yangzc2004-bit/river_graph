"""Fill large-network geometry gaps with bounded upstream navigation fragments."""
from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from build_doc_river_planform_v1 import API, CACHE, ROOT, fetch, ingest, save_json

from river_graph.topology.river_planform import COLUMNS, UpstreamNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--distance", type=int, default=100)
    args = parser.parse_args()
    records = json.loads((ROOT / "analysis/acquisition_records.json").read_text())
    target = np.unique(np.concatenate([np.load(CACHE / "members_full" / f"comid_{r['comid']}.npz")["comids"]
                    for r in records.values() if r["status"] != "complete"]))
    n = UpstreamNetwork(pd.read_parquet("cache/nldplus_vaa.parquet", columns=list(COLUMNS)))
    minor = n.frame.dnminorhyd.fillna(0).to_numpy(np.int64)
    connection = sqlite3.connect(CACHE / "flowlines.sqlite")
    used, events = set(), []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range((args.requests+1)//2):
            available = np.asarray([c for (c,) in connection.execute("SELECT comid FROM lines")], dtype=np.int64)
            missing = np.setdiff1d(target, available)
            if not len(missing):
                break
            present = np.intersect1d(target, available)
            mi = n.order[np.searchsorted(n.sorted_comid, missing)]
            pi = n.order[np.searchsorted(n.sorted_comid, present)]
            frontier = mi[np.isin(n.downstream[mi], n.hydro[pi]) | np.isin(minor[mi], n.hydro[pi])]
            frontier = frontier[np.argsort(n.cumulative[frontier])[::-1]]
            chosen = [int(i) for i in frontier if int(n.comid[i]) not in used][:min(2, args.requests-len(events))]
            if not chosen:
                break
            futures = []
            for i in chosen:
                cid = int(n.comid[i])
                used.add(cid)
                distance = args.distance if n.cumulative[i] > 20000 else 9999
                url = f"{API}/{cid}/navigation/UT/flowlines?f=json&distance={distance}"
                path = CACHE / "downloads" / f"fragment_ut_comid_{cid}_d{distance}.json.gz"
                futures.append((cid, distance, url, pool.submit(fetch, url, path)))
            for cid, distance, url, future in futures:
                event = {"comid": cid, "distance_km": distance, "url": url, "missing_before": len(missing)}
                try:
                    value = future.result()
                    ids = {int(f["properties"]["nhdplus_comid"]) for f in value["features"]}
                    event.update(status="ok", returned=len(ids), filled_missing=len(ids & set(map(int, missing))))
                    ingest(connection, value["features"])
                except (RuntimeError, ValueError, KeyError) as error:
                    event.update(status="error", error=str(error))
                events.append(event)
                save_json(ROOT / "fragment_acquisition.json", {"requests": events,
                    "rule": "missing upstream frontier; largest upstream channel support first; 100-km fragments for large candidates"})
                print(json.dumps(event), flush=True)
    connection.close()


if __name__ == "__main__":
    main()
