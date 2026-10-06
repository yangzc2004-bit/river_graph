"""Cache official gauge drainage areas independently of NHD station snapping."""
from __future__ import annotations

import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from river_graph.data.nwis import NWIS_SITE_URL, _get, read_rdb
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_mechanisms_v1")
CACHE = Path("data/raw/river_mechanisms_v1")


def fetch(batch):
    index, sites = batch
    p = CACHE/f"gauge_metadata_batch{index:02d}.rdb"
    url = NWIS_SITE_URL+"?"+urllib.parse.urlencode({"format": "rdb", "sites": ",".join(sites), "siteOutput": "expanded"})
    if not p.exists():
        payload = _get(url, timeout=45)
        if "drain_area_va" not in payload or "agency_cd\t" not in payload:
            raise ValueError("expanded site metadata response absent")
        p.write_text(payload)
    frame = read_rdb(p)
    if not set(sites).issubset(frame.site_no):
        raise ValueError(f"missing requested station in {p}")
    return frame, {"file": str(p), "url": url, "sha256": sha256_file(p), "n_stations": len(frame)}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    sites = sorted(pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str}).site_no)
    batches = [(i//75, sites[i:i+75]) for i in range(0, len(sites), 75)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, batches))
    frame = pd.concat([r[0] for r in results], ignore_index=True)
    if frame.site_no.duplicated().any():
        raise ValueError("duplicate official gauge")
    frame["drainage_area_km2"] = pd.to_numeric(frame.drain_area_va, errors="coerce")*2.589988110336
    out = ROOT/"metadata"
    out.mkdir(exist_ok=True)
    frame[["site_no", "station_nm", "huc_cd", "drain_area_va", "drainage_area_km2", "dec_lat_va", "dec_long_va"]].to_csv(out/"usgs_gauge_metadata.csv", index=False)
    (out/"sources.json").write_text(json.dumps({"provider": "USGS NWIS expanded Site Service", "raw_area_unit": "square miles",
        "conversion_to_km2": 2.589988110336, "files": [r[1] for r in results]}, indent=2)+"\n")
    print(f"Official metadata: {len(frame)} stations, {frame.drainage_area_km2.notna().sum()} reported drainage areas")


if __name__ == "__main__":
    main()
