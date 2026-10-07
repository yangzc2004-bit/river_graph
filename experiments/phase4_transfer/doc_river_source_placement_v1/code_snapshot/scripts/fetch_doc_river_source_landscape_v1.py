"""Acquire per-reach catchment and riparian landscapes from EPA StreamCat."""
from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_source_placement_v1")
CACHE = Path("data/raw/river_source_placement_v1")
API = "https://api.epa.gov/StreamCat/streams/metrics"
REGIONS = ("05", "06", "07", "08", "10L", "10U", "11")
METRICS = "pctwdwet2019,pcthbwet2019,pctconif2019,pctdecid2019,pctmxfst2019"


def acquire(job):
    region, aoi = job
    path = CACHE/f"region{region}_{aoi}.json"
    params = {"name": METRICS, "aoi": aoi, "region": f"Region{region}"}
    if aoi == "cat":
        params.update(showareasqkm="true", showpctfull="true")
    # These exact successful probes can be resumed without another large request.
    probe = CACHE/f"probe_region05{'cat' if aoi == 'cat' else 'rpplain'}.json"
    if region == "05" and not path.exists() and probe.exists():
        value = json.loads(probe.read_text())
        if value.get("items") and "comid" in value["items"][0]:
            shutil.copyfile(probe, path)
    failures = []
    for attempt in range(2):
        try:
            if path.exists():
                value = json.loads(path.read_text())
            else:
                response = requests.post(API, data=params, timeout=(20, 240))
                response.raise_for_status()
                value = response.json()
            items = value.get("items", [])
            if not items or "comid" not in items[0]:
                raise ValueError("StreamCat returned no COMID metrics")
            frame = pd.DataFrame(items)
            expected = [f"{metric}{aoi}" for metric in METRICS.split(",")]
            if set(expected)-set(frame) or frame.comid.duplicated().any():
                raise ValueError("incomplete columns or duplicate regional COMID")
            if not path.exists():
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(value, separators=(",", ":")))
                temp.replace(path)
            print(f"{region}/{aoi}: {len(frame):,} reaches", flush=True)
            return frame, {"region": region, "aoi": aoi, "status": "ok", "url": API,
                "request": params, "file": str(path), "sha256": sha256_file(path), "rows": len(frame)}
        except (requests.RequestException, ValueError) as error:
            failures.append(str(error))
            if attempt == 0:
                time.sleep(3)
    print(f"{region}/{aoi}: acquisition failed", flush=True)
    return None, {"region": region, "aoi": aoi, "status": "error", "url": API,
                  "request": params, "errors": failures}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    membership_paths = sorted(Path("data/raw/river_planform_v1/members_full").glob("*.npz"))
    union = set()
    for path in membership_paths:
        with np.load(path) as z:
            union.update(z["comids"].tolist())
    jobs = [(region, aoi) for region in REGIONS for aoi in ("cat", "catrp100")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, jobs))
    frames = {}
    for aoi in ("cat", "catrp100"):
        parts = [f for f, receipt in results if f is not None and receipt["aoi"] == aoi]
        if not parts:
            raise RuntimeError(f"no {aoi} landscapes available")
        combined = pd.concat(parts, ignore_index=True)
        duplicated = combined[combined.comid.duplicated(False)]
        for _, group in duplicated.groupby("comid"):
            if any(group[c].dropna().nunique() > 1 for c in group if c != "comid"):
                raise ValueError("regional duplicate COMID has conflicting values")
        frames[aoi] = combined.drop_duplicates("comid")
        frames[aoi] = frames[aoi][frames[aoi].comid.isin(union)]
    landscape = frames["cat"].merge(frames["catrp100"], on="comid", how="outer", validate="one_to_one")
    landscape = landscape.sort_values("comid").reset_index(drop=True)
    landscape.to_parquet(CACHE/"reach_landscape.parquet", index=False)
    (ROOT/"landscape_sources.json").write_text(json.dumps({"provider": "US EPA StreamCat",
        "land_cover_year": 2019, "requests": [r for _, r in results],
        "upstream_reaches_requested": len(union), "reaches_acquired": len(landscape),
        "riparian_pctfull": "API fails for catrp100 + showpctfull; not inferred from catchment completeness",
        "membership_hashes": {str(p): sha256_file(p) for p in membership_paths},
        "landscape_file": str(CACHE/"reach_landscape.parquet"),
        "landscape_sha256": sha256_file(CACHE/"reach_landscape.parquet")}, indent=2)+"\n")
    print(f"Saved {len(landscape):,}/{len(union):,} upstream reach landscapes", flush=True)


if __name__ == "__main__":
    main()
