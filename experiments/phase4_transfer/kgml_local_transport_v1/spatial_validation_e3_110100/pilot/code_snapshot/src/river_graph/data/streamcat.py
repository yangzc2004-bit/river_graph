"""EPA StreamCat API client: catchment/watershed metrics keyed by NHDPlusV2 COMID.

API: POST https://api.epa.gov/StreamCat/streams/metrics with form fields
name=<metrics>, aoi=ws|cat, comid=<comma list>. Responses are JSON; results
are cached per batch on disk so reruns resume.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

API_URL = "https://api.epa.gov/StreamCat/streams/metrics"
USER_AGENT = "river-graph-research"
BATCH = 50  # comids per request
SLEEP = 2.5  # API throttle: 30 req/min

# v1 ecological context metrics (design_m4.md M5); aoi=ws = full upstream
# watershed aggregates
METRICS = [
    "pctconif2019", "pctdecid2019", "pctmxfst2019",   # forest
    "pctcrop2019", "pcthay2019",                       # agriculture
    "pcturbhi2019", "pcturbmd2019", "pcturblo2019", "pcturbop2019",  # urban
    "pctwdwet2019", "pcthbwet2019",                    # wetland
    "precip9120", "tmean9120",                         # climate normals
    "om",                                              # soil organic matter
    "elev",                                            # elevation
    "bfi",                                             # baseflow index
]


def fetch_metrics(comids: list[str], cache_dir: str | Path) -> pd.DataFrame:
    """Fetch ws-level metrics for comids in batches (cached, resumable)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    batches = [comids[i:i + BATCH] for i in range(0, len(comids), BATCH)]
    for i, batch in enumerate(batches):
        out = cache_dir / f"sc_{i:04d}.json"
        if out.exists() and out.stat().st_size > 30:
            data = json.loads(out.read_text(encoding="utf-8"))
        else:
            body = f"name={','.join(METRICS)}&aoi=ws&comid={','.join(batch)}"
            req = urllib.request.Request(
                API_URL, data=body.encode(),
                headers={"User-Agent": USER_AGENT,
                         "Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    if data.get("items", [{}])[0].get("output") is None:
                        break
                    raise RuntimeError(f"unexpected payload: {str(data)[:120]}")
                except Exception as e:  # noqa: BLE001
                    print(f"  sc batch {i} attempt {attempt + 1} failed: {e}")
                    time.sleep(10 * (attempt + 1))
            else:
                print(f"  sc batch {i} FAILED, skipping (rerun to retry)")
                continue
            out.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(SLEEP)
        items = [it for it in data.get("items", []) if isinstance(it, dict) and it.get("comid")]
        frames.append(pd.DataFrame(items))
        if (i + 1) % 3 == 0:
            print(f"  streamcat {i + 1}/{len(batches)} batches")
    if not frames:
        raise RuntimeError("no StreamCat data fetched")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("comid")
    return df
