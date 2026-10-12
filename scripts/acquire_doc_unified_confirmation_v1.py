"""Bounded station-specific temperature acquisition for a fresh external case."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import build_doc_external_inputs_v1 as builder
import pandas as pd

from river_graph.data.wqp import _payload_looks_complete, _query_url
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_unified_generalization_v1/external_build")
CODE = "02030103"


def temperature_by_station(code, raw, first_year, last_year):
    sites = pd.read_csv(ROOT/"external_availability"/code/"stations.csv", dtype={"site_no": str}).site_no.tolist()
    destination = raw/"temperature_by_site"
    destination.mkdir(exist_ok=True)
    # A real no-content response is represented by the canonical CSV header;
    # transport failures remain errors rather than fabricated missing values.
    template = next(Path("data/raw/doc_external_inputs_v1/02040104").glob("temperature_*.csv"))
    header = template.read_bytes().split(b"\n", 1)[0]+b"\n"
    def fetch(site):
        path = destination/f"{site}.csv"
        if path.exists() and _payload_looks_complete(path.read_bytes(), None):
            return path
        url = _query_url(site, f"01-01-{first_year}", f"12-31-{last_year}", ["Temperature, water"])
        last_error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "river-graph-research", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    status = response.status
                    payload = response.read(30_000_001)
                    declared = response.headers.get("Content-Length")
                if status == 204:
                    payload = header
                if len(payload) > 30_000_000 or not _payload_looks_complete(payload, declared if status != 204 else None):
                    raise ValueError("station temperature response incomplete or exceeds bound")
                path.write_bytes(payload)
                path.with_suffix(".request.json").write_text(json.dumps({"url": url, "status": status,
                    "sha256": sha256_file(path), "attempt": attempt+1}, indent=2)+"\n")
                return path
            except (OSError, ValueError, urllib.error.HTTPError) as error:
                last_error = error
        raise RuntimeError(f"temperature unavailable due to network failure at {site}: {last_error}")
    result = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path in pool.map(fetch, sites):
            result.append(path)
            print(json.dumps({"stage": "station_temperature", "completed": len(result), "stations": len(sites)}), flush=True)
    return result


if __name__ == "__main__":
    builder.fetch_temperature = temperature_by_station
    builder.build(CODE, ROOT)
