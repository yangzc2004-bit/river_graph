#!/usr/bin/env python3
"""Bounded WQP station-metadata probe. Never downloads observation values.

Station counts are screening evidence only: they cannot establish 10,000
QC station-months, graph connectivity, or a passed external data gate.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from river_graph.experiments.transfer import file_hash

REQUESTS = {"doc": {"pCode": "00681"}, "ph": {"characteristicName": "pH"},
            "spec_conductance": {"characteristicName": "Specific conductance"}}


def probe(analyte: str, params: dict, cache: Path) -> dict:
    params = {**params, "countrycode": "US", "siteType": "Stream", "providers": "NWIS",
              "mimeType": "csv"}
    url = "https://www.waterqualitydata.us/wqx3/Station/search?" + urllib.parse.urlencode(params)
    record = {"analyte": analyte, "url": url, "status": "pending",
              "requested_at": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "river-graph-research",
                                                       "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = response.read(20_000_001)
            record["http_status"] = response.status
            declared = response.headers.get("Content-Length")
        if len(payload) > 20_000_000:
            raise ValueError("metadata exceeds 20 MB probe limit")
        if declared and int(declared) != len(payload):
            raise ValueError("incomplete HTTP response")
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
        if not rows or len(rows[0]) < 3 or not payload.endswith(b"\n"):
            raise ValueError("not a complete station CSV")
        if any(len(r) != len(rows[0]) for r in rows[1:]):
            raise ValueError("malformed CSV rows")
        if not any("identifier" in c.lower() for c in rows[0]):
            raise ValueError("missing station-identifier header")
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"{analyte}_stations.csv"
        target.write_bytes(payload)
        record.update({"status": "metadata_received", "rows": len(rows) - 1,
                       "path": str(target), "sha256": file_hash(target), "columns": rows[0]})
    except Exception as exc:  # noqa: BLE001 - network failure is pending, never absence
        record.update({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase4_transfer/data_gate/external_probe.json")
    ap.add_argument("--cache", default="data/raw/transfer_external_inventory")
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(probe, a, p, Path(args.cache)) for a, p in REQUESTS.items()]
        results = [f.result() for f in futures]
    report = {
        "scope": "national stream-station metadata only; not target concentration results",
        "source": "https://www.waterqualitydata.us/beta/webservices_documentation/",
        "interpretation": "Unavailable requests mean unknown, not no eligible external basin. "
                          "No basin is selected and no training is authorized by this probe.",
        "requests": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("analyte", "status")} for r in results]))
    return 0 if all(r["status"] == "metadata_received" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
