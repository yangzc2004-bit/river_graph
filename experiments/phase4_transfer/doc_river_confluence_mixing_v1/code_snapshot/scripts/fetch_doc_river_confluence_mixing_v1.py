"""Inspect public landing metadata for the real-confluence mixing archive."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/river_confluence_mixing_v1"
OUT = ROOT / "experiments/phase4_transfer/doc_river_confluence_mixing_v1"
SOURCES = {
    "datacite": "https://api.datacite.org/dois/10.13012/B2IDB-5324086_V1",
    "landing": "https://databank.illinois.edu/datasets/IDB-5324086",
    "public_library_landing": "https://aws-databank-alb.library.illinois.edu/datasets/IDB-5324086",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        path = RAW / f"{name}.json" if name == "datacite" else RAW / f"{name}.html"
        if path.exists():
            records.append({"source": name, "url": url, "path": str(path.relative_to(ROOT)),
                            "status": "cached"})
            print(f"{name}: cached", flush=True)
            continue
        try:
            response = requests.get(url, timeout=18)
            status = response.status_code
            record = {"source": name, "url": url, "http_status": status,
                      "resolved_url": response.url}
            if response.ok:
                path.write_bytes(response.content)
                record.update(status="retrieved", path=str(path.relative_to(ROOT)))
            else:
                record.update(status="unavailable", response_excerpt=response.text[:300])
        except requests.RequestException as error:
            record = {"source": name, "url": url, "status": "unavailable",
                      "error": str(error)}
        records.append(record)
        print(f"{name}: {record['status']}", flush=True)
    (OUT / "access_records.json").write_text(json.dumps({
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
