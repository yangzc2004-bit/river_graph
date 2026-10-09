"""Retrieve the separately published, public NEON DOC/absorbance archive."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW = Path("data/raw/river_wholeform_evidence_v1")
OUT = Path("experiments/phase4_transfer/doc_river_wholeform_evidence_v1")
RECORD_URL = "https://zenodo.org/api/records/20039155"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW / "zenodo_record_20039155.json"
    if not metadata_path.exists():
        response = requests.get(RECORD_URL, timeout=(15, 45))
        response.raise_for_status()
        metadata_path.write_text(json.dumps(response.json(), indent=2) + "\n")
    record = json.loads(metadata_path.read_text())
    files = []
    for item in record["files"]:
        name = item["key"]
        if name not in {"absNEON_output.csv", "absNEON_output_older_samples.csv"}:
            raise ValueError(f"Unexpected public archive object: {name}")
        path = RAW / name
        url = item["links"]["self"]
        attempts = []
        if not path.exists():
            # The public record download is the same CC-BY object. Use it only
            # for transient API/server failures, never to bypass a denied read.
            public_url = f"https://zenodo.org/records/20039155/files/{name}?download=1"
            content = None
            for candidate_url in (url, public_url):
                response = requests.get(candidate_url, timeout=(15, 45))
                attempts.append({"url": candidate_url, "http_status": response.status_code})
                if response.ok:
                    content = response.content
                    url = candidate_url
                    break
                if response.status_code not in {500, 502, 503, 504}:
                    response.raise_for_status()
            if content is None:
                raise RuntimeError(f"Public archive server unavailable: {attempts}")
        else:
            content = path.read_bytes()
        observed = "md5:" + hashlib.md5(content).hexdigest()
        if observed != item["checksum"] or len(content) != item["size"]:
            raise ValueError(f"Published file identity differs: {name}")
        if not path.exists():
            path.write_bytes(content)
        files.append({"name": name, "url": url, "size_bytes": len(content),
                      "provider_checksum": observed,
                      "sha256": hashlib.sha256(content).hexdigest(), "attempts": attempts})
        print(f"Retrieved {name}: {len(content)} bytes", flush=True)
    manifest = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "record_url": RECORD_URL, "record_doi": record.get("doi"),
                "metadata": record["metadata"], "files": files,
                "original_neon_api_accessed": False,
                "authority": "Public NEON-authored derived archive; original lab quality tables not supplied"}
    (OUT / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
