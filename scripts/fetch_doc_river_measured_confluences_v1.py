"""Retrieve public confluence observations and methods without executing source R."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase4_transfer/doc_river_measured_confluences_v1"
RAW = ROOT / "data/raw/river_measured_confluences_v1"
RESOURCE = "c5e687fa040e4707ba922002bafd18fd"
NAMES = {
    "Plontetal_WRR_database.csv",
    "Plontetal_WRR_database_metadata.csv",
    "Plontetal_WRR_Fall2021_sumdata.csv",
    "Plontetal_WRR_Fall2021_widthdepth_notrib.csv",
    "Plontetal_WRR_BGCNonConservMixConfluence.R",
    "Plontetal_WRR_BGCNonConservativeMixingConfluences_SupplementaryMaterials.pdf",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://www.hydroshare.org/hsapi/resource/{RESOURCE}/files/"
    objects, inventories = [], []
    while url:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        page = response.json()
        inventories.append(page)
        for item in page["results"]:
            if item["file_name"] not in NAMES:
                continue
            path = RAW / item["file_name"]
            href = item["url"].replace("http://", "https://", 1)
            if not path.exists():
                download = requests.get(href, timeout=45)
                download.raise_for_status()
                path.write_bytes(download.content)
            data = path.read_bytes()
            if len(data) != item["size"]:
                raise ValueError(f"Provider size mismatch: {path.name}")
            if hashlib.md5(data).hexdigest() != item["checksum"]:
                raise ValueError(f"Provider checksum mismatch: {path.name}")
            objects.append({"path": str(path.relative_to(ROOT)), "url": href,
                            "bytes": len(data), "sha256": sha256_file(path),
                            "licence": "CC BY 4.0"})
            print(f"Saved {path.name} ({len(data)} bytes)", flush=True)
        url = page["next"]
    if {Path(obj["path"]).name for obj in objects} != NAMES:
        raise ValueError("Selected public objects missing from complete inventory")
    (RAW / "inventory.json").write_text(json.dumps(inventories, indent=2) + "\n")
    (OUT / "retrieval_manifest.json").write_text(json.dumps({
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": f"https://www.hydroshare.org/resource/{RESOURCE}/",
        "citation": "Plont (2022), Plont_WRR_BGCNonconservativemixingConfluences_Data",
        "selection": "All five confluences, both campaigns; study plan before DOC calculation",
        "inventory_count": sum(len(page["results"]) for page in inventories),
        "objects": objects,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
