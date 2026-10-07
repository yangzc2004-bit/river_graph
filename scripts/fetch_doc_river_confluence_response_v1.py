"""Retrieve mapped branch/receiver flow and an independent confluence DOC archive."""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fetch_doc_river_event_observations_v1 import get, metadata

from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase4_transfer/doc_river_confluence_response_v1"
RAW = ROOT / "data/raw/river_confluence_response_v1"
PARENT = ROOT / "data/raw/river_event_observations_v1"
RESOURCE = "c5e687fa040e4707ba922002bafd18fd"
INVENTORY = f"https://www.hydroshare.org/hsapi/resource/{RESOURCE}/files/"
SITES = {"C1", "C2", "C4", "C6", "C7", "C9", "C10", "C12", "C13", "C14", "C16"}
PLONT_FILES = {
    "Plontetal_WRR_database.csv",
    "Plontetal_WRR_database_metadata.csv",
    "Plontetal_WRR_Fall2021_sumdata.csv",
    "Plontetal_WRR_Fall2021_widthdepth_notrib.csv",
    "Plontetal_WRR_BGCNonConservMixConfluence.R",
    "Plontetal_WRR_BGCNonConservativeMixingConfluences_SupplementaryMaterials.pdf",
}


def fetch_flow(member: dict) -> dict:
    record = metadata(member["res"])
    path = RAW / "flow_30min" / record["fileName"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        oid = record["accessUrl"].rstrip("/").rsplit("/", 1)[-1]
        response = get("https://data.fieldsites.se/licence_accept", params={"ids": json.dumps([oid])})
        if "text/csv" not in response.headers.get("Content-Type", ""):
            raise ValueError(f"Not a CSV: {response.url}")
        path.write_bytes(response.content)
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Flow download size mismatch: {path.name}")
    metapath = path.with_suffix(".metadata.json")
    metapath.write_text(json.dumps(record, indent=2) + "\n")
    site = re.search(r"-(C\d+)_", path.name).group(1)
    print(f"Retrieved {site} 30-minute flow", flush=True)
    return {"kind": "flow_30min", "site": site, "url": record["latestVersion"],
            "pid": record["pid"], "path": str(path.relative_to(ROOT)),
            "metadata_path": str(metapath.relative_to(ROOT)), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "licence": "SITES CC BY 4.0"}


def fetch_plont(item: dict) -> dict:
    path = RAW / "plont" / item["file_name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    url = item["url"].replace("http://", "https://", 1)
    if not path.exists():
        path.write_bytes(get(url).content)
    if path.stat().st_size != item["size"] or hashlib.md5(path.read_bytes()).hexdigest() != item["checksum"]:
        raise ValueError(f"HydroShare content mismatch: {path.name}")
    print(f"Retrieved Plont {path.name}", flush=True)
    return {"kind": "confluence_archive", "url": url, "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "provider_md5": item["checksum"], "licence": "CC BY 4.0",
            "resource": f"https://www.hydroshare.org/resource/{RESOURCE}/"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "retrieval_manifest.json"
    if manifest_path.exists():
        for item in json.loads(manifest_path.read_text())["objects"]:
            if sha256_file(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Changed downloaded source: {item['path']}")
        print("Existing public downloads verified; nothing fetched")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    first = get(INVENTORY).json()
    pages = math.ceil(first["count"] / len(first["results"]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        rest = list(pool.map(lambda page: get(INVENTORY, params={"page": page}).json(), range(2, pages+1)))
    inventory = {"count": first["count"], "results": first["results"] + [r for page in rest for r in page["results"]]}
    (RAW / "hydroshare_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    selected = [item for item in inventory["results"] if item["file_name"] in PLONT_FILES]
    if len(selected) != len(PLONT_FILES):
        raise ValueError("Missing or ambiguous confluence archive file")
    members = json.loads((PARENT / "flow_collection.json").read_text())["members"]
    flow = [m for m in members if m["name"].startswith("SITES_WB-Q_")
            and m["name"].endswith("_30min.csv")
            and re.search(r"-(C\d+)_", m["name"]).group(1) in SITES]
    if len(flow) != len(SITES):
        raise ValueError("Expected eleven mapped flow sites")
    with ThreadPoolExecutor(max_workers=3) as pool:
        objects = list(pool.map(fetch_plont, selected))
        objects += list(pool.map(fetch_flow, flow))
    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "selection": "All eleven sites in four previously mapped confluences; all five Plont confluences and both seasons",
        "hydroshare_inventory": str((RAW / "hydroshare_inventory.json").relative_to(ROOT)),
        "hydroshare_inventory_sha256": sha256_file(RAW / "hydroshare_inventory.json"),
        "sites_collection_sha256": sha256_file(PARENT / "flow_collection.json"),
        "objects": sorted(objects, key=lambda x: x["path"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(objects)} source objects")


if __name__ == "__main__":
    main()
