"""Retrieve public paired salt/DOC observations and separately archived processing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase4_transfer/doc_river_tracer_process_v1"
RAW = ROOT / "data/raw/river_tracer_process_v1"
RESOURCE = "988f0d0aa46249b2b654145cf5fbf895"
LAB_FILES = {
    "13CAdditionData_CrestonGlucose_20190808_analyzed.csv",
    "13CAdditionData_CrestonGlucose_20190815_analyzed.csv",
    "13CAdditionData_CrestonLeachate_20190809_analyzed.csv",
    "site_data.csv",
}
AUTHOR_FILES = [
    "README.md", "data_process.R", "doc_functions.R", "site_data.csv",
    "data_doc.csv", "DOC_tracer_code.R", "TTfromNaCl/output_TTQ_Creston2019.csv",
    "plots.R",
]


def get(url: str) -> requests.Response:
    response = requests.get(url, timeout=50)
    response.raise_for_status()
    return response


def download(item: dict) -> dict:
    path = RAW / item["relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        content = get(item["url"]).content
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(content)
        temporary.replace(path)
    content = path.read_bytes()
    if "provider_md5" in item and (
        len(content) != item["bytes"] or hashlib.md5(content).hexdigest() != item["provider_md5"]
    ):
        raise ValueError(f"Public source integrity mismatch: {path.name}")
    result = {**item, "path": str(path.relative_to(ROOT)), "bytes": len(content),
              "sha256": sha256_file(path)}
    result.pop("relative_path")
    print(f"Retrieved {path.relative_to(RAW)}", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "retrieval_manifest.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text())["objects"]:
            path = ROOT / item["path"]
            if not path.exists():
                # A clean checkout retains the manifest but not gitignored raw
                # inputs. Restore its exact pinned URL, never today's main ref.
                download({**item, "relative_path": str(path.relative_to(RAW))})
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"Changed local public source: {item['path']}")
        print("Existing paired-tracer downloads verified")
        return
    inventory_url = f"https://www.hydroshare.org/hsapi/resource/{RESOURCE}/files/"
    inventory = get(inventory_url).json()
    entries = list(inventory["results"])
    while inventory.get("next"):
        inventory = get(inventory["next"]).json()
        entries.extend(inventory["results"])
    inventory_path = RAW / "hydroshare_inventory.json"
    inventory_path.write_text(json.dumps({"count": len(entries), "results": entries}, indent=2) + "\n")
    chosen = [item for item in entries if item["file_name"] in LAB_FILES]
    if len(chosen) != len(LAB_FILES):
        raise ValueError("Missing or duplicate primary paired-tracer source")
    # Public Git refs remain available when GitHub's unauthenticated API quota
    # is exhausted. Download explicitly inspected paths at this fixed commit.
    refs = subprocess.run(
        ["git", "ls-remote", "https://github.com/robohall/DOC_uptake.git", "refs/heads/main"],
        check=True, text=True, capture_output=True, timeout=50,
    ).stdout.splitlines()
    if len(refs) != 1:
        raise ValueError("Ambiguous public author branch")
    commit, ref = refs[0].split()
    if len(commit) != 40 or ref != "refs/heads/main":
        raise ValueError("Invalid public author commit")
    items = [
        {"kind": "original_analysed_observations", "relative_path": f"blaine/{item['file_name']}",
         "url": item["url"].replace("http://", "https://", 1), "bytes": item["size"],
         "provider_md5": item["checksum"], "license": "CC BY 4.0"}
        for item in chosen
    ]
    items += [
        {"kind": "author_processing_reference", "relative_path": f"author/{name}",
         "url": f"https://raw.githubusercontent.com/robohall/DOC_uptake/{commit}/{quote(name, safe='/')}",
         "author_commit": commit, "license": "public author code; no repository license file located"}
        for name in AUTHOR_FILES
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        objects = list(pool.map(download, items))
    payload = {"retrieved_at": datetime.now(timezone.utc).isoformat(),
               "resource": f"https://www.hydroshare.org/resource/{RESOURCE}/",
               "selection": "All three archived Blaine additions and both sites; processing references separately identified",
               "inventory_path": str(inventory_path.relative_to(ROOT)),
               "inventory_sha256": sha256_file(inventory_path), "author_commit": commit,
               "objects": sorted(objects, key=lambda item: item["path"])}
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(objects)} public source records")


if __name__ == "__main__":
    main()
