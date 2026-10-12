"""Acquire and inspect public Aqua-OC inputs without altering frozen experiments.

This is data preparation, not a claim of complete Aqua-OC reproduction.
Run with ``uv run python scripts/prepare_aqua_oc_data_v1.py <command>``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from river_graph.data.wqp import extract_doc_obs, load_station_results

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/aqua_oc_reproduction_v1"
OUTPUT = ROOT / "experiments/phase4_transfer/aqua_oc_reproduction_v1"
HEADERS = {"User-Agent": "river-graph-research/0.1"}


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fetch_json(url: str, name: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    payload = response.json()
    write_json(CACHE / "metadata" / f"{name}.json", payload)
    return payload


def metadata() -> None:
    sources = {
        "aquasat": "https://api.figshare.com/v2/articles/8139383",
        "grqa": "https://zenodo.org/api/records/5101057",
        "aqua_oc_followup": "https://zenodo.org/api/records/20337044",
    }
    def one(item):
        name, url = item
        try:
            result = fetch_json(url, name)
            files = [{k: f.get(k) for k in ("id", "name", "key", "size", "download_url",
                                          "links", "checksum", "computed_md5")}
                     for f in result.get("files", [])]
            return {"source": name, "url": url, "status": "ok", "files": files,
                    "description": result.get("description", result.get("metadata", {}).get(
                        "description", ""))}
        except requests.RequestException as exc:
            return {"source": name, "url": url, "status": "failed", "error": str(exc)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipt = list(pool.map(one, sources.items()))
    write_json(OUTPUT / "source_inventory.json", receipt)
    print(json.dumps([{**{key: value for key, value in row.items()
                         if key not in ("description", "files")},
                       "relevant_files": [f for f in row.get("files", []) if
                                          row["source"] != "grqa" or
                                          f.get("key") in ("DOC_GRQA.csv", "README.txt")]}
                      for row in receipt], indent=2))


def download_file(source: str, entry: dict) -> dict:
    name = entry.get("name", entry.get("key"))
    url = entry.get("download_url") or entry["links"].get("content", entry["links"]["self"])
    path = CACHE / "raw" / source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    size = entry.get("size")
    checksum = entry.get("computed_md5") or entry.get("checksum")
    if not path.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        headers = dict(HEADERS)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with requests.get(url, headers=headers, timeout=(30, 45), stream=True) as response:
            response.raise_for_status()
            append = bool(offset and response.status_code == 206)
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError(f"Invalid range response: {name}")
            with partial.open("ab" if append else "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        if size is not None and partial.stat().st_size != size:
            raise RuntimeError(f"Truncated download {name}: {partial.stat().st_size} != {size}")
        if checksum and digest(partial, "md5") != checksum.removeprefix("md5:"):
            raise RuntimeError(f"Provider MD5 mismatch on download: {name}")
        partial.rename(path)
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"Incorrect cached size: {name}")
    if checksum:
        checksum = checksum.removeprefix("md5:")
        if digest(path, "md5") != checksum:
            raise RuntimeError(f"Provider MD5 mismatch: {name}")
    return {"source": source, "file": str(path.relative_to(ROOT)), "url": url,
            "bytes": path.stat().st_size, "sha256": digest(path),
            "provider_md5": checksum, "retrieved_at": datetime.now(timezone.utc).isoformat()}


def download() -> None:
    entries = []
    for source in ("aquasat", "grqa"):
        metadata_path = CACHE / "metadata" / f"{source}.json"
        if not metadata_path.exists():
            print(f"Metadata unavailable: {source}", flush=True)
            continue
        payload = json.loads(metadata_path.read_text())
        for entry in payload["files"]:
            name = entry.get("name", entry.get("key", ""))
            if source == "grqa" and name not in ("DOC_GRQA.csv", "README.txt"):
                continue
            entries.append((source, entry))
    def one(pair):
        receipt = download_file(*pair)
        write_json(CACHE / "receipts" / f"{Path(receipt['file']).name}.json", receipt)
        print(json.dumps(receipt), flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(one, entries))
    write_json(OUTPUT / "download_receipts.json", receipts)


def local_inventory() -> None:
    nodes = pd.read_csv(ROOT / "data/processed/graph_nodes_graphfix_st357.csv",
                        dtype={"site_no": str, "huc_cd": str})
    observations = []
    source_files = []
    for site in nodes.site_no:
        path = ROOT / "data/raw/wqp_results" / f"{site}.csv"
        if not path.exists():
            continue
        frame = extract_doc_obs(load_station_results(path))
        observations.append(frame)
        source_files.append({"site_no": site, "sha256": digest(path), "rows": len(frame)})
    raw = pd.concat(observations, ignore_index=True)
    # Preserve individual observations; do not silently substitute daily averages for the
    # original monthly mean, whose sample weighting is fixed by the canonical dataset.
    raw["sample_row"] = raw.groupby(["site_no", "date"]).cumcount()
    raw = raw.merge(nodes[["site_no", "dec_lat_va", "dec_long_va", "huc_cd"]],
                    on="site_no", validate="many_to_one")
    write_json(CACHE / "local_wqp_sources.json", source_files)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "local_doc_samples.parquet"
    raw.to_parquet(path, index=False)
    credential_paths = {
        "earth_engine_credentials": Path.home() / ".config/earthengine/credentials",
        "google_application_default_credentials": Path.home() / ".config/gcloud/application_default_credentials.json",
        "cds_api_config": Path.home() / ".cdsapirc",
    }
    satellite_window = raw.date.between("1984-01-01", "2020-12-31")
    inventory = {
        "nodes": len(nodes), "raw_samples": len(raw), "stations_with_samples": raw.site_no.nunique(),
        "unique_station_dates": len(raw[["site_no", "date"]].drop_duplicates()),
        "dates": [str(raw.date.min().date()), str(raw.date.max().date())],
        "samples_in_paper_1984_2020_window": int(satellite_window.sum()),
        "stations_in_paper_window": raw.loc[satellite_window, "site_no"].nunique(),
        "duplicate_station_dates": int(raw.duplicated(["site_no", "date"]).sum()),
        "negative_doc": int((raw.doc < 0).sum()),
        "missing_coordinates": int(raw[["dec_lat_va", "dec_long_va"]].isna().any(axis=1).sum()),
        "sample_file": str(path.relative_to(ROOT)), "sample_file_sha256": digest(path),
        "credential_file_exists_only": {key: value.exists() for key, value in credential_paths.items()},
        "installed_modules": {key: importlib.util.find_spec(key) is not None for key in
                              ("ee", "cdsapi", "rasterio", "xarray", "zarr", "pystac_client", "tensorflow")},
    }
    write_json(OUTPUT / "local_inventory.json", inventory)
    print(json.dumps(inventory, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metadata", "download", "local"))
    args = parser.parse_args()
    {"metadata": metadata, "download": download, "local": local_inventory}[args.command]()


if __name__ == "__main__":
    main()
