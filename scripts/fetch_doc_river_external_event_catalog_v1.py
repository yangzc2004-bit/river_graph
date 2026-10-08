"""Retrieve public Arctic event packages and preserve their EML definitions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_external_event_catalog_v1")
RAW = Path("data/raw/river_external_event_catalog_v1")
PACKAGES = {"timeseries": "doi:10.18739/A2JH3D482", "storms": "doi:10.18739/A2C53F37J"}
SERVICE = "https://arcticdata.io/metacat/d1/mn/v2/object/"


def text(node, path):
    element = node.find(path)
    return " ".join(element.itertext()).strip() if element is not None else ""


def fetch_kervidy():
    """Retain the public version's chemistry and source measurement description."""
    manifest_path = ROOT/"kervidy_retrieval_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Kervidy retrieval is recorded; reuse its existing files")
    url = "https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/?persistentId=doi:10.57745/OFOUWE"
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    metadata = RAW/"kervidy_metadata.json"
    if metadata.exists() and metadata.read_bytes() != response.content:
        raise ValueError("Previously retrieved metadata differs; preserve and investigate versions")
    metadata.write_bytes(response.content)
    version = response.json()["data"]["latestVersion"]
    files = {"spectroDataverse.txt": "kervidy_spectro.txt", "DataVerseSpectro.pdf": "kervidy_readme.pdf"}
    entities = []
    for record in version["files"]:
        source = record["dataFile"]
        if source["filename"] not in files:
            continue
        entity_url = f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{source['id']}"
        path = RAW/files[source["filename"]]
        if not path.exists():
            result = requests.get(entity_url, timeout=60)
            result.raise_for_status()
            path.write_bytes(result.content)
        if path.stat().st_size != source["filesize"]:
            raise ValueError("Downloaded file size differs from repository record")
        checksum = source.get("checksum", {})
        if checksum.get("type") == "MD5":
            import hashlib
            if hashlib.md5(path.read_bytes()).hexdigest() != checksum["value"]:
                raise ValueError("Cached file differs from published file checksum")
        entities.append({"filename": source["filename"], "path": str(path), "url": entity_url,
                         "sha256": sha256_file(path), "bytes": path.stat().st_size})
    fields = version["metadataBlocks"]["citation"]["fields"]
    title = next(f["value"] for f in fields if f["typeName"] == "title")
    manifest_path.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
        "pid": "doi:10.57745/OFOUWE", "title": title,
        "version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
        "metadata_sha256": sha256_file(metadata), "entities": entities}, indent=2)+"\n")
    print("Kervidy retrieval recorded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("arctic", "kervidy"), default="arctic")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if args.source == "kervidy":
        fetch_kervidy()
        return
    manifest_path = ROOT/"retrieval_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Public object retrieval is recorded; reuse its existing files")
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "packages": [], "entities": []}
    for kind, pid in PACKAGES.items():
        url = SERVICE+quote(pid, safe="")
        response = requests.get(url, timeout=40)
        response.raise_for_status()
        folder = RAW/kind
        folder.mkdir(exist_ok=True)
        metadata = folder/"metadata.xml"
        metadata.write_bytes(response.content)
        eml = ET.fromstring(response.content)
        dataset = eml.find("dataset")
        if dataset is None:
            raise ValueError("Expected public EML dataset")
        manifest["packages"].append({"kind": kind, "pid": pid, "url": url,
            "title": text(dataset, "title"), "abstract": text(dataset, "abstract"),
            "rights": text(dataset, "intellectualRights"), "metadata": str(metadata),
            "metadata_sha256": sha256_file(metadata)})
        for table in dataset.findall("dataTable"):
            name = text(table, "physical/objectName")
            if Path(name).name != name:
                raise ValueError("Repository entity must have a plain filename")
            entity_url = text(table, "physical/distribution/online/url")
            if not entity_url.startswith("https://"):
                raise ValueError("Expected an explicit public HTTPS entity URL")
            path = folder/name
            if not path.exists():
                result = requests.get(entity_url, timeout=60)
                result.raise_for_status()
                path.write_bytes(result.content)
            attributes = []
            for attribute in table.findall("attributeList/attribute"):
                attributes.append({"name": text(attribute, "attributeName"),
                    "definition": text(attribute, "attributeDefinition"),
                    "units": text(attribute, "measurementScale/ratio/unit") or text(attribute, "measurementScale/interval/unit"),
                    "missing_codes": [text(m, "code") for m in attribute.findall("missingValueCode")]})
            manifest["entities"].append({"kind": kind, "entity_name": text(table, "entityName"),
                "entity_description": text(table, "entityDescription"), "filename": name,
                "path": str(path), "url": entity_url, "sha256": sha256_file(path),
                "bytes": path.stat().st_size, "attributes": attributes})
            print(kind, name, path.stat().st_size, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("Retrieval complete", flush=True)


if __name__ == "__main__":
    main()
