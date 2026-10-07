"""Download public Krycklan DOC/flow objects for an event-resolution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/river_event_observations_v1"
OUT = ROOT / "experiments/phase4_transfer/doc_river_event_observations_v1"
COLLECTIONS = {
    "chemistry": (
        "https://meta.fieldsites.se/collections/368_zFIGZFTzg1Ynt5tNEGPv/"
        "Svartberget%20-%20Krycklan%20stream%20water%20chemistry.json"
    ),
    "flow": (
        "https://meta.fieldsites.se/collections/2M74g6oPabfeXXoOkaiGbFs2/"
        "Svartberget%20-%20Krycklan%20stream%20water%20balance.json"
    ),
}
GEOMETRY = {
    "streams.geojson": "https://ttiwarir.github.io/krycklan-map/data/streams.geojson",
    "regular_catchments.geojson": (
        "https://ttiwarir.github.io/krycklan-map/data/regular_catchments.geojson"
    ),
    "map_script.js": "https://ttiwarir.github.io/krycklan-map/script.js",
}
ALTERNATIVES = {
    "REPO_DOC_final_2.xlsx": "https://researchdata.cab.unipd.it/803/4/REPO_DOC_final_2.xlsx",
    "readme.txt": "https://researchdata.cab.unipd.it/803/2/readme.txt",
}


class MetadataLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href and href.endswith(".json"):
            self.links.append(href)


def get(url: str, **kwargs: object) -> requests.Response:
    response = requests.get(url, timeout=40, **kwargs)
    response.raise_for_status()
    return response


def metadata(url: str) -> dict:
    page = get(url)
    links = MetadataLinks()
    links.feed(page.text)
    candidates = [x for x in links.links if ".csv.json" in x]
    if len(candidates) != 1:
        raise ValueError(f"Expected one metadata link: {url}")
    record = get(urljoin(url, candidates[0])).json()
    latest = record.get("latestVersion", url)
    if latest.rstrip("/") != url.rstrip("/"):
        return metadata(latest)
    return record


def fetch_object(task: tuple[str, dict]) -> dict:
    kind, member = task
    record = metadata(member["res"])
    name = record["fileName"]
    path = RAW / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        object_id = record["accessUrl"].rstrip("/").split("/")[-1]
        response = get(
            "https://data.fieldsites.se/licence_accept",
            params={"ids": json.dumps([object_id])},
        )
        if "text/csv" not in response.headers.get("Content-Type", ""):
            raise ValueError(f"Not a CSV: {response.url}")
        path.write_bytes(response.content)
    content = path.read_bytes()
    expected = record["size"]
    if len(content) != expected:
        raise ValueError(f"Size mismatch: {name}")
    (path.with_suffix(".metadata.json")).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    result = {
        "kind": kind,
        "collection_object": member["res"],
        "object_url": record["latestVersion"],
        "pid": record["pid"],
        "filename": name,
        "path": str(path.relative_to(ROOT)),
        "site": re.search(r"-C(\d+)_", name).group(1),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "metadata": str(path.with_suffix(".metadata.json").relative_to(ROOT)),
        "licence": "CC BY 4.0; SITES attribution retained",
    }
    print(f"Retrieved {kind} C{result['site']}: {len(content):,} bytes", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--alternatives", action="store_true", help="Retrieve the public Turbolo optical case")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if args.alternatives:
        objects = []
        for name, url in ALTERNATIVES.items():
            path = RAW / "turbolo" / name
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_bytes(get(url).content)
            objects.append({"url": url, "path": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size})
        record = {
            "landing_page": "https://researchdata.cab.unipd.it/803/",
            "paper": "https://doi.org/10.1029/2022WR034397",
            "licence": "CC BY 4.0",
            "evidence_type": "corrected_fDOM_derived_DOC_not_independent_laboratory_truth",
            "objects": objects,
        }
        (OUT / "alternative_retrieval.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return
    if (OUT / "retrieval_manifest.json").exists():
        raise FileExistsError("Retrieval is frozen; use existing raw files for analysis")
    tasks = []
    for kind, url in COLLECTIONS.items():
        collection = get(url).json()
        (RAW / f"{kind}_collection.json").write_text(json.dumps(collection, indent=2) + "\n")
        for member in collection["members"]:
            if kind == "chemistry" or (
                member["name"].startswith("SITES_WB-Q_")
                and member["name"].lower().endswith("_daily.csv")
            ):
                tasks.append((kind, member))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        objects = list(pool.map(fetch_object, tasks))
    for name, url in GEOMETRY.items():
        path = RAW / name
        content = get(url).content
        if name.endswith(".geojson"):
            json.loads(content)
        path.write_bytes(content)
        objects.append({
            "kind": "geometry", "object_url": url,
            "filename": name, "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
            "licence": "SLU-linked Krycklan Explorer; provenance retained",
        })
        print(f"Retrieved {name}: {len(content):,} bytes", flush=True)
    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "selection": "All chemistry members and daily discharge collection members",
        "collections": COLLECTIONS,
        "objects": objects,
    }
    (OUT / "retrieval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
