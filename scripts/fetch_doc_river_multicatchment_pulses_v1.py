"""Acquire two independently published DOC/Q archives and original river maps."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/river_multicatchment_pulses_v1"
OUT = ROOT / "experiments/phase4_transfer/doc_river_multicatchment_pulses_v1"
RESOURCE = "9be43573ba754ec1b3650ce233fc99de"
RB = f"https://www.hydroshare.org/resource/{RESOURCE}/"
PAPERS = {
    "rappbode": "https://bg.copernicus.org/articles/16/4497/2019/",
    "bouleau": "https://hess.copernicus.org/articles/27/3935/2023/",
}


class Assets(HTMLParser):
    """Read linked public assets without an optional HTML parsing dependency."""

    def __init__(self, text):
        super().__init__()
        self.links, self.images, self.pdfs = [], [], []
        self.anchor = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.anchor = [attrs["href"], ""]
        if tag == "img" and "src" in attrs:
            self.images.append(attrs.get("data-webversion", attrs["src"]))
        if tag == "meta" and attrs.get("name") == "citation_pdf_url":
            self.pdfs.append(attrs["content"])

    def handle_data(self, data):
        if self.anchor is not None:
            self.anchor[1] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None:
            self.links.append(tuple(self.anchor))
            self.anchor = None


def get(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response


def save(url, path, kind, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(get(url).content)
    if extra.get("provider_md5") and hashlib.md5(path.read_bytes()).hexdigest() != extra["provider_md5"]:
        raise ValueError(f"Provider checksum mismatch: {path.name}")
    if extra.get("provider_bytes") and path.stat().st_size != extra["provider_bytes"]:
        raise ValueError(f"Provider size mismatch: {path.name}")
    return {"url": url, "path": str(path.relative_to(ROOT)), "kind": kind,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "licence": "CC BY 4.0", **extra}


def pangaea(number):
    url = f"https://doi.pangaea.de/10.1594/PANGAEA.{number}"
    page = save(url, RAW / "bouleau" / f"{number}.html", "metadata")
    assets = Assets((ROOT / page["path"]).read_text())
    links = [href for href, label in assets.links
             if "Download dataset as tab-delimited text" in label]
    if len(links) != 1:
        raise ValueError(f"Missing or ambiguous official text download: {number}")
    text = save(urljoin(url, links[0]), RAW / "bouleau" / f"{number}.tsv", "hourly_doc_flow",
                citation=f"Prijac (2023), doi:10.1594/PANGAEA.{number}")
    return [page, text]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "retrieval_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for obj in saved["objects"]:
            if sha256_file(ROOT / obj["path"]) != obj["sha256"]:
                raise ValueError(f"Changed source: {obj['path']}")
        for site, url in PAPERS.items():
            assets = Assets((RAW / site / "paper.html").read_text())
            image = next(urljoin(url, src) for src in assets.images if "-f01" in src)
            path = RAW / site / "published_river_map_web.png"
            if str(path.relative_to(ROOT)) not in {obj["path"] for obj in saved["objects"]}:
                saved["objects"].append(save(image, path, "published_figure_1_web"))
        manifest.write_text(json.dumps(saved, indent=2) + "\n")
        print("Existing independent catchment sources verified")
        return
    inventory = get(f"https://www.hydroshare.org/hsapi/resource/{RESOURCE}/files/").json()
    if inventory["next"] is not None or inventory["count"] != 2:
        raise ValueError("Unexpected Rappbode resource inventory")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "rappbode_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    jobs = [(item["url"].replace("http://", "https://", 1),
             RAW / "rappbode" / item["file_name"], "quarter_hour_processed_doc_flow",
             {"provider_md5": item["checksum"], "provider_bytes": item["size"],
              "citation": "Musolff (2024), doi:10.4211/hs.9be43573ba754ec1b3650ce233fc99de"})
            for item in inventory["results"]]
    jobs += [(url, RAW / site / "paper.html", "methods_and_published_river_map", {})
             for site, url in PAPERS.items()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        objects = list(pool.map(lambda job: save(*job[:3], **job[3]), jobs))
        for bundle in pool.map(pangaea, [959043, 959044]):
            objects.extend(bundle)
    for site, url in PAPERS.items():
        assets = Assets((RAW / site / "paper.html").read_text())
        pdfs = [urljoin(url, href) for href, label in assets.links
                if href.endswith(".pdf") and label.strip() == "Article"]
        # Copernicus' article asset is also explicitly embedded as citation_pdf_url.
        pdfs.extend(assets.pdfs)
        if not pdfs:
            raise ValueError(f"Original paper PDF not linked: {site}")
        objects.append(save(pdfs[0], RAW / site / "paper.pdf", "paper"))
        images = [urljoin(url, src) for src in assets.images
                  if "-f01" in src and src.endswith(".png")]
        if images:
            objects.append(save(images[0], RAW / site / "published_river_map_web.png", "published_figure_1_web"))
    manifest.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
        "selection": "Both independent DOC/Q archives fixed in study_plan.md before pulse measurements",
        "objects": objects, "unavailable": [
            {"case": "Vermont NEWRnet", "resource": "https://www.hydroshare.org/resource/faac1672244c407e9c9c8644c8211fd6/",
             "reason": "Discoverable only; page explicitly denies access to content files"},
            {"case": "Wood Brook", "resource": "https://figshare.com/articles/dataset/Nutrient_export_during_storm_events/4884401",
             "reason": "Public metadata API returned HTTP 403"}]}, indent=2) + "\n")
    print(f"Saved {len(objects)} independent catchment source objects")


if __name__ == "__main__":
    main()
