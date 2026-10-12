"""Record public-source access and inspect author-released repository metadata."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from prepare_aqua_oc_data_v1 import CACHE, HEADERS, OUTPUT, write_json


def main() -> None:
    urls = {
        "aquasat_version2_api": "https://api.figshare.com/v2/articles/8139383/versions/2",
        "aquasat_page": "https://figshare.com/articles/dataset/AquaSat/8139383",
        "aquasat_cdn_api": "https://api.figshare.com/v2/articles/8139383/files",
        "author_repositories": "https://api.github.com/users/qqq196988/repos?per_page=100",
        "aquasat_repository": "https://api.github.com/repos/GlobalHydrologyLab/watersat/contents",
        "nature_followup": "https://www.nature.com/articles/s41561-026-02034-w",
        "landsat_catalog": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2",
        "era5land_timeseries": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=download",
    }
    def one(pair):
        name, url = pair
        try:
            response = requests.get(url, headers=HEADERS, timeout=45)
            response.raise_for_status()
            path = CACHE / "metadata" / f"probe_{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text)
            result = {"name": name, "url": url, "status": response.status_code,
                      "bytes": len(response.content), "cache": str(path)}
            if "api.github.com" in url:
                payload = response.json()
                result["entries"] = [{k: x.get(k) for k in ("name", "description", "html_url")}
                                     for x in payload] if isinstance(payload, list) else payload
            elif name.startswith("aquasat"):
                result["file_links"] = sorted(set(re.findall(
                    r'https[^"\s<>]+(?:ndownloader|files\.figshare|file/download)[^"\s<>]*',
                    response.text)))[:20]
                if isinstance(response.json() if "json" in response.headers.get("Content-Type", "")
                              else None, dict):
                    result["files"] = response.json().get("files", [])
            elif name == "nature_followup":
                for key in ("Code availability", "Data availability"):
                    position = response.text.find(key)
                    result[key] = re.sub("<[^>]+>", " ", response.text[position:position + 4500])[:2200]
            return result
        except requests.RequestException as exc:
            return {"name": name, "url": url, "status": "failed", "error": str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, urls.items()))
    write_json(OUTPUT / "public_access_probes.json", results)
    for row in results:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
