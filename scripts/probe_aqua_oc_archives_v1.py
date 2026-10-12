"""Probe a public Landsat matchup and ERA5-Land access, never print credentials."""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from prepare_aqua_oc_data_v1 import CACHE, HEADERS, OUTPUT, digest, write_json


def archive_probes() -> list[dict]:
    urls = {
        "figshare_download": "https://figshare.com/ndownloader/articles/8139383/versions/2",
        "figshare_institution": "https://colostate.figshare.com/articles/dataset/AquaSat/8139383",
        "cds_arco": "https://arco.datastores.ecmwf.int/cadl-arco-geo-007/arco/reanalysis_era5_land/sfc-2m-temperature/geoChunked.zarr/.zmetadata",
        "destine_era5land_daily": "https://data.earthdatahub.destine.eu/era5/era5-land-daily-utc-v1.zarr/.zmetadata",
    }
    def one(pair):
        name, url = pair
        try:
            with requests.get(url, headers=HEADERS, timeout=35, stream=True) as response:
                result = {"name": name, "url": url, "status": response.status_code,
                          "content_type": response.headers.get("Content-Type"),
                          "content_length": response.headers.get("Content-Length")}
                # Metadata reads are bounded. Do not download a global archive for a probe.
                if response.ok and name in ("cds_arco", "destine_era5land_daily"):
                    payload = response.json()
                    write_json(CACHE / "metadata" / f"{name}.json", payload)
                    result["variables"] = [key for key in payload.get("metadata", {})
                                           if key.endswith("/.zarray")]
                elif not response.ok:
                    result["response_excerpt"] = next(response.iter_content(500), b"").decode(
                        "utf-8", errors="replace")[:300]
                return result
        except requests.RequestException as exc:
            return {"name": name, "url": url, "status": "failed", "error": str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, urls.items()))


def landsat_probe() -> dict:
    import numpy as np
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window

    frame = pd.read_parquet(CACHE / "local_doc_samples.parquet")
    # Fixed, DOC-value-blind probe at a large river station. No result-driven selection.
    nodes = pd.read_csv(CACHE.parents[1] / "data/processed/graph_nodes_graphfix_st357.csv",
                        dtype={"site_no": str})
    reach = pd.read_csv(CACHE.parents[1] / "data/processed/reach_attributes.csv", dtype={"site_no": str})
    nodes = nodes.merge(reach[["site_no", "totdasqkm"]], on="site_no", validate="one_to_one")
    candidates = nodes[nodes.station_nm.str.contains("Mississippi River", case=False)].sort_values(
        "totdasqkm", ascending=False)
    station = candidates.site_no.iloc[0] if not candidates.empty else frame.site_no.iloc[0]
    rows = frame[(frame.site_no == station) & frame.date.between("2000-01-01", "2020-12-31")]
    row = rows.sort_values("date").iloc[0]
    search = {
        "collections": ["landsat-c2-l2"], "limit": 100,
        "intersects": {"type": "Point", "coordinates": [float(row.dec_long_va), float(row.dec_lat_va)]},
        "datetime": "2000-01-01T00:00:00Z/2020-12-31T23:59:59Z",
        "sortby": [{"field": "datetime", "direction": "asc"}],
    }
    response = requests.post("https://planetarycomputer.microsoft.com/api/stac/v1/search",
                             json=search, headers=HEADERS, timeout=45)
    response.raise_for_status()
    payload = response.json()
    write_json(CACHE / "metadata/landsat_probe_search.json", payload)
    items = []
    for item in payload["features"]:
        if item["properties"].get("platform") not in ("landsat-5", "landsat-7"):
            continue
        overpass = pd.Timestamp(item["properties"]["datetime"]).tz_localize(None).normalize()
        distances = (rows.date - overpass).abs().dt.days
        if distances.min() <= 1:
            item["matched_sample_date"] = str(rows.loc[distances.idxmin(), "date"].date())
            items.append(item)
    result = {"site_no": station, "native_sensor_candidates_on_first_catalog_page": len(items),
              "search_query": search, "note": "Connectivity probe only; not full four-detector QC."}
    if not items:
        return result
    item = min(items, key=lambda value: value["properties"]["datetime"])
    token_response = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/sign",
                                  params={"href": item["assets"]["blue"]["href"]},
                                  headers=HEADERS, timeout=45)
    token_response.raise_for_status()
    href = token_response.json()["href"]
    # Signed URL is used in memory only; no tokens are logged or committed.
    with rasterio.Env(GDAL_HTTP_TIMEOUT="40", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif"), rasterio.open(href) as src:
        x, y = transform("EPSG:4326", src.crs, [row.dec_long_va], [row.dec_lat_va])
        r, c = src.index(x[0], y[0])
        values = src.read(1, window=Window(c - 1, r - 1, 3, 3))
    raw_path = CACHE / "pilot_landsat_blue.npy"
    np.save(raw_path, values)
    result.update({"scene_id": item["id"], "satellite_date": item["properties"]["datetime"],
                   "sample_date": item["matched_sample_date"],
                   "blue_window_shape": list(values.shape), "raw_window_sha256": digest(raw_path),
                   "unsigned_asset": item["assets"]["blue"]["href"]})
    return result


def main() -> None:
    results = archive_probes()
    try:
        results.append({"name": "landsat_pixel_probe", "status": "ok", **landsat_probe()})
    except Exception as exc:  # noqa: BLE001 - record provider failures without leaking SAS tokens.
        # GDAL errors may contain a signed URL, so retain only the exception type here.
        results.append({"name": "landsat_pixel_probe", "status": "failed", "error_type": type(exc).__name__})
    for directory in ("official_global_river_poc", "official_aquasat"):
        checkout = CACHE / directory
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                                check=True, text=True, capture_output=True)
        files = sorted(str(path.relative_to(checkout)) for path in checkout.rglob("*")
                       if path.is_file() and ".git" not in path.relative_to(checkout).parts)
        results.append({"name": directory, "checkout_commit": result.stdout.strip(),
                        "file_count": len(files),
                        "implementation_files": [name for name in files if name.endswith((".py", ".ipynb"))]})
    write_json(OUTPUT / "archive_access_probes.json", results)
    for row in results:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
