"""NLDI (Network Linked Data Index) client.

NLDI links NWIS sites to the NHDPlus network and provides downstream
navigation along flowlines. This is how USGS stations get embedded into
the river graph.

Base URL: https://api.water.usgs.gov/nldi/linked-data
(service moved from labs.waterdata.usgs.gov in 2025)

All network results are cached on disk (one JSON per station); only
definitive responses (200/404) are cached, so transient failures retry
on the next run.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
USER_AGENT = "river-graph-research"


def _fetch_json(url: str, cache_path: Path) -> dict | None:
    """GET JSON with disk caching. Returns None for definitive 404s."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    data = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = {"_not_found": True}
                break
            if e.code == 429:  # rate limited: back off, do not cache
                time.sleep(15 * (attempt + 1))
                continue
            raise  # other server errors: do not cache, retry next run
    if data is None:
        raise RuntimeError(f"rate limited after retries: {url}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.5)  # be polite
    return data


def get_station_comid(site_no: str, cache_dir: str | Path) -> str | None:
    """NHDPlus COMID of the reach a NWIS site is indexed to (None if unlinked)."""
    url = f"{NLDI_BASE}/nwissite/USGS-{site_no}"
    data = _fetch_json(url, Path(cache_dir) / f"{site_no}.json")
    if not data or data.get("_not_found"):
        return None
    features = data.get("features") or []
    if not features:
        return None
    comid = features[0]["properties"].get("comid")
    return str(comid) if comid is not None else None


def snap_point_to_comid(lat: float, lon: float, cache_dir: str | Path) -> str | None:
    """Snap an arbitrary point to the nearest NHDPlus reach COMID.

    Fallback for stations missing from the NLDI nwissite index.
    """
    url = f"{NLDI_BASE}/comid/position?coords=POINT({lon}%20{lat})"
    data = _fetch_json(url, Path(cache_dir) / f"pt_{lat}_{lon}.json")
    if not data or data.get("_not_found"):
        return None
    features = data.get("features") or []
    if not features:
        return None
    comid = features[0]["properties"].get("comid")
    return str(comid) if comid is not None else None


def match_stations(
    site_numbers: list[str],
    cache_dir: str | Path,
    coords: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match stations to COMIDs. Returns DataFrame(site_no, comid).

    Tries the NLDI nwissite index first; stations it does not index fall
    back to snapping their coordinates to the nearest reach (requires
    ``coords`` with columns site_no, dec_lat_va, dec_long_va).
    Stations that still fail get comid=None so the caller can report them.
    """
    coord_map = (
        coords.set_index("site_no")[["dec_lat_va", "dec_long_va"]] if coords is not None else None
    )
    rows = []
    n = len(site_numbers)
    for i, site_no in enumerate(site_numbers):
        try:
            comid = get_station_comid(site_no, cache_dir)
            if comid is None and coord_map is not None and site_no in coord_map.index:
                lat, lon = coord_map.loc[site_no].astype(float)
                comid = snap_point_to_comid(lat, lon, Path(cache_dir).parent / "snap")
        except Exception as e:  # noqa: BLE001 - transient network errors
            print(f"  match failed for {site_no}: {e}")
            continue  # not cached; retried next run
        rows.append({"site_no": site_no, "comid": comid})
        if (i + 1) % 50 == 0:
            print(f"  matched {i + 1}/{n}")
    return pd.DataFrame(rows)


def get_downstream_flowline_comids(
    comid: str, cache_dir: str | Path, distance_km: int = 500
) -> list[str]:
    """Ordered COMIDs of downstream-mainstem flowlines (own reach first).

    Navigates by COMID rather than by site so it also works for stations
    that are not in the NLDI nwissite index (coordinate-snapped ones).
    """
    url = f"{NLDI_BASE}/comid/{comid}/navigation/DM/flowlines?distance={distance_km}"
    data = _fetch_json(url, Path(cache_dir) / f"comid_{comid}_dm{distance_km}.json")
    if not data or data.get("_not_found"):
        return []
    return [
        str(f["properties"]["nhdplus_comid"])
        for f in data.get("features", [])
        if f.get("properties", {}).get("nhdplus_comid") is not None
    ]
