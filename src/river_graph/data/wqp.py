"""Water Quality Portal (WQP) access for actual observation values.

The NWIS catalog gives sample *counts*; the actual DOC concentrations come
from WQP. The bulk endpoint is unreliable (frequent 500/overload), so all
fetches here are per-station, disk-cached, and resume-safe.

Schema notes (WQP 3.0 narrow profile):
- Result_Characteristic: 'Organic carbon', 'Temperature, water', ...
- Result_SampleFraction: 'Dissolved' marks DOC (USGS pcode 00681)
- Result_Measure / Result_MeasureUnit: value and unit
- Result_ResultDetectionCondition: non-null marks censored values
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

WQP_RESULT_URL = "https://www.waterqualitydata.us/wqx3/Result/search"
USER_AGENT = "river-graph-research"

# characteristic name -> our short name
CHARACTERISTICS = {
    "Organic carbon": "organic_carbon",
    "Temperature, water": "temperature",
    "pH": "ph",
    "Specific conductance": "spec_conductance",
}

NARROW_COLS = {
    "Location_Identifier": "site_no",
    "Activity_StartDate": "date",
    "Result_Characteristic": "characteristic",
    "Result_SampleFraction": "fraction",
    "Result_Measure": "value",
    "Result_MeasureUnit": "unit",
    "Result_ResultDetectionCondition": "detection_condition",
    "USGSpcode": "usgs_pcode",
}


def _query_url(site_no: str) -> str:
    params = [
        ("siteid", f"USGS-{site_no}"),
        ("dataProfile", "narrow"),
        ("mimeType", "csv"),
    ] + [("characteristicName", c) for c in CHARACTERISTICS]
    return f"{WQP_RESULT_URL}?{urllib.parse.urlencode(params)}"


def fetch_station_results(
    site_no: str, cache_dir: str | Path, max_attempts: int = 6
) -> Path | None:
    """Download one station's results CSV (cached). None on persistent failure.

    Only definitive responses are cached: a valid CSV (possibly header-only,
    meaning the station has no results for these characteristics) or a 404.
    Server overloads (500/429) and timeouts are retried with backoff.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{site_no}.csv"
    if out.exists():
        return out
    req = urllib.request.Request(_query_url(site_no), headers={"User-Agent": USER_AGENT})
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if text.startswith("Org_Identifier"):
                out.write_text(text, encoding="utf-8")
                return out
            # server-side truncation marker: retry
            print(f"  {site_no}: malformed response, retry {attempt + 1}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                out.write_text("Org_Identifier\n", encoding="utf-8")  # definitive empty
                return out
            print(f"  {site_no}: HTTP {e.code}, retry {attempt + 1}")
        except Exception as e:  # noqa: BLE001 - timeouts, SSL resets, ...
            print(f"  {site_no}: {e}, retry {attempt + 1}")
        time.sleep(min(20 * (attempt + 1), 120))
    print(f"  {site_no}: FAILED after {max_attempts} attempts")
    return None


def load_station_results(path: str | Path) -> pd.DataFrame:
    """Parse a cached narrow-profile CSV into a tidy long table."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df.rename(columns={k: v for k, v in NARROW_COLS.items() if k in df.columns})
    if "characteristic" not in df.columns:  # header-only (no results)
        return pd.DataFrame(columns=list(NARROW_COLS.values()) + ["variable"])
    # WQP prefixes NWIS site ids ("USGS-05331000"); the graph uses bare numbers
    df["site_no"] = df["site_no"].str.removeprefix("USGS-")
    df["variable"] = df["characteristic"].map(CHARACTERISTICS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["variable"])


def extract_doc_obs(results: pd.DataFrame) -> pd.DataFrame:
    """DOC observations: USGS pcode 00681 (dissolved fraction, mg/L, uncensored).

    WQP 3.0 does not use the literal fraction "Dissolved" for these records;
    00681 appears as "Filtered field and/or lab". Where the pcode is missing
    we fall back to that semantic filter.
    """
    doc = results[results["variable"] == "organic_carbon"]
    by_pcode = doc["usgs_pcode"] == "00681"
    by_fraction = (
        doc["usgs_pcode"].isna()
        & doc["fraction"].str.lower().isin(["dissolved", "filtered field and/or lab"])
    )
    doc = doc[
        (by_pcode | by_fraction)
        & (doc["unit"].str.lower() == "mg/l")
        & (doc["detection_condition"].isna())
    ]
    return doc[["site_no", "date", "value"]].dropna().rename(columns={"value": "doc"})


def extract_covariate_obs(results: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Uncensored observations of one covariate (temperature/ph/spec_conductance)."""
    sub = results[
        (results["variable"] == variable) & (results["detection_condition"].isna())
    ]
    return sub[["site_no", "date", "value"]].dropna().rename(columns={"value": variable})
