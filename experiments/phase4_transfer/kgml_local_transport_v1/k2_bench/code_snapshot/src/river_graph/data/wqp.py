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

import io
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
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


def _query_url(
    site_no: str,
    start_date_lo: str | None = None,
    start_date_hi: str | None = None,
    characteristics: list[str] | None = None,
) -> str:
    params = [
        ("siteid", f"USGS-{site_no}"),
        ("dataProfile", "narrow"),
        ("mimeType", "csv"),
    ] + [("characteristicName", c) for c in (characteristics or CHARACTERISTICS)]
    if start_date_lo is not None:
        params.append(("startDateLo", start_date_lo))
    if start_date_hi is not None:
        params.append(("startDateHi", start_date_hi))
    return f"{WQP_RESULT_URL}?{urllib.parse.urlencode(params)}"


def _payload_looks_complete(raw: bytes, declared_length: str | None) -> bool:
    """A truncated response still starts with the CSV header.

    The previous check only looked at the first characters, so a body cut off
    mid-download was cached as if it were the whole station history. Three
    stations ended up holding 2, 11 and 15 DOC samples where the NWIS catalog
    reports 432, 382 and 445, and the missing history propagated into every
    later dataset build.

    Two independent structural signals are used: the declared Content-Length
    must have been delivered, and the record structure must close cleanly (the
    last record must have the same number of fields as the header).
    """
    if not raw:
        return False
    if declared_length is not None:
        try:
            if len(raw) < int(declared_length):
                return False
        except ValueError:
            pass
    import csv
    import io

    text = raw.decode("utf-8", errors="replace")
    if not text.startswith("Org_Identifier"):
        return False
    if not text.endswith("\n"):
        return False
    reader = csv.reader(io.StringIO(text))
    try:
        width = len(next(reader))
    except StopIteration:
        return False
    last = width
    for row in reader:
        last = len(row)
    return last == width


def cached_doc_sample_count(path: str | Path) -> int:
    """DOC samples a cached file actually contains, after the frozen filters."""
    frame = load_station_results(path)
    if frame.empty:
        return 0
    return len(extract_doc_obs(frame))


def _date_windows(years: int = 10) -> list[tuple[str, str]]:
    """Ten-year WQP windows used when an all-history request is truncated."""
    last_year = datetime.now(timezone.utc).year + 1
    out = []
    # The NWIS inventory for this project starts in 1958.  Avoiding empty
    # pre-1950 requests also avoids a known WQP timeout path.
    for start in range(1950, last_year + 1, years):
        end = min(start + years - 1, last_year)
        out.append((f"01-01-{start:04d}", f"12-31-{end:04d}"))
    return out


def _fetch_complete_window(
    site_no: str, lo: str, hi: str, characteristic: str
) -> bytes | None:
    """Fetch one bounded WQP result window, returning only complete CSV bytes."""
    req = urllib.request.Request(
        _query_url(site_no, lo, hi, [characteristic]),
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                declared = resp.headers.get("Content-Length")
            if _payload_looks_complete(raw, declared):
                return raw
            print(f"  {site_no}: incomplete window {lo}..{hi}, retry {attempt + 1}")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                print(f"  {site_no}: WQP rate limited; cooling down 60s")
                time.sleep(60)
                continue
            print(f"  {site_no}: window {lo}..{hi}: HTTP {exc.code}, "
                  f"retry {attempt + 1}")
        except Exception as exc:  # noqa: BLE001 - provider failures vary
            print(f"  {site_no}: window {lo}..{hi}: {exc}, retry {attempt + 1}")
        time.sleep(10 * (attempt + 1))
    return None


def _fetch_segmented_station(site_no: str, out: Path, min_doc_samples: int) -> Path | None:
    """Recover a long station by combining bounded WQP result windows."""
    frames = []
    for characteristic in CHARACTERISTICS:
        for lo, hi in _date_windows():
            raw = _fetch_complete_window(site_no, lo, hi, characteristic)
            if raw is None:
                continue
            frames.append(pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False))
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    merged.to_csv(out, index=False)
    have = cached_doc_sample_count(out)
    print(f"  {site_no}: segmented recovery has {have} DOC samples "
          f"(catalog expects {min_doc_samples})")
    return out if have >= min_doc_samples else None


def fetch_station_results(
    site_no: str,
    cache_dir: str | Path,
    max_attempts: int = 6,
    min_doc_samples: int | None = None,
) -> Path | None:
    """Download one station's results CSV (cached). None on persistent failure.

    Only definitive responses are cached: a structurally complete CSV
    (possibly header-only, meaning the station has no results for these
    characteristics) or a 404. Server overloads (500/429) and timeouts are
    retried with backoff.

    min_doc_samples is an independent expectation, normally the sample count
    the NWIS series catalog reports for this station. A response that arrives
    intact but short of that count is treated as incomplete and retried, and an
    existing cache file that is short of it is refetched instead of reused.
    Transport truncation cannot be detected from the header, so this semantic
    check is what actually protects the history.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{site_no}.csv"
    if out.exists() and min_doc_samples is None:
        return out
    if out.exists():
        have = cached_doc_sample_count(out)
        if have >= int(min_doc_samples):
            return out
        print(f"  {site_no}: cached file holds {have} DOC samples, catalog says "
              f"{min_doc_samples}; refetching")
        out.unlink()
    req = urllib.request.Request(_query_url(site_no), headers={"User-Agent": USER_AGENT})
    best: bytes | None = None
    incomplete_or_short = False
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                declared = resp.headers.get("Content-Length")
                raw = resp.read()
            if not _payload_looks_complete(raw, declared):
                incomplete_or_short = True
                print(f"  {site_no}: incomplete response "
                      f"({len(raw)} of {declared} bytes), retry {attempt + 1}")
                if attempt >= 1:
                    break
                time.sleep(min(20 * (attempt + 1), 120))
                continue
            out.write_bytes(raw)
            if min_doc_samples is None:
                return out
            have = cached_doc_sample_count(out)
            if have >= int(min_doc_samples):
                return out
            incomplete_or_short = True
            print(f"  {site_no}: response holds {have} DOC samples, catalog says "
                  f"{min_doc_samples}, retry {attempt + 1}")
            if best is None or len(raw) > len(best):
                best = raw
            out.unlink()
            if attempt >= 1:
                break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                out.write_text("Org_Identifier\n", encoding="utf-8")  # definitive empty
                return out
            if e.code == 429:
                print(f"  {site_no}: WQP rate limited; cooling down 60s")
                time.sleep(60)
                continue
            incomplete_or_short = True
            print(f"  {site_no}: HTTP {e.code}, retry {attempt + 1}")
            if attempt >= 1:
                break
        except Exception as e:  # noqa: BLE001 - timeouts, SSL resets, ...
            incomplete_or_short = True
            print(f"  {site_no}: {e}, retry {attempt + 1}")
            if attempt >= 1:
                break
        time.sleep(min(20 * (attempt + 1), 120))
    if incomplete_or_short and min_doc_samples is not None:
        recovered = _fetch_segmented_station(site_no, out, int(min_doc_samples))
        if recovered is not None:
            return recovered
    if best is not None:
        # keep the longest response seen rather than losing the station, and
        # say so: the caller can decide whether to use a short station
        out.write_bytes(best)
        print(f"  {site_no}: kept the longest response ({len(best)} bytes) but it "
              f"is still short of the catalog count")
        return out
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


def extract_covariate_obs(
    results: pd.DataFrame, variable: str, audit: list | None = None
) -> pd.DataFrame:
    """Uncensored observations of one covariate that pass the quality rules.

    Filtering by characteristic and detection condition is not enough: the
    cached provider records contain values that cannot be real measurements,
    including a water temperature of 1310 deg C at station 05357225 in June
    2017 that reached the model inputs. Every candidate row is therefore
    classified by river_graph.data.quality first, and only accepted rows are
    returned. Rejected rows are marked missing, never repaired.

    Pass a list as the audit argument to also collect the full audited frame
    (raw value, unit, rejection reason) for the build report.
    """
    from river_graph.data.quality import apply_rules

    sub = results[
        (results["variable"] == variable) & (results["detection_condition"].isna())
    ]
    candidate = sub[["site_no", "date", "value"]].copy()
    candidate["unit"] = sub["unit"].values if "unit" in sub.columns else None
    audited = apply_rules(candidate, variable)
    if audit is not None:
        audit.append(audited)
    accepted = audited[audited["qc_status"] == "accepted"]
    out = accepted[["site_no", "date", "value"]].dropna()
    return out.rename(columns={"value": variable})
