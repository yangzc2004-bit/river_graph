"""USGS NWIS data access.

Two building blocks used by the data pipeline:

- DOC site inventory: which NWIS sites in the study region have dissolved
  organic carbon (pcode 00681) water-quality samples.
- Series catalog: per site x parameter sample counts and begin/end dates,
  used to profile data availability before downloading actual values.

The NWIS site service is far more reliable than the WQP bulk result endpoint,
so availability profiling goes through here; actual values come from WQP.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
NWIS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
DOC_PCODE = "00681"
DISCHARGE_PCODE = "00060"  # discharge, cubic feet per second
USER_AGENT = "river-graph-research"


def _get(url: str, timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def read_rdb(path: str | Path) -> pd.DataFrame:
    """Read an NWIS RDB file, dropping comment lines and the format row."""
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    # the row after the header holds column formats like "5s", "15s"
    return df[~df.iloc[:, 0].str.fullmatch(r"\d+s")].reset_index(drop=True)


def fetch_doc_sites(huc2_list: list[str], out_dir: str | Path) -> pd.DataFrame:
    """Download (once) and combine the inventory of sites with DOC samples.

    One RDB file per HUC2 region is stored in ``out_dir``; existing files are
    reused. Returns the combined site table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for huc2 in huc2_list:
        f = out_dir / f"nwis_doc_sites_huc{huc2}.rdb"
        if f.exists() and f.stat().st_size > 1000:
            continue
        url = (
            f"{NWIS_SITE_URL}?format=rdb&huc={huc2}&parameterCd={DOC_PCODE}"
            "&siteStatus=all&hasDataTypeCd=qw"
        )
        f.write_text(_get(url), encoding="utf-8")
        time.sleep(1)
    frames = [read_rdb(f) for f in sorted(out_dir.glob("nwis_doc_sites_huc*.rdb"))]
    return pd.concat(frames, ignore_index=True).drop_duplicates("site_no")


def fetch_qw_catalog(
    site_numbers: list[str],
    out_dir: str | Path,
    batch_size: int = 150,
) -> None:
    """Download the qw series catalog (per site x parameter counts) in batches.

    One RDB per batch is written to ``out_dir``; existing non-trivial files are
    skipped, so reruns resume where they stopped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [site_numbers[i:i + batch_size] for i in range(0, len(site_numbers), batch_size)]
    for i, batch in enumerate(batches):
        out = out_dir / f"batch_{i:04d}.rdb"
        if out.exists() and out.stat().st_size > 1000:
            continue
        url = (
            f"{NWIS_SITE_URL}?format=rdb&sites={','.join(batch)}"
            f"&seriesCatalogOutput=true&hasDataTypeCd=qw&parameterCd={DOC_PCODE}"
        )
        for attempt in range(4):
            try:
                text = _get(url)
                if "agency_cd" not in text:
                    raise RuntimeError(f"unexpected response: {text[:120]}")
                out.write_text(text, encoding="utf-8")
                print(f"batch {i + 1}/{len(batches)} OK")
                break
            except Exception as e:  # noqa: BLE001 - network errors vary
                print(f"batch {i} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"catalog batch {i} failed after 4 attempts")
        time.sleep(1)


def load_qw_catalog(catalog_dir: str | Path) -> pd.DataFrame:
    """Combine batch RDBs into one typed catalog DataFrame."""
    files = sorted(Path(catalog_dir).glob("batch_*.rdb"))
    if not files:
        raise FileNotFoundError(f"no batch_*.rdb under {catalog_dir}")
    cat = pd.concat([read_rdb(f) for f in files], ignore_index=True)
    cat["count_nu"] = pd.to_numeric(cat["count_nu"], errors="coerce")
    # mixed input formats across data types -> per-element parsing
    cat["begin_date"] = pd.to_datetime(cat["begin_date"], errors="coerce", format="mixed")
    cat["end_date"] = pd.to_datetime(cat["end_date"], errors="coerce", format="mixed")
    return cat


def fetch_daily_discharge(
    site_numbers: list[str],
    cache_dir: str | Path,
    batch_size: int = 25,
) -> None:
    """Download daily mean discharge (pcode 00060, statCd 00003) in batches.

    One RDB per batch under ``cache_dir``; the filename carries a hash of the
    batch's site list so different site orderings never collide. Failed
    batches are skipped (logged) rather than fatal — reruns resume.
    """
    import hashlib

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    batches = [site_numbers[i:i + batch_size] for i in range(0, len(site_numbers), batch_size)]
    failed = 0
    for i, batch in enumerate(batches):
        digest = hashlib.md5(",".join(batch).encode()).hexdigest()[:8]
        out = cache_dir / f"dv_{i:04d}_{digest}.rdb"
        if out.exists() and out.stat().st_size > 500:
            continue
        url = (
            f"{NWIS_DV_URL}?format=rdb&sites={','.join(batch)}"
            f"&parameterCd={DISCHARGE_PCODE}&statCd=00003&startDT=1900-01-01"
        )
        for attempt in range(4):
            try:
                text = _get(url, timeout=300)
                if "agency_cd" not in text:
                    raise RuntimeError(f"unexpected response: {text[:120]}")
                out.write_text(text, encoding="utf-8")
                print(f"dv batch {i + 1}/{len(batches)} OK")
                break
            except Exception as e:  # noqa: BLE001
                print(f"dv batch {i} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            print(f"dv batch {i} SKIPPED after 4 attempts (rerun to retry)")
            failed += 1
        time.sleep(1)
    if failed:
        print(f"warning: {failed} dv batches skipped")


def _read_rdb_blocks(path: str | Path, value_suffix: str | None = None) -> pd.DataFrame:
    """Parse a multi-site RDB where each site block has its own header.

    Blocks can differ in column count, and per-site value columns are named
    ``<ts_id>_<pcode>_<stat>`` (site-specific), so blocks must NOT be
    column-aligned. If ``value_suffix`` is given, the column ending with it
    is renamed to "value" per block and only
    (agency_cd, site_no, datetime, value) are returned.
    """
    lines = [
        ln for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if not ln.startswith("#")
    ]
    header_idx = [i for i, ln in enumerate(lines) if ln.startswith("agency_cd")]
    frames = []
    for k, start in enumerate(header_idx):
        end = header_idx[k + 1] if k + 1 < len(header_idx) else len(lines)
        block = lines[start:end]
        if len(block) < 3:  # header + format row + >=1 data row
            continue
        df = pd.read_csv(io.StringIO("\n".join(block)), sep="\t", dtype=str)
        df = df[~df.iloc[:, 0].str.fullmatch(r"\d+s")]
        if value_suffix is not None:
            value_cols = [c for c in df.columns if c.endswith(value_suffix)]
            if not value_cols:
                continue
            df = df[["agency_cd", "site_no", "datetime", value_cols[0]]].rename(
                columns={value_cols[0]: "value"}
            )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_daily_discharge(cache_dir: str | Path) -> pd.DataFrame:
    """Combine daily-discharge RDBs into (site_no, date, discharge_cfs)."""
    files = sorted(Path(cache_dir).glob("dv_*.rdb"))
    if not files:
        raise FileNotFoundError(f"no dv_*.rdb under {cache_dir}")
    frames = []
    for f in files:
        df = _read_rdb_blocks(f, value_suffix="_00060_00003")
        if not df.empty:
            frames.append(
                df[["site_no", "datetime", "value"]].rename(
                    columns={"datetime": "date", "value": "discharge_cfs"}
                )
            )
    if not frames:
        return pd.DataFrame(columns=["site_no", "date", "discharge_cfs"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    out["discharge_cfs"] = pd.to_numeric(out["discharge_cfs"], errors="coerce")
    return out.dropna(subset=["date", "discharge_cfs"])
