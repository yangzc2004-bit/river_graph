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

import time
import urllib.request
from pathlib import Path

import pandas as pd

NWIS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
DOC_PCODE = "00681"
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
