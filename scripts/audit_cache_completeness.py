"""Check whether the cached provider responses are actually complete.

    python scripts/audit_cache_completeness.py
    python scripts/audit_cache_completeness.py --redownload data/raw/wqp_control

Why this exists: 571 cache files were present, but three stations turned out to
hold 2, 11 and 15 DOC samples where the NWIS series catalog reports 432, 382 and
445. A cache file can exist and still be missing most of its history, so "the
file is there" is not evidence that the input is complete.

The audit cross-checks every cached station against the catalog sample count,
which is an independent source: the catalog comes from the NWIS site service,
the values come from WQP. A station whose cached DOC count is far below its
catalog count is flagged, and the shortfall is quantified so the coverage
difference in a rebuilt dataset can be explained rather than guessed at.

--redownload fetches the flagged stations again into a NEW directory and records
the request URL, time, HTTP status, byte count and sha256 of each response, so a
control download can be compared with what is on disk without overwriting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.data.wqp import (
    USER_AGENT,
    _query_url,
    cached_doc_sample_count,
    extract_doc_obs,
    load_station_results,
)

CACHE = ROOT / "data/raw/wqp_results"
CATALOG = ROOT / "data/processed/doc_site_inventory.csv"
NODES = ROOT / "data/processed/graph_nodes.csv"


def analyse(cache: Path, catalog_path: Path) -> tuple[list[dict], dict]:
    catalog = pd.read_csv(catalog_path, dtype={"site_no": str})
    # a station can appear on more than one catalog row; the largest reported
    # sample count is the least we should be able to find in its cache
    catalog = catalog.groupby("site_no")["count_nu"].max()
    sites = list(pd.read_csv(NODES, dtype={"site_no": str})["site_no"])
    rows = []
    for site in sites:
        path = cache / (site + ".csv")
        expected = int(catalog[site]) if site in catalog.index else None
        if not path.is_file():
            rows.append({"site_no": site, "cached": False, "bytes": 0,
                         "rows": 0, "doc_rows": 0, "catalog_doc_samples": expected,
                         "shortfall": expected or 0, "first_date": "", "last_date": ""})
            continue
        frame = load_station_results(path)
        doc = extract_doc_obs(frame)
        rows.append(
            {
                "site_no": site,
                "cached": True,
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "doc_rows": len(doc),
                "catalog_doc_samples": expected,
                "shortfall": (expected or 0) - len(doc),
                "first_date": str(frame["date"].min())[:10] if len(frame) else "",
                "last_date": str(frame["date"].max())[:10] if len(frame) else "",
            }
        )
    table = pd.DataFrame(rows)
    flagged = table[(table.shortfall > 0) & table.cached]
    summary = {
        "stations": len(table),
        "cached_files": int(table.cached.sum()),
        "missing_files": int((~table.cached).sum()),
        "stations_below_catalog": len(flagged),
        "total_missing_doc_samples": int(flagged.shortfall.sum()),
        "worst": flagged.nlargest(10, "shortfall")[
            ["site_no", "bytes", "doc_rows", "catalog_doc_samples", "shortfall"]
        ].to_dict("records"),
    }
    return rows, summary


def redownload(sites: list[str], target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    log = []
    for site in sites:
        url = _query_url(site)
        out = target / (site + ".csv")
        record = {
            "site_no": site,
            "url": url,
            "requested_at": time.time(),
            "attempt": 0,
        }
        for attempt in range(3):
            record["attempt"] = attempt + 1
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=300) as response:
                    record["http_status"] = response.status
                    record["content_length"] = response.headers.get("Content-Length")
                    body = response.read()
                record["bytes"] = len(body)
                record["sha256"] = hashlib.sha256(body).hexdigest()
                out.write_bytes(body)
                record["stored"] = out.as_posix()
                record["ok"] = True
                break
            except urllib.error.HTTPError as exc:
                record["http_status"] = exc.code
                record["error"] = "HTTP " + str(exc.code)
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
            record["ok"] = False
            time.sleep(5 * (attempt + 1))
        log.append(record)
        print(f"  {site}: ok={record.get('ok')} bytes={record.get('bytes')} "
              f"status={record.get('http_status')}", flush=True)
    return log




def promote(control_dir: Path, cache: Path, out_dir: Path) -> list[dict]:
    """Keep whichever copy of a station holds more DOC samples.

    A control download can return LESS than what is already cached: the two
    stations 05455100 and 445548111032200 came back with fewer samples than the
    files on disk. Promotion is therefore per station and by measured content,
    never by "the newer file must be better", and every decision is logged.
    """
    decisions = []
    for control in sorted(control_dir.glob("*.csv")):
        site = control.stem
        existing = cache / (site + ".csv")
        have = cached_doc_sample_count(existing) if existing.is_file() else -1
        new = cached_doc_sample_count(control)
        keep_control = new > have
        if keep_control:
            if existing.is_file():
                backup = cache / (site + ".csv.replaced")
                shutil.copy2(existing, backup)
            shutil.copy2(control, existing)
        decisions.append(
            {
                "site_no": site,
                "previous_doc_samples": have,
                "control_doc_samples": new,
                "kept": "control" if keep_control else "existing",
            }
        )
    (out_dir / "cache_promotion_log.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    kept = sum(1 for d in decisions if d["kept"] == "control")
    print(f"promoted {kept} of {len(decisions)} control downloads")
    return decisions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/raw/wqp_results")
    ap.add_argument("--catalog", default="data/processed/doc_site_inventory.csv")
    ap.add_argument("--out", default="data/processed/cache_completeness")
    ap.add_argument("--threshold", type=int, default=5,
                    help="flag a station short by more than this many samples")
    ap.add_argument("--redownload", default=None,
                    help="re-fetch flagged or named stations into this NEW directory")
    ap.add_argument("--sites", nargs="*", default=None,
                    help="explicit station list for --redownload")
    ap.add_argument("--promote", default=None,
                    help="keep the more complete copy of each control download")
    args = ap.parse_args()

    rows, summary = analyse(ROOT / args.cache, ROOT / args.catalog)
    table = pd.DataFrame(rows)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "cache_completeness.csv", index=False)
    (out_dir / "cache_completeness_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.redownload:
        if args.sites:
            sites = args.sites
        else:
            short = table[(table.shortfall > args.threshold) & table.cached]
            sites = list(short.sort_values("shortfall", ascending=False)["site_no"])
        print(f"control download of {len(sites)} stations -> {args.redownload}")
        log = redownload(sites, ROOT / args.redownload)
        (out_dir / "control_download_log.json").write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.promote:
        promote(ROOT / args.promote, ROOT / args.cache, out_dir)
        rows, summary = analyse(ROOT / args.cache, ROOT / args.catalog)
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "cache_completeness.csv", index=False)
        (out_dir / "cache_completeness_report.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("after promotion:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
