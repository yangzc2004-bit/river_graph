"""R1: audit every cached covariate against the pre-registered quality rules.

    python scripts/audit_covariates.py

Reads the per-station WQP caches and writes

    data/processed/covariate_quality_report.json   rules, counts, examples
    data/processed/covariate_rejections.csv        every rejected raw row

without building or touching any dataset version.  The rules live in
river_graph.data.quality and depend only on units and physical plausibility,
never on model scores or on any evaluation label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.data.quality import (
    RULE_VERSION,
    RULES,
    rejection_sample,
    rules_digest,
    summarize,
    write_report,
)
from river_graph.data.wqp import (
    extract_covariate_obs,
    load_station_results,
)

COVARIATES = ("temperature", "ph", "spec_conductance")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/raw/wqp_results")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    cache = ROOT / args.cache
    out_dir = ROOT / args.out
    files = sorted(cache.glob("*.csv"))
    if not files:
        raise SystemExit("no cached station files under " + str(cache))

    frames = []
    for path in files:
        frame = load_station_results(path)
        if not frame.empty:
            frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    print(f"stations={len(files)} rows={len(results):,}")

    audit_frames = []
    accepted = {
        name: extract_covariate_obs(results, name, audit=audit_frames)
        for name in COVARIATES
    }
    audited = pd.concat(audit_frames, ignore_index=True)
    summary = summarize(audited)
    report = {
        "rule_version": RULE_VERSION,
        "rules_sha256": rules_digest(),
        "rules": RULES,
        "summary": summary,
        "per_variable": {
            name: summarize(audited[audited.variable == name])
            for name in COVARIATES
        },
        "accepted_rows": {name: len(frame) for name, frame in accepted.items()},
        "sample": rejection_sample(audited),
        "trace": {
            "cached_station_files": len(files),
            "read_from": cache.as_posix(),
        },
        "doc_note": (
            "DOC is reported through the existing frozen unit/detection "
            "filters only; no value-range rule is applied to the target."
        ),
    }
    write_report(out_dir / "covariate_quality_report.json", report)
    rejected = audited[audited.qc_status == "rejected"]
    rejected_path = out_dir / "covariate_rejections.csv"
    rejected.to_csv(rejected_path, index=False)

    print(f"\nrule version {RULE_VERSION}  sha256 {report['rules_sha256'][:16]}")
    for name in COVARIATES:
        part = report["per_variable"][name]
        print(
            f"  {name:17s} rows={part['rows']:7d} accepted={part['accepted']:7d} "
            f"rejected={part['rejected']:6d}"
        )
        for key, count in part["by_reason"].items():
            print(f"      {key}: {count}")
    for key, rows in report["sample"].items():
        print(f"  example {key}: {rows[0] if rows else 'none'}")
    print(f"\nwrote {out_dir / 'covariate_quality_report.json'}")
    print(f"wrote {rejected_path} ({len(rejected)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
