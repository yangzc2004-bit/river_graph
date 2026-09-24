"""Audit one external WQP candidate before graph construction or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from river_graph.experiments.transfer import ANALYTES
from river_graph.experiments.transfer_data import external_raw_inventory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--huc8", required=True)
    ap.add_argument("--station-dir", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    station_paths = {a: args.station_dir / f"{a}_stations.csv" for a in ANALYTES}
    result_paths = {a: args.result_dir / f"{a}_results.csv" for a in ANALYTES}
    limits = {
        "minimum_stations": 50, "minimum_months": 36,
        "minimum_station_months_per_analyte": 10000,
    }
    report = external_raw_inventory(args.huc8, station_paths, result_paths, limits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eligible": report["eligible"], "out": str(args.out)}))
    return 0 if report["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
