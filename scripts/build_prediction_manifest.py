"""A0: freeze the Phase-A analysis input list (prediction manifest).

Every analysis in Phase A must be able to name the exact prediction files it
was allowed to read. This script writes that allow-list, together with the
conflicts and gaps it deliberately excludes, so a later reader can tell which
numbers came from which batch.

Usage:
    python scripts/build_prediction_manifest.py
    python scripts/build_prediction_manifest.py --date 20260912
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FROZEN = Path("experiments/frozen_results")
CURRENT = Path("experiments/predictions")
HISTORICAL = Path("experiments/predictions_historical")
ANALYSIS = Path("experiments/analysis")


def git(*args: str) -> str:
    res = subprocess.run(["git", *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=False)
    return res.stdout.strip() if res.returncode == 0 else ""


def today() -> str:
    return datetime.now(timezone.utc).date().strftime("%Y%m%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=today())
    args = ap.parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    frozen = pd.read_csv(FROZEN / "benchmark.csv")
    multiseed = pd.read_csv(FROZEN / "benchmark_multiseed.csv")
    verification = ANALYSIS / f"artifact_verification_{args.date}.csv"
    vdf = pd.read_csv(verification) if verification.exists() else None

    status: dict[str, str] = {}
    if vdf is not None:
        status = {r["file"]: r["status"] for _, r in vdf.iterrows()}

    entries = []
    for p in sorted(CURRENT.glob("*.parquet")):
        key = (str(pd.read_parquet(p, columns=["model"])["model"].iloc[0]),
               p.stem.split("__", 1)[1])
        st = status.get(p.name, "unverified")
        in_frozen = bool(((frozen["model"] == key[0]) &
                          (frozen["mask"] == key[1])).any())
        in_multi = bool(((multiseed["model"] == key[0]) &
                         (multiseed["mask"] == key[1])).any())
        entries.append({
            "file": p.name, "batch": "frozen_phase0", "status": st,
            "in_frozen_table": in_frozen, "in_multiseed_table": in_multi,
            "used_for_phase_a": st.startswith("verified"),
        })

    historical = []
    for p in sorted(HISTORICAL.glob("*.parquet")):
        historical.append({
            "file": p.name, "batch": "recovered_256d08e",
            "used_for_phase_a": True,
            "reason": "overwritten or deleted after the frozen batch; "
                      "recomputed metrics match the frozen table",
        })

    conflicts = [
        {
            "file": e["file"],
            "problem": ("recomputed test metrics disagree with the frozen "
                        "table; the file was overwritten after the freeze by a "
                        "different model run"),
            "used_for_phase_a": False,
            "authoritative_value": "experiments/frozen_results/benchmark.csv",
        }
        for e in entries if e["status"] == "conflict"
    ]
    gaps = [
        {
            "file": e["file"],
            "problem": "no finite test-set prediction; metrics are NaN by design",
            "used_for_phase_a": True,
            "note": "report n and coverage; do not compare error magnitudes "
                    "against full-coverage models",
        }
        for e in entries if e["status"] == "zero_coverage"
    ]

    manifest = {
        "manifest_date": args.date,
        "commit": git("rev-parse", "HEAD"),
        "commit_subject": git("log", "-1", "--format=%s", "HEAD"),
        "materials": {
            "frozen_benchmark": str(FROZEN / "benchmark.csv"),
            "frozen_multiseed": str(FROZEN / "benchmark_multiseed.csv"),
            "prediction_dir": str(CURRENT),
            "historical_dir": str(HISTORICAL),
            "mask_dir": "experiments/masks",
            "masks": sorted(p.stem for p in Path("experiments/masks").glob("*.npz")),
        },
        "counts": {
            "current_predictions": len(entries),
            "used_for_phase_a": sum(e["used_for_phase_a"] for e in entries),
            "historical_recovered": len(historical),
            "conflicts": len(conflicts),
            "zero_coverage": len(gaps),
            "frozen_table_rows": len(frozen),
            "multiseed_table_rows": len(multiseed),
        },
        "predictions": entries,
        "historical_recovered": historical,
        "conflicts": conflicts,
        "zero_coverage": gaps,
        "new_five_seed_predictions": {
            "available": False,
            "note": ("No per-cell predictions exist for H1/H2/H2X training "
                     "seeds 0-4; five-seed prediction ensembling, residual maps "
                     "and per-station intervals are therefore not possible and "
                     "are listed as not-run."),
        },
        "excluded_from_analysis": {
            "env_group_runs": [
                "H2X_hydro_river", "H2X_landcover_river",
                "H2X_climate_river", "H2X_soil_topo_river",
            ],
            "reason": ("env_groups column selection is known to be wrong "
                       "(src/river_graph/models/gcn.py); the default full-feature "
                       "H2X path is unaffected. Group rankings cannot support "
                       "mechanism conclusions."),
        },
    }

    out = FROZEN / f"prediction_manifest_{args.date}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"manifest -> {out}")
    for key, value in manifest["counts"].items():
        print(f"  {key:26} {value}")


if __name__ == "__main__":
    main()
