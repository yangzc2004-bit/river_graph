"""Recompute the frozen benchmark table from stored predictions.

Zero retraining: reads experiments/predictions/*.parquet and rebuilds
experiments/frozen_results/benchmark.csv with the current metric set.

Guard rail
----------
The frozen table has 154 rows but only a subset of them has a prediction file
in experiments/predictions/ (A0 explains why: commit 2b67bf0 deleted 69 of
them, and the G0_gcn_none file was overwritten by a different run). Rewriting
the table from the parquet directory would silently *drop* every row without a
prediction and silently *replace* rows whose stored prediction no longer
matches. Both are data loss, so this script reports the damage and refuses to
write unless the caller accepts it explicitly.

Usage:
    python scripts/recompute_metrics.py --check
    python scripts/recompute_metrics.py --allow-row-loss
    python scripts/recompute_metrics.py --allow-row-loss --allow-changed-rows
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.predictions import PRED_DIR

FROZEN = Path("experiments/frozen_results/benchmark.csv")
TOL = 1e-4


def recompute() -> pd.DataFrame:
    rows = []
    for f in sorted(PRED_DIR.glob("*.parquet")):
        model, mask = f.stem.split("__")
        df = pd.read_parquet(f)
        test = df[df["split"] == "test"]
        m = metrics(test["y_true"].values, test["y_pred"].values)
        rows.append({"model": model, "mask": mask, **m})
    return pd.DataFrame(rows)


def diff_report(current: pd.DataFrame, new: pd.DataFrame) -> dict:
    cur_keys = set(zip(current["model"], current["mask"]))
    new_keys = set(zip(new["model"], new["mask"]))
    dropped = sorted(cur_keys - new_keys)
    added = sorted(new_keys - cur_keys)

    cur = current.set_index(["model", "mask"])
    newi = new.set_index(["model", "mask"])
    changed = []
    for key in sorted(cur_keys & new_keys):
        a, b = cur.loc[key], newi.loc[key]
        for col in ("mae", "r2", "rmse"):
            if col not in cur.columns or col not in newi.columns:
                continue
            va, vb = a[col], b[col]
            if pd.isna(va) and pd.isna(vb):
                continue
            if abs(float(va) - float(vb)) > TOL:
                changed.append({
                    "model": key[0], "mask": key[1], "column": col,
                    "frozen": float(va), "recomputed": float(vb),
                    "delta": float(vb) - float(va),
                })
                break
    return {"dropped": dropped, "added": added, "changed": changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what a rewrite would change, write nothing")
    ap.add_argument("--allow-row-loss", action="store_true",
                    help="permit dropping frozen rows that have no prediction")
    ap.add_argument("--allow-changed-rows", action="store_true",
                    help="permit replacing rows whose stored prediction no "
                         "longer reproduces the frozen value")
    args = ap.parse_args()

    if not FROZEN.exists():
        raise SystemExit(f"{FROZEN} not found; nothing to compare against")
    current = pd.read_csv(FROZEN)
    new = recompute()
    report = diff_report(current, new)

    print(f"frozen rows  : {len(current)}")
    print(f"recomputed   : {len(new)}")
    print(f"would drop   : {len(report['dropped'])} rows")
    print(f"would add    : {len(report['added'])} rows")
    print(f"would change : {len(report['changed'])} rows")
    for row in report["changed"]:
        print(f"  {row['model']}__{row['mask']} {row['column']}: "
              f"{row['frozen']:.6f} -> {row['recomputed']:.6f} "
              f"({row['delta']:+.6f})")
    if report["dropped"]:
        print("\ndropped rows come from predictions that are no longer in "
              f"{PRED_DIR}/; see experiments/analysis/"
              "evidence_inventory_20260912.md and "
              "scripts/recover_historical_predictions.py")

    if args.check:
        print("\n--check: nothing written")
        return

    blocking = []
    if report["dropped"] and not args.allow_row_loss:
        blocking.append(f"{len(report['dropped'])} rows would be dropped "
                        "(--allow-row-loss)")
    if report["changed"] and not args.allow_changed_rows:
        blocking.append(f"{len(report['changed'])} rows would change "
                        "(--allow-changed-rows)")
    if blocking:
        print("\nrefusing to overwrite the frozen table:")
        for item in blocking:
            print(f"  - {item}")
        raise SystemExit(2)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = FROZEN.with_name(f"benchmark.pre_recompute_{stamp}.csv")
    shutil.copy2(FROZEN, backup)
    new.to_csv(FROZEN, index=False)
    print(f"\nbackup       : {backup}")
    print(f"recomputed {len(new)} rows -> {FROZEN}")


if __name__ == "__main__":
    main()
