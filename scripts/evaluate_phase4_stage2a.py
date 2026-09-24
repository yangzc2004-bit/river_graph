"""Evaluate corrected Stage-2A same-analyte diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.transfer import object_hash


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cluster_bootstrap(frame: pd.DataFrame, reps: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap basin-month clusters after averaging task seeds."""
    rng = np.random.default_rng(seed)
    values = frame["delta_mae"].to_numpy(float)
    observed = float(values.mean())
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        draws[i] = float(values[rng.integers(0, len(values), size=len(values))].mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def _shared_month_bootstrap(
    frame: pd.DataFrame, reps: int, seed: int
) -> tuple[float, float, float, int]:
    """Bootstrap shared calendar months while keeping basins equally weighted."""
    rng = np.random.default_rng(seed)
    by_basin = {b: g.set_index("month")["delta_mae"] for b, g in frame.groupby("basin")}
    months = sorted(set().union(*(set(s.index) for s in by_basin.values())))
    observed = float(np.mean([s.mean() for s in by_basin.values()]))
    draws, empty_redraws = [], 0
    while len(draws) < reps:
        sampled = rng.choice(months, size=len(months), replace=True)
        means = []
        for series in by_basin.values():
            values = series.reindex(sampled).dropna()
            if values.empty:
                break
            means.append(float(values.mean()))
        if len(means) != len(by_basin):
            empty_redraws += 1
            continue
        draws.append(float(np.mean(means)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi), empty_redraws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="experiments/phase4_transfer/stage2a_same_analyte_v2")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.input_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    cells_path = root / "cells.parquet"
    task_path = root / "task_manifest.json"
    if _file_hash(cells_path) != manifest["cells_sha256"]:
        raise SystemExit("cells hash does not match manifest")
    task_manifest = json.loads(task_path.read_text())
    if object_hash(task_manifest) != manifest["task_manifest_sha256"]:
        raise SystemExit("task manifest hash does not match manifest")
    cells = pd.read_parquet(cells_path)
    if cells["visibility_role"].ne("same_analyte_support_diagnostic").any():
        raise SystemExit("unexpected visibility role")
    if cells["config_hash"].nunique() != 1 or cells["config_hash"].iat[0] != manifest["config_hash"]:
        raise SystemExit("config hash mismatch")
    rows, pooled = [], []
    methods = ("climatology", "local_mean", "mean_bias", "analytic_blend")
    for analyte in sorted(cells.analyte.unique()):
        for basin in sorted(cells.loc[cells.analyte.eq(analyte), "basin"].unique()):
            sub = cells[cells.analyte.eq(analyte) & cells.basin.eq(basin)]
            sub = sub[sub.k.isin([0, 5])].copy()
            for method in methods:
                sub["mae"] = np.abs(sub[f"pred_{method}"] - sub.y_true)
                # Average query cells within task-month, then task seeds within
                # calendar month. Seeds are repeatizations, not new clusters.
                task = sub.groupby(["task_seed", "task_index", "month", "k"], as_index=False)["mae"].mean()
                seed_mean = task.groupby(["month", "k"], as_index=False)["mae"].mean()
                wide = seed_mean.pivot(index="month", columns="k", values="mae").dropna(subset=[0, 5])
                wide = wide.reset_index().rename(columns={0: "k0_mae", 5: "k5_mae"})
                wide["delta_mae"] = wide.k5_mae - wide.k0_mae
                estimate, lo, hi = _cluster_bootstrap(wide, args.reps, 42 + len(rows))
                rows.append({
                    "analyte": analyte, "basin": basin, "method": method,
                    "n_basin_month": len(wide),
                    "k0_mae": float(wide.k0_mae.mean()), "k5_mae": float(wide.k5_mae.mean()),
                    "delta_mae": estimate, "ci95_lo": lo, "ci95_hi": hi,
                    "direction_negative": bool(estimate < 0),
                    "ci_excludes_zero": bool(hi < 0),
                })
                wide["analyte"], wide["basin"], wide["method"] = analyte, basin, method
                pooled.append(wide[["analyte", "basin", "month", "delta_mae", "method"]])
    basin_metrics = pd.DataFrame(rows)
    pooled_tasks = pd.concat(pooled, ignore_index=True)
    # Equal basin weights within each analyte; units are never pooled across
    # analytes in this diagnostic.
    pooled_rows = []
    for (analyte, method), group in pooled_tasks.groupby(["analyte", "method"]):
        estimate, lo, hi, redraws = _shared_month_bootstrap(group, args.reps, 1000 + len(pooled_rows))
        sub = basin_metrics[
            basin_metrics.analyte.eq(analyte) & basin_metrics.method.eq(method)
        ]
        pooled_rows.append({
            "analyte": analyte, "method": method,
            "n_basins": int(group.basin.nunique()), "n_unique_months": int(group.month.nunique()),
            "delta_mae": estimate,
            "ci95_lo": lo, "ci95_hi": hi,
            "basins_negative": int(sub.direction_negative.sum()),
            "basins_ci_excludes_zero": int(sub.ci_excludes_zero.sum()),
            "empty_month_redraws": redraws,
        })
    basin_metrics.to_csv(root / "basin_metrics.csv", index=False)
    pooled_metrics = pd.DataFrame(pooled_rows)
    pooled_metrics.to_csv(root / "pooled_metrics.csv", index=False)
    verdict = {
        "version": "phase4_stage2a_same_analyte_evaluation_v2",
        "role": "diagnostic_only_not_heldout_analyte_transfer",
        "cluster": "shared calendar month after averaging task seeds",
        "q90": "not evaluated in this first corrected pass; any tail endpoint requires a source-derived threshold",
        "pooled": "equal basin weights within analyte; no cross-analyte MAE pooling",
        "evaluator_sha256": _file_hash(Path(__file__)),
        "stage2_gate_pass": False,
        "stage3_unlocked": False,
        "reason": "full Stage-2 controls and target-analyte isolation remain incomplete",
    }
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict))


if __name__ == "__main__":
    raise SystemExit(main())
