"""Evaluate the frozen, no-training Stage-2 baseline products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_equal_basin(tasks: pd.DataFrame, seed: int, reps: int = 2000) -> tuple[float, float, float]:
    """Cluster bootstrap task-months, giving each HUC6 equal pooled weight."""
    rng = np.random.default_rng(seed)
    by_basin = {b: g["delta"].to_numpy(float) for b, g in tasks.groupby("basin")}
    observed = float(np.mean([v.mean() for v in by_basin.values()]))
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        draws[i] = np.mean([
            values[rng.integers(0, len(values), size=len(values))].mean()
            for values in by_basin.values()
        ])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/phase4_transfer/stage2_baselines_v1/all_cells.parquet")
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/stage2_baselines_v1")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    cells = pd.read_parquet(args.input)
    rows, pooled, gate = [], [], {}
    for analyte in sorted(cells.analyte.unique()):
        sub_a = cells[cells.analyte.eq(analyte)]
        for basin in sorted(sub_a.basin.unique()):
            sub = sub_a[sub_a.basin.eq(basin)]
            # Collapse each fixed task-month to one paired error before the
            # cluster bootstrap. Query cells within a task are not independent.
            task = (
                sub[sub.k.isin([0, 5])]
                .groupby(["task_seed", "task_index", "month", "k"], as_index=False)
                .apply(lambda x: pd.Series({"mae": np.mean(np.abs(x.pred_analytic_blend - x.y_true))}))
                .reset_index(drop=True)
            )
            wide = task.pivot_table(index=["task_seed", "task_index", "month"], columns="k", values="mae")
            wide = wide.dropna(subset=[0, 5]).reset_index()
            wide["delta"] = wide[5] - wide[0]
            wide["basin"] = basin
            estimate, lo, hi = bootstrap_equal_basin(wide, seed=42 + len(rows), reps=args.reps)
            k0 = float(wide[0].mean())
            basin_row = {
                "analyte": analyte, "basin": basin, "n_task_month": len(wide),
                "k0_mae": k0, "k5_mae": float(wide[5].mean()),
                "delta_mae": estimate, "ci95_lo": lo, "ci95_hi": hi,
                "relative_reduction_pct": float(-estimate / k0 * 100) if k0 else None,
                "direction_negative": bool(estimate < 0),
                "ci_excludes_zero": bool(hi < 0),
            }
            rows.append(basin_row)
        basin_frame = pd.DataFrame([r for r in rows if r["analyte"] == analyte])
        task_all = []
        for basin in sorted(sub_a.basin.unique()):
            sub = sub_a[sub_a.basin.eq(basin) & sub_a.k.isin([0, 5])]
            task = (
                sub.groupby(["basin", "task_seed", "task_index", "month", "k"], as_index=False)
                .apply(lambda x: pd.Series({"mae": np.mean(np.abs(x.pred_analytic_blend - x.y_true))}))
                .reset_index(drop=True)
            )
            wide = task.pivot_table(index=["basin", "task_seed", "task_index", "month"], columns="k", values="mae")
            wide = wide.dropna(subset=[0, 5]).reset_index()
            wide["delta"] = wide[5] - wide[0]
            task_all.append(wide[["basin", "delta"]])
        all_tasks = pd.concat(task_all, ignore_index=True)
        estimate, lo, hi = bootstrap_equal_basin(all_tasks, seed=1000 + len(pooled), reps=args.reps)
        pooled.append({
            "analyte": analyte, "n_basins": basin_frame.shape[0],
            "delta_mae": estimate, "ci95_lo": lo, "ci95_hi": hi,
            "basins_negative": int(basin_frame.direction_negative.sum()),
            "basins_ci_excludes_zero": int(basin_frame.ci_excludes_zero.sum()),
            "gate_pass": bool(hi < 0 and basin_frame.ci_excludes_zero.sum() >= 3),
        })
        gate[analyte] = pooled[-1]["gate_pass"]
    pd.DataFrame(rows).to_csv(out / "basin_metrics.csv", index=False)
    pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
    report = {
        "version": "phase4_stage2_baseline_evaluation_v1",
        "input": str(args.input), "bootstrap_reps": args.reps,
        "primary_method": "analytic_blend", "cluster": "task-month",
        "pooling": "equal HUC6 basin weights",
        "analyte_gate": gate, "stage2_gate_pass": sum(gate.values()) >= 2,
        "training_started": False,
    }
    (out / "baseline_verdict.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
