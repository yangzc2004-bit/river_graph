"""Evaluate the frozen Stage-2 transfer gate from existing metric tables.

This is a metrics-only gate report.  It consumes Stage-2B deterministic
baselines and Stage-2C control metrics, never datasets or prediction rows, and
therefore cannot create support-shuffle evidence that was not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from river_graph.experiments.transfer import file_hash

try:
    from scripts.evaluate_phase4_stage2_information_decomposition import (
        TASK_KEYS,
        _pair,
        _read_metrics,
        _summary,
    )
except ModuleNotFoundError:  # direct execution from the scripts directory
    from evaluate_phase4_stage2_information_decomposition import (
        TASK_KEYS,
        _pair,
        _read_metrics,
        _summary,
    )

BASELINE_DIR = Path("experiments/phase4_transfer/stage2b_cross_basin_v1")
CONTROLS_DIR = Path("experiments/phase4_transfer/stage2c_controls_v1_1")
OUT_DIR = Path("experiments/phase4_transfer/stage2_gate_v1")
TASK_MANIFEST = Path("experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json")


def _selected_baseline_pair(baselines: pd.DataFrame, controls: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for choice in selected.itertuples(index=False):
        analyte, basin, method = str(choice.analyte), str(choice.basin).zfill(6), str(choice.selected_method)
        comparator = controls[
            controls["analyte"].eq(analyte)
            & controls["basin"].eq(basin)
            & controls["model_name"].eq("h2x_full")
            & controls["k"].eq(5)
        ]
        reference = baselines[
            baselines["analyte"].eq(analyte)
            & baselines["basin"].eq(basin)
            & baselines["model_name"].eq(method)
            & baselines["k"].eq(0)
        ]
        rows.append(
            _pair(
                comparator,
                reference,
                comparator_name="h2x_full_k5",
                reference_name=f"selected_{method}_k0",
                label="h2x_k5_vs_source_selected_simple_baseline",
                k=5,
                ignore_k=True,
            )
        )
    if not rows:
        raise ValueError("selected baseline table is empty")
    return pd.concat(rows, ignore_index=True)


def _k_curve_checks(controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    h2x = controls[controls["model_name"].eq("h2x_full")]
    for analyte, group in h2x.groupby("analyte", sort=True):
        by_basin = group.groupby(["basin", "k"], as_index=False)["mae"].mean()
        pooled = by_basin.groupby("k")["mae"].mean()
        rows.append(
            {
                "analyte": analyte,
                "check": "h2x_k5_no_worse_than_k0_and_k1",
                "k0_mae": float(pooled.loc[0]),
                "k1_mae": float(pooled.loc[1]),
                "k5_mae": float(pooled.loc[5]),
                "pass": bool(pooled.loc[5] <= pooled.loc[0] and pooled.loc[5] <= pooled.loc[1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--controls-dir", default=str(CONTROLS_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    if args.reps < 100:
        raise SystemExit("--reps must be at least 100")
    baseline_root, controls_root, out = map(Path, (args.baseline_dir, args.controls_dir, args.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    baselines = _read_metrics(baseline_root / "task_metrics.csv", "Stage-2B")
    controls = _read_metrics(controls_root / "task_metrics.csv", "Stage-2C")
    if set(map(tuple, baselines[TASK_KEYS].drop_duplicates().to_numpy())) != set(
        map(tuple, controls[TASK_KEYS].drop_duplicates().to_numpy())
    ):
        raise ValueError("Stage-2B and Stage-2C task inventories differ")
    selected = pd.read_csv(baseline_root / "selected_baselines.csv")
    if len(selected) != 15 or selected[["analyte", "basin"]].duplicated().any():
        raise ValueError("selected-baseline table must contain one row per analyte and basin")
    pair = _selected_baseline_pair(baselines, controls, selected)
    rows = _summary(
        pair,
        label="h2x_k5_vs_source_selected_simple_baseline",
        value="delta_mae",
        comparator="h2x_full_k5",
        reference="source_selected_simple_baseline_k0",
        reps=args.reps,
        seed_start=900,
        k=5,
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "gate_metrics.csv", index=False)
    curve = _k_curve_checks(controls)
    curve.to_csv(out / "k_curve_checks.csv", index=False)
    pooled = metrics[metrics["scope"].eq("analyte_pooled")].copy()
    pooled_pass = pooled[
        pooled["relative_reduction_pct"].ge(10.0) & pooled["ci95_hi"].lt(0)
    ]["analyte"].tolist()
    direction = (
        metrics[metrics["scope"].eq("analyte_basin")]
        .groupby("analyte")["direction_improves"]
        .sum()
    )
    direction_pass = direction[direction.ge(3)].index.tolist()
    no_harm = bool(
        metrics[metrics["scope"].eq("analyte_basin")]["relative_reduction_pct"].ge(-5.0).all()
    )
    verdict = {
        "version": "phase4_stage2_gate_v1",
        "status": "completed_metrics_only_gate_report",
        "query_labels_opened_by_this_evaluator": False,
        "primary_relative_reduction_pass_analytes": pooled_pass,
        "primary_direction_pass_analytes": direction_pass,
        "k_curve_pass_analytes": curve.loc[curve["pass"], "analyte"].tolist(),
        "no_harm_pass": no_harm,
        "support_value_shuffle": "not_run",
        "support_site_shuffle": "not_run",
        "stage2_unlocked": False,
        "stage3_unlocked": False,
        "missing_or_failed_requirements": [
            "support_value_shuffle",
            "support_site_shuffle",
        ],
        "interpretation": "H2X K=5 versus the source-selected simple baseline is reported; the feasibility gate remains locked until support integrity controls are run and all no-harm criteria are satisfied.",
        "baseline_metrics_sha256": file_hash(baseline_root / "task_metrics.csv"),
        "selected_baselines_sha256": file_hash(baseline_root / "selected_baselines.csv"),
        "controls_metrics_sha256": file_hash(controls_root / "task_metrics.csv"),
        "task_manifest_sha256": file_hash(TASK_MANIFEST),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (out / "STATUS.md").write_text(
        "# Stage 2 transfer gate\n\n"
        "Metrics-only report under the frozen same-month spatial HUC6 task. "
        "The H2X K=5 comparison to source-selected simple baselines is "
        "reported, but support value/site shuffles were not run and the gate "
        "remains locked.\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
