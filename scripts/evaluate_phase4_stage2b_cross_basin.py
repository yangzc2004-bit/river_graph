"""Evaluate the frozen no-training cross-basin baseline products.

Only this evaluator opens ``y_true`` from target query rows.  It writes
descriptive baseline diagnostics and deliberately does not unlock a model
training stage: EcoRF, H2X, and matched no-graph/no-ecology controls remain
separate required arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.transfer import file_hash, object_hash

SUPPORT_METHODS = ("local_mean", "mean_bias", "analytic_blend")


def _canonical_basin(value) -> str:
    code = str(value)
    return code.zfill(6) if code.isdigit() else code


def _task_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["abs_error"] = np.abs(frame["y_pred"] - frame["y_true"])
    frame["log1p_abs_error"] = np.abs(
        np.log1p(np.maximum(frame["y_pred"], 0.0))
        - np.log1p(np.maximum(frame["y_true"], 0.0))
    )
    keys = ["analyte", "basin", "task_seed", "task_index", "month", "k", "model_name"]
    return frame.groupby(keys, as_index=False).agg(
        mae=("abs_error", "mean"), log1p_mae=("log1p_abs_error", "mean"), n=("y_true", "size")
    )


def _bootstrap_months(
    frame: pd.DataFrame, *, value: str, reps: int, seed: int
) -> tuple[float, float, float]:
    """Shared calendar-month cluster bootstrap, equal basin weights."""
    by_basin = {
        str(basin): group.groupby("month", as_index=False)[value].mean().set_index("month")[value]
        for basin, group in frame.groupby("basin")
    }
    months = sorted(set().union(*(set(series.index) for series in by_basin.values())))
    if not months or not by_basin:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean([series.mean() for series in by_basin.values()]))
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < reps:
        sampled = rng.choice(months, size=len(months), replace=True)
        basin_means = []
        for series in by_basin.values():
            values = series.reindex(sampled).dropna()
            if values.empty:
                break
            basin_means.append(float(values.mean()))
        if len(basin_means) == len(by_basin):
            draws.append(float(np.mean(basin_means)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="experiments/phase4_transfer/stage2b_cross_basin_v1")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.input_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    predictions_path = root / "predictions.parquet"
    if file_hash(predictions_path) != manifest["predictions_sha256"]:
        raise SystemExit("prediction hash does not match manifest")
    predictions = pd.read_parquet(predictions_path)
    if predictions.empty:
        raise SystemExit("prediction product is empty")
    if predictions["query_labels_used_for_prediction"].astype(bool).any():
        raise SystemExit("query labels were marked as used during prediction")
    if predictions["visibility_role"].nunique() != 1:
        raise SystemExit("unexpected visibility roles")
    if predictions["source_q90_threshold"].isna().any():
        raise SystemExit("missing source-derived Q90 threshold")
    predictions["analyte"] = predictions["analyte"].astype(str)
    predictions["basin"] = predictions["basin"].map(_canonical_basin)
    selected_path = root / "selected_baselines.csv"
    if file_hash(selected_path) != manifest["selected_baselines_sha256"]:
        raise SystemExit("selected baseline hash does not match manifest")
    selected = pd.read_csv(selected_path)
    selected["analyte"] = selected["analyte"].astype(str)
    selected["basin"] = selected["basin"].map(_canonical_basin)
    if selected[["analyte", "basin"]].duplicated().any():
        raise SystemExit("duplicate selected baseline")
    task = _task_mae(predictions)
    task.to_csv(root / "task_metrics.csv", index=False)

    # K curve is kept per analyte and HUC6; native-unit errors are never pooled
    # across analytes.
    k_curve = (
        task.groupby(["analyte", "basin", "k", "model_name"], as_index=False)
        .agg(mae=("mae", "mean"), log1p_mae=("log1p_mae", "mean"), n_task_month=("mae", "size"))
    )
    k_curve.to_csv(root / "k_curve.csv", index=False)

    # Source-selected baseline is joined by analyte/HUC6 and remains constant
    # over K. The comparison is paired at each fixed task-month.
    baseline_method = selected.rename(columns={"selected_method": "baseline_method"})[
        ["analyte", "basin", "baseline_method"]
    ]
    task_key = ["analyte", "basin", "task_seed", "task_index", "month", "k"]
    base = task.merge(baseline_method, on=["analyte", "basin"], how="inner")
    base = base[base.model_name.eq(base.baseline_method)]
    if base.empty:
        raise SystemExit("selected baseline did not match prediction metrics")
    base = base.rename(columns={"mae": "baseline_mae", "log1p_mae": "baseline_log1p_mae"})[
        task_key + ["baseline_method", "baseline_mae", "baseline_log1p_mae"]
    ]
    adaptations = task[task.model_name.isin(SUPPORT_METHODS)].merge(base, on=task_key, how="inner")
    adaptations["delta_mae"] = adaptations["mae"] - adaptations["baseline_mae"]
    adaptations["relative_reduction_pct"] = np.where(
        adaptations["baseline_mae"] > 0,
        -adaptations["delta_mae"] / adaptations["baseline_mae"] * 100.0,
        np.nan,
    )
    primary_rows = []
    for (analyte, basin, method), group in adaptations[adaptations.k.eq(5)].groupby(
        ["analyte", "basin", "model_name"]
    ):
        estimate, lo, hi = _bootstrap_months(
            group, value="delta_mae", reps=args.reps, seed=42 + len(primary_rows)
        )
        baseline_mae = float(group["baseline_mae"].mean())
        primary_rows.append(
            {
                "analyte": analyte,
                "basin": basin,
                "method": method,
                "n_task_month": len(group),
                "baseline_method": str(group["baseline_method"].iat[0]),
                "baseline_mae": baseline_mae,
                "adapted_mae": float(group.mae.mean()),
                "delta_mae": estimate,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "relative_reduction_pct": float(-estimate / baseline_mae * 100) if baseline_mae else None,
                "direction_improves": bool(estimate < 0),
                "ci_excludes_zero": bool(hi < 0),
            }
        )
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(root / "primary_k5_metrics.csv", index=False)

    # Source-derived tail diagnostic. The threshold is never chosen from the
    # query values; n<20 is explicitly unstable as required by the protocol.
    tail = predictions[predictions.y_true >= predictions.source_q90_threshold].copy()
    tail_metrics = (
        tail.groupby(["analyte", "basin", "k", "model_name"], as_index=False)
        .agg(tail_mae=("y_pred", lambda x: float(np.mean(np.abs(x - tail.loc[x.index, "y_true"])))), n_tail=("y_true", "size"))
    )
    if not tail_metrics.empty:
        tail_metrics["unstable_n_lt_20"] = tail_metrics.n_tail < 20
    tail_metrics.to_csv(root / "tail_metrics.csv", index=False)

    verdict = {
        "version": "phase4_stage2b_cross_basin_evaluation_v1",
        "role": "nonlearning_baseline_diagnostic_only",
        "methods": list(manifest["methods"]),
        "primary_comparison": "K=5 support baseline versus source-validation-selected source-only climatology",
        "cluster": "calendar month after task-level query averaging; equal HUC6 weights",
        "tail_threshold": "source-train Q90 per analyte/basin/seed; query truth only scores membership",
        "training_started": False,
        "stage2_unlocked": False,
        "stage3_unlocked": False,
        "baseline_signal_gate": "descriptive_only_until_EcoRF_H2X_and_matched_controls_are_complete",
        "missing_required_arms": ["EcoRF", "single_analyte_H2X", "no_graph", "no_ecology", "missingness_decomposition"],
        "primary_rows": len(primary),
        "evaluator_sha256": file_hash(Path(__file__)),
        "manifest_hash": object_hash(manifest),
    }
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (root / "STATUS.md").write_text(
        "# Stage 2B cross-basin baseline status\n\n"
        "This artifact is a deterministic, no-training diagnostic. It uses the "
        "frozen cross-basin task roles and source-validation baseline selection. "
        "It does not unlock Stage 3: EcoRF, single-analyte H2X, matched no-graph "
        "and no-ecology controls, and the missingness decomposition are still required.\n\n"
        "Target query labels are opened only by this final evaluator for scoring; "
        "they are not used in prediction or selection.\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
