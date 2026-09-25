"""Evaluate completed Stage-2C control products after finalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.transfer import (
    ANALYTES,
    DATASETS,
    array,
    file_hash,
    load_bundle,
)


def _bootstrap_months(frame: pd.DataFrame, reps: int = 2000, seed: int = 42):
    by_basin = {
        str(b): g.groupby("month")["delta_mae"].mean()
        for b, g in frame.groupby("basin")
    }
    months = sorted(set().union(*(set(s.index) for s in by_basin.values())))
    observed = float(np.mean([s.mean() for s in by_basin.values()]))
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < reps:
        sample = rng.choice(months, len(months), replace=True)
        means = []
        for series in by_basin.values():
            values = series.reindex(sample).dropna()
            if values.empty:
                break
            means.append(float(values.mean()))
        if len(means) == len(by_basin):
            draws.append(float(np.mean(means)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="experiments/phase4_transfer/stage2c_controls_v1_1")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.input_dir)
    plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
    if plan.get("status") != "executed" or not plan.get("training_started"):
        raise SystemExit("Stage-2C execution plan is not finalized")
    completed = json.loads((root / "completed_units.json").read_text(encoding="utf-8"))
    if len(completed) != int(plan["n_units"]):
        raise SystemExit("completed unit count does not match the execution plan")
    units = {u["unit"]: u for u in plan["units"]}
    frames = []
    for item in completed:
        unit = units.get(item["unit"])
        if unit is None or item["config_hash"] != unit["config_hash"]:
            raise SystemExit(f"unknown or mismatched completed unit: {item.get('unit')}")
        path = root / "predictions" / f"{item['unit']}.parquet"
        if file_hash(path) != item["prediction_sha256"]:
            raise SystemExit(f"prediction hash mismatch: {item['unit']}")
        frame = pd.read_parquet(path)
        if frame.empty or frame["config_hash"].nunique() != 1:
            raise SystemExit(f"invalid prediction product: {item['unit']}")
        if frame["config_hash"].iat[0] != unit["config_hash"]:
            raise SystemExit(f"prediction config mismatch: {item['unit']}")
        if frame["query_labels_used_for_prediction"].astype(bool).any():
            raise SystemExit(f"query label flag set: {item['unit']}")
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    datasets, summaries, _nodes = load_bundle(
        DATASETS,
        "data/processed/graph_nodes_graphfix_st357.csv",
        "data/processed/graph_edges_graphfix_st357.csv",
    )
    expected_dataset_hash = {
        u["analyte"]: u["config"]["dataset_sha256"] for u in plan["units"]
    }
    for analyte in ANALYTES:
        if summaries[analyte]["sha256"] != expected_dataset_hash[analyte]:
            raise SystemExit(f"dataset hash mismatch: {analyte}")
    labels = {a: array(datasets[a]["y"]) for a in ANALYTES}
    # Query labels are opened only here, for final scoring.
    predictions["y_true"] = [
        float(labels[row.analyte].ravel()[int(row.flat)])
        for row in predictions.itertuples(index=False)
    ]
    predictions["abs_error"] = np.abs(predictions["y_pred"] - predictions["y_true"])
    task = (
        predictions.groupby(
            ["model_name", "analyte", "basin", "task_seed", "task_index", "month", "k"],
            as_index=False,
        )["abs_error"].mean()
        .rename(columns={"abs_error": "mae"})
    )
    task.to_csv(root / "task_metrics.csv", index=False)
    k_curve = task.groupby(["model_name", "analyte", "basin", "k"], as_index=False).agg(
        mae=("mae", "mean"), n_task_month=("mae", "size")
    )
    k_curve.to_csv(root / "k_curve.csv", index=False)
    primary = []
    for (arm, analyte, basin), group in task.groupby(["model_name", "analyte", "basin"]):
        wide = group[group["k"].isin([0, 5])].pivot_table(
            index=["task_seed", "task_index", "month"], columns="k", values="mae"
        ).dropna(subset=[0, 5]).reset_index()
        if wide.empty:
            continue
        wide["delta_mae"] = wide[5] - wide[0]
        estimate, lo, hi = _bootstrap_months(
            wide.assign(basin=basin), reps=args.reps, seed=42 + len(primary)
        )
        primary.append(
            {
                "model_name": arm,
                "analyte": analyte,
                "basin": basin,
                "k0_mae": float(wide[0].mean()),
                "k5_mae": float(wide[5].mean()),
                "delta_mae": estimate,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "relative_reduction_pct": float(-estimate / wide[0].mean() * 100),
                "direction_improves": bool(estimate < 0),
                "ci_excludes_zero": bool(hi < 0),
                "n_task_month": len(wide),
            }
        )
    primary_frame = pd.DataFrame(primary)
    primary_frame.to_csv(root / "primary_k5_metrics.csv", index=False)
    source_selection = pd.read_csv(root / "source_selection.csv")
    verdict = {
        "version": "phase4_stage2c_controls_evaluation_v1_1",
        "status": "completed_control_diagnostic",
        "training_started": True,
        "query_labels_used_for_prediction": False,
        "query_labels_opened_for_scoring": True,
        "n_units": len(completed),
        "arms": sorted(predictions["model_name"].unique().tolist()),
        "analytes": sorted(predictions["analyte"].unique().tolist()),
        "primary_rows": len(primary_frame),
        "source_selection_rows": len(source_selection),
        "stage2_unlocked": False,
        "stage3_unlocked": False,
        "missing_required": ["missingness_decomposition"],
        "note": "Control results are descriptive until the frozen information decomposition is complete.",
    }
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
