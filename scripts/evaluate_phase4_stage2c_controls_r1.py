"""Fail-closed Stage-2C evaluation after a completed v1.1 pilot.

This version audits the label-free prediction products against the frozen
task manifest before it opens hidden query labels for scoring.  It writes to a
separate ``evaluation_r1`` directory and never rewrites the original
descriptive v1.1 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
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

PLAN = "experiments/phase4_transfer/stage2c_controls_v1_1/execution_plan.json"
TASKS = "experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json"
NODES = "data/processed/graph_nodes_graphfix_st357.csv"
EDGES = "data/processed/graph_edges_graphfix_st357.csv"
REQUIRED_ARMS = {"ecorf", "h2x_full", "h2x_no_graph", "h2_no_ecology"}
KEY_COLUMNS = ("task_index", "month", "month_index", "k", "flat", "support_count")


def _bootstrap_months(frame: pd.DataFrame, reps: int, seed: int):
    by_basin = {str(b): g.groupby("month")["delta_mae"].mean() for b, g in frame.groupby("basin")}
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


def expected_product_keys(task_manifest: dict, basin: str, seed: int, analyte: str):
    tasks = [
        t
        for t in task_manifest["tasks"]
        if str(t["basin"]) == str(basin)
        and int(t["seed"]) == int(seed)
        and str(t["analyte"]) == str(analyte)
    ]
    return {
        (
            int(t["task_index"]),
            str(t["month"]),
            int(t["month_index"]),
            int(t["k"]),
            int(q),
            int(t["support_count"]),
        )
        for t in tasks
        for q in t["query_cells"]
    }


def validate_prediction_frame(frame: pd.DataFrame, unit: dict, expected: set[tuple]) -> None:
    forbidden = {"y_true", "y", "target", "abs_error", "query_label"}
    if forbidden.intersection(frame.columns):
        raise ValueError(f"label column present: {sorted(forbidden.intersection(frame.columns))}")
    if frame.empty or not np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all():
        raise ValueError(f"empty or non-finite predictions: {unit['unit']}")
    if frame["config_hash"].nunique() != 1 or str(frame["config_hash"].iat[0]) != unit["config_hash"]:
        raise ValueError(f"config identity mismatch: {unit['unit']}")
    if frame["query_labels_used_for_prediction"].astype(bool).any():
        raise ValueError(f"query label flag set: {unit['unit']}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"duplicate product key: {unit['unit']}")
    got = set(frame[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    if got != expected:
        raise ValueError(
            f"task inventory mismatch {unit['unit']}: expected {len(expected)}, got {len(got)}"
        )
    for column, value in (
        ("model_name", unit["arm"]),
        ("analyte", unit["analyte"]),
        ("basin", unit["basin"]),
        ("task_seed", int(unit["seed"])),
    ):
        if not frame[column].astype(str).eq(str(value)).all():
            raise ValueError(f"unit column mismatch {unit['unit']}: {column}")


def audit_label_free_products(root: Path, plan: dict, task_manifest: dict) -> tuple[pd.DataFrame, dict]:
    units = {u["unit"]: u for u in plan["units"]}
    completed = json.loads((root / "completed_units.json").read_text(encoding="utf-8"))
    if len(completed) != int(plan["n_units"]):
        raise ValueError("completed unit count does not match plan")
    frames = []
    selection_hashes = {}
    for item in completed:
        unit = units.get(item["unit"])
        if unit is None or item.get("config_hash") != unit["config_hash"]:
            raise ValueError(f"unknown or mismatched completed unit: {item.get('unit')}")
        pred_path = root / "predictions" / f"{unit['unit']}.parquet"
        selection_path = root / "selection" / f"{unit['unit']}.json"
        if file_hash(pred_path) != item.get("prediction_sha256"):
            raise ValueError(f"prediction hash mismatch: {unit['unit']}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("unit") != unit["unit"] or selection.get("config_hash") != unit["config_hash"]:
            raise ValueError(f"selection identity mismatch: {unit['unit']}")
        if not np.isfinite(float(selection["source_val_mae"])):
            raise ValueError(f"invalid source selection metric: {unit['unit']}")
        selection_hashes[unit["unit"]] = file_hash(selection_path)
        frame = pd.read_parquet(pred_path)
        expected = expected_product_keys(task_manifest, unit["basin"], unit["seed"], unit["analyte"])
        validate_prediction_frame(frame, unit, expected)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    audit = {
        "status": "pass",
        "query_labels_opened": False,
        "n_units": len(completed),
        "n_rows": len(merged),
        "selection_sha256": selection_hashes,
        "checks": [
            "completed_unit_hashes",
            "selection_sidecar_identity",
            "expected_arm_analyte_basin_seed_set",
            "exact_task_index_k_flat_month_support_inventory",
            "duplicate_key_rejection",
            "finite_prediction_values",
            "forbidden_label_columns",
            "query_label_flag_false",
        ],
    }
    (root / "evaluation_r1_pre_score_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return merged, audit


def _primary_metrics(task: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    for (arm, analyte, basin), group in task.groupby(["model_name", "analyte", "basin"]):
        wide = group[group["k"].isin([0, 5])].pivot_table(
            index=["task_seed", "task_index", "month"], columns="k", values="mae"
        ).dropna(subset=[0, 5]).reset_index()
        if wide.empty:
            continue
        wide["delta_mae"] = wide[5] - wide[0]
        estimate, lo, hi = _bootstrap_months(
            wide.assign(basin=basin), reps=reps, seed=42 + len(rows)
        )
        rows.append(
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
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="experiments/phase4_transfer/stage2c_controls_v1_1")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.input_dir)
    eval_root = root / "evaluation_r1"
    eval_root.mkdir(exist_ok=True)
    plan_path = root / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "executed" or not plan.get("training_started"):
        raise SystemExit("execution plan is not finalized")
    task_manifest = json.loads(Path(TASKS).read_text(encoding="utf-8"))
    if file_hash(Path(TASKS)) != plan["task_manifest_sha256"]:
        raise SystemExit("task manifest hash mismatch")
    predictions, audit = audit_label_free_products(root, plan, task_manifest)
    predictions.to_parquet(eval_root / "label_free_predictions_audited.parquet", index=False)

    # Target query values are consumed for scoring only after the label-free
    # product audit succeeds. The runner may have loaded complete arrays for
    # mask validation; this evaluator does not claim those arrays were never
    # loaded, only that query values were not passed to fit or prediction.
    datasets, _summaries, _nodes = load_bundle(DATASETS, NODES, EDGES)
    labels = {a: array(datasets[a]["y"]) for a in ANALYTES}
    predictions["y_true"] = [float(labels[row.analyte].ravel()[int(row.flat)]) for row in predictions.itertuples(index=False)]
    predictions["abs_error"] = np.abs(predictions["y_pred"] - predictions["y_true"])
    task = (
        predictions.groupby(
            ["model_name", "analyte", "basin", "task_seed", "task_index", "month", "k"],
            as_index=False,
        )["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    task.to_csv(eval_root / "task_metrics.csv", index=False)
    k_curve = task.groupby(["model_name", "analyte", "basin", "k"], as_index=False).agg(
        mae=("mae", "mean"), n_task_month=("mae", "size")
    )
    k_curve.to_csv(eval_root / "k_curve.csv", index=False)
    primary = _primary_metrics(task, args.reps)
    primary.to_csv(eval_root / "primary_k5_metrics.csv", index=False)
    own_script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    verdict = {
        "version": "phase4_stage2c_controls_evaluation_v1_1_r1",
        "status": "completed_control_diagnostic_with_fail_closed_inventory",
        "query_labels_used_for_prediction": False,
        "query_labels_opened_for_scoring": True,
        "n_units": int(plan["n_units"]),
        "n_rows": len(predictions),
        "arms": sorted(predictions["model_name"].unique().tolist()),
        "analytes": sorted(predictions["analyte"].unique().tolist()),
        "primary_rows": len(primary),
        "stage2_unlocked": False,
        "stage3_unlocked": False,
        "missing_required": [
            "information_decomposition",
            "support_value_shuffle",
            "support_site_shuffle",
            "source_selected_baseline_gate",
        ],
        "execution_plan_sha256_finalized": file_hash(plan_path),
        "original_plan_ack_sha256": json.loads((root / "manifest.json").read_text())["execution_plan_sha256"],
        "evaluation_script_sha256": own_script_sha,
        "task_manifest_sha256": file_hash(Path(TASKS)),
        "node_sha256": file_hash(NODES),
        "edge_sha256": file_hash(EDGES),
        "visibility_schema_limitation": "historical product role cross_basin_target_analyte_source_visible is retained; it is not the route role enum",
        "note": "This is a fail-closed control diagnostic. It does not by itself pass the Stage-2 feasibility gate.",
        "pre_score_audit": audit,
    }
    (eval_root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    (eval_root / "manifest.json").write_text(
        json.dumps(
            {
                "version": verdict["version"],
                "verdict_sha256": hashlib.sha256((eval_root / "verdict.json").read_bytes()).hexdigest(),
                "task_manifest_sha256": verdict["task_manifest_sha256"],
                "execution_plan_sha256_finalized": verdict["execution_plan_sha256_finalized"],
                "evaluation_script_sha256": own_script_sha,
                "query_labels_opened_for_scoring": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
