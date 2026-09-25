"""Audit and score the frozen Stage-2 support-integrity products.

The evaluator first checks label-free products and shuffle identities.  Only
after that audit passes does it open target query labels for one final score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.support_integrity import (
    SHUFFLE_MODES,
)
from river_graph.experiments.transfer import (
    ANALYTES,
    DATASETS,
    array,
    file_hash,
    load_bundle,
    object_hash,
)

TASKS = Path("experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
EDGES = Path("data/processed/graph_edges_graphfix_st357.csv")


def _expected_records(task_manifest: dict, basin: str, seed: int, analyte: str):
    return [
        t
        for t in task_manifest["tasks"]
        if str(t["basin"]) == str(basin)
        and int(t["seed"]) == int(seed)
        and str(t["analyte"]) == str(analyte)
    ]


def _audit_products(root: Path, plan: dict, task_manifest: dict, shuffle: dict) -> tuple[pd.DataFrame, dict]:
    if object_hash(shuffle) != plan["shuffle_manifest_sha256"]:
        raise ValueError("shuffle manifest hash mismatch")
    if file_hash(TASKS) != plan["task_manifest_sha256"]:
        raise ValueError("task manifest hash mismatch")
    completed = json.loads((root / "completed_units.json").read_text(encoding="utf-8"))
    if len(completed) != int(plan["n_units"]):
        raise ValueError("completed unit count does not match frozen plan")
    units = {u["unit"]: u for u in plan["units"]}
    shuffle_records = {
        (str(r["analyte"]), str(r["basin"]), int(r["seed"]), int(r["task_index"]), int(r["k"]), str(r["mode"])): r
        for r in shuffle["records"]
    }
    frames = []
    forbidden = {"y_true", "y", "target", "abs_error", "query_label"}
    for item in completed:
        unit = units.get(item["unit"])
        if unit is None or item.get("config_hash") != unit["config_hash"]:
            raise ValueError(f"unknown or mismatched unit: {item.get('unit')}")
        pred_path = root / "predictions" / f"{unit['unit']}.parquet"
        side_path = root / "sidecars" / f"{unit['unit']}.json"
        if file_hash(pred_path) != item.get("prediction_sha256") or file_hash(side_path) != item.get("sidecar_sha256"):
            raise ValueError(f"artifact hash mismatch: {unit['unit']}")
        side = json.loads(side_path.read_text(encoding="utf-8"))
        for key in ("config_hash", "task_manifest_sha256", "shuffle_manifest_sha256", "support_integrity_spec_sha256", "runtime_snapshot_hash"):
            if side.get(key) != (unit["config_hash"] if key == "config_hash" else plan.get(key.replace("support_integrity_spec_sha256", "support_integrity_spec_sha256"))):
                # The explicit checks below give clearer messages for fields
                # that are not copied into the plan under the same name.
                if key == "task_manifest_sha256" and side.get(key) == plan[key]:
                    continue
                if key == "shuffle_manifest_sha256" and side.get(key) == plan[key]:
                    continue
                if key == "support_integrity_spec_sha256" and side.get(key) == plan[key]:
                    continue
                if key == "runtime_snapshot_hash" and side.get(key) == plan[key]:
                    continue
                raise ValueError(f"sidecar identity mismatch: {unit['unit']} {key}")
        frame = pd.read_parquet(pred_path)
        if forbidden.intersection(frame.columns):
            raise ValueError(f"forbidden label column in {unit['unit']}")
        if frame["config_hash"].nunique() != 1 or str(frame["config_hash"].iat[0]) != unit["config_hash"]:
            raise ValueError(f"config hash mismatch in {unit['unit']}")
        if frame["query_labels_used_for_prediction"].astype(bool).any():
            raise ValueError(f"query-label flag set in {unit['unit']}")
        if set(frame["support_mode"].unique()) != set(SHUFFLE_MODES):
            raise ValueError(f"support mode inventory mismatch in {unit['unit']}")
        expected = _expected_records(task_manifest, unit["basin"], int(unit["seed"]), unit["analyte"])
        expected_keys = {
            (int(t["task_index"]), int(t["k"]), int(q), mode)
            for t in expected
            for q in t["query_cells"]
            for mode in SHUFFLE_MODES
        }
        got_keys = set(frame[["task_index", "k", "flat", "support_mode"]].itertuples(index=False, name=None))
        if got_keys != expected_keys:
            raise ValueError(f"task inventory mismatch in {unit['unit']}")
        for row in frame.itertuples(index=False):
            key = (unit["analyte"], unit["basin"], int(unit["seed"]), int(row.task_index), int(row.k), str(row.support_mode))
            record = shuffle_records.get(key)
            if row.support_mode != "true" and record is None:
                raise ValueError(f"missing shuffle record: {key}")
            status = "identifiable" if row.support_mode == "true" else record["status"]
            if status == "identifiable" and not np.isfinite(float(row.y_pred)):
                raise ValueError(f"non-finite identifiable prediction: {key}")
            if status != "identifiable" and np.isfinite(float(row.y_pred)):
                raise ValueError(f"non-identifiable prediction has a value: {key}")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    audit = {
        "status": "pass",
        "query_labels_opened": False,
        "n_units": len(completed),
        "n_rows": len(merged),
        "checks": [
            "completed_unit_hashes",
            "sidecar_identity",
            "forbidden_label_columns",
            "query_label_flag_false",
            "exact_task_mode_query_inventory",
            "identifiable_shuffle_predictions_finite",
            "non_identifiable_shuffle_predictions_null",
        ],
    }
    (root / "evaluation_pre_score_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return merged, audit


def _bootstrap_delta(frame: pd.DataFrame, *, reps: int, seed: int) -> tuple[float, float, float]:
    by_basin = {str(b): g.groupby("month")["delta_mae"].mean() for b, g in frame.groupby("basin")}
    months = sorted(set().union(*(set(s.index) for s in by_basin.values())))
    observed = float(np.mean([s.mean() for s in by_basin.values()]))
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < reps:
        sample = rng.choice(months, len(months), replace=True)
        means = [series.reindex(sample).dropna().mean() for series in by_basin.values()]
        if all(np.isfinite(means)):
            draws.append(float(np.mean(means)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return observed, float(lo), float(hi)


def _score(root: Path, predictions: pd.DataFrame, *, reps: int) -> dict:
    datasets, _summaries, _nodes = load_bundle(DATASETS, NODES, EDGES)
    labels = {a: array(datasets[a]["y"]).astype(float) for a in ANALYTES}
    predictions = predictions.copy()
    predictions["y_true"] = [float(labels[r.analyte].ravel()[int(r.flat)]) for r in predictions.itertuples(index=False)]
    predictions["abs_error"] = np.abs(predictions["y_pred"] - predictions["y_true"])
    finite = predictions[predictions["y_pred"].notna()].copy()
    task = (
        finite.groupby(["support_mode", "analyte", "basin", "task_seed", "task_index", "month", "k"], as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    task.to_csv(root / "task_metrics.csv", index=False)
    rows = []
    ident_rows = []
    for mode in ("value_shuffle", "site_shuffle"):
        for analyte in ANALYTES:
            for basin in sorted(task["basin"].unique()):
                true = task[(task["support_mode"] == "true") & (task["analyte"] == analyte) & (task["basin"] == basin) & (task["k"] == 5)]
                shuffled = task[(task["support_mode"] == mode) & (task["analyte"] == analyte) & (task["basin"] == basin) & (task["k"] == 5)]
                key = ["task_seed", "task_index", "month"]
                wide = true.merge(shuffled, on=key, suffixes=("_true", "_shuffle")).copy()
                if wide.empty:
                    continue
                wide["delta_mae"] = wide["mae_true"] - wide["mae_shuffle"]
                # This comparison is already within one fixed basin.  Keep an
                # explicit basin column so the shared bootstrap helper retains
                # its HUC6-stratified interface.
                wide["basin"] = basin
                estimate, lo, hi = _bootstrap_delta(wide, reps=reps, seed=700 + len(rows))
                rows.append({
                    "support_mode": mode,
                    "analyte": analyte,
                    "basin": basin,
                    "true_mae": float(wide["mae_true"].mean()),
                    "shuffle_mae": float(wide["mae_shuffle"].mean()),
                    "delta_mae_true_minus_shuffle": estimate,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "relative_reduction_pct": float(-estimate / wide["mae_shuffle"].mean() * 100),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "true_better": bool(estimate < 0),
                    "n_task_month": len(wide),
                    "unstable_n_lt20": bool(len(wide) < 20),
                })
        available = predictions[(predictions["support_mode"] == mode) & predictions["y_pred"].notna() & (predictions["k"] == 5)]
        total = predictions[(predictions["support_mode"] == mode) & (predictions["k"] == 5)]
        ident_rows.append({"support_mode": mode, "identifiable_query_rows": len(available), "total_query_rows": len(total), "identifiable_fraction": float(len(available) / max(len(total), 1))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "shuffle_metrics.csv", index=False)
    pd.DataFrame(ident_rows).to_csv(root / "identifiability.csv", index=False)
    pooled = []
    for (mode, analyte), group in metrics.groupby(["support_mode", "analyte"]):
        stable = group[~group["unstable_n_lt20"]].copy()
        if stable.empty:
            continue
        # Native-unit deltas are only averaged within an analyte.  DOC, pH,
        # and conductance are never combined into one numeric mean.
        estimate = float(stable["delta_mae_true_minus_shuffle"].mean())
        pooled.append({
            "support_mode": mode,
            "analyte": analyte,
            "mean_cell_delta_mae": estimate,
            "cells_true_better": int(stable["true_better"].sum()),
            "cells_ci_excludes_zero": int(stable["ci_excludes_zero"].sum()),
            "stable_cells": len(stable),
            "unstable_cells_excluded": int(group["unstable_n_lt20"].sum()),
        })
    verdict = {
        "version": "phase4_stage2_support_integrity_v1",
        "status": "completed_bounded_audit",
        "query_labels_opened_after_pre_score_audit": True,
        "support_integrity_gate": "descriptive_only",
        "stage2_unlocked": False,
        "stage3_unlocked": False,
        "interpretation": "Support-shuffle evidence is reported as an integrity audit. It cannot erase the previously frozen DOC no-harm failure or unlock a new architecture/phase.",
        "pooled_cell_summary": pooled,
        "task_metrics_sha256": file_hash(root / "task_metrics.csv"),
        "shuffle_metrics_sha256": file_hash(root / "shuffle_metrics.csv"),
    }
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="experiments/phase4_transfer/stage2_support_integrity_v1")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    if args.reps < 100:
        raise SystemExit("--reps must be at least 100")
    root = Path(args.input_dir)
    plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
    if plan.get("status") != "executed":
        raise SystemExit("execution plan is not finalized")
    task_manifest = json.loads(TASKS.read_text(encoding="utf-8"))
    shuffle = json.loads((root / "shuffle_manifest.json").read_text(encoding="utf-8"))
    predictions, _audit = _audit_products(root, plan, task_manifest, shuffle)
    predictions.to_parquet(root / "label_free_predictions_audited.parquet", index=False)
    verdict = _score(root, predictions, reps=args.reps)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
