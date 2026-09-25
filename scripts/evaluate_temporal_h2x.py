"""Audit H2X-T run products and compare matched snapshot/temporal runs.

The evaluator is intentionally metrics-only after the run files exist.  It
recomputes test metrics from each full-grid parquet, verifies V3 provenance,
and writes a compact comparison table for the T2 pilot or T3 formal run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import config_hash, run_identity_sha256


def _expected_runs(plan: dict) -> list[str]:
    return [
        f"{model}__{analyte}__{mask}__seed{seed}"
        for analyte in plan["analytes"]
        for mask in plan["masks"]
        for seed in plan["seeds"]
        for model in plan["models"]
    ]


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return bool(np.isclose(a, b, rtol=tol, atol=tol, equal_nan=True))


def audit_run(root: Path, run: str) -> dict:
    run_dir = root / "runs" / run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(run_dir / "full_grid.parquet")
    test = frame[frame["split"].eq("test")]
    if test.empty or test["y_true"].isna().any():
        raise ValueError(f"{run}: test rows are missing finite labels")
    recomputed = metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
    metric_keys = ("rmse", "mae", "r2", "log_rmse", "log_mae", "log_r2", "pbias", "n")
    drift = [key for key in metric_keys if not _close(float(stored[key]), float(recomputed[key]))]
    version = int(meta.get("config_hash_version") or 0)
    if version != 3:
        raise ValueError(f"{run}: expected provenance schema v3, got {version}")
    if meta.get("config_hash") != config_hash(meta["config"], version=3):
        raise ValueError(f"{run}: config hash does not recompute")
    runtime_sha = meta.get("runtime_code_snapshot_sha256")
    if meta["config"].get("runtime_snapshot_hash") != runtime_sha:
        raise ValueError(f"{run}: runtime snapshot is not bound in config")
    if meta.get("run_identity_sha256") != run_identity_sha256(
        meta["config_hash"], meta["started_at"], runtime_sha
    ):
        raise ValueError(f"{run}: run identity does not recompute")
    if drift:
        raise ValueError(f"{run}: metric drift in {drift}")
    return {
        "run": run,
        "model_name": meta["model_name"],
        "analyte": meta["config"]["target_analyte"],
        "mask": meta["mask_name"],
        "seed": int(meta["config"]["seed"]),
        **{key: float(stored[key]) for key in metric_keys},
        "rows_full_grid": len(frame),
        "rows_test": len(test),
        "identity": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/phase4_transfer/temporal_h2x_v1")
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), default=None)
    args = parser.parse_args()
    root = Path(args.root)
    plan = json.loads((root / "run_plan.json").read_text(encoding="utf-8"))
    if args.stage is not None and plan.get("stage") != args.stage:
        raise SystemExit(f"run plan stage is {plan.get('stage')!r}, not {args.stage!r}")
    expected = _expected_runs(plan)
    rows = []
    missing = []
    failures = {}
    for run in expected:
        if not (root / "runs" / run / "meta.json").is_file():
            missing.append(run)
            continue
        try:
            rows.append(audit_run(root, run))
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            # Report every failed unit, then fail closed.
            failures[run] = str(exc)
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values("run").to_csv(root / "audit_metrics.csv", index=False)
    comparison = pd.DataFrame()
    if {"h2x", "h2x_t"}.issubset(set(plan["models"])):
        keys = ["analyte", "mask", "seed"]
        left = result[result["model_name"].eq("h2x")].set_index(keys)
        right = result[result["model_name"].eq("h2x_t")].set_index(keys)
        common = left.index.intersection(right.index)
        comparison = pd.DataFrame(index=common).reset_index()
        for metric_name in ("mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2"):
            comparison[f"h2x_{metric_name}"] = [left.loc[k, metric_name] for k in common]
            comparison[f"h2x_t_{metric_name}"] = [right.loc[k, metric_name] for k in common]
            comparison[f"delta_{metric_name}"] = (
                comparison[f"h2x_t_{metric_name}"] - comparison[f"h2x_{metric_name}"]
            )
        comparison.to_csv(root / "snapshot_vs_temporal.csv", index=False)
    verdict = {
        "version": "temporal_h2x_audit_v1",
        "stage": plan["stage"],
        "expected_runs": len(expected),
        "audited_runs": len(rows),
        "missing_runs": missing,
        "failed_runs": failures,
        "identity_complete": not missing and not failures,
        "comparison_rows": len(comparison),
        "status": "pass" if not missing and not failures else "fail",
    }
    (root / "audit_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict))
    if verdict["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
