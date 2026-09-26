"""Audit H2X-T run products and compare matched snapshot/temporal runs.

The evaluator is intentionally metrics-only after the run files exist.  It
recomputes test metrics from each full-grid parquet, verifies V3 provenance,
and writes a compact comparison table for the T2 pilot or T3 formal run.
"""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import (
    config_hash,
    run_identity_sha256,
    sha256_file,
)


@cache
def _dataset(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


@cache
def _hash(path: str) -> str:
    return sha256_file(path)


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


def audit_run(root: Path, run: str, plan: dict | None = None) -> dict:
    run_dir = root / "runs" / run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(run_dir / "full_grid.parquet")
    config = meta["config"]
    for role in ("dataset", "mask"):
        identity = meta[role]
        if _hash(identity["path"]) != identity["sha256"]:
            raise ValueError(f"{run}: {role} content hash mismatch")
        if config[f"{role}_sha256"] != identity["sha256"]:
            raise ValueError(f"{run}: {role} config binding mismatch")
    if _hash(str(run_dir / "full_grid.parquet")) != meta["full_grid_sha256"]:
        raise ValueError(f"{run}: prediction content hash mismatch")
    if stored["config_hash"] != meta["config_hash"]:
        raise ValueError(f"{run}: metrics bind a different configuration")
    expected_name = (
        f"{meta['model_name']}__{config['target_analyte']}__"
        f"{meta['mask_name']}__seed{config['seed']}"
    )
    if expected_name != run:
        raise ValueError(f"{run}: run name disagrees with sidecar")
    if plan is not None:
        for key in ("max_epochs", "patience"):
            if config[key] != plan[key]:
                raise ValueError(f"{run}: {key} differs from execution plan")
    ds = _dataset(meta["dataset"]["path"])
    y, observed = np.asarray(ds["y"]), np.asarray(ds["y_mask"], dtype=bool)
    n, t = y.shape
    if len(frame) != n * t or frame[["station", "month"]].duplicated().any():
        raise ValueError(f"{run}: incomplete or duplicate full grid")
    if not np.array_equal(frame.station.to_numpy(), np.repeat(ds["site_no"], t)):
        raise ValueError(f"{run}: station alignment differs")
    if not np.array_equal(frame.month.to_numpy(), np.tile(np.asarray(ds["months"], str), n)):
        raise ValueError(f"{run}: month alignment differs")
    if not np.isfinite(frame.y_pred).all():
        raise ValueError(f"{run}: nonfinite predictions")
    if not np.array_equal(frame.observed.to_numpy(), observed.ravel()):
        raise ValueError(f"{run}: observed mask differs")
    if not np.array_equal(frame.y_true.to_numpy()[observed.ravel()], y[observed]):
        raise ValueError(f"{run}: observed labels differ")
    roles = np.full(n * t, "", dtype=object)
    with np.load(meta["mask"]["path"], allow_pickle=False) as masks:
        for role in ("train", "val", "test", "context"):
            if role not in masks:
                continue
            cells = masks[role]
            if not observed.ravel()[cells].all() or (roles[cells] != "").any():
                raise ValueError(f"{run}: invalid/overlapping role cells")
            roles[cells] = role
        threshold = float(np.quantile(y.ravel()[masks["train"]], 0.90))
    if not np.array_equal(frame.split.to_numpy(), roles):
        raise ValueError(f"{run}: product roles differ from mask")
    observed_product = pd.read_parquet(run_dir / "observed_predictions.parquet")
    if not frame[frame.observed].reset_index(drop=True).equals(observed_product.reset_index(drop=True)):
        raise ValueError(f"{run}: observed export differs from full grid")
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
    tail = test[test.y_true >= threshold]
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
        "runtime_snapshot_hash": runtime_sha,
        "dataset_hash": config["dataset_sha256"],
        "mask_hash": config["mask_sha256"],
        "q90_threshold": threshold,
        "q90_n": len(tail),
        "q90_mae": float((tail.y_pred - tail.y_true).abs().mean()),
        "q90_unstable": len(tail) < 20,
        "epochs_run": meta.get("training", {}).get("epochs_run"),
        "target_transform_declared": config["target_transform"],
        "training_protocol": config.get("training_protocol"),
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
            rows.append(audit_run(root, run, plan))
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            # Report every failed unit, then fail closed.
            failures[run] = str(exc)
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values("run").to_csv(root / "audit_metrics.csv", index=False)
    comparison = pd.DataFrame()
    if not result.empty and {"h2x", "h2x_t"}.issubset(set(plan["models"])):
        keys = ["analyte", "mask", "seed"]
        left = result[result["model_name"].eq("h2x")].set_index(keys)
        right = result[result["model_name"].eq("h2x_t")].set_index(keys)
        common = left.index.intersection(right.index)
        for key in common:
            for field in ("dataset_hash", "mask_hash", "rows_test", "runtime_snapshot_hash"):
                if left.loc[key, field] != right.loc[key, field]:
                    failures[str(key)] = f"paired {field} mismatch"
            if left.loc[key, "training_protocol"] != right.loc[key, "training_protocol"]:
                failures[str(key)] = "paired training protocol mismatch"
        comparison = pd.DataFrame(index=common).reset_index()
        for metric_name in ("mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2", "q90_mae"):
            comparison[f"h2x_{metric_name}"] = [left.loc[k, metric_name] for k in common]
            comparison[f"h2x_t_{metric_name}"] = [right.loc[k, metric_name] for k in common]
            comparison[f"delta_{metric_name}"] = (
                comparison[f"h2x_t_{metric_name}"] - comparison[f"h2x_{metric_name}"]
            )
        comparison.to_csv(root / "snapshot_vs_temporal.csv", index=False)
        comparison["relative_mae_reduction_pct"] = 100 * (1 - comparison.h2x_t_mae / comparison.h2x_mae)
        comparison["temporal_better"] = comparison.delta_mae < 0
        comparison.to_csv(root / "snapshot_vs_temporal.csv", index=False)
        summary = comparison.groupby(["analyte", "mask"]).agg(
            h2x_mae=("h2x_mae", "mean"), h2x_t_mae=("h2x_t_mae", "mean"),
            better_seeds=("temporal_better", "sum"), seeds=("seed", "count"),
            h2x_q90_mae=("h2x_q90_mae", "mean"), h2x_t_q90_mae=("h2x_t_q90_mae", "mean"),
        )
        summary["relative_mae_reduction_pct"] = 100 * (1 - summary.h2x_t_mae / summary.h2x_mae)
        summary.to_csv(root / "pilot_summary.csv")
        e2 = summary.reset_index().query("mask in ['e2a_strict', 'e2b_partial']")
        e1 = summary.reset_index().query("mask == 'e1_r20_seed42'")
        pilot_verdict = {
            "identity_and_reproducibility": not missing and not failures,
            "e2_analytes_with_positive_mean_reduction": int(
                e2.groupby("analyte")["relative_mae_reduction_pct"].mean().gt(0).sum()
            ),
            "e2_gate_required": 2,
            "e1_analytes_with_positive_mean_reduction": int(
                e1.groupby("analyte")["relative_mae_reduction_pct"].mean().gt(0).sum()
            ),
            "e1_gate_required_no_systematic_degradation": True,
            "status": "pass" if (
                not missing and not failures
                and e2.groupby("analyte")["relative_mae_reduction_pct"].mean().gt(0).sum() >= 2
                and e1.groupby("analyte")["relative_mae_reduction_pct"].mean().gt(0).all()
            ) else "fail",
            "scope": "T2 pilot feasibility gate; not authorization of T3.",
        }
        (root / "pilot_verdict.json").write_text(
            json.dumps(pilot_verdict, indent=2) + "\n", encoding="utf-8"
        )
    verdict = {
        "version": "temporal_h2x_audit_v2",
        "stage": plan["stage"],
        "expected_runs": len(expected),
        "audited_runs": len(rows),
        "missing_runs": missing,
        "failed_runs": failures,
        "identity_complete": not missing and not failures,
        "comparison_rows": len(comparison),
        "status": "pass" if not missing and not failures else "fail",
        "scope": "Artifact integrity only; not approval of training-protocol fidelity or T3.",
    }
    (root / "audit_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict))
    if verdict["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
