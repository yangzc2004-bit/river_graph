"""Validation-only affine stacking of ExtraTrees context and KGML residuals.

The two temporal experts are combined in log target space.  Coefficients are
fit on the validation view for each seed and missingness family, then frozen
for the terminal test view.  Spatial families use the selected ExtraTrees
context expert alone.  This is a post-processing comparison over saved
predictions; it does not read terminal test labels when fitting or selecting
the stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_et_hybrid_gate import (
    MASKS,
    TEMPORAL_MASKS,
    merge_experts,
)


def _fit_log_affine(frame: pd.DataFrame) -> np.ndarray:
    c = np.log1p(np.clip(frame["context_pred"].to_numpy(dtype=float), 0.0, None))
    r = np.log1p(np.clip(frame["residual_pred"].to_numpy(dtype=float), 0.0, None))
    y = np.log1p(np.clip(frame["y_true"].to_numpy(dtype=float), 0.0, None))
    x = np.column_stack([np.ones(len(frame)), c, r])
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _predict_log_affine(frame: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    c = np.log1p(np.clip(frame["context_pred"].to_numpy(dtype=float), 0.0, None))
    r = np.log1p(np.clip(frame["residual_pred"].to_numpy(dtype=float), 0.0, None))
    x = np.column_stack([np.ones(len(frame)), c, r])
    return np.maximum(0.0, np.expm1(x @ beta))


def _fit_geo_alpha(frame: pd.DataFrame) -> float:
    c = np.log1p(np.clip(frame["context_pred"].to_numpy(dtype=float), 0.0, None))
    r = np.log1p(np.clip(frame["residual_pred"].to_numpy(dtype=float), 0.0, None))
    y = frame["y_true"].to_numpy(dtype=float)
    grid = np.linspace(0.0, 1.0, 201)
    losses = [np.abs(y - np.maximum(0.0, np.expm1((1.0 - a) * c + a * r))).mean()
              for a in grid]
    return float(grid[int(np.argmin(losses))])


def _predict_geo(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    c = np.log1p(np.clip(frame["context_pred"].to_numpy(dtype=float), 0.0, None))
    r = np.log1p(np.clip(frame["residual_pred"].to_numpy(dtype=float), 0.0, None))
    return np.maximum(0.0, np.expm1((1.0 - alpha) * c + alpha * r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[dict] = []
    for seed in args.seeds:
        for mask in args.masks:
            try:
                val = merge_experts(seed, mask, "val")
                test = merge_experts(seed, mask, "test")
            except (FileNotFoundError, ValueError) as exc:
                missing.append({"seed": seed, "mask": mask, "reason": str(exc)})
                continue

            yv = val["y_true"].to_numpy(dtype=float)
            yt = test["y_true"].to_numpy(dtype=float)
            context_val = val["context_pred"].to_numpy(dtype=float)
            context_test = test["context_pred"].to_numpy(dtype=float)
            context_val_mae = float(np.abs(yv - context_val).mean())
            context_test_mae = float(np.abs(yt - context_test).mean())
            if mask not in TEMPORAL_MASKS:
                rows.append({
                    "seed": seed, "mask": mask,
                    "n_test": len(test),
                    "selected_variant": "context",
                    "val_mae": context_val_mae,
                    "test_mae": context_test_mae,
                    "context_test_mae": context_test_mae,
                    "beta0": np.nan, "beta_context": np.nan, "beta_residual": np.nan,
                    "geo_alpha": np.nan,
                })
                continue

            beta = _fit_log_affine(val)
            geo_alpha = _fit_geo_alpha(val)
            val_pred = {
                "context": context_val,
                "residual": val["residual_pred"].to_numpy(dtype=float),
                "geo_blend": _predict_geo(val, geo_alpha),
                "log_affine": _predict_log_affine(val, beta),
            }
            test_pred = {
                "context": context_test,
                "residual": test["residual_pred"].to_numpy(dtype=float),
                "geo_blend": _predict_geo(test, geo_alpha),
                "log_affine": _predict_log_affine(test, beta),
            }
            val_mae = {name: float(np.abs(yv - pred).mean()) for name, pred in val_pred.items()}
            selected = min(val_mae, key=val_mae.get)
            coeff = beta if selected == "log_affine" else np.array([np.nan, np.nan, np.nan])
            rows.append({
                "seed": seed, "mask": mask,
                "n_test": len(test),
                "selected_variant": selected,
                "val_mae": val_mae[selected],
                "test_mae": float(np.abs(yt - test_pred[selected]).mean()),
                "context_test_mae": context_test_mae,
                "beta0": coeff[0], "beta_context": coeff[1], "beta_residual": coeff[2],
                "geo_alpha": geo_alpha,
                **{f"val_{name}_mae": value for name, value in val_mae.items()},
                **{f"test_{name}_mae": float(np.abs(yt - pred).mean())
                   for name, pred in test_pred.items()},
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(args.out_dir / "seed_family_metrics.csv", index=False)
    pd.DataFrame(missing).to_csv(args.out_dir / "missing_runs.csv", index=False)
    if detail.empty:
        raise RuntimeError("no complete runs found")
    family = detail.groupby("mask", as_index=False).agg(
        n_seeds=("seed", "count"),
        selected_test_mae=("test_mae", "mean"),
        context_test_mae=("context_test_mae", "mean"),
    )
    family["gain_vs_context_pct"] = 100.0 * (
        family["context_test_mae"] - family["selected_test_mae"]
    ) / family["context_test_mae"]
    pooled = pd.DataFrame([{
        "rows": len(detail),
        "selected_mae_equal_seed_family": float(detail["test_mae"].mean()),
        "context_mae_equal_seed_family": float(detail["context_test_mae"].mean()),
        "gain_equal_seed_family_pct": float(100.0 * (
            detail["context_test_mae"].mean() - detail["test_mae"].mean()
        ) / detail["context_test_mae"].mean()),
        "selected_mae_cell_weighted": float(np.average(detail["test_mae"], weights=detail["n_test"])),
        "context_mae_cell_weighted": float(np.average(detail["context_test_mae"], weights=detail["n_test"])),
    }])
    family.to_csv(args.out_dir / "family_metrics.csv", index=False)
    pooled.to_csv(args.out_dir / "pooled_summary.csv", index=False)
    (args.out_dir / "verdict.md").write_text(
        "# Validation-only affine expert stack\n\n"
        "The log-space affine stack is fit on validation predictions only. "
        "The selected variant is the lowest-validation-MAE candidate among "
        "context, residual, geometric blend, and affine stack; terminal test "
        "labels are never used for fitting or selection.\n\n"
        + family.to_string(index=False) + "\n\n" + pooled.to_string(index=False) + "\n"
    )
    print(family.to_string(index=False))
    print(pooled.to_string(index=False))
    if missing:
        print(f"Skipped incomplete runs: {len(missing)}")


if __name__ == "__main__":
    main()
