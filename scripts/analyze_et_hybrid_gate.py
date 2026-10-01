"""Evaluate a validation-selected ExtraTrees context expert plus ET residual expert.

This is a small post-hoc comparison over already-fitted predictions.  The
context model is selected from its validation table separately for each seed
and missingness family.  For the two temporal families, a constant blend
weight is fitted on validation predictions only and then evaluated once on
the terminal test rows.  Spatial families use the context expert alone until
the corresponding residual expert exists.

The script is intentionally independent of model training so that it can be
run while residual jobs finish and produces an honest partial report when a
seed is not yet complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MASKS = ("e1_r20_seed42", "e2a_strict", "e2b_partial", "e3_spatial_seed42")
TEMPORAL_MASKS = {"e2a_strict", "e2b_partial"}
DEFAULT_SEEDS = (42, 43, 44)
CONTEXT_ROOT = Path("experiments/phase4_transfer/rf_context_model_selection_v1")
RESIDUAL_ROOT = Path("experiments/phase4_transfer/kgml_local_transport_v1")


def context_root(seed: int) -> Path:
    return CONTEXT_ROOT / f"all_masks_seed{seed}"


def residual_root(seed: int) -> Path:
    if seed == 42:
        return RESIDUAL_ROOT / "extra_trees_residual_pilot"
    return RESIDUAL_ROOT / f"extra_trees_residual_seed{seed}"


def _selected_model(seed: int, mask: str) -> str:
    table = pd.read_csv(context_root(seed) / "validation_selection.csv")
    row = table.loc[table["mask"].eq(mask)]
    if len(row) != 1:
        raise ValueError(f"expected one selected context model for seed={seed}, mask={mask}")
    return str(row.iloc[0]["selected_model"])


def load_context(seed: int, mask: str) -> dict[str, pd.DataFrame]:
    model = _selected_model(seed, mask)
    path = context_root(seed) / "runs" / f"{mask}__seed{seed}" / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["model"].eq(model)].copy()
    if frame.empty:
        raise ValueError(f"selected context model has no rows: {path} / {model}")
    result = {}
    for role in ("val", "test"):
        out = frame.loc[frame["role"].eq(role), ["cell", "y_true", "y_pred"]].copy()
        out = out.rename(columns={"y_pred": "context_pred"})
        if out["cell"].duplicated().any():
            raise ValueError(f"duplicate context cells for {seed}/{mask}/{role}")
        result[role] = out
    return result


def load_residual(seed: int, mask: str) -> dict[str, pd.DataFrame]:
    if mask not in TEMPORAL_MASKS:
        return {}
    root = residual_root(seed) / "runs"
    matches = sorted(root.glob(f"residual_nomsg__doc__{mask}__seed{seed}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"residual run missing for seed={seed}, mask={mask}")
    run = matches[0]
    result = {}
    for role in ("val", "test"):
        path = run / f"{role}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        out = frame[["cell", "y_true", "final_pred"]].copy()
        out = out.rename(columns={"final_pred": "residual_pred"})
        if out["cell"].duplicated().any():
            raise ValueError(f"duplicate residual cells for {seed}/{mask}/{role}")
        result[role] = out
    return result


def merge_experts(seed: int, mask: str, role: str) -> pd.DataFrame:
    context = load_context(seed, mask)[role]
    out = context.copy()
    if mask in TEMPORAL_MASKS:
        residual = load_residual(seed, mask)[role]
        out = out.merge(residual, on="cell", suffixes=("", "_residual"), validate="one_to_one")
        if not np.allclose(out["y_true"], out["y_true_residual"], rtol=0, atol=1e-5):
            raise ValueError(f"context/residual y_true mismatch for {seed}/{mask}/{role}")
        out = out.drop(columns=["y_true_residual"])
    return out


def transformed_blend(context: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    z_context = np.log1p(np.clip(context, 0.0, None))
    z_residual = np.log1p(np.clip(residual, 0.0, None))
    return np.expm1((1.0 - alpha) * z_context + alpha * z_residual)


def fit_alpha(frame: pd.DataFrame) -> float:
    if "residual_pred" not in frame:
        return 0.0
    grid = np.linspace(0.0, 1.0, 101)
    losses = [
        np.abs(frame["y_true"] - transformed_blend(
            frame["context_pred"].to_numpy(), frame["residual_pred"].to_numpy(), alpha
        )).mean()
        for alpha in grid
    ]
    return float(grid[int(np.argmin(losses))])


def station_ci(frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray,
               draws: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    # This optional diagnostic is only used when station labels are available;
    # the main comparison remains cell-weighted MAE.
    if "station" not in frame:
        return (float("nan"), float("nan"), float("nan"))
    stations = frame["station"].to_numpy()
    _, inverse = np.unique(stations, return_inverse=True)
    counts = np.bincount(inverse)
    diff = np.abs(frame["y_true"].to_numpy() - baseline) - np.abs(frame["y_true"].to_numpy() - candidate)
    means = np.bincount(inverse, weights=diff) / counts
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(means), size=(draws, len(means)))
    boot = means[sample].mean(axis=1)
    return float(means.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
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
            alpha = fit_alpha(val)
            test_pred = (
                transformed_blend(test["context_pred"].to_numpy(), test["residual_pred"].to_numpy(), alpha)
                if "residual_pred" in test
                else test["context_pred"].to_numpy()
            )
            y = test["y_true"].to_numpy()
            context_mae = float(np.abs(y - test["context_pred"]).mean())
            residual_mae = float(np.abs(y - test["residual_pred"]).mean()) if "residual_pred" in test else np.nan
            hybrid_mae = float(np.abs(y - test_pred).mean())
            rows.append({
                "seed": seed,
                "mask": mask,
                "context_model": _selected_model(seed, mask),
                "n_val": len(val),
                "n_test": len(test),
                "alpha_residual": alpha,
                "context_mae": context_mae,
                "residual_mae": residual_mae,
                "hybrid_mae": hybrid_mae,
                "hybrid_gain_vs_context_pct": 100.0 * (context_mae - hybrid_mae) / context_mae,
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(args.out_dir / "seed_family_metrics.csv", index=False)
    pd.DataFrame(missing).to_csv(args.out_dir / "missing_runs.csv", index=False)
    if detail.empty:
        raise RuntimeError("no complete runs found")

    family = detail.groupby("mask", as_index=False).agg(
        n_seeds=("seed", "count"),
        context_mae=("context_mae", "mean"),
        residual_mae=("residual_mae", "mean"),
        hybrid_mae=("hybrid_mae", "mean"),
        alpha_mean=("alpha_residual", "mean"),
        alpha_sd=("alpha_residual", "std"),
    )
    family["hybrid_gain_vs_context_pct"] = 100 * (family["context_mae"] - family["hybrid_mae"]) / family["context_mae"]
    # Equal family weighting is the same high-level view used by the existing
    # router summaries; cell-weighted pooled MAE is reported alongside it.
    pooled = pd.DataFrame([{
        "n_seed_family_rows": len(detail),
        "pooled_context_mae": float(detail.context_mae.mean()),
        "pooled_hybrid_mae": float(detail.hybrid_mae.mean()),
        "pooled_hybrid_gain_vs_context_pct": float(100 * (detail.context_mae.mean() - detail.hybrid_mae.mean()) / detail.context_mae.mean()),
        "cell_weighted_context_mae": float(np.average(detail.context_mae, weights=detail.n_test)),
        "cell_weighted_hybrid_mae": float(np.average(detail.hybrid_mae, weights=detail.n_test)),
        "cell_weighted_hybrid_gain_vs_context_pct": float(
            100 * (np.average(detail.context_mae, weights=detail.n_test)
                   - np.average(detail.hybrid_mae, weights=detail.n_test))
            / np.average(detail.context_mae, weights=detail.n_test)
        ),
        "mean_temporal_alpha": float(detail.loc[detail["mask"].isin(TEMPORAL_MASKS), "alpha_residual"].mean()),
    }])
    family.to_csv(args.out_dir / "family_metrics.csv", index=False)
    pooled.to_csv(args.out_dir / "pooled_summary.csv", index=False)
    (args.out_dir / "verdict.md").write_text(
        "# ExtraTrees context + residual hybrid\n\n"
        "Context model selection is validation-only and performed separately per seed and family. "
        "For temporal families, the transformed-space blend weight is selected on validation predictions; "
        "test predictions are then evaluated once. Missing runs are listed in `missing_runs.csv`.\n\n"
        "## Family means\n\n" + family.to_string(index=False) +
        "\n\n## Pooled seed-family view\n\n" + pooled.to_string(index=False) + "\n"
    )
    print(family.to_string(index=False))
    print(pooled.to_string(index=False))
    if missing:
        print(f"Skipped incomplete runs: {len(missing)}")


if __name__ == "__main__":
    main()
