"""Fit a validation-only three-expert DOC stack from saved predictions.

Experts are RF-local, RF-context, and the causal local residual model.  The
stack uses one non-negative simplex weight per missingness family and only
aggregates already completed five-seed products; it does not fit on test
labels or retrain any base model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMILIES = ("e1_r20_seed42", "e2a_strict", "e2b_partial", "e3_spatial_seed42")
SEEDS = (42, 43, 44, 45, 46)
ARMS = ("rf_context", "residual_nomsg")
ROOTS = {
    "e1_r20_seed42": (
        Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_pilot"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_confirmation_rf"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/unified_fusion_expert_completion_residual"),
    ),
    "e2a_strict": (
        Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/unified_fusion_expert_completion_rf"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_confirmation_residual"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/unified_fusion_expert_completion_residual"),
    ),
    "e2b_partial": (
        Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_pilot"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/unified_fusion_expert_completion_rf"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_confirmation_residual"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/unified_fusion_expert_completion_residual"),
    ),
    "e3_spatial_seed42": (
        Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_confirmation_rf"),
        Path("experiments/phase4_transfer/kgml_local_transport_v1/unified_fusion_expert_completion_residual"),
    ),
}


def find_run(family: str, arm: str, seed: int) -> Path:
    matches = []
    for root in ROOTS[family]:
        for path in root.glob("runs/*/meta.json"):
            meta = json.loads(path.read_text())
            cfg = meta["config"]
            if cfg["mask"] == family and cfg["arm"] == arm and cfg["seed"] == seed:
                matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one {family}/{arm}/seed{seed}, found {len(matches)}")
    return matches[0]


def load_expert(family: str, arm: str, role: str) -> pd.DataFrame:
    parts = []
    for seed in SEEDS:
        frame = pd.read_parquet(find_run(family, arm, seed) / f"{role}_predictions.parquet")
        columns = ["cell", "station", "month_index", "y_true", "local_pred", "context_pred", "final_pred"]
        parts.append(frame[columns].assign(seed=seed))
    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby(["cell", "station", "month_index", "y_true"], as_index=False).agg(
        local_pred=("local_pred", "mean"), context_pred=("context_pred", "mean"),
        final_pred=("final_pred", "mean"),
    )


def fit_weights(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    grid = np.linspace(0.0, 1.0, 101)
    best_loss, best = float("inf"), None
    # RF-local, RF-context, causal residual; simplex grid is deterministic.
    for a in grid:
        for b in grid[grid <= 1.0 - a + 1e-12]:
            weights = np.array([a, b, 1.0 - a - b])
            loss = float(np.abs(y - pred @ weights).mean())
            if loss < best_loss:
                best_loss, best = loss, weights
    assert best is not None
    return best, best_loss


def analyze(out_dir: Path) -> None:
    rows, weights_rows = [], []
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        val_rf = load_expert(family, "rf_context", "val")
        test_rf = load_expert(family, "rf_context", "test")
        val_res = load_expert(family, "residual_nomsg", "val")
        test_res = load_expert(family, "residual_nomsg", "test")
        if not val_rf.cell.equals(val_res.cell) or not test_rf.cell.equals(test_res.cell):
            raise ValueError(f"cell mismatch for {family}")
        val_pred = np.column_stack([val_rf.local_pred, val_rf.context_pred, val_res.final_pred])
        test_pred = np.column_stack([test_rf.local_pred, test_rf.context_pred, test_res.final_pred])
        weights, val_mae = fit_weights(val_pred, val_rf.y_true.to_numpy())
        test_y = test_rf.y_true.to_numpy()
        pred = test_pred @ weights
        weights_rows.append({"family": family, "w_rf_local": weights[0], "w_rf_context": weights[1],
                             "w_residual": weights[2], "val_mae": val_mae})
        for name, value in (("rf_local", test_pred[:, 0]), ("rf_context", test_pred[:, 1]),
                            ("residual", test_pred[:, 2]), ("stack", pred)):
            rows.append({"family": family, "model": name, "n": len(test_y),
                         "mae": float(np.abs(test_y - value).mean()),
                         "rmse": float(np.sqrt(np.mean((test_y - value) ** 2)))})
    results = pd.DataFrame(rows)
    weights = pd.DataFrame(weights_rows)
    results.to_csv(out_dir / "family_metrics.csv", index=False)
    weights.to_csv(out_dir / "weights.csv", index=False)
    pooled = results.groupby("model").apply(
        lambda frame: pd.Series({"n": frame.n.sum(), "mae": (frame.n * frame.mae).sum() / frame.n.sum()}),
        include_groups=False,
    ).reset_index()
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    (out_dir / "verdict.md").write_text(
        "# DOC three-expert validation stack\n\n"
        "Weights were selected on each family validation split using a 0.01 simplex grid, "
        "then applied once to the five-seed terminal predictions.\n\n"
        + weights.to_string(index=False) + "\n\n## Pooled test\n\n"
        + pooled.to_string(index=False) + "\n"
    )
    print(weights.to_string(index=False))
    print(pooled.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
