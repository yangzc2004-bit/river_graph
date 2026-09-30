"""Fit a low-dimensional fusion gate over completed DOC expert predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_gate_pilot import (
    ROOTS,
    blend,
    encode,
    feature_frame,
    load_mask,
    station_bootstrap_gain,
)

from river_graph.models.unified_fusion import UnifiedSpatiotemporalFusion


def fit_family_gate(
    x_train: np.ndarray,
    z_rf: np.ndarray,
    z_local: np.ndarray,
    z_y: np.ndarray,
    init_alpha: float,
    *,
    steps: int = 500,
) -> UnifiedSpatiotemporalFusion:
    model = UnifiedSpatiotemporalFusion(x_train.shape[1], init_alpha=init_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
    features = torch.as_tensor(x_train, dtype=torch.float32)
    rf = torch.as_tensor(z_rf, dtype=torch.float32)
    local = torch.as_tensor(z_local, dtype=torch.float32)
    target = torch.as_tensor(z_y, dtype=torch.float32)
    for _ in range(steps):
        _alpha, fused = model(rf, local, features)
        loss = torch.mean((fused - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("observable", "disagreement"), default="disagreement")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries, effects = [], []
    pooled_y, pooled_rf, pooled_local, pooled_fused = [], [], [], []
    for mask, root in ROOTS.items():
        val = load_mask(root, mask, "val")
        test = load_mask(root, mask, "test")
        val_features, _ = feature_frame(val, mask)
        test_features, _ = feature_frame(test, mask)
        x_train, x_test = encode(val_features, test_features, args.feature_set)
        val_rf = val["rf_context"].final_pred.to_numpy()
        val_local = val["residual_nomsg"].final_pred.to_numpy()
        val_y = val["rf_context"].y_true.to_numpy()
        test_rf = test["rf_context"].final_pred.to_numpy()
        test_local = test["residual_nomsg"].final_pred.to_numpy()
        test_y = test["rf_context"].y_true.to_numpy()
        grid = np.linspace(0.0, 1.0, 101)
        val_losses = []
        for alpha in grid:
            pred = blend(pd.DataFrame({"rf_context": val_rf, "residual_nomsg": val_local}),
                         np.full(len(val_rf), alpha))
            val_losses.append(np.abs(val_y - pred).mean())
        init_alpha = float(grid[int(np.argmin(val_losses))])
        z_rf = np.log1p(np.clip(val_rf, 0, None))
        z_local = np.log1p(np.clip(val_local, 0, None))
        z_y = np.log1p(np.clip(val_y, 0, None))
        model = fit_family_gate(x_train, z_rf, z_local, z_y, init_alpha)
        with torch.no_grad():
            test_alpha, fused_z = model(
                torch.as_tensor(np.log1p(np.clip(test_rf, 0, None)), dtype=torch.float32),
                torch.as_tensor(np.log1p(np.clip(test_local, 0, None)), dtype=torch.float32),
                torch.as_tensor(x_test, dtype=torch.float32),
            )
        gate = test_alpha.numpy()
        fused = np.expm1(fused_z.numpy())
        rf_mae = float(np.abs(test_y - test_rf).mean())
        local_mae = float(np.abs(test_y - test_local).mean())
        fused_mae = float(np.abs(test_y - fused).mean())
        summaries.append({
            "mask": mask,
            "n": len(test_y),
            "feature_set": args.feature_set,
            "init_alpha": init_alpha,
            "test_alpha_mean": float(gate.mean()),
            "test_alpha_sd": float(gate.std()),
            "rf_context_mae": rf_mae,
            "residual_nomsg_mae": local_mae,
            "fusion_mae": fused_mae,
        })
        pooled_y.append(test_y)
        pooled_rf.append(test_rf)
        pooled_local.append(test_local)
        pooled_fused.append(fused)
        for name, baseline in (("rf_context", test_rf), ("residual_nomsg", test_local)):
            gain, low, high = station_bootstrap_gain(
                test_y, fused, baseline, test["rf_context"].station.to_numpy(), seed=42
            )
            effects.append({"mask": mask, "baseline": name, "gain_mae": gain,
                            "ci_lo": low, "ci_hi": high})

    summary = pd.DataFrame(summaries)
    effect = pd.DataFrame(effects)
    summary.to_csv(args.out_dir / "fusion_summary.csv", index=False)
    effect.to_csv(args.out_dir / "fusion_effects.csv", index=False)
    pooled = pd.DataFrame({"y": np.concatenate(pooled_y), "rf": np.concatenate(pooled_rf),
                           "local": np.concatenate(pooled_local), "fused": np.concatenate(pooled_fused)})
    pooled.to_csv(args.out_dir / "expert_test_values.csv", index=False)
    pd.DataFrame([
        {"system": "fusion", "mae": float(np.abs(pooled.y - pooled.fused).mean())},
        {"system": "rf_context", "mae": float(np.abs(pooled.y - pooled.rf).mean())},
        {"system": "residual_nomsg", "mae": float(np.abs(pooled.y - pooled.local).mean())},
    ]).to_csv(args.out_dir / "fusion_pooled.csv", index=False)
    (args.out_dir / "fusion_verdict.md").write_text(
        "# Trainable unified fusion pilot\n\n"
        "The two expert predictions were frozen. A separate observable-feature gate was fitted for each family using validation labels only, then evaluated on the terminal test predictions.\n\n"
        + summary.to_string(index=False) + "\n\n## Station-clustered effects\n\n"
        + effect.to_string(index=False) + "\n"
    )
    print(summary.to_string(index=False))
    print(effect.to_string(index=False))


if __name__ == "__main__":
    main()
