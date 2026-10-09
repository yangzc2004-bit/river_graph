"""Fit a validation-only feature gate over three saved DOC experts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_three_expert_stack import FAMILIES, load_expert


def features(frame: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    month = 2 * np.pi * frame.month_index.to_numpy() / 12.0
    z = np.log1p(np.clip(preds, 0.0, None))
    numeric = np.column_stack([
        np.sin(month), np.cos(month), z,
        z[:, 0] - z[:, 1], z[:, 0] - z[:, 2], z[:, 1] - z[:, 2],
    ]).astype(np.float32)
    numeric = (numeric - numeric.mean(0)) / np.maximum(numeric.std(0), 1e-6)
    return np.column_stack([np.ones(len(frame), dtype=np.float32), numeric])


def fit_gate(x: np.ndarray, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    torch.manual_seed(42)
    xt = torch.as_tensor(x)
    z = torch.as_tensor(np.log1p(np.clip(pred, 0.0, None)), dtype=torch.float32)
    target = torch.as_tensor(np.log1p(np.clip(y, 0.0, None)), dtype=torch.float32)
    logits = torch.zeros((x.shape[1], 3), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits[0] = torch.log(torch.tensor([0.2, 0.6, 0.2]))
    opt = torch.optim.Adam((logits,), lr=0.04, weight_decay=1e-3)
    for _ in range(700):
        weight = torch.softmax(xt @ logits, dim=1)
        fused = (weight * z).sum(1)
        loss = torch.mean((fused - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return logits.detach().numpy()


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, weights = [], []
    for family in FAMILIES:
        val_rf = load_expert(family, "rf_context", "val")
        test_rf = load_expert(family, "rf_context", "test")
        val_res = load_expert(family, "residual_nomsg", "val")
        test_res = load_expert(family, "residual_nomsg", "test")
        val_pred = np.column_stack([val_rf.local_pred, val_rf.context_pred, val_res.final_pred])
        test_pred = np.column_stack([test_rf.local_pred, test_rf.context_pred, test_res.final_pred])
        x_val = features(val_rf, val_pred)
        x_test = features(test_rf, test_pred)
        params = fit_gate(x_val, val_pred, val_rf.y_true.to_numpy())
        with torch.no_grad():
            w = torch.softmax(torch.as_tensor(x_test) @ torch.as_tensor(params), dim=1).numpy()
        fused = np.expm1((w * np.log1p(np.clip(test_pred, 0.0, None))).sum(1))
        y = test_rf.y_true.to_numpy()
        for name, value in (("rf_local", test_pred[:, 0]), ("rf_context", test_pred[:, 1]),
                            ("residual", test_pred[:, 2]), ("three_expert_gate", fused)):
            rows.append({"family": family, "model": name, "n": len(y),
                         "mae": float(np.abs(y - value).mean()),
                         "rmse": float(np.sqrt(np.mean((y - value) ** 2)))})
        weights.append({"family": family, "w_rf_local_mean": w[:, 0].mean(),
                        "w_rf_context_mean": w[:, 1].mean(), "w_residual_mean": w[:, 2].mean(),
                        "w_rf_local_sd": w[:, 0].std(), "w_rf_context_sd": w[:, 1].std(),
                        "w_residual_sd": w[:, 2].std()})
    results = pd.DataFrame(rows)
    weight_table = pd.DataFrame(weights)
    pooled = results.groupby("model").apply(
        lambda f: pd.Series({"n": f.n.sum(), "mae": (f.n * f.mae).sum() / f.n.sum()}),
        include_groups=False,
    ).reset_index()
    results.to_csv(out_dir / "family_metrics.csv", index=False)
    weight_table.to_csv(out_dir / "weights.csv", index=False)
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    (out_dir / "verdict.md").write_text(
        "# DOC three-expert feature gate\n\n"
        "A linear softmax gate was fitted separately on each family validation split in log1p space and evaluated once on the five-seed terminal ensemble.\n\n"
        + weight_table.to_string(index=False) + "\n\n## Pooled test\n\n" + pooled.to_string(index=False) + "\n"
    )
    print(weight_table.to_string(index=False))
    print(pooled.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
