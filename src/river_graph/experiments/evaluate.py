"""Benchmark evaluation: run a model over mask files, report RMSE/MAE/R2.

All evaluation happens in mg/L space (predictions and labels are stored
log1p-transformed; expm1 before metrics), on test cells only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MASKS_DIR = Path("experiments/masks")


def load_dataset(path: str | Path = "data/processed/mississippi_graph_v02.pt") -> dict:
    return torch.load(path, weights_only=False)


def load_mask(name: str, masks_dir: Path = MASKS_DIR) -> dict[str, np.ndarray]:
    with np.load(masks_dir / f"{name}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / R2 in mg/L space + log1p-space variants.

    DOC is long-tailed; log-space metrics reflect typical-condition skill
    and are robust to rare extreme-regime episodes (see
    experiments/analysis/extreme/).
    """
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "log_rmse": float("nan"), "log_mae": float("nan"),
                "log_r2": float("nan"), "n": 0}
    yt, yp = y_true[ok], y_pred[ok]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    lt, lp = np.log1p(yt), np.log1p(yp)
    log_rmse = float(np.sqrt(np.mean((lt - lp) ** 2)))
    log_mae = float(np.mean(np.abs(lt - lp)))
    lss_res = float(np.sum((lt - lp) ** 2))
    lss_tot = float(np.sum((lt - lt.mean()) ** 2))
    log_r2 = 1.0 - lss_res / lss_tot if lss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2,
            "log_rmse": log_rmse, "log_mae": log_mae, "log_r2": log_r2,
            "n": int(ok.sum())}


def evaluate(
    model,
    dataset: dict,
    mask_names: list[str],
    masks_dir: Path = MASKS_DIR,
) -> dict[str, dict[str, float]]:
    """Run ``model`` over each mask; results keyed by mask name.

    ``model`` must implement fit_predict(dataset, split) -> (N, T) ndarray
    of predictions in mg/L space (NaN where not predicted).
    """
    results = {}
    for name in mask_names:
        split = load_mask(name, masks_dir)
        pred = model.fit_predict(dataset, split)
        y = dataset["y"].numpy()  # mg/L
        flat_test = split["test"]
        results[name] = metrics(y.ravel()[flat_test], pred.ravel()[flat_test])
    return results
