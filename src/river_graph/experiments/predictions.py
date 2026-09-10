"""Frozen prediction storage (Phase 0 of the paper-production stage).

After the benchmark is frozen, every model's per-cell predictions are stored
as parquet under experiments/predictions/. All downstream analysis
(residual maps, extreme-cell breakdowns, uncertainty) reads these files —
no retraining, no benchmark drift.

Row schema: station, month, y_true (mg/L), y_pred (mg/L), split role,
plus model / mask / dataset_version metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRED_DIR = Path("experiments/predictions")


def save_predictions(
    pred: np.ndarray,
    dataset: dict,
    split: dict[str, np.ndarray],
    model: str,
    mask_name: str,
    dataset_version: str,
    out_dir: Path = PRED_DIR,
) -> Path:
    """Store per-cell predictions for every observed cell."""
    y = dataset["y"].numpy()
    obs_mask = dataset["y_mask"].numpy()
    sites = dataset["site_no"]
    months = dataset["months"]
    n, t = y.shape

    role = np.full(n * t, "", dtype=object)
    for key in ("train", "val", "test", "context"):
        if key in split and len(split[key]):
            role[split[key]] = key
    role = role.reshape(n, t)

    rows = []
    for i in range(n):
        cells = np.flatnonzero(obs_mask[i])
        if len(cells) == 0:
            continue
        rows.append(pd.DataFrame({
            "station": sites[i],
            "month": [months[j] for j in cells],
            "y_true": y[i, cells],
            "y_pred": pred[i, cells],
            "split": role[i, cells],
        }))
    df = pd.concat(rows, ignore_index=True)
    df["model"] = model
    df["mask"] = mask_name
    df["dataset_version"] = dataset_version

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model}__{mask_name}.parquet"
    df.to_parquet(out, index=False)
    return out


def load_predictions(model: str, mask_name: str, out_dir: Path = PRED_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / f"{model}__{mask_name}.parquet")
