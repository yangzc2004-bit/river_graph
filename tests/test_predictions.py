"""Tests for prediction freezing (synthetic, no training)."""

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.predictions import load_predictions, save_predictions


def _ds():
    n, t = 4, 6
    ymask = torch.zeros(n, t, dtype=torch.bool)
    ymask[0, 0] = ymask[1, 1] = ymask[2, 2] = True
    return {
        "site_no": ["A", "B", "C", "D"],
        "months": [f"2020-0{j + 1}-01" for j in range(t)],
        "y": torch.arange(n * t, dtype=torch.float32).reshape(n, t),
        "y_mask": ymask,
    }


def test_save_load_roundtrip(tmp_path):
    ds = _ds()
    n, t = ds["y"].shape
    pred = np.full((n, t), 1.5)
    split = {"train": np.array([0]), "test": np.array([t + 1, 2 * t + 2])}
    save_predictions(pred, ds, split, "M", "m1", "v02", out_dir=tmp_path)
    df = load_predictions("M", "m1", out_dir=tmp_path)
    assert len(df) == 3  # one per observed cell
    assert set(df["split"]) == {"train", "test"}
    assert (df["model"] == "M").all() and (df["dataset_version"] == "v02").all()
    assert df["y_pred"].eq(1.5).all()


def test_only_observed_cells_stored(tmp_path):
    ds = _ds()
    n, t = ds["y"].shape
    out = save_predictions(np.zeros((n, t)), ds, {"train": np.array([0])},
                           "M", "m2", "v02", out_dir=tmp_path)
    df = pd.read_parquet(out)
    assert len(df) == int(ds["y_mask"].sum())
