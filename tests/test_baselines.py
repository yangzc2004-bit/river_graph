"""Smoke tests for baselines + evaluation on a tiny synthetic dataset."""

import numpy as np

from river_graph.baselines.baselines import StationMean
from river_graph.experiments.evaluate import evaluate, metrics


def _tiny_dataset():
    import torch

    rng = np.random.default_rng(0)
    n, t = 6, 20
    y = rng.normal(1.5, 0.4, (n, t)).astype(np.float32)  # log-space values
    ymask = torch.tensor(rng.random((n, t)) < 0.5)
    return {
        "site_no": [f"s{i}" for i in range(n)],
        "months": [str(m) for m in np.arange("2010-01", "2011-09", dtype="datetime64[M]")],
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "y": torch.tensor(y),
        "y_mask": ymask,
        "x": torch.tensor(rng.normal(size=(n, t, 2)).astype(np.float32)),
        "x_mask": torch.tensor(rng.random((n, t, 2)) < 0.8),
        "feature_channels": ["temperature", "discharge"],
        "static": torch.tensor(rng.uniform(30, 48, (n, 2)).astype(np.float32)),
    }


def _split(ds, test_frac=0.3, seed=1):
    obs = np.flatnonzero(ds["y_mask"].numpy().ravel())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(obs)
    k = round(len(obs) * test_frac)
    return {"train": perm[k:], "val": perm[:0], "test": perm[:k]}


def test_metrics_perfect_prediction():
    m = metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert m["rmse"] == 0.0 and m["r2"] == 1.0 and m["n"] == 2


def test_metrics_ignores_nan():
    m = metrics(np.array([1.0, np.nan]), np.array([1.5, 9.0]))
    assert m["n"] == 1


def test_station_mean_fits_and_predicts(tmp_path):
    ds = _tiny_dataset()
    split = _split(ds)
    pred = StationMean().fit_predict(ds, split)
    assert pred.shape == ds["y"].shape
    # prediction for a station equals expm1 of its train mean in log space
    y = np.log1p(ds["y"].numpy())
    tr = np.zeros(y.size, dtype=bool)
    tr[split["train"]] = True
    tr = tr.reshape(y.shape)
    i = 0
    expected = np.expm1(np.mean(y[i][tr[i]]))
    assert abs(pred[i, 0] - expected) < 1e-4


def test_evaluate_runs_end_to_end(tmp_path):
    import numpy as np

    ds = _tiny_dataset()
    split = _split(ds)
    np.savez(tmp_path / "m1.npz", **split)
    res = evaluate(StationMean(), ds, ["m1"], masks_dir=tmp_path)
    assert "m1" in res and res["m1"]["rmse"] >= 0
