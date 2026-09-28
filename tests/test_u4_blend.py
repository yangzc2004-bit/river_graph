"""Scientific contracts for the blend diagnostic and paired analysis."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _module(monkeypatch, name):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module(name)


def test_rf_fit_features_hide_all_validation_and_test_labels(monkeypatch):
    module = _module(monkeypatch, "run_temporal_rf_upgrade")
    n, t = 4, 6
    data = {
        "y": torch.arange(n * t, dtype=torch.float32).reshape(n, t),
        "x": torch.ones(n, t, 2), "x_mask": torch.ones(n, t, 2),
        "regime": torch.ones(n, 13), "static": torch.ones(n, 2),
        "months": [f"2020-{m:02d}-01" for m in range(1, t + 1)],
        "edge_index": torch.tensor([[0, 1, 2], [2, 2, 3]]),
    }
    split = {"train": np.arange(12), "val": np.arange(12, 18),
             "test": np.arange(18, 24)}
    before = module._features(data, split, module.FIT_VISIBILITY)
    data["y"].reshape(-1)[np.r_[split["val"], split["test"]]] += 1000
    after = module._features(data, split, module.FIT_VISIBILITY)
    np.testing.assert_array_equal(before, after)


def test_station_ci_does_not_gain_precision_by_duplicating_seeds(monkeypatch):
    module = _module(monkeypatch, "analyze_u4_blend")
    errors = pd.DataFrame({"station": [0, 0, 1, 1, 2], "cell": [0, 1, 2, 3, 4],
                           "delta": [0.1, 0.2, -0.8, -0.4, 0.5]})
    single = module.cluster_interval(errors)
    repeated = module.cluster_interval(pd.concat([errors] * 5, ignore_index=True))
    np.testing.assert_allclose(single, repeated)
    assert single[0] < 0 < single[1]
