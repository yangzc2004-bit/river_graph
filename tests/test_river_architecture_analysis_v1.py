"""Independent arithmetic examples for the reported comparison metrics."""
import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("analyze_doc_river_architecture_comparison_v1")


def test_native_metrics_arithmetic(analysis):
    frame = pd.DataFrame({"y_true": [1., 3.], "y_pred": [3., 1.], "station": ["a", "b"],
                          "cell": [0, 1], "high_doc": [False, True]})
    result = analysis.metrics(frame)
    assert result["mae"] == 2.
    assert result["rmse"] == 2.
    assert result["r2"] == -3.
    assert result["bias"] == 0.
    assert result["q90_bias"] == -2.
    assert result["unique_cells"] == 2


def test_seed_repeats_do_not_create_new_bootstrap_clusters(analysis):
    rows = []
    for i, region in enumerate(["a", "b"]):
        for seed in (42, 43):
            for model, prediction in (("local", 2.), ("candidate", 1.5)):
                rows.append({"target_huc4": region, "station": region, "cell": i,
                             "seed": seed, "model_name": model, "y_true": 1., "y_pred": prediction})
    frame = pd.DataFrame(rows)
    result = analysis.paired_bootstrap(frame, "target_huc4", "local", "candidate", repeats=100)
    assert result["clusters"] == 2
    assert result["gain_percent_interval"] == [50., 50.]
    repeated = analysis.paired_bootstrap(pd.concat([frame, frame]), "target_huc4", "local", "candidate", repeats=100)
    assert repeated == result
