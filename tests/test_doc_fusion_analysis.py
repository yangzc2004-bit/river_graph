"""Independent aggregation examples: seed repeats and geographical pairing."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def analyzer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("doc_fusion_analysis", scripts /
                                                     "analyze_doc_fusion_component_comparison_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_seed_error_average_is_not_an_ensemble_metric():
    frame = pd.DataFrame({"y_true": [2., 2., 4., 4.], "y_pred": [1., 3., 3., 5.],
                          "station": ["a", "a", "b", "b"], "cell": [0, 0, 1, 1],
                          "high_doc": [False, False, True, True]})
    result = analyzer().metrics(frame)
    assert result["mae"] == result["rmse"] == result["q90_mae"] == 1.
    assert result["r2"] == 0.
    assert result["unique_cells"] == 2 and result["query_rows"] == 4
    assert (frame.groupby("cell").y_pred.mean() == frame.groupby("cell").y_true.mean()).all()


def test_geographical_bootstrap_averages_seed_errors_before_cluster_sampling():
    rows = []
    for basin in range(5):
        for cell in range(basin + 1):
            for seed in (42, 43, 44):
                for model, error in (("base", 2.), ("candidate", 1.)):
                    rows.append({"target_huc4": str(basin), "station": f"{basin}_{cell}",
                                 "cell": basin * 10 + cell, "seed": seed, "model_name": model,
                                 "y_true": 3., "y_pred": 3. + error})
    result = analyzer().interval(pd.DataFrame(rows), "base", "candidate")
    assert result["clusters"] == 5
    assert result["gain_percent_interval"] == pytest.approx([50., 50.])
