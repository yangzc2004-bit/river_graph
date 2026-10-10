"""Query erasure is target-only and propagates coherently into later history."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.river_architecture_comparison import covariate_inputs


@pytest.mark.parametrize("scenario", ["query_current_temperature_hidden", "query_current_hydrology_hidden"])
def test_query_month_erasure_preserves_nonquery_history_and_source_values(scenario):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("time_current_missing",
            scripts / "evaluate_doc_time_current_missing_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    data = {"x": torch.arange(36).reshape(3, 6, 2).float(),
            "x_mask": torch.ones(3, 6, 2, dtype=torch.bool),
            "regime": torch.ones(3, 13), "static": torch.ones(3, 2),
            "months": [f"2020-{i:02d}" for i in range(1, 7)],
            "feature_channels": ["temperature", "discharge"], "y": torch.ones(3, 6),
            "y_mask": torch.ones(3, 6, dtype=torch.bool)}
    cells = np.array([2 * 6 + 2, 2 * 6 + 5])
    altered = module.query_masked_inputs(data, cells, scenario)
    assert "y" not in altered and "y_mask" not in altered
    assert altered["x_mask"][:2].all() and np.asarray(data["x_mask"]).all()
    assert altered["x_mask"][2, [0, 1, 3, 4]].all()
    assert not altered["x_mask"][2, [2, 5], 0].any()
    original = covariate_inputs(data, [0, 1], lookback=3)
    changed = covariate_inputs(altered, [0, 1], lookback=3)
    assert original["normalization"] == changed["normalization"]
    assert torch.equal(original["windows"][:, :2], changed["windows"][:, :2])
    # Month 2 is hidden at the query and stays hidden in month 3's history.
    assert changed["windows"][2, 2, -1, 2] == 0
    assert changed["windows"][3, 2, -2, 2] == 0
    assert changed["windows"][3, 2, -1, 2] == 1
    if scenario == "query_current_hydrology_hidden":
        assert not changed["windows"][2, 2, -1, :4].any()
    else:
        assert changed["windows"][2, 2, -1, 3] == 1
