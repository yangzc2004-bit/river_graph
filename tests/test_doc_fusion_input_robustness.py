"""Target-only dynamic mask interventions preserve source values and scaling."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.river_architecture_comparison import covariate_inputs


@pytest.mark.parametrize("scenario", ["target_temperature_hidden", "target_dynamic_hydrology_hidden"])
def test_target_mask_intervention_keeps_source_normalization_and_excludes_DOC(scenario):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("doc_input_robustness",
            scripts / "evaluate_doc_fusion_input_robustness_v1.py")
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
    altered = module.masked_covariates(data, np.array([2]), scenario)
    assert "y" not in altered and "y_mask" not in altered
    assert np.asarray(data["x_mask"]).all()
    original = covariate_inputs(data, [0, 1], lookback=3)
    changed = covariate_inputs(altered, [0, 1], lookback=3)
    assert original["normalization"] == changed["normalization"]
    assert torch.equal(original["windows"][:, :2], changed["windows"][:, :2])
    assert not changed["windows"][:, 2, :, 0].any()
    assert not changed["windows"][:, 2, :, 2].any()
    assert torch.equal(changed["environment"], original["environment"])
    assert torch.equal(changed["season"], original["season"])
    if scenario == "target_dynamic_hydrology_hidden":
        assert not changed["windows"][:, 2, :, :4].any()
    else:
        assert torch.equal(changed["windows"][:, 2, :, 1], original["windows"][:, 2, :, 1])
