"""Source-derived calibration respects information roles and can be restored."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from river_graph.models.concentration_hydro_calibration import (
    ConcentrationHydroCalibration,
    hydro_calibration_inputs,
)


def data():
    rng = np.random.default_rng(5)
    x = rng.uniform(1., 15., (8, 18, 2))
    mask = rng.random(x.shape) > .15
    return {"x": x, "x_mask": mask, "y": rng.uniform(1., 10., x.shape[:2]),
            "ph": rng.uniform(6., 8., x.shape[:2])}


def test_hydro_inputs_ignore_chemistry_hidden_values_and_future():
    dataset = data()
    expected = hydro_calibration_inputs(dataset)
    changed = copy.deepcopy(dataset)
    changed["y"][:] = 1e30
    changed["ph"][:] = -1e30
    changed["x"][~changed["x_mask"]] = np.nan
    np.testing.assert_array_equal(hydro_calibration_inputs(changed), expected)
    changed["x"][:, 11:] *= 100
    np.testing.assert_array_equal(hydro_calibration_inputs(changed)[:, :11], expected[:, :11])


@pytest.mark.parametrize("mode", ["log_affine", "concentration_hydro"])
def test_calibration_recovers_bias_save_load_and_zero_identity(mode):
    dataset = data()
    h = hydro_calibration_inputs(dataset).reshape(-1, 6)
    p = np.linspace(.5, 12., len(h))
    y = np.expm1(np.log1p(p)+.2)
    model = ConcentrationHydroCalibration(mode).fit(p, y, h)
    assert np.abs(model.predict(p, h)-y).mean() < .2*np.abs(p-y).mean()
    state = json.loads(json.dumps(model.to_dict()))
    loaded = ConcentrationHydroCalibration.from_dict(state)
    np.testing.assert_array_equal(loaded.predict(p, h), model.predict(p, h))
    loaded.coefficients_[:] = 0
    np.testing.assert_array_equal(loaded.predict(p, h), p)
    assert np.isfinite(model.predict(p, h)).all()
    with pytest.raises(ValueError, match="source-training"):
        model.fit(p, y, h, fit_role="target_test")


def test_source_statistics_unchanged_by_query_distribution():
    h = hydro_calibration_inputs(data()).reshape(-1, 6)
    p = np.linspace(1., 10., len(h))
    model = ConcentrationHydroCalibration().fit(p, p+.4, h)
    before = model.to_dict()
    query = h.copy()
    query[:, :2] *= 1000
    predicted = model.predict(p*20, query)
    assert model.to_dict() == before
    assert np.isfinite(predicted).all() and (predicted >= 0).all()
