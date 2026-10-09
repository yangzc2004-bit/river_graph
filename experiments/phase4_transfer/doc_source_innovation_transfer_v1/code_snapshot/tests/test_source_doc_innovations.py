"""Information isolation and causal sparse source transfer."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    corrected_prediction,
    select_correction,
    source_residual_grid,
)


def _library():
    months = pd.period_range("2000-01", periods=48, freq="M").astype(str).to_numpy()
    ecology = np.zeros((3, 9))
    ecology[1:, 0] = [1., 2.]
    residual = np.repeat(np.arange(4.), 12)[None, :] * np.array([[2.], [-1.], [3.]])
    residual[1, 25] = np.nan
    return SourceDOCInnovationLibrary().fit(["a", "b", "c"], months, ecology, residual)


def test_source_labels_selected_before_residual_statistics():
    truth = np.arange(72.).reshape(3, 24)
    oof = np.log1p(np.ones_like(truth))
    source = np.array([0, 5, 24, 27])
    ids, values = source_residual_grid(truth, oof, source)
    permitted = np.zeros(truth.size, bool)
    permitted[source] = True
    truth.ravel()[~permitted], oof.ravel()[~permitted] = np.nan, np.inf
    again_ids, again = source_residual_grid(truth, oof, source)
    np.testing.assert_array_equal(ids, again_ids)
    np.testing.assert_array_equal(values, again)
    model = SourceDOCInnovationLibrary().fit(["a", "b"],
        pd.period_range("2000-01", periods=24, freq="M").astype(str), np.zeros((2, 9)), values)
    assert model.to_dict()["source_observations"] == 4
    with pytest.raises(ValueError, match="unique source cells"):
        source_residual_grid(truth, oof, np.r_[source, source[0]])


def test_same_month_real_and_past_control_hand_calculation_and_self_exclusion():
    model = _library()
    past, dates = model.historical_control()
    assert np.isnan(past[:, :12]).all()
    assert (dates[np.isfinite(past)] < np.broadcast_to(model.month_ordinals_, past.shape)[np.isfinite(past)]).all()
    result = model.predict_components(["a"], model.ecology_[:1], ["2001-01", "1999-12", "2002-02"])
    assert result["donors"][0]["source_stations"] == ["b", "c"]
    weights = np.exp(-np.array([1., 2.])/model.sigma_)
    assert result["support_count"][0, 0] == 2
    expected = weights @ model.innovations_[1:, 12]/(1+weights.sum())
    assert result["real_innovation"][0, 0] == expected
    assert result["historical_innovation"][0, 0] == weights @ model.innovations_[1:, 0]/(1+weights.sum())
    assert result["support_count"][0, 2] == 1
    for key in ("real_innovation", "historical_innovation", "support_count", "weight_mass"):
        assert result[key][0, 1] == 0


def test_future_innovations_do_not_change_current_or_past_inference():
    model = _library()
    months = model.months_[:30]
    original = model.predict_components(["new"], np.zeros((1, 9)), months)
    changed = deepcopy(model)
    changed.innovations_[:, 30:] = 10000.
    changed.innovations_[0, 32:] = np.nan
    actual = changed.predict_components(["new"], np.zeros((1, 9)), months)
    for key in original:
        if key != "donors":
            np.testing.assert_array_equal(actual[key], original[key])
    # Current source evidence genuinely matters once prior seasonal support exists.
    changed.innovations_[:, 29] += 100.
    actual = changed.predict_components(["new"], np.zeros((1, 9)), months)
    assert actual["real_innovation"][0, -1] != original["real_innovation"][0, -1]
    np.testing.assert_array_equal(actual["historical_innovation"], original["historical_innovation"])


def test_save_load_arbitrary_receivers_and_zero_alpha(tmp_path):
    model = _library()
    path = tmp_path/"library.npz"
    model.save(path)
    restored = SourceDOCInnovationLibrary.load(path)
    names, ecology, months = ["new-42", "new-99"], np.ones((2, 9)), ["2003-05", "2001-02"]
    expected = model.predict_components(names, ecology, months)
    actual = restored.predict_components(names, ecology, months)
    assert actual["donors"] == expected["donors"]
    for key in actual:
        if key != "donors":
            np.testing.assert_array_equal(actual[key], expected[key])
    base = np.array([0., 1., 2.])
    innovation = np.array([-4., 3., 1.])
    np.testing.assert_array_equal(corrected_prediction(base, innovation, 0), base)
    np.testing.assert_array_equal(corrected_prediction(base, innovation, 1), [0., 4., 3.])
    chosen = select_correction(base, innovation, np.array([0., 4., 3.]))
    assert chosen["alpha"] == 1.
    with pytest.raises(ValueError, match="source_validation"):
        select_correction(base, innovation, base, selection_role="test")
