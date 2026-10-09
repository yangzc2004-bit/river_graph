"""Antecedent hydro states use physical current/past observations only."""
from __future__ import annotations

import copy

import numpy as np

from river_graph.models.antecedent_hydro_states import antecedent_hydro_states


def data():
    x = np.full((2, 36, 2), 10.)
    x[:, 12, 1] = 20.
    x[:, 24, 1] = 30.
    return {"x": x, "x_mask": np.ones_like(x, dtype=bool), "y": np.full((2, 36), 2.)}


def test_seasonal_anomaly_known_values_and_flow_unit_invariance():
    dataset = data()
    actual = antecedent_hydro_states(dataset)["full"]
    np.testing.assert_array_equal(actual[:, 24, 0], [.5, .5])
    np.testing.assert_array_equal(actual[:, 24, 1], [1., 1.])
    scaled = copy.deepcopy(dataset)
    scaled["x"][..., 1] *= 35.3146667
    np.testing.assert_allclose(actual, antecedent_hydro_states(scaled)["full"], rtol=1e-6, atol=1e-7)


def test_hidden_targets_and_future_inputs_do_not_change_past_states():
    dataset = data()
    expected = antecedent_hydro_states(dataset)["full"]
    changed = copy.deepcopy(dataset)
    changed["y"][:] = 1e30
    np.testing.assert_array_equal(expected, antecedent_hydro_states(changed)["full"])
    changed["x"][:, 30:] += 100.
    np.testing.assert_array_equal(expected[:, :30], antecedent_hydro_states(changed)["full"][:, :30])


def test_missing_negative_and_zero_flow_keep_value_validity_distinct():
    dataset = data()
    dataset["x_mask"][0] = False
    dataset["x"][0] = np.nan
    dataset["x"][1, :, 1] = -1.
    result = antecedent_hydro_states(dataset)
    assert not result["full"][0].any() and not result["full"][1, :, :6].any()
    dataset["x"][1, :, 1] = 0.
    zero = antecedent_hydro_states(dataset)["full"]
    assert not zero[1, :, :6].any() and np.isfinite(zero).all()
    np.testing.assert_array_equal(result["availability_only"][..., (1, 3, 5, 7)], result["full"][..., (1, 3, 5, 7)])
