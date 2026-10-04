"""The chemical residual operator is bounded and uses active support only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from river_graph.models.chemical_residual_kernel import (
    chemical_residual_kernel,
    source_chemical_distance_scale,
)


def problem():
    return (np.array([[0., 0.], [1., 1.], [3., 2.], [9., 9.]]),
            np.array([[0., 0.], [1., 0.], [2., 2.]]),
            np.array([True, True, True, False]), np.ones(3, bool), np.array([1., -2., 4.]))


def test_exact_formula_bounds_and_shift_invariance():
    q, s, qa, sa, r = problem()
    output = chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=2., bandwidth_scale=.5, eta=.75)
    weights = np.exp(-np.sum((q[qa, None]-s[None])**2, axis=2)/2)
    weights /= weights.sum(1, keepdims=True)
    expected = np.zeros(len(q)); expected[qa] = .75*(weights @ (r-r.mean()))
    np.testing.assert_allclose(output, expected, atol=1e-15)
    assert np.all(output >= .75*(r-r.mean()).min()) and np.all(output <= .75*(r-r.mean()).max())
    np.testing.assert_allclose(chemical_residual_kernel(q, s, qa, sa, r+17, source_distance_scale=2.,
                                                      bandwidth_scale=.5, eta=.75), output, atol=1e-15)
    np.testing.assert_allclose(chemical_residual_kernel(q+8, s+8, qa, sa, r, source_distance_scale=2.,
                                                      bandwidth_scale=.5, eta=.75), output, atol=1e-15)


def test_inactive_placeholders_do_not_change_output():
    q, s, qa, sa, r = problem()
    sa[-1] = False
    expected = chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=1)
    q[~qa] = np.nan; s[~sa] = np.nan; r[~sa] = np.nan
    actual = chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=1)
    np.testing.assert_array_equal(actual, expected)
    assert actual[~qa].tolist() == [0.]
    # Centering excludes the inactive support, not merely its kernel weight.
    np.testing.assert_array_equal(actual, chemical_residual_kernel(q, s[:2], qa, sa[:2], r[:2], source_distance_scale=1))


def test_degenerate_zero_and_one_support_exact_zero():
    q, s, qa, sa, r = problem()
    for active in [np.zeros(3, bool), np.array([True, False, False])]:
        np.testing.assert_array_equal(chemical_residual_kernel(q, s, qa, active, r, source_distance_scale=1), np.zeros(4))
    np.testing.assert_array_equal(chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=1, eta=0), np.zeros(4))
    np.testing.assert_array_equal(chemical_residual_kernel(q, s, qa, sa, np.ones(3), source_distance_scale=1), np.zeros(4))
    np.testing.assert_array_equal(chemical_residual_kernel(q, np.ones_like(s), qa, sa, [.1, .2, .3],
                                                          source_distance_scale=1), np.zeros(4))
    empty = chemical_residual_kernel(q, np.empty((0, 2)), qa, np.array([], bool), np.array([]), source_distance_scale=1)
    np.testing.assert_array_equal(empty, np.zeros(4))


def test_permutation_equivariance_and_future_query_independence():
    q, s, qa, sa, r = problem()
    reference = chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=.6)
    ip, jp = [2, 0, 3, 1], [1, 2, 0]
    actual = chemical_residual_kernel(q[ip], s[jp], qa[ip], sa[jp], r[jp], source_distance_scale=.6)
    np.testing.assert_allclose(actual, reference[ip], atol=1e-15)
    q[2:] *= 20
    actual = chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=.6)
    np.testing.assert_array_equal(actual[:2], reference[:2])


def test_distant_query_does_not_underflow_all_weights():
    q = np.array([[1e6, 0.]])
    s = np.array([[0., 0.], [1., 0.]])
    output = chemical_residual_kernel(q, s, [1], [1, 1], [2., 4.], source_distance_scale=1)
    np.testing.assert_array_equal(output, [1.])


def test_source_scale_uses_positive_within_station_distances_only():
    chemical = np.array([[[0., 0.], [0., 0.], [3., 0.]], [[100., 0.], [100., 4.], [100., 8.]],
                         [[999., 999.], [1000., 1000.], [1001., 1001.]]])
    active = np.ones((3, 3), bool)
    state = source_chemical_distance_scale(chemical, active, [0, 1], source_role="source_training")
    assert state["distance_scale"] == np.median([3., 3., 4., 8., 4.])
    assert state["n_pairs"] == 6 and state["n_positive_pairs"] == 5 and not state["fallback"]
    json.dumps(state, allow_nan=False)
    chemical[2] = np.nan
    # Target availability is not inspected during source bandwidth fitting.
    flags = active.astype(float); flags[2] = np.nan
    assert source_chemical_distance_scale(chemical, flags, [1, 0], source_role="source_training") == state


def test_deterministic_even_month_sampling_and_fallback():
    chemical = np.zeros((2, 80, 2)); chemical[0, :, 0] = np.arange(80)
    active = np.ones((2, 80), bool); active[:, ::4] = False
    state = source_chemical_distance_scale(chemical, active, [0], source_role="source_training")
    available = np.flatnonzero(active[0])
    expected = available[np.linspace(0, len(available)-1, 24, dtype=np.int64)].tolist()
    assert state["station_inventory"][0]["sampled_month_indices"] == expected
    assert state["n_sampled_months"] == 24 and state["n_pairs"] == 276
    constant = source_chemical_distance_scale(chemical, active, [1], source_role="source_training")
    assert constant["distance_scale"] == 1 and constant["fallback"]
    active[1] = False
    absent = source_chemical_distance_scale(chemical, active, [1], source_role="source_training")
    assert absent["fallback"] and absent["n_pairs"] == 0


def test_explicit_source_role_and_bad_inputs_rejected():
    q, s, qa, sa, r = problem()
    for kwargs in [{"eta": -1}, {"eta": 1.1}, {"bandwidth_scale": 0}, {"source_distance_scale": np.inf}]:
        with pytest.raises(ValueError):
            chemical_residual_kernel(q, s, qa, sa, r, **{"source_distance_scale": 1, **kwargs})
    with pytest.raises(ValueError, match="binary"):
        chemical_residual_kernel(q, s, [.5]*len(q), sa, r, source_distance_scale=1)
    q[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        chemical_residual_kernel(q, s, qa, sa, r, source_distance_scale=1)
    chemical, active = np.zeros((2, 3, 2)), np.ones((2, 3), bool)
    with pytest.raises(ValueError, match="source_training"):
        source_chemical_distance_scale(chemical, active, [0], source_role="source_validation")
    for stations in [[0, 0], [2], [-1], [.5]]:
        with pytest.raises(ValueError, match="indices"):
            source_chemical_distance_scale(chemical, active, stations, source_role="source_training")
    with pytest.raises(ValueError, match="24"):
        source_chemical_distance_scale(chemical, active, [0], source_role="source_training", max_months_per_station=25)
