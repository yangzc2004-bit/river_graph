"""Causal, visibility and scale contracts for the fixed discharge block."""

import copy

import numpy as np
import pytest

from river_graph.models.causal_flow_features import build_causal_flow_features


def dataset(flow, observed=None):
    flow = np.atleast_2d(np.asarray(flow, dtype=float))
    mask = np.ones_like(flow, dtype=bool) if observed is None else np.atleast_2d(observed).astype(bool)
    return {"x": np.stack([np.full_like(flow, 15.0), flow], axis=-1),
            "x_mask": np.stack([np.ones_like(mask), mask], axis=-1),
            "y": np.full_like(flow, 2.0), "y_mask": np.ones_like(mask)}


def test_hand_calculated_signed_features_and_named_partition():
    result = build_causal_flow_features(dataset([2, 4, 6, 8, 4, 0, -2, 2, 6]))
    x = result["full"]
    assert x.shape == (1, 9, 10)
    assert x.dtype == np.float32
    assert result["version"] == 1
    assert result["age_cap_months"] == result["history_months"] == 12
    assert result["min_history_observations"] == 3
    np.testing.assert_allclose(x[0, 3], [.5, 1, 1 / 7, 1, .6, 1, 3 / 12, 0, 1, 1], rtol=1e-7)
    # At t6, prior mean=4 and mean absolute flow=4: (-2-4)/(4+6)=-.6.
    np.testing.assert_allclose(x[0, 6, :6], [-.6, 1, -1, 1, -1, 1], rtol=1e-7)
    np.testing.assert_array_equal(x[0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    values, freshness = result["value_feature_indices"], result["freshness_feature_indices"]
    assert values == [0, 2, 4]
    assert freshness == [1, 3, 5, 6, 7, 8, 9]
    assert sorted(values + freshness) == list(range(10))
    assert set(result["flag_feature_indices"]) <= set(freshness)
    assert len(set(result["feature_names"])) == 10
    ablation = x.copy()
    ablation[..., values] = 0
    np.testing.assert_array_equal(ablation[..., freshness], x[..., freshness])
    assert np.max(np.abs(x[..., values])) <= 1
    assert np.min(x[..., freshness]) >= 0 and np.max(x[..., freshness]) <= 1


@pytest.mark.parametrize("factor", [0.028316846592, 1e-6, 1e6])
def test_positive_unit_conversion_invariance(factor):
    rng = np.random.default_rng(28)
    data = dataset(rng.normal(0, 20, (3, 24)), rng.random((3, 24)) > .2)
    expected = build_causal_flow_features(data)["full"]
    data["x"][..., 1] *= factor
    actual = build_causal_flow_features(data)["full"]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_future_flow_and_mask_changes_cannot_affect_present_or_past():
    data = dataset(np.arange(1, 31))
    before = build_causal_flow_features(data)["full"]
    data["x"][:, 17:, 1] = -5000
    data["x_mask"][:, 17:, 1] = False
    after = build_causal_flow_features(data)["full"]
    np.testing.assert_array_equal(after[:, :17], before[:, :17])
    assert not np.array_equal(after[:, 17:], before[:, 17:])


def test_masked_hydro_values_are_ignored_even_if_nonfinite():
    data = dataset([2, 4, 6, 8, 10, 12, 14], [True, False, True, True, False, True, True])
    before = build_causal_flow_features(data)["full"]
    data["x"][0, 1, 1], data["x"][0, 4, 1] = np.nan, np.inf
    after = build_causal_flow_features(data)["full"]
    np.testing.assert_array_equal(after, before)
    assert after[0, 4, 0] == after[0, 4, 1] == 0
    assert after[0, 2, 3] == 0  # The lag-1 flow at t1 is unavailable.


def test_no_doc_temperature_or_fitted_statistics_are_read():
    data = dataset([1, 2, 4, 8, 0, -2, -4, 8])
    before = build_causal_flow_features(data)["full"]
    changed = copy.deepcopy(data)
    changed["y"][:] = 10000
    changed["y_mask"][:] = False
    changed["x"][..., 0] = np.nan
    changed["x_mask"][..., 0] = False
    np.testing.assert_array_equal(build_causal_flow_features(changed)["full"], before)
    # A dict with only these two fields also works: no target/scaler/date lookup.
    minimal = {"x": changed["x"], "x_mask": changed["x_mask"]}
    np.testing.assert_array_equal(build_causal_flow_features(minimal)["full"], before)


def test_zero_and_negative_flows_have_explicit_scale_validity():
    zero = build_causal_flow_features(dataset(np.zeros(16)))["full"]
    np.testing.assert_array_equal(zero[..., [0, 1, 2, 3, 4, 5]], 0)
    assert zero[0, 12, 6] == 1  # Zeros are observed, despite no relative scale.
    np.testing.assert_array_equal(zero[..., [8, 9]], 1)
    negative = build_causal_flow_features(dataset([-2, -4, -6, -8]))["full"]
    np.testing.assert_allclose(negative[0, 3, :6], [-.5, 1, -1 / 7, 1, -.6, 1], rtol=1e-7)
    crossing = build_causal_flow_features(dataset([0, 0, 0, -2]))["full"]
    np.testing.assert_array_equal(crossing[0, 3, :6], [0, 0, -1, 1, -1, 1])


def test_prior_window_excludes_current_requires_three_and_drops_old_observations():
    data = dataset([1000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    full = build_causal_flow_features(data)["full"]
    assert full[0, 2, 1] == 0
    assert full[0, 3, 1] == 1
    # At t13, prior t1..t12 are all1; t0=1000 is no longer in the reference.
    np.testing.assert_allclose(full[0, 13, 0], .5)
    data["x_mask"][0, 1:11, 1] = False
    sparse = build_causal_flow_features(data)["full"]
    assert sparse[0, 13, 1] == 0  # Only two observed values in the past12.
    np.testing.assert_allclose(sparse[0, 13, 6], 2 / 12)


def test_flow_age_uses_current_observation_and_caps_with_explicit_never_seen_flag():
    observed = np.zeros(20, dtype=bool)
    observed[2] = observed[18] = True
    data = dataset(np.arange(20), observed)
    x = build_causal_flow_features(data)["full"][0]
    np.testing.assert_array_equal(x[:2, 7:10], 0)
    np.testing.assert_array_equal(x[2, 7:10], [0, 1, 1])
    np.testing.assert_allclose(x[3, 7:10], [1 / 12, 1, 0])
    np.testing.assert_array_equal(x[14:18, 7], 1)
    np.testing.assert_array_equal(x[18, 7:10], [0, 1, 1])
    result = build_causal_flow_features(data, age_cap_months=3)
    assert result["age_cap_months"] == 3
    assert result["history_months"] == 12
    capped = result["full"][0]
    assert capped[5, 7] == 1


def test_invalid_visible_flow_and_age_cap_are_rejected():
    data = dataset([1, 2, np.nan])
    with pytest.raises(ValueError, match="observed discharge"):
        build_causal_flow_features(data)
    with pytest.raises(ValueError, match="positive"):
        build_causal_flow_features(dataset([1]), age_cap_months=0)
