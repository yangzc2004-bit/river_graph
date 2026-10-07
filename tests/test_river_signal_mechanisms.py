"""Exact mechanisms need correct mixing, causal input and observation accounting."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_signal_mechanisms import (
    arrival_opportunity,
    flow_budget_records,
    mixing_records,
    variance_decomposition,
)


def records():
    n = 36
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "month_index": np.arange(n), "pair_id": "a_b_t", "target": "t",
        "source_a": "a", "source_b": "b", "huc4": "0101", "component": 0, "cluster": 3,
        "doc_a_now": 5+rng.uniform(0, 2, n), "doc_b_now": 5+rng.uniform(0, 2, n),
        "doc_a_previous": 5+rng.uniform(0, 2, n), "doc_b_previous": 5+rng.uniform(0, 2, n),
        "y_true": 5+rng.uniform(0, 2, n), "weight_a": .3, "path_a_km": 10., "path_b_km": 30.,
        "source_drainage_coverage": .9})


def test_variance_identity_and_normalized_counterfactual():
    a = np.array([-2., -1., 1., 2.])
    b = np.array([1., -1., 2., -2.])
    r = variance_decomposition(a, b, .3)
    assert r["mixture_buffer_fraction"] == pytest.approx(r["amplitude_buffer_fraction"]+r["asynchronous_buffer_fraction"])
    assert r["equal_amplitude_buffer_fraction"] == pytest.approx(.42*(1-r["source_rho"]))
    assert r["balanced_equal_amplitude_buffer_fraction"] >= r["equal_amplitude_buffer_fraction"]
    same = variance_decomposition(a, a, .5)
    assert same["mixture_buffer_fraction"] == pytest.approx(0.)
    opposite = variance_decomposition(a, -a, .5)
    assert opposite["mixture_variance"] == pytest.approx(0.)
    assert opposite["equal_amplitude_buffer_fraction"] == pytest.approx(1.)


def test_common_calendar_projection_keeps_native_mixing_linear():
    frame = records()
    metrics, product, fitted = mixing_records(frame)
    np.testing.assert_allclose(product.mixture_anomaly, .3*product.source_a_anomaly+.7*product.source_b_anomaly, atol=1e-12)
    changed = frame.copy()
    changed["y_true"] *= 100
    _, replay, _ = mixing_records(changed)
    np.testing.assert_array_equal(product.mixture_anomaly, replay.mixture_anomaly)
    assert len(metrics) == 1 and fitted[0]["design_rank"] == 4


def test_arrival_identity_zero_opportunities_and_outlet_independence():
    frame = records()
    result = arrival_opportunity(frame, 100.)
    np.testing.assert_allclose(result.branch_arrival_proxy-result.mean_delay_proxy, result.arrival_gap, atol=1e-12)
    changed = frame.copy()
    changed["y_true"] = 99999.
    np.testing.assert_array_equal(result.arrival_gap, arrival_opportunity(changed, 100.).arrival_gap)
    changed["path_a_km"] = changed.path_b_km
    np.testing.assert_array_equal(arrival_opportunity(changed, 100.).arrival_gap, 0.)
    changed = frame.copy()
    changed["doc_b_previous"] = changed.doc_b_now+(changed.doc_a_previous-changed.doc_a_now)
    np.testing.assert_allclose(arrival_opportunity(changed, 100.).arrival_gap, 0., atol=1e-12)
    with pytest.raises(ValueError, match="causal paths"):
        arrival_opportunity(frame, 5.)


def test_flow_screen_does_not_infer_missing_or_nonpositive_discharge():
    frame = records().iloc[:4].copy()
    q = np.array([1., np.nan, 0., -1.])
    result = flow_budget_records(frame, q, np.ones(4), np.full(4, 2.))
    np.testing.assert_array_equal(result.three_flows_measured, [True, False, False, False])
    np.testing.assert_array_equal(result.budget_screen, [True, False, False, False])
    assert result.flow_mixture_doc.iloc[0] == pytest.approx((frame.doc_a_now.iloc[0]+frame.doc_b_now.iloc[0])/2)
    assert result.flow_mixture_doc.iloc[1:].isna().all()


def test_degenerate_sources_are_explicitly_unidentified():
    r = variance_decomposition(np.ones(5), np.ones(5), .5)
    assert np.isnan(r["source_rho"]) and np.isnan(r["mixture_buffer_fraction"])
    with pytest.raises(ValueError):
        variance_decomposition(np.ones(5), np.ones(4), .5)
    with pytest.raises(ValueError):
        variance_decomposition(np.ones(5), np.ones(5), 1.)
