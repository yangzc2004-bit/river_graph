"""Scientific identities, observed populations and synthetic river routing."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_mechanisms import (
    bootstrap_receivers,
    branch_status,
    concentration_mix,
    covariance_identity,
    detrend,
    drainage_order_consistent,
    gauge_mapping_status,
    mixing_summary,
    permitted_doc,
    receiver_summary,
    shared_station_components,
    simulate_networks,
)


def test_sources_are_independent_not_two_stations_on_one_branch():
    area = {1: 2., 2: 3., 3: 4., 4: 5.}
    independent = branch_status(1, 3, [1, 2], [3, 4], area)
    assert independent["branch_status"] == "independent"
    assert independent["area_overlap_fraction"] == 0
    assert branch_status(1, 3, [1, 2], [1, 2, 3], area)["branch_status"] == "nested"
    assert branch_status(1, 3, [1, 2], [2, 3], area)["branch_status"] == "shared_catchment"
    assert branch_status(1, 1, [1, 2], [1, 2], area)["branch_status"] == "nested"


def test_official_drainage_area_catches_wrongly_snapped_small_control_stream():
    # Hull Hollow's official 0.22 square miles cannot receive two basins of
    # 0.98 and 1.01 square miles (USGS report WRI 85-4197).
    assert not drainage_order_consistent(.98, 1.01, .22)
    assert drainage_order_consistent(.3, .6, 1.)
    assert gauge_mapping_status(2., .22) == "gross_area_mismatch"
    assert gauge_mapping_status(2., 1.9) == "area_consistent"
    assert gauge_mapping_status(2., np.nan) == "official_area_missing"


def test_native_concentration_positive_weights_and_covariance_identity():
    np.testing.assert_allclose(concentration_mix([2., 8.], [8., 2.], [1., 3.], [3., 1.]), [6.5, 6.5])
    assert np.isnan(concentration_mix(2., 8., -1., 3.))
    assert np.isnan(concentration_mix(2., 8., 0., 3.))
    assert np.isnan(concentration_mix(2., np.nan, 1., 3.))
    a, b = np.asarray([1.2, 3.4]), np.asarray([2.8, 4.6])
    qa, qb = np.asarray([.3, 1.7], np.float32), np.asarray([1.1, .9], np.float32)
    expected = (a*qa.astype(float)+b*qb.astype(float))/(qa.astype(float)+qb.astype(float))
    np.testing.assert_array_equal(concentration_mix(a, b, qa, qb), expected)
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 50))
    expected, actual = covariance_identity(a, b, .3)
    np.testing.assert_allclose(expected, actual, rtol=1e-12)


def test_hidden_cells_do_not_enter_any_mechanism_response():
    dataset = {"y": np.arange(12.).reshape(3, 4)}
    cells = np.asarray([0, 2, 6, 11])
    initial = permitted_doc(dataset, cells)
    hidden = np.ones(12, bool)
    hidden[cells] = False
    dataset["y"].ravel()[hidden] = 1e8
    np.testing.assert_array_equal(initial, permitted_doc(dataset, cells))
    with pytest.raises(ValueError, match="unique"):
        permitted_doc(dataset, [0, 0])


def test_mixing_reference_is_fixed_not_the_best_observed_branch():
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    a = 4+np.sin(np.arange(24))
    b = 8+np.sin(np.arange(24))
    frame = pd.DataFrame({"doc_a": a, "doc_b": b, "doc_target": (a+b)/2})
    result = mixing_summary(frame, weights=(1., 1.), dates=dates)
    assert result["mixture_mae"] == 0
    np.testing.assert_allclose(result["reference_mae"], 2.)
    assert result["within_source_range_fraction"] == 1
    seasonal = 2*np.sin(2*np.pi*dates.month.to_numpy()/12)+dates.year.to_numpy()*.01
    np.testing.assert_allclose(detrend(seasonal, dates), 0, atol=1e-10)


def test_receivers_not_combinations_define_bootstrap_and_effect():
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({"doc_a": np.ones(24), "doc_b": np.full(24, 3.), "doc_target": np.full(24, 2.)})
    metrics = mixing_summary(frame, weights=(1., 1.), dates=dates)
    cases = pd.DataFrame([{**metrics, "pair_id": f"p{i}", "target": "a" if i < 20 else "b",
                           "huc4": "10" if i < 20 else "11", "cluster": 3,
                           "population": "monthly", "weighting": "area"} for i in range(21)])
    receivers = receiver_summary(cases)
    assert len(receivers) == 2 and receivers.n_combinations.sum() == 21
    result = bootstrap_receivers(receivers, 30)
    assert result.n_receivers.eq(2).all()
    assert result.loc[result.metric.eq("mae_gain_pct"), "estimate"].eq(100).all()


def test_shared_station_sets_are_joint_units_even_across_receiver_regions():
    labels = shared_station_components([("r1", "a", "b"), ("r2", "b", "c"), ("r3", "d", "e")])
    assert labels["r1"] == labels["r2"] == labels["a"] == labels["c"]
    assert labels["r3"] != labels["r1"]


def test_synthetic_networks_conserve_water_and_distinguish_processing():
    series, summary = simulate_networks()
    assert summary.n_sources.eq(8).all() and summary.n_reaches.eq(14).all()
    np.testing.assert_allclose(series.outlet_flow, 1., atol=1e-12)
    assert np.isfinite(series.doc_normalized).all() and series.doc_normalized.gt(0).all()
    for _, frame in summary.groupby(["topology", "forcing"]):
        assert frame.set_index("removal_per_step").loc[.04, "mean_doc"] < frame.set_index("removal_per_step").loc[0., "mean_doc"]
    # Balanced synchronous sources follow exactly the same series: mixing alone
    # must not manufacture a decline in concentration or suppress their pulse.
    balanced = series[(series.topology == "balanced") & (series.forcing == "synchronous") & series.removal_per_step.eq(0)]
    assert balanced.doc_normalized.max() > 7
