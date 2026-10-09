"""Physical and observational controls for the geometry/export link."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_geometry_budget import (
    observed_window_budget,
    route_water_carbon,
)


def test_constant_concentration_remains_constant_during_asynchronous_water_pulses():
    metrics, trace = route_water_carbon([.7, 1.8], [.2, .8], common=.5,
                                      dispersed=True, concentration_amplitude=0)
    np.testing.assert_array_equal(trace.concentration, np.full(len(trace), 5.))
    np.testing.assert_allclose(trace.carbon_flux, 5 * trace.water_flux, rtol=0, atol=1e-12)
    assert metrics["output_extra_carbon"] == 0
    assert np.isnan(metrics["carbon_excess_sd"])


def test_conservative_path_and_common_spreading_preserve_carbon_and_water():
    for parameters in ({}, {"path_factor": 0}, {"aligned": True},
                       {"common": .4, "dispersed": True}):
        metrics, trace = route_water_carbon([.5, 1.3], [.5, .5], **parameters)
        assert metrics["retained_extra_carbon_fraction"] == pytest.approx(1, abs=1e-12)
        assert metrics["output_extra_water"] == pytest.approx(metrics["input_extra_water"], abs=1e-12)
        assert trace.concentration.max() <= 6 + 1e-12
        np.testing.assert_allclose(trace.carbon_flux / trace.water_flux, trace.concentration)


def test_equal_arrivals_recover_one_source_concentration_not_sum_of_two():
    _, trace = route_water_carbon([1, 1], [.5, .5])
    assert trace.concentration.max() == pytest.approx(6, abs=1e-10)
    assert trace.water_flux.max() == pytest.approx(4, abs=1e-10)


def test_shared_repartition_is_null_without_spreading_or_processing():
    a, _ = route_water_carbon([1.003, 2.007], [.3, .7], common=.213)
    b, _ = route_water_carbon([1.003, 2.007], [.3, .7], common=.807)
    for metric in ("concentration_peak_excess", "carbon_excess_centroid", "carbon_excess_sd", "output_extra_carbon"):
        assert a[metric] == pytest.approx(b[metric], abs=1e-10)


def test_shared_processing_reduces_carbon_but_preserves_water():
    a, _ = route_water_carbon([1, 2], [.3, .7], common=.8, dispersed=True)
    b, _ = route_water_carbon([1, 2], [.3, .7], common=.8, dispersed=True, reaction_rate=.25)
    assert b["output_extra_carbon"] < a["output_extra_carbon"]
    assert b["output_extra_water"] == pytest.approx(a["output_extra_water"])
    assert b["retained_extra_carbon_fraction"] == pytest.approx((1 + .25 * .8 / 4) ** -4, rel=1e-5)


def test_carbon_spreading_matches_independent_variance_decomposition():
    paths, weights = np.array([.8, 1.9]), np.array([.35, .65])
    amplitude, sigma = 3., .15
    m, _ = route_water_carbon(paths, weights, common=.5, dispersed=True)
    # g + A*g² has mixture variances sigma² and sigma²/2 and relative
    # integrated weight A/sqrt(2). Path and common variances then add.
    source_variance = sigma**2 * (1 + amplitude / (2 * np.sqrt(2))) / (1 + amplitude / np.sqrt(2))
    path_variance = weights @ (paths - weights @ paths) ** 2
    predicted = source_variance + path_variance + m["shared_kernel_sd"]**2
    assert m["carbon_excess_sd"]**2 == pytest.approx(predicted, abs=1e-5)
    assert m["carbon_excess_centroid"] == pytest.approx(weights @ paths, abs=1e-5)


def hourly_case():
    return pd.DataFrame({"clock": pd.date_range("2020-01-01", periods=5, freq="h"),
                         "q_m3_s": [1., 2., 3., 2., 1.], "doc_mg_l": [5., 5., 5., 5., 5.]})


def budget(frame):
    return observed_window_budget(frame, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 04:00"),
                                  pd.Timestamp("2020-01-01 02:00"), pd.Timestamp("2020-01-01 03:00"), 5., 2.)


def test_hourly_carbon_units_and_no_concentration_enhancement():
    result = budget(hourly_case())
    assert result["budget_complete"]
    assert result["water_volume_m3"] == 28800
    assert result["carbon_kg"] == 144
    assert result["carbon_yield_kg_km2"] == 72
    assert result["signed_concentration_extra_kg"] == 0
    assert result["carbon_minus_water_after_peak_share"] == pytest.approx(0)


def test_missing_clock_and_missing_target_do_not_receive_filled_budgets():
    missing_clock = hourly_case().drop(index=2)
    assert not budget(missing_clock)["budget_complete"]
    missing_doc = hourly_case()
    missing_doc.loc[2, "doc_mg_l"] = np.nan
    result = budget(missing_doc)
    assert not result["budget_complete"]
    assert "carbon_kg" not in result


def test_signed_export_decomposition_accounts_for_negative_concentration_changes():
    frame = hourly_case()
    frame["doc_mg_l"] = [5., 4., 5., 7., 5.]
    result = budget(frame)
    assert result["negative_concentration_extra_kg"] > 0
    assert result["positive_concentration_extra_kg"] > 0
    assert result["carbon_kg"] == pytest.approx(result["constant_concentration_carbon_kg"]
                                               + result["signed_concentration_extra_kg"])


def test_noncausal_paths_and_duplicate_observations_rejected():
    with pytest.raises(ValueError, match="Causal"):
        route_water_carbon([-1, 2], [.5, .5])
    with pytest.raises(ValueError, match="Shared"):
        route_water_carbon([1, 2], [.5, .5], common=1.1)
    frame = hourly_case()
    with pytest.raises(ValueError, match="Canonical"):
        budget(pd.concat([frame, frame.iloc[[0]]]))
