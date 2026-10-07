import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_morphology_effect import (
    MATCH_COVARIATES,
    geometry_routing,
    morphology_pairs,
    paired_response_summary,
)


def matching_panel():
    rows = []
    for i in range(8):
        rows.append({"station": str(i), "cluster": 1 if i % 2 == 0 else 3, "huc4": str(i//2),
                     "log_polygon_area": 5+.05*(i % 2), "wetland": 10., "forest": 40.,
                     "agriculture": 30., "urban": 5., "precipitation": 900.,
                     "climate_temperature": 10., "latitude": 40., "longitude": -90.,
                     "median_year": 2000., "doc_median": float(i % 2)+2})
    return pd.DataFrame(rows)


def test_matching_ignores_doc_shape_and_excludes_nested_networks():
    p = matching_panel()
    pairs, _ = morphology_pairs(p, 1, 3)
    assert len(pairs) == 4
    modified = p.assign(doc_median=-999, mainstem_sinuosity=1000, route_mean_scaled=-999)
    altered, _ = morphology_pairs(modified, 1, 3)
    pd.testing.assert_frame_equal(pairs, altered)
    nested = np.zeros((len(p), len(p)), bool)
    nested[0, 1] = True
    excluded, _ = morphology_pairs(p, 1, 3, nested=nested)
    assert len(excluded) == 3
    assert "0" not in set(excluded.station_a)
    assert not excluded.station_a.duplicated().any()
    assert not excluded.station_b.duplicated().any()


def test_calipers_and_geography_are_hard_constraints():
    p = matching_panel()
    p.loc[1, "log_polygon_area"] += np.log(3)
    p.loc[3, "wetland"] += 6
    p.loc[5, "huc4"] = "different"
    selected, balance = morphology_pairs(p, 1, 3)
    assert len(selected) == 1
    assert selected.station_a.iloc[0] == "6"
    assert set(balance.covariate) == set(MATCH_COVARIATES)
    p.loc[0, "wetland"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        morphology_pairs(p, 1, 3)


def test_paired_difference_uses_joint_region_resampling_and_correct_direction():
    p = matching_panel()
    pairs, _ = morphology_pairs(p, 1, 3)
    result, evidence = paired_response_summary(pairs, p, {"doc_median": "raw"}, draws=100)
    np.testing.assert_allclose(result[["difference_b_minus_a", "ci_low", "ci_high"]], 1.)
    assert result.n_huc4.iloc[0] == 4
    assert len(evidence) == 4


def test_conservative_routing_keeps_signal_mass_but_changes_peak():
    single, profile, pulse = geometry_routing([1, 1], [0, 0], 100)
    spread, _, mixed = geometry_routing([1, 1], [0, 1], 100)
    np.testing.assert_allclose(single["routing_pulse_mass_ratio"], 1)
    np.testing.assert_allclose(spread["routing_pulse_mass_ratio"], 1)
    np.testing.assert_allclose(single["routing_pulse_peak"], 1)
    np.testing.assert_allclose(spread["routing_pulse_peak"], .5, atol=1e-10)
    assert spread["routing_pulse_spread"] > single["routing_pulse_spread"]
    np.testing.assert_allclose(profile.area_mass.sum(), 1)
    np.testing.assert_allclose(pulse.routed_anomaly, pulse.no_delay_anomaly)
    np.testing.assert_allclose(mixed.routed_anomaly.sum(), pulse.routed_anomaly.sum())


def test_dimensionless_geometry_invariant_to_length_and_area_rescaling():
    a, profile, pulse = geometry_routing([1, 3], [2, 8], 10)
    b, resized, output = geometry_routing([4, 12], [4, 16], 40)
    for term in ("route_mean_scaled", "route_distance_cv", "routing_pulse_peak", "routing_pulse_spread"):
        np.testing.assert_allclose(a[term], b[term])
    pd.testing.assert_frame_equal(profile, resized)
    pd.testing.assert_frame_equal(pulse, output)


def test_unreachable_routes_remain_explicit_and_bad_area_is_rejected():
    result, profile, _ = geometry_routing([1, 3], [2, np.inf], 10)
    assert result["route_area_coverage"] == .25
    assert result["n_reachable_reaches"] == 1
    np.testing.assert_allclose(profile.area_mass.sum(), 1)
    with pytest.raises(ValueError, match="nonnegative"):
        geometry_routing([-1, 3], [2, 8], 10)


def test_single_region_does_not_create_a_degenerate_confidence_interval():
    p = matching_panel().iloc[:2]
    pairs, _ = morphology_pairs(p, 1, 3)
    result, _ = paired_response_summary(pairs, p, {"doc_median": "raw"}, draws=20)
    assert pd.isna(result.ci_low.iloc[0]) and pd.isna(result.ci_high.iloc[0])
    assert result.interval_status.iloc[0] == "not_estimable_single_HUC4"
