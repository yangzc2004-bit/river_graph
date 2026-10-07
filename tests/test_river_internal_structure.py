"""Structure assignment is independent of DOC and scenario mean travel time."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_internal_structure import (
    assign_profiles,
    classify_geometry,
    path_distribution,
    representatives,
    routing_scenarios,
)


def geometry():
    return pd.DataFrame({"station": ["a", "b", "c", "d", "e"], "comid": range(5), "cluster": [1, 1, 2, 3, 2],
        "tributary_balance": [.1, .1, .3, .3, 0.], "path_cv": [.2, .8, .2, .8, .4],
        "n_junctions": [2, 3, 1, 4, 0], "basin_area_km2": [50., 60., 70., 80., 20.]})


def test_four_profiles_and_junction_free_case():
    g = geometry()
    np.testing.assert_array_equal(assign_profiles(g, .2, .5).profile, [1, 2, 3, 4, 0])
    g.loc[0, "tributary_balance"] = .2
    assert assign_profiles(g, .2, .5).profile.iloc[0] == 3


def test_geometry_thresholds_and_medoid_ignore_doc():
    g = geometry()
    first, cuts = classify_geometry(g)
    g["doc_mean"] = [10000., 0., 1., 2., 3.]
    second, changed_cuts = classify_geometry(g)
    assert cuts == changed_cuts
    np.testing.assert_array_equal(first.profile, second.profile)
    pd.testing.assert_frame_equal(representatives(first), representatives(second))
    with pytest.raises(ValueError, match="one geometric vote"):
        classify_geometry(pd.concat([g, g.iloc[:1]]))


def test_area_weighted_paths_scale_invariance_and_missing_route_coverage():
    area, distance = np.array([1., 3., 2.]), np.array([2., 4., np.inf])
    w, d, metrics = path_distribution(area, distance)
    w2, d2, m2 = path_distribution(10*area, 20*distance)
    np.testing.assert_allclose(w, w2)
    np.testing.assert_allclose(d, d2)
    assert np.dot(w, d) == pytest.approx(1.)
    assert metrics["path_cv"] == pytest.approx(m2["path_cv"])
    assert metrics["represented_area_fraction"] == pytest.approx(4/6)
    with pytest.raises(ValueError, match="mean path"):
        path_distribution([1.], [0.])


def test_pulse_spread_changes_at_fixed_mean_and_conserved_mass():
    w, d, _ = path_distribution([1., 1.], [1., 3.])
    table, _ = routing_scenarios(w, d)
    np.testing.assert_allclose(table.mean_travel_delay, 1., atol=1e-12)
    np.testing.assert_allclose(table.anomaly_mass_fraction, 1., atol=1e-12)
    np.testing.assert_allclose(table.steady_doc, 5., atol=1e-12)
    assert table.pulse_sd.tolist() == sorted(table.pulse_sd, reverse=True)
    assert table.set_index("scenario").loc["zero_spread", "pulse_peak"] == pytest.approx(1.)
