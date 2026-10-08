"""Analytic tributary trees protect non-overlapping geometry attribution."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from river_graph.analysis.river_junction_layout import (
    arrival_scenarios,
    condition_geometric_mainstem,
    partition_mainstem,
)


def tree():
    paths = SimpleNamespace(comids=np.array([10, 20, 30, 40, 50, 60]),
        successor=np.array([-1, 0, 1, 0, 3, 1]), length=np.array([2., 4., 2., 2., 4., 2.]),
        area=np.array([1., 1., 1., 2., 3., 2.]), distance=np.array([1., 4., 7., 3., 6., 7.]))
    return paths, np.arange(1, 7), np.array([24., 16., 8., 6., 4., 2.])


def test_first_attachment_and_incremental_partition():
    p, h, a = tree()
    result = partition_mainstem(p, h, a)
    np.testing.assert_array_equal(result.mainstem_comids, [10, 20, 30])
    u = result.units.set_index("unit_comid")
    assert len(u) == 5
    assert u.loc[40, "area_km2"] == 5
    assert u.loc[40, "n_reaches"] == 2
    assert u.loc[40, "entry_comid"] == 10
    assert u.loc[60, "entry_comid"] == 20
    assert u.loc[40, "entry_position"] == .25
    assert u.loc[60, "entry_position"] == .75
    assert u.area_km2.sum() == p.area.sum()
    assert result.descriptors["lateral_area_fraction"] == .7
    assert result.descriptors["lateral_entry_mean"] == pytest.approx((5*.25+2*.75)/7)
    assert u.loc[40, "mean_path_km"] == pytest.approx(4.8)


def test_two_nested_variance_identities():
    p, h, a = tree()
    r = partition_mainstem(p, h, a)
    d = r.descriptors
    w = p.area/p.area.sum()
    expected = w@((p.distance-w@p.distance)**2)
    assert d["path_variance_km2"] == pytest.approx(expected)
    assert d["within_unit_variance_km2"] == pytest.approx((2*(3-4.8)**2+3*(6-4.8)**2)/10)
    assert d["within_unit_variance_fraction"]+d["between_unit_variance_fraction"] == pytest.approx(1.)
    assert (d["entry_trunk_variance_fraction"]+d["mean_branch_variance_fraction"]
            +d["entry_branch_covariance_fraction"]) == pytest.approx(d["between_unit_variance_fraction"])


def test_row_permutation_keeps_physical_partition():
    p, h, a = tree()
    original = partition_mainstem(p, h, a)
    order = np.array([4, 0, 5, 3, 2, 1])
    inverse = np.argsort(order)
    perm = SimpleNamespace(**{name: getattr(p, name)[order] for name in ("comids", "area", "length", "distance")})
    perm.successor = np.where(p.successor[order] >= 0, inverse[p.successor[order]], -1)
    changed = partition_mainstem(perm, h[order], a[order])
    np.testing.assert_array_equal(original.mainstem_comids, changed.mainstem_comids)
    np.testing.assert_allclose(original.units.sort_values("unit_comid").area_km2,
                               changed.units.sort_values("unit_comid").area_km2)
    for name, value in original.descriptors.items():
        assert changed.descriptors[name] == pytest.approx(value, nan_ok=True)


def test_aligned_unit_means_remove_only_between_variance():
    p, h, a = tree()
    r = partition_mainstem(p, h, a)
    f, curves = arrival_scenarios(r, keep_curves=True)
    f = f.set_index("scenario")
    scale = r.descriptors["mean_path_km"]**2
    assert f.at["unit_means_aligned", "analytic_delay_variance"] == pytest.approx(
        r.descriptors["within_unit_variance_km2"]/scale)
    assert f.at["within_unit_collapsed", "analytic_delay_variance"] == pytest.approx(
        r.descriptors["between_unit_variance_km2"]/scale)
    assert f.at["unit_means_aligned", "pulse_centroid"] == pytest.approx(f.at["common_translation", "pulse_centroid"])
    assert f.at["actual_paths", "pulse_peak"] == f.at["common_translation", "pulse_peak"]
    assert f.at["actual_paths", "duration_80"] == f.at["common_translation", "duration_80"]
    np.testing.assert_allclose(f.anomaly_mass_fraction, 1., atol=1e-12)
    actual = curves[curves.scenario.eq("actual_paths")]
    translated = curves[curves.scenario.eq("common_translation")]
    np.testing.assert_array_equal(actual.outlet_anomaly, translated.outlet_anomaly)
    np.testing.assert_array_equal(actual.centered_time, translated.centered_time)


def test_entry_labels_cannot_change_same_paths_outlet_response():
    p, h, a = tree()
    original = partition_mainstem(p, h, a)
    relabelled = deepcopy(original)
    relabelled.units["entry_position"] = 1-relabelled.units.entry_position
    relabelled.junctions["entry_position"] = 1-relabelled.junctions.entry_position
    first, _ = arrival_scenarios(original)
    second, _ = arrival_scenarios(relabelled)
    np.testing.assert_array_equal(first.pulse_peak, second.pulse_peak)
    np.testing.assert_array_equal(first.pulse_sd, second.pulse_sd)


def test_no_lateral_tributaries_and_zero_area_reaches():
    p, h, a = tree()
    p = SimpleNamespace(**{name: getattr(p, name)[:3] for name in vars(p)})
    p.area[1] = 0
    r = partition_mainstem(p, h[:3], a[:3])
    assert r.junctions.empty
    assert r.descriptors["n_lateral_units"] == 0
    assert np.isnan(r.descriptors["lateral_entry_mean"])
    assert r.descriptors["within_unit_variance_fraction"] == 0
    assert r.units.n_reaches.sum() == 3
    f, _ = arrival_scenarios(r)
    assert np.isfinite(f.pulse_peak).all()


def test_original_mainstem_mismatch_is_exposed():
    p, h, a = tree()
    r = partition_mainstem(p, h, a, geometric_mainstem=[10, 40, 20, 30])
    assert r.descriptors["original_mainstem_link_mismatches"] == 1
    assert r.descriptors["mainstem_reach_jaccard"] == .75


def test_invalid_routing_and_nonunique_areas_rejected():
    p, h, a = tree()
    p.distance[4] += 1
    with pytest.raises(ValueError, match="contiguous saved route"):
        partition_mainstem(p, h, a)
    p, h, a = tree()
    p.comids[4] = p.comids[3]
    with pytest.raises(ValueError, match="unique reach"):
        partition_mainstem(p, h, a)


def test_mapped_trunk_conditioning_uses_real_links_and_keeps_saved_arrays():
    p, h, a = tree()
    # Saved shortest route bypasses reach 40 for source 50.
    p.successor[4] = 0
    p.distance[4] = 4.
    primary = np.where(p.successor >= 0, h[p.successor], 0)
    secondary = np.zeros(6, int)
    secondary[4] = h[3]
    conditioned = condition_geometric_mainstem(p, h, [10, 40, 50], primary, secondary)
    assert p.successor[4] == 0 and p.distance[4] == 4.
    assert conditioned.successor[4] == 3 and conditioned.distance[4] == 6.
    np.testing.assert_array_equal(conditioned.successor[np.arange(6) != 4], p.successor[np.arange(6) != 4])
    r = partition_mainstem(conditioned, h, a, prescribed_mainstem=[10, 40, 50])
    np.testing.assert_array_equal(r.mainstem_comids, [10, 40, 50])
    assert r.units.area_km2.sum() == p.area.sum()


def test_nonedge_trunk_conditioning_rejected():
    p, h, _ = tree()
    primary = np.where(p.successor >= 0, h[p.successor], 0)
    with pytest.raises(ValueError, match="real downstream"):
        condition_geometric_mainstem(p, h, [10, 60], primary, np.zeros(6))
