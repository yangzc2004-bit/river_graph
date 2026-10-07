"""Known-angle mapped fixtures and original matched-pair analysis."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from river_graph.analysis.river_confluence_geometry import (
    angle_degrees,
    local_line,
    measure_junction,
    paired_form_contrasts,
    total_turning,
)


def fixture_lines(reverse=False):
    reaches = pd.DataFrame({"segment": ["branch_a", "branch_b", "common"],
                           "sequence": [0, 0, 0], "comid": [1, 2, 3],
                           "start_measure": [100., 100., 100.], "end_measure": [0., 0., 0.]})
    coords = {1: [(-600., 600.), (0., 0.)], 2: [(600., 600.), (0., 0.)], 3: [(0., 0.), (0., -600.)]}
    return reaches, {i: LineString(xy[::-1] if reverse else xy) for i, xy in coords.items()}


def test_true_junction_angle_and_flow_deflection():
    reaches, lines = fixture_lines()
    measured, local = measure_junction(reaches, lines, .7)
    np.testing.assert_allclose(measured.incoming_angle_deg, 90., atol=1e-10)
    np.testing.assert_allclose(measured.branch_a_deflection_deg, 45., atol=1e-10)
    np.testing.assert_allclose(measured.area_weighted_deflection_deg, 45., atol=1e-10)
    np.testing.assert_allclose(measured.downstream_sinuosity, 1., atol=1e-12)
    np.testing.assert_allclose(measured.downstream_turning_deg, 0., atol=1e-10)
    assert set(local) == {100., 250., 500.}


def test_cached_coordinate_order_does_not_change_geometry():
    r, a = fixture_lines()
    _, b = fixture_lines(reverse=True)
    one, _ = measure_junction(r, a, .7)
    two, _ = measure_junction(r, b, .7)
    pd.testing.assert_frame_equal(one, two)


def test_short_common_trunk_not_extended_to_requested_scale():
    r, a = fixture_lines()
    a[3] = LineString([(0., 0.), (0., -150.)])
    f, _ = measure_junction(r, a, .5)
    assert f.status.tolist() == ["measured", "incomplete_local_geometry", "incomplete_local_geometry"]
    assert f.incoming_angle_deg.iloc[1:].isna().all()
    assert f.common_status.iloc[1] == "mapped_route_shorter_than_scale"


def test_only_local_gaps_control_measurability():
    parts = [LineString([(0., 0.), (150., 0.)]), LineString([(200., 0.), (500., 0.)])]
    line, status, gap = local_line(parts, 100.)
    assert status == "measured" and line.length == 100. and gap == 0.
    line, status, gap = local_line(parts, 250.)
    assert line is None and status == "mapped_gap_exceeds_tolerance" and gap == 50.


def test_small_gap_is_explicitly_included_in_length():
    parts = [LineString([(0., 0.), (100., 0.)]), LineString([(105., 0.), (300., 0.)])]
    line, status, gap = local_line(parts, 250.)
    assert status == "measured" and gap == 5.
    np.testing.assert_allclose(line.coords[-1], (250., 0.))


def test_turning_sampling_and_units():
    line = LineString([(0., 0.), (100., 0.), (100., 100.)])
    assert total_turning(line, step=25.) == pytest.approx(90.)
    assert angle_degrees((1., 0.), (-1., 0.)) == pytest.approx(180.)
    with pytest.raises(ValueError, match="nonzero"):
        angle_degrees((0., 0.), (1., 0.))
    almost_exact = LineString([(0., 0.), (0., 100.+1e-12)])
    assert total_turning(almost_exact, step=25.) == pytest.approx(0.)


def test_junction_endpoint_gap_is_not_hidden_by_parallel_branches():
    r, a = fixture_lines()
    a[3] = LineString([(0., -30.), (0., -600.)])
    f, _ = measure_junction(r, a, .5)
    assert f.status.eq("junction_gap_exceeds_tolerance").all()
    assert f.junction_gap_m.eq(30.).all()


def test_matched_comparison_uses_original_pairs_and_broad_minus_elongated():
    base = {"scale_m": 250., "status": "measured"}
    metrics = ("incoming_angle_deg", "branch_a_deflection_deg", "branch_b_deflection_deg",
               "area_weighted_deflection_deg", "downstream_sinuosity", "downstream_turning_deg")
    g = pd.DataFrame([base | {"station": station} | {m: value for m in metrics}
                      for station, value in (("a", 1.), ("b", 3.), ("c", 4.), ("d", 7.))])
    p = pd.DataFrame({"station_a": ["a", "c"], "station_b": ["b", "d"], "huc4": ["01", "02"]})
    summary, detail = paired_form_contrasts(g, p, draws=500)
    assert summary.n_pairs.eq(2).all() and summary.n_huc4.eq(2).all()
    np.testing.assert_allclose(summary.mean_difference, 2.5)
    assert set(detail.broad_minus_elongated) == {2., 3.}
    mixed = p.assign(class_a=[1, 2], class_b=[3, 3])
    with pytest.raises(ValueError, match="elongated/broad"):
        paired_form_contrasts(g, mixed, draws=500)
