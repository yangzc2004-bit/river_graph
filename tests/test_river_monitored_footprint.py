"""Branch/trunk identities, partial station paths and source-only timing."""
import numpy as np
import pandas as pd
import pytest
from test_river_structure_atlas import toy_network

from river_graph.analysis.river_monitored_footprint import (
    cropped_corridor,
    geometry_examples,
    matched_pulses,
    partition_footprint,
    timing_components,
)
from river_graph.analysis.river_observed_transport import concentration_proxy
from river_graph.topology.river_structure import ReachNetwork


def test_partition_preserves_partial_lengths_and_unique_common_trunk():
    m, reaches = partition_footprint(ReachNetwork(toy_network()), (1, 50), (2, 100), (4, 50), .25)
    assert m["branch_a_km"] == 1 and m["branch_b_km"] == 3 and m["common_km"] == 6.5
    assert m["path_a_km"] == 7.5 and m["path_b_km"] == 9.5
    assert m["unique_corridor_km"] == 10.5
    assert m["branch_a_storage_fraction"] == 1
    assert not reaches.comid.duplicated().any()
    assert reaches[reaches.segment.eq("common")].comid.tolist() == [3, 4]
    assert m["total_path_cv"] == pytest.approx((1-m["common_fraction"])*m["independent_branch_cv"])


def test_nested_station_pair_and_disconnected_direction_are_rejected():
    n = ReachNetwork(toy_network())
    with pytest.raises(ValueError, match="independent"):
        partition_footprint(n, (1, 100), (3, 100), (4, 0), .5)
    with pytest.raises(ValueError, match="no primary downstream"):
        partition_footprint(n, (4, 100), (2, 100), (3, 0), .5)
    with pytest.raises(ValueError, match="shares"):
        partition_footprint(n, (1, 100), (2, 100), (4, 0), 1.)


def test_shared_trunk_translates_pulse_without_damping_it():
    table, curves = matched_pulses(8., 10., 6., .3)
    table = table.set_index("scenario")
    assert table.loc["actual", "pulse_peak"] == pytest.approx(table.loc["no_common_trunk", "pulse_peak"], abs=1e-14)
    assert table.loc["actual", "pulse_sd"] == pytest.approx(table.loc["no_common_trunk", "pulse_sd"], abs=1e-14)
    a = curves.query("scenario == 'actual'")
    b = curves.query("scenario == 'no_common_trunk'")
    np.testing.assert_allclose(a.outlet_anomaly, b.outlet_anomaly, atol=1e-14)
    np.testing.assert_allclose(a.relative_time.to_numpy()-b.relative_time.to_numpy(), 6/9.4)
    np.testing.assert_allclose(table.anomaly_mass_fraction, 1., atol=1e-12)
    assert table.loc["equal_branches", "pulse_peak"] == 1.
    assert table.loc["actual", "pulse_peak"] < 1.
    assert table.loc["equal_branches", "pulse_centroid"] == pytest.approx(table.loc["actual", "pulse_centroid"])


def test_timing_decomposition_and_hidden_receiver_labels():
    f = pd.DataFrame({"doc_a_now": [2., 4.], "doc_b_now": [5., 6.], "doc_a_previous": [3., 7.],
        "doc_b_previous": [4., 8.], "weight_a": [.3, .4], "path_a_km": [8., 9.], "path_b_km": [10., 12.],
        "common_km": [6., 5.], "y_true": [1., 8.]})
    result = timing_components(f, 12.)
    np.testing.assert_allclose(result.mean_delay_proxy, concentration_proxy(f, "mean_delay", 1., 12.))
    np.testing.assert_allclose(result.actual_branch_proxy, concentration_proxy(f, "branch_arrival", 1., 12.))
    np.testing.assert_allclose(result.actual_branch_proxy-result.same_month_proxy,
        result.shared_delay_input+result.equal_branch_delay_input+result.differential_arrival_input)
    f["y_true"] = [-999., np.nan]
    changed = timing_components(f, 12.)
    pd.testing.assert_frame_equal(result.drop(columns="y_true"), changed.drop(columns="y_true"))
    # Updating the later source observation cannot alter the earlier record.
    f.loc[1, "doc_a_now"] = 999.
    np.testing.assert_array_equal(result.drop(columns="y_true").iloc[0], timing_components(f, 12.).drop(columns="y_true").iloc[0])


def test_equal_paths_or_equal_source_changes_remove_differential_arrival():
    f = pd.DataFrame({"doc_a_now": [2., 2.], "doc_b_now": [5., 5.], "doc_a_previous": [3., 3.],
        "doc_b_previous": [7., 6.], "weight_a": [.3, .4], "path_a_km": [10., 8.], "path_b_km": [10., 12.],
        "common_km": [6., 5.]})
    np.testing.assert_allclose(timing_components(f, 12.).differential_arrival_input, 0., atol=1e-14)


def test_geometric_example_selection_ignores_doc():
    f = pd.DataFrame({"pair_id": list("abcd"), "independent_branch_cv": [.1, .9, .1, .9],
        "common_fraction": [.2, .2, .8, .8], "mean_total_km": [20., 40., 20., 40.]})
    before, cuts = geometry_examples(f)
    f["doc_gain"] = [999., -1., 0., 1000.]
    after, second_cuts = geometry_examples(f)
    pd.testing.assert_frame_equal(before, after)
    assert cuts == second_cuts and len(after) == 4


def test_map_crop_uses_station_measures_with_reversed_coordinate_order():
    from shapely.geometry import LineString

    _, reaches = partition_footprint(ReachNetwork(toy_network()), (1, 50), (2, 100), (4, 50), .25)
    # Alternate head/outlet coordinate order, including the receiver reach.
    lines = {1: LineString([(0, 0), (-4, 2)]), 2: LineString([(4, 2), (0, 0)]),
             3: LineString([(0, -4), (0, 0)]), 4: LineString([(0, -9), (0, -4)])}
    parts, gap = cropped_corridor(reaches, lines)
    assert gap == 0.
    np.testing.assert_allclose(parts[1].coords[0], [-2, 1])
    np.testing.assert_allclose(parts[1].coords[-1], [0, 0])
    np.testing.assert_allclose(parts[4].coords[0], [0, -4])
    np.testing.assert_allclose(parts[4].coords[-1], [0, -6.5])
    assert parts[1].length == pytest.approx(lines[1].length/2)
