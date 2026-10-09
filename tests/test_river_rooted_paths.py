import pytest
from shapely.geometry import LineString

from river_graph.analysis.river_rooted_paths import rooted_paths


def test_path_lengths_and_common_route_on_known_tree():
    reaches = [("left", LineString([(0, 1), (0, 0)])),
               ("right", LineString([(1, 0), (0, 0)])),
               ("shared", LineString([(0, 0), (0, -2)]))]
    _, paths, summary = rooted_paths(reaches, (0, -2))
    assert [p["path_length_m"] for p in paths] == [3, 3]
    assert summary["shared_terminal_length_m"] == 2
    assert summary["all_digitized_towards_known_outlet"]


def test_rooted_direction_does_not_assume_digitization_order():
    reaches = [("reverse", LineString([(0, -2), (0, 0)]))]
    edges, paths, summary = rooted_paths(reaches, (0, -2))
    assert not summary["all_digitized_towards_known_outlet"]
    assert edges[0]["upstream"] == (0, 0)
    assert paths[0]["path_length_m"] == 2


def test_no_bridging_disconnected_geometries():
    reaches = [("a", LineString([(0, 1), (0, 0)])),
               ("b", LineString([(0.01, 0), (0, -2)]))]
    with pytest.raises(ValueError, match="connected tree"):
        rooted_paths(reaches, (0, -2))
