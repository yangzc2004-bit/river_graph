import pandas as pd
import pytest

from river_graph.analysis.river_cosampling_pairs import disjoint_pair_inventory


def test_non_frontier_pairs_exclude_nested_catchments_but_keep_parallel_branches():
    f = pd.DataFrame({"station": ["a", "b", "c", "d"], "comid": [1, 2, 3, 4],
        "downstream_candidate_comid": [3, 3, -1, -1], "frontier": [False, False, True, True],
        "routed_area_km2": [1., 2., 4., 3.], "reported_area_km2": [1., 2., 4., 3.]})
    dates = pd.date_range("2000-01-01", periods=90)
    pairs = disjoint_pair_inventory(f, {s: dates for s in f.station}, dates, receiver_area=10.)
    assert set(zip(pairs.source_a, pairs.source_b)) == {("a", "b"), ("a", "d"), ("b", "d"), ("c", "d")}
    assert pairs.both_nearest_frontier.sum() == 1
    f.loc[f.comid.eq(3), "downstream_candidate_comid"] = 1
    with pytest.raises(ValueError):
        disjoint_pair_inventory(f, {s: dates for s in f.station}, dates, receiver_area=10.)
