import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_source_placement import (
    UpstreamDistances,
    source_placement,
)


def test_channel_distance_direction_secondary_and_duplicate_edges():
    f = pd.DataFrame({"comid": [1, 2, 3, 4], "hydroseq": [10, 20, 30, 40],
                      "dnhydroseq": [0, 10, 20, 0], "dnminorhyd": [0, 10, 0, 10],
                      "lengthkm": [2., 4., 6., 8.]})
    network = UpstreamDistances(f)
    np.testing.assert_array_equal(network.distances(1, [1, 2, 3, 4]), [1., 4., 9., 6.])
    assert np.isinf(network.distances(3, [1])[0])
    with pytest.raises(ValueError, match="not in routing"):
        network.index(99)


def test_same_amount_different_positions_is_distinguishable():
    a, d = [1., 1., 1.], [1., 30., 100.]
    near = source_placement(a, d, [100., 0., 0.], [0., 0., 100.])
    far = source_placement(a, d, [0., 0., 100.], [100., 0., 0.])
    assert near["wetland_area_km2"] == far["wetland_area_km2"] == 1.
    assert near["wetland_distance_ratio"] < far["wetland_distance_ratio"]
    assert near["wetland_near_excess"] > far["wetland_near_excess"]


def test_missing_cover_and_unreachable_area_remain_missing():
    r = source_placement([2., 1., 1.], [1., np.inf, 10.], [50., 60., np.nan], [25., 30., 40.])
    assert r["represented_area_fraction"] == .5
    assert r["wetland_area_km2"] == 1.
    r = source_placement([1., 1.], [1., 4.], [0., 0.], [30., 20.])
    assert np.isnan(r["wetland_distance_ratio"])
    assert np.isnan(r["wetland_near_excess"])


def test_riparian_enrichment_uses_matched_area_and_detects_incomplete_rows():
    r = source_placement([1., 3.], [1., 100.], [10., 30.], [50., 50.],
        riparian_area=[.1, .3], riparian_wetland=[40., 60.], riparian_forest=[20., np.nan])
    assert np.isclose(r["wetland_riparian_enrichment"], 30.)
    assert np.isnan(r["forest_riparian_enrichment"])
    assert np.isclose(r["forest_riparian_area_coverage"], .25)


def test_invalid_area_and_alignment_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        source_placement([-1.], [1.], [10.], [10.])
    with pytest.raises(ValueError, match="aligned"):
        source_placement([1.], [1., 2.], [10.], [10.])
