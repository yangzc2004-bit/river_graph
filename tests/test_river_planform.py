"""Analytical checks for real whole-network planform, independent of DOC."""
import numpy as np
import pandas as pd
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon

from river_graph.analysis.river_planform_typology import classify, physical_features
from river_graph.topology.river_planform import (
    UpstreamNetwork,
    metric_geometry_features,
    network_features,
    project_geometry,
    project_lines,
    straight_toy_network,
)


def toy_topology():
    return UpstreamNetwork(pd.DataFrame({
        "comid": [1, 2, 3, 4], "hydroseq": [10, 20, 30, 40],
        "dnhydroseq": [0, 10, 10, 0], "arbolatesu": [4, 2, 1, 5],
        "totdasqkm": [4, 2, 1, 5], "streamorde": [2, 1, 1, 1],
        "divergence": [1, 1, 1, 2],
    }))


def test_primary_upstream_membership_and_mainstem():
    net = toy_topology()
    indices, main = net.membership(1)
    assert net.comid[indices].tolist() == [1, 2, 3]
    assert net.comid[main].tolist() == [1, 2]
    with pytest.raises(ValueError, match="not found"):
        net.membership(99)


def test_secondary_connections_restore_real_upstream_branches():
    frame = toy_topology().frame.copy()
    frame["dnminorhyd"] = [0, 0, 0, 10]
    net = UpstreamNetwork(frame)
    indices, main = net.membership(1)
    assert set(net.comid[indices]) == {1, 2, 3, 4}
    assert net.comid[main].tolist() == [1, 4]


def test_analytical_lengths_basin_and_balance():
    lines = straight_toy_network()
    basin = Polygon([(-150, 0), (150, 0), (150, 300), (-150, 300)])
    result = metric_geometry_features(lines, basin, [1, 2])
    assert result["basin_area_km2"] == pytest.approx(.09)
    assert result["basin_aspect"] == pytest.approx(1)
    total = 100 + 200*np.sqrt(2)
    assert result["channel_length_km"] == pytest.approx(total/1000)
    assert result["mainstem_share"] == pytest.approx((100+100*np.sqrt(2))/total)
    assert result["mainstem_gap_max_m"] == 0


def test_shape_rotation_translation_and_line_orientation_invariance():
    lines = straight_toy_network()
    basin = Polygon([(-150, 0), (150, 0), (150, 300), (-150, 300)])
    original = metric_geometry_features(lines, basin, [1, 2])
    moved = {cid: translate(rotate(line, 63, origin=(0, 0)), 300, 900) for cid, line in lines.items()}
    moved[2] = type(moved[2])(list(moved[2].coords)[::-1])
    b = translate(rotate(basin, 63, origin=(0, 0)), 300, 900)
    result = metric_geometry_features(moved, b, [1, 2])
    for key in original:
        if not key.startswith("outlet_"):
            assert result[key] == pytest.approx(original[key], abs=1e-9)


def test_missing_geometry_is_rejected():
    net = toy_topology()
    indices, main = net.membership(1)
    with pytest.raises(ValueError, match="complete expected"):
        network_features(net, indices, main, {1: straight_toy_network()[1]},
                         Polygon([(-80, 40), (-79, 40), (-79, 41), (-80, 41)]))


def test_empty_mainstem_rejected():
    with pytest.raises(ValueError, match="mainstem required"):
        metric_geometry_features(straight_toy_network(), Polygon([(0, 0), (1, 0), (1, 1)]), [])


def test_bounded_bulk_projection_matches_scalar_operation_exactly():
    lines = {i: LineString([(-90+i*.01, 35), (-90+i*.01, 36)]) for i in range(7)}
    bulk = project_lines(lines, chunk=3)
    for cid, line in lines.items():
        np.testing.assert_array_equal(np.asarray(bulk[cid].coords), np.asarray(project_geometry(line).coords))


def test_branch_absence_is_structural_zero():
    frame = pd.DataFrame({"basin_aspect": [2.], "network_axis_ratio": [3.],
        "drainage_density": [.7], "junction_frequency": [0.], "basin_compactness": [.4],
        "mainstem_share": [1.], "hierarchy_order": [1.], "side_imbalance": [np.nan],
        "tributary_alignment": [np.nan], "confluence_position": [np.nan],
        "tributary_balance": [0.], "n_mainstem_junctions": [0]})
    result = physical_features(frame)
    assert np.isfinite(result.to_numpy()).all()
    assert result.side_imbalance.iloc[0] == result.tributary_alignment.iloc[0] == 0


def classification_fixture():
    rng = np.random.default_rng(12)
    group = np.repeat([0, 1, 2], 25)
    return pd.DataFrame({"station": [f"s{i:03d}" for i in range(75)], "comid": np.arange(75),
        "n_reaches": 100, "basin_aspect": np.exp(.3+group*.7+rng.normal(0, .04, 75)),
        "network_axis_ratio": np.exp(.4+group*.6+rng.normal(0, .05, 75)),
        "basin_compactness": .8-group*.2+rng.normal(0, .02, 75),
        "drainage_density": .6+group*.3+rng.uniform(0, .04, 75),
        "junction_frequency": .05+group*.03+rng.uniform(0, .005, 75),
        "mainstem_share": .8-group*.3+rng.uniform(0, .02, 75),
        "hierarchy_order": 2+group+rng.uniform(0, .02, 75),
        "side_imbalance": .15+group*.25+rng.uniform(0, .02, 75),
        "tributary_alignment": .3+group*.2+rng.uniform(0, .02, 75),
        "confluence_position": .15+group*.2+rng.uniform(0, .02, 75),
        "tributary_balance": .1+group*.1+rng.uniform(0, .02, 75), "n_mainstem_junctions": 10})


def test_classification_uses_geometry_not_doc_and_selects_real_examples():
    frame = classification_fixture()
    first = classify(frame.assign(doc=0), draws=2)
    second = classify(frame.assign(doc=np.arange(75)*100), draws=2)
    np.testing.assert_array_equal(first["classes"].cluster, second["classes"].cluster)
    assert first["selected_k"] == 3
    assert first["representatives"].station.isin(frame.station).all()
    assert first["representatives"].groupby("cluster").size().eq(3).all()
    assert len(first["stability"]) == 12


def test_small_shape_sample_is_not_definitive_taxonomy():
    with pytest.raises(ValueError, match="need 30"):
        classify(classification_fixture().iloc[:29], draws=1)
