"""Whole-network physical identities and waterbody segmentation controls."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_storage_transport import gaussian_storage_response
from river_graph.analysis.river_whole_storage import (
    StoragePaths,
    network_responses,
    storage_transfer,
)


def line(lengths=(1., 2., 1.), storage=(False, True, True), ids=(0, 77, 77)):
    n = len(lengths)
    f = pd.DataFrame({"comid": np.arange(1, n+1), "hydroseq": np.arange(n, 0, -1),
        "dnhydroseq": np.arange(n-1, -1, -1), "dnminorhyd": np.zeros(n), "lengthkm": lengths,
        "areasqkm": [1.]+[0.]*(n-1),
        "wbareatype": ["LakePond" if s else None for s in storage], "wbareacomi": ids})
    lengths = np.asarray(lengths)
    distance = np.cumsum(lengths[::-1])[::-1]-lengths/2
    return StoragePaths(f, n, distance)


def test_contiguous_waterbody_is_one_reservoir():
    p = line()
    x = p.inputs()
    np.testing.assert_allclose(x["storage_total"], [3.])
    np.testing.assert_allclose(x["storage_variance"], [9.])
    assert p.descriptors()["mean_serial_elements"] == 1


def test_segmentation_does_not_add_storage_elements():
    p = line()
    split = line((1., 1., 1., 1.), (False, True, True, True), (0, 77, 77, 77))
    a, _ = network_responses(p, sigmas=(.15,), fractions=(0., .5, 1.))
    b, _ = network_responses(split, sigmas=(.15,), fractions=(0., .5, 1.))
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b), check_exact=True)


def test_distinct_serial_waterbodies_are_not_lumped():
    p = line(ids=(0, 77, 78))
    np.testing.assert_allclose(p.inputs()["storage_variance"], [5.])
    assert p.descriptors()["mean_serial_elements"] == 2
    assert p.descriptors()["lumped_storage_variance"] == 9


def test_noncontiguous_same_id_remains_separate():
    p = line(storage=(True, False, True), ids=(77, 0, 77))
    np.testing.assert_allclose(p.inputs()["storage_variance"], [.25+1.])


def test_causal_storage_matches_analytic_single_reservoir():
    p = line()
    rows, curves = network_responses(p, sigmas=(.15,), fractions=(.5,), keep_curves=True)
    row = rows[0]
    trace = pd.DataFrame(curves)
    expected = gaussian_storage_response(trace.time.to_numpy(), 3.5-1.5, .15, 1.5)
    np.testing.assert_allclose(trace.outlet_anomaly, expected, atol=1e-10)
    assert abs(row["numerical_centroid"]-3.5) < 1e-8
    assert abs(row["numerical_sd"]-np.sqrt(.15**2+1.5**2)) < 1e-8
    assert abs(row["anomaly_area_fraction"]-1) < 1e-10
    assert row["steady_gain"] == 1.


def test_no_waterbody_is_exact_baseline_at_all_fractions():
    p = line(storage=(False, False, False), ids=(0, 0, 0))
    rows, _ = network_responses(p, sigmas=(.15,))
    for metric in ("pulse_peak", "pulse_sd", "duration_80", "numerical_centroid"):
        assert len({r[metric] for r in rows}) == 1


def test_within_reach_sources_preserve_mean_and_traversed_storage():
    p = line(storage=(True, True, True), ids=(77, 77, 77))
    x = p.inputs(points=5)
    np.testing.assert_allclose(x["storage_total"], x["delay"])
    assert x["weights"]@x["delay"] == pytest.approx(3.5)
    assert x["weights"]@((x["delay"]-3.5)**2) == pytest.approx(.08)


def test_storage_frequency_has_unit_gain_and_causal_delay_budget():
    p = line()
    value = storage_transfer(p, [0., 1.], 1.)
    np.testing.assert_array_equal(value[:, 0], np.ones(len(p.tau)))
    assert np.all(np.abs(value[:, 1]) <= 1+1e-12)
    x = p.inputs(points=5)
    assert np.all(x["delay"] >= x["storage_total"])


@pytest.mark.parametrize("fraction", [-.1, 1.1])
def test_invalid_storage_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        storage_transfer(line(), [0.], fraction)


def test_shortest_secondary_link_and_primary_tie_preference():
    f = pd.DataFrame({"comid": [1, 2, 3], "hydroseq": [3, 2, 1],
        "dnhydroseq": [2, 1, 0], "dnminorhyd": [1, 0, 0], "lengthkm": [1., 2., 1.],
        "areasqkm": [1., 0., 0.], "wbareatype": [None]*3, "wbareacomi": [0]*3})
    p = StoragePaths(f, 3, [1.5, 2., .5])
    assert p.secondary[0] and p.successor[0] == 2
    f["lengthkm"] = [1., 0., 1.]
    p = StoragePaths(f, 3, [1.5, 1., .5])
    assert not p.secondary[0] and p.successor[0] == 1


def test_changed_saved_distance_is_rejected():
    f = pd.DataFrame({"comid": [1, 2], "hydroseq": [2, 1], "dnhydroseq": [1, 0],
        "dnminorhyd": [0, 0], "lengthkm": [1., 1.], "areasqkm": [1., 0.],
        "wbareatype": [None, None], "wbareacomi": [0, 0]})
    with pytest.raises(ValueError, match="no successor"):
        StoragePaths(f, 2, [2., .5])


def test_unknown_waterbody_identity_is_rejected():
    with pytest.raises(ValueError, match="physical waterbody IDs"):
        line(ids=(0, 0, 0))
