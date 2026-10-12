"""Meaningful mask, leakage, interpolation and ordinary-kriging controls."""
import json
from pathlib import Path

import numpy as np
import pytest

from river_graph.experiments.doc_interpolation import (
    Interpolator,
    _ordinary,
    temporal_masks,
)


def fixture_support():
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    distance = np.linalg.norm(xy[:, None]-xy[None, :], axis=-1)
    cells = np.array([0, 4, 5, 9, 15])
    return Interpolator(cells, [1., 9., 2., 3., 4.], (4, 5), distance, distance)


def test_time_gap_core_is_fixed_and_support_nested():
    masks = temporal_masks(np.ones((4, 90), bool), 42)
    for gap in (6, 12):
        np.testing.assert_array_equal(masks[3]["test"], masks[gap]["test"])
        np.testing.assert_array_equal(masks[3]["val"], masks[gap]["val"])
        assert set(masks[gap]["train"]) <= set(masks[3]["train"])
        assert not set(masks[gap]["train"]) & set(masks[gap]["test"])
        assert not set(masks[gap]["train"]) & set(masks[gap]["val"])
    for node, val, test in masks[12]["intervals"]:
        assert abs(val-test) >= 24
        source = masks[12]["train"][masks[12]["train"]//90 == node] % 90
        assert len(source) >= 6
        assert source.min() < test-5 and source.max() > test+6


def test_sparse_unbracketed_stations_are_not_fake_temporal_queries():
    observed = np.zeros((2, 100), bool)
    observed[0, [0, 99]] = True
    masks = temporal_masks(observed, 42)
    assert masks[12]["test"].size == 0
    np.testing.assert_array_equal(masks[12]["train"], [0, 99])


def test_hidden_label_cannot_enter_interpolation_support():
    model = fixture_support()
    with pytest.raises(ValueError, match="intersects"):
        model.predict("linear_time", {}, [0])
    predicted, covered, _ = model.predict("linear_time", {}, [2])
    np.testing.assert_allclose(predicted, [5.])
    assert covered.all()
    for family in ("linear_time", "pchip_time"):
        predicted, covered, _ = model.predict(family, {}, [11])
        assert not covered.any()
        assert np.isfinite(predicted).all()  # declared seasonal fallback


def test_idw_and_kriging_reproduce_constant_field_with_missing_support():
    model = fixture_support()
    model.values[:] = 7.
    for family, recipe, cells in (
        ("euclidean_idw", {"k": 3, "power": 2.}, [10]),
        ("ordinary_kriging", {"k": 3, "range": 10., "nugget": .2}, [10]),
        ("spatiotemporal_kriging", {"k": 5, "max_lag": 5, "space_scale": 10., "time_scale": 2., "nugget": .2}, [11]),
        ("spatiotemporal_idw", {"k": 5, "max_lag": 5, "space_scale": 10., "time_scale": 2., "power": 2.}, [11])):
        prediction, intrinsic, _ = model.predict(family, recipe, cells)
        assert intrinsic.all()
        np.testing.assert_allclose(prediction, 7., atol=1e-10)


def test_ordinary_kriging_matches_augmented_lagrange_system():
    covariance = np.array([[1.2, .4, .2], [.4, 1.2, .3], [.2, .3, 1.2]])
    cross, values = np.array([.5, .6, .8]), np.array([2., 4., 9.])
    augmented = np.block([[covariance, np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]])
    weights = np.linalg.solve(augmented, np.r_[cross, 1.])[:3]
    np.testing.assert_allclose(_ordinary(covariance, cross, values), weights @ values)


def test_no_month_donors_reports_abstention_and_explicit_fallback():
    model = fixture_support()
    _, coverage, raw = model.predict("euclidean_idw", {"k": 5, "power": 2.}, [12])
    assert not coverage.any() and np.isnan(raw).all()


@pytest.mark.skipif(not Path("experiments/phase4_transfer/doc_interpolation_v1/protocol.json").exists(),
                    reason="local frozen interpolation protocol has not been generated")
def test_frozen_protocol_keeps_label_boundaries_and_original_paper():
    root = Path("experiments/phase4_transfer/doc_interpolation_v1")
    protocol = json.loads((root/"protocol.json").read_text())
    assert protocol["joint_primary"] == ["mae", "log1p_mae"]
    assert protocol["prior_paper_endpoints_modified"] is False
    from river_graph.experiments.provenance import sha256_file
    for task, expected in protocol["mask_sha256"].items():
        path = root/"masks"/f"{task}.npz"
        assert sha256_file(path) == expected
        with np.load(path) as mask:
            assert not set(mask["train"]) & set(mask["test"])
            assert not set(mask["train"]) & set(mask["val"])
            if task.startswith("space"):
                months = 654
                assert not set(mask["train"]//months) & set(mask["test"]//months)
