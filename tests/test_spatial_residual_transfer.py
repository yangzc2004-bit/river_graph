"""Tests for the source-station spatial residual transfer baseline."""

import numpy as np
import torch

from river_graph.models.kgml_local_transport import fit_rf_artifacts
from river_graph.models.spatial_residual_transfer import (
    SIMILARITY_FEATURE_NAMES,
    SpatialResidualTransfer,
    build_similarity_features,
)


def toy_bundle():
    torch.manual_seed(9)
    n, t = 8, 6
    y = 1.0 + torch.rand(n, t) * 4.0
    return {
        "y": y,
        "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.rand(n, t, 2),
        "x_mask": torch.ones(n, t, 2),
        "static": torch.rand(n, 2),
        "regime": torch.rand(n, 13),
        "months": [f"2020-{j + 1:02d}-01" for j in range(t)],
        "edge_index": torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                                     [1, 2, 3, 4, 5, 6, 7]]),
        "edge_attr": torch.rand(7, 6),
    }, {
        "train": np.arange(0, 24),
        "context": np.array([], dtype=np.int64),
        "val": np.arange(24, 30),
        "test": np.arange(30, 48),
    }


def test_similarity_features_are_label_free_and_aligned():
    data, _ = toy_bundle()
    first = build_similarity_features(data)
    changed = {**data, "y": data["y"].clone()}
    changed["y"].reshape(-1)[30:] += 1000
    np.testing.assert_array_equal(first, build_similarity_features(changed))
    assert first.shape == (48, len(SIMILARITY_FEATURE_NAMES))
    assert np.isfinite(first).all()


def test_transfer_fits_only_source_cells_and_ignores_heldout_labels():
    data, split = toy_bundle()
    rf = fit_rf_artifacts(data, split, target_transform="log1p", seed=42,
                          n_estimators=3, n_jobs=1)
    model = SpatialResidualTransfer(seed=42, n_estimators=3, n_jobs=1)
    model.fit(data, split, rf=rf)
    np.testing.assert_array_equal(model.source_cells, split["train"])
    assert set(model.source_stations) == set((split["train"] // data["y"].shape[1]).tolist())
    first = model.predict_components(("train", "val", "context"))["final_pred"]
    changed = {**data, "y": data["y"].clone()}
    changed["y"].reshape(-1)[split["test"]] += 1000
    rf2 = fit_rf_artifacts(changed, split, target_transform="log1p", seed=42,
                           n_estimators=3, n_jobs=1)
    model2 = SpatialResidualTransfer(seed=42, n_estimators=3, n_jobs=1)
    model2.fit(changed, split, rf=rf2)
    second = model2.predict_components(("train", "val", "context"))["final_pred"]
    np.testing.assert_array_equal(first, second)


def test_zero_source_oof_is_rejected():
    data, split = toy_bundle()
    split = {**split, "train": np.array([], dtype=np.int64)}
    model = SpatialResidualTransfer(seed=42, n_estimators=2, n_jobs=1)
    try:
        model.fit(data, split)
    except ValueError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("empty source train set should fail")
