"""Temporal refits keep observed histories while excluding query supervision."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_doc_temporal_compatibility_v1 import (
    fit_temporal_backbone,
    temporal_arrays,
    temporal_label_view,
)

from river_graph.models.unified_doc import UnifiedDOCReconstructor


def test_temporal_label_view_preserves_only_train_val_and_explicit_context():
    dataset = {"y": np.arange(42.).reshape(3, 14), "ph": np.ones((3, 14)),
               "spec_conductance": np.ones((3, 14))*100}
    split = {"train": np.array([0, 1, 14, 15]), "val": np.array([9, 23]),
             "context": np.array([12, 26]), "test": np.array([11, 25, 40])}
    first = temporal_label_view(dataset, split)
    altered = copy.deepcopy(dataset)
    altered["y"].ravel()[split["test"]] = 1e20
    np.testing.assert_array_equal(first["y"], temporal_label_view(altered, split)["y"])
    assert "ph" not in first and "spec_conductance" not in first
    assert np.isnan(first["y"].ravel()[split["test"]]).all()
    np.testing.assert_array_equal(first["y"].ravel()[split["context"]], dataset["y"].ravel()[split["context"]])
    assert np.isfinite(dataset["y"]).all()  # No mutation of the scoring dataset.


def test_temporal_arrays_hide_whole_source_station_but_keep_legitimate_local_history():
    n, t = 3, 14
    full = np.zeros((n, t, 23), np.float32)
    full[:, 0, 8:10] = 1  # Visible old DOC; hidden query slots remain zero.
    features = {"source_station_ids": np.array([0, 1]), "source_raw": np.zeros((2, t, 23)),
        "source_age": np.ones((2, t)), "source_support": np.zeros((2, t, 3)),
        "source_env": np.zeros((2, 9)), "full_raw": full, "full_age": np.ones((n, t)),
        "full_support": np.zeros((n, t, 3)), "env": np.zeros((n, 9))}
    extra = {"source_extra": np.zeros((2, t, 30)), "full_extra": np.zeros((n, t, 30))}
    split = {"train": np.array([0, 14]), "val": np.array([9, 23]), "test": np.array([11, 25]),
             "context": np.empty(0, int)}
    inputs, arrays = temporal_arrays(features, extra, np.zeros((n, t, 8)),
        {"y": np.ones((n, t))}, split, np.zeros((n, t)), np.ones((n, t)))
    assert not arrays[0]["raw"][..., 8:10].any()
    assert arrays[4]["raw"][:, 0, 8:10].all()
    np.testing.assert_array_equal(inputs["raw"], full)
    assert arrays[3].sum() == 2 and arrays[7].sum() == 2
    assert not arrays[7].ravel()[split["test"]].any()


def test_temporal_backbone_accepts_sparse_val_sites_and_restores_predictions(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(14)
    n, t = 6, 16
    dataset = {"site_no": np.array([f"station{i}" for i in range(n)]),
        "months": np.arange("2000-01", "2001-05", dtype="datetime64[M]"),
        "y": rng.uniform(1, 6, (n, t)), "y_mask": np.ones((n, t), bool),
        "x": rng.uniform(1, 10, (n, t, 2)), "x_mask": np.ones((n, t, 2), bool),
        "regime": rng.uniform(0, 1, (n, 13)), "static": rng.uniform(1, 10, (n, 2)),
        "edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), "edge_attr": rng.uniform(0, 1, (4, 2))}
    split = {"train": np.flatnonzero(np.tile(np.arange(t) < 10, (n, 1))),
        "val": np.flatnonzero(np.tile(np.isin(np.arange(t), [10, 11]), (n, 1))),
        "test": np.flatnonzero(np.tile(np.arange(t) > 11, (n, 1))), "context": np.empty(0, int)}
    view = temporal_label_view(dataset, split)
    expert = fit_temporal_backbone(view, split, seed=42, epochs=1, n_estimators=4)
    prediction = expert.predict_components()["hybrid_pred"]
    assert np.isfinite(prediction).all()
    assert len(split["val"]) == 12  # Every site has only two validation labels.
    expert.save(tmp_path/"backbone")
    restored = UnifiedDOCReconstructor.load(tmp_path/"backbone", view, split)
    np.testing.assert_allclose(restored.predict_components()["hybrid_pred"], prediction, atol=1e-10)
