"""Tests for support-encoder helpers."""

from __future__ import annotations

import numpy as np

from river_graph.models.support_encoder import (
    FEATURE_DIM,
    analytic_blend,
    encode_predict,
    fit_support_encoder,
    residual_idw,
    sample_episodes,
    support_features,
)


def test_support_features_shape():
    y = np.arange(12, dtype=float).reshape(4, 3)
    pred0 = y * 0.5
    hops = lambda a, b: 1 if a != b else 0
    feat = support_features(pred0, y, [0, 3], 6, hops, t=3)
    assert feat.shape == (FEATURE_DIM,)


def test_residual_idw_moves_toward_support():
    y = np.zeros((3, 2))
    y[0, 0] = 10.0
    y[1, 0] = 10.0
    pred0 = np.zeros((3, 2))
    hops = lambda a, b: abs(a - b)
    # t=2: support cells 0 (row0) and 2 (row1); query cell 4 (row2)
    out = residual_idw(pred0, y, [0, 2], [4], hops, t=2)
    assert out[4] > 5.0


def test_analytic_blend_between_base_and_mean():
    y = np.zeros((3, 1))
    y[0, 0] = 8.0
    y[1, 0] = 8.0
    pred0 = np.full((3, 1), 2.0)
    hops = lambda a, b: 0 if a == b else 1
    out = analytic_blend(pred0, y, [0, 1], [2], hops, t=1)
    # tight + close support → almost pure local_mean (8.0)
    assert 7.0 < out[2] <= 8.0


def test_sample_and_fit_encoder():
    rng = np.random.default_rng(0)
    n, t = 12, 20
    y = rng.normal(5, 1, size=(n, t))
    y_mask = rng.random((n, t)) > 0.3
    y = np.where(y_mask, y, 0.0)
    pred0 = y * 0.8
    hops = lambda a, b: abs(a - b)
    batch = sample_episodes(
        pred0, y, y_mask, list(range(n)), hops, t, n_episodes=32, seed=0
    )
    assert batch.feats.shape[0] > 0
    model = fit_support_encoder(batch, epochs=5, seed=0)
    support = [i for i in range(n * t) if y_mask.ravel()[i]][:3]
    query = [i for i in range(n * t) if y_mask.ravel()[i]][3:5]
    out = encode_predict(model, pred0, y, support, query, hops, t)
    assert set(out) == set(query)
    y_flat = y.ravel()
    local = float(y_flat[support].mean())
    p0 = pred0.ravel()
    for q, val in out.items():
        lo, hi = sorted([float(p0[q]), local])
        assert lo - 1e-4 <= val <= hi + 1e-4
