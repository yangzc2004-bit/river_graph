"""Contract tests for the H2X-T spatial/temporal extension."""

import numpy as np
import torch

from river_graph.experiments.temporal_h2x import (
    H2XTemporalModel,
    build_temporal_inputs,
    fill_target_channel,
)
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.temporal import TemporalTransportGCNImputer


def _toy_dataset():
    torch.manual_seed(7)
    n, t = 5, 7
    return {
        "y": torch.rand(n, t) * 3.0,
        "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.rand(n, t, 2),
        "x_mask": torch.ones(n, t, 2, dtype=torch.bool),
        "months": [f"2020-{j:02d}-01" for j in range(1, t + 1)],
        "site_no": [str(i) for i in range(n)],
        "static": torch.rand(n, 2),
        "regime": torch.rand(n, 13),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "edge_attr": torch.rand(3, 6),
    }


def _split(n=5, t=7):
    flat = np.arange(n * t)
    return {
        "train": flat[:15],
        "val": flat[15:20],
        "test": flat[20:],
    }


def test_transport_encode_decode_preserves_forward():
    torch.manual_seed(1)
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    model = TransportGCNImputer(10, 6, hidden=16, layers=2)
    model.eval()
    with torch.no_grad():
        old = model(x, ei, ea)
        new = model.predict_from_hidden(model.encode_nodes(x, ei, ea))
    assert torch.equal(old, new)


def test_temporal_windows_are_causal_and_cover_all_months():
    torch.manual_seed(2)
    spatial = TransportGCNImputer(10, 6, hidden=16, layers=2)
    temporal = TemporalTransportGCNImputer(
        spatial, lookback=3, temporal_hidden=16, chunk_months=2
    )
    h = torch.randn(7, 5, 16)
    temporal.eval()
    with torch.no_grad():
        first = temporal._temporal_windows(h)
        changed_future = h.clone()
        changed_future[4:] += 100.0
        second = temporal._temporal_windows(changed_future)
    assert first.shape == (7, 5)
    assert torch.equal(first[:4], second[:4])
    assert not torch.equal(first[4:], second[4:])


def test_batched_month_encoding_matches_independent_spatial_calls():
    torch.manual_seed(4)
    spatial = TransportGCNImputer(10, 6, hidden=16, layers=2)
    temporal = TemporalTransportGCNImputer(
        spatial, lookback=3, temporal_hidden=16, chunk_months=2
    )
    x = torch.randn(5, 5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    env = torch.randn(5, 4)
    temporal.eval()
    with torch.no_grad():
        batched = temporal.encode_months(x, ei, ea, env)
        independent = torch.stack(
            [spatial.encode_nodes(x[j], ei, ea, env) for j in range(x.shape[0])]
        )
    assert torch.allclose(batched, independent, atol=1e-6, rtol=1e-6)


def test_history_ablation_modes_keep_current_step_and_are_valid():
    torch.manual_seed(5)
    spatial = TransportGCNImputer(10, 6, hidden=16, layers=2)
    x = torch.randn(5, 4, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    for mode in ("shuffle", "hydro_only"):
        temporal = TemporalTransportGCNImputer(
            spatial, lookback=3, temporal_hidden=16, chunk_months=2,
            history_ablation=mode,
        )
        out = temporal(x, ei, ea)
        assert out.shape == (5, 4)


def test_temporal_input_visibility_fills_only_visible_target_cells():
    dataset = _toy_dataset()
    split = _split()
    inputs = build_temporal_inputs(
        dataset, split, target_transform="log1p", env_encoder=True
    )
    visible = torch.zeros_like(inputs.y_model, dtype=torch.bool)
    visible.reshape(-1)[[0, 3]] = True
    xt = fill_target_channel(inputs.xt_static, inputs.y_model, visible)
    assert xt.shape[0:2] == (dataset["y"].shape[1], dataset["y"].shape[0])
    assert xt[0, 0, 9].item() == 1.0
    assert xt[0, 0, 8].item() != 0.0
    assert xt[0, 1, 9].item() == 0.0
    assert xt[0, 1, 8].item() == 0.0


def test_temporal_model_smoke_fit_predict_all_months():
    dataset = _toy_dataset()
    split = _split()
    model = H2XTemporalModel(
        seed=42,
        lookback=3,
        temporal_hidden=16,
        hidden=16,
        max_epochs=2,
        patience=1,
    )
    pred = model.fit_predict(dataset, split)
    assert pred.shape == tuple(dataset["y"].shape)
    assert np.isfinite(pred).all()
    assert np.std(pred) > 0
