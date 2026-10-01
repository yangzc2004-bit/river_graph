"""Contracts for sparse causal upstream edge--lag attention."""

import numpy as np
import torch

from river_graph.experiments.graph_upgrade_v2 import GraphUpgradeModel
from river_graph.models.river_attention import RiverLagAttention


def test_river_lag_attention_shape_and_lag_mass():
    torch.manual_seed(7)
    model = RiverLagAttention(8, 6, num_heads=2, dropout=0.0)
    hidden = torch.randn(9, 4, 8)
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]])
    attrs = torch.randn(3, 6)
    x = torch.randn(9, 4, 10)
    support = torch.rand(9, 4, 3)
    out, diag = model(hidden, edge, attrs, x_seq=x, support_seq=support,
                      return_diagnostics=True)
    assert out.shape == hidden.shape
    assert diag["lag_mass"].shape == (9, 4, 2, 5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        diag["lag_mass"][:, 1:, :].sum(-1),
        torch.ones(9, 3, 2),
        atol=1e-6, rtol=1e-6,
    )
    # The first month has only lag zero candidates.
    assert torch.count_nonzero(diag["lag_mass"][0, ..., 1:]) == 0


def test_river_lag_attention_does_not_read_future_sources():
    torch.manual_seed(8)
    model = RiverLagAttention(8, 6, num_heads=2, dropout=0.0).eval()
    hidden = torch.randn(10, 4, 8)
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]])
    attrs = torch.randn(3, 6)
    with torch.no_grad():
        first = model(hidden, edge, attrs)
        changed = hidden.clone()
        changed[5:] += 100.0
        second = model(changed, edge, attrs)
    torch.testing.assert_close(first[:5], second[:5], atol=0, rtol=0)


def test_graph_upgrade_river_attention_smoke():
    from tests.test_graph_upgrade_v2 import _split, _toy_dataset

    data = _toy_dataset()
    model = GraphUpgradeModel(
        mechanism="m1", temporal_operator="gru_attention_river",
        edge_direction="upstream", lookback=13, seed=42,
        hidden=8, temporal_hidden=8, max_epochs=1, patience=1,
        chunk_months=4,
    )
    pred = model.fit_predict(data, _split())
    assert pred.shape == data["y"].shape
    assert np.isfinite(pred).all()
