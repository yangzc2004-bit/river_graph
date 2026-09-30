"""Contracts for the causal observation-attention upgrade."""

import torch

from river_graph.models.graph_upgrade import (
    CausalObservationAttention,
    ObservationAwareAttentionTemporalTransportGCNImputer,
    ObservationAwareTemporalTransportGCNImputer,
)
from river_graph.models.hydro import TransportGCNImputer


def test_causal_observation_attention_shape_and_normalization():
    torch.manual_seed(101)
    t, n, h, lookback = 7, 3, 8, 4
    module = CausalObservationAttention(h, lookback=lookback, num_heads=2)
    hidden = torch.randn(t, n, h)
    age = torch.rand(t, n)
    support = torch.rand(t, n, 3)
    valid_history = torch.ones(t, n, dtype=torch.bool)
    module.eval()
    output, diag = module(
        hidden, age, support, valid_history, return_diagnostics=True
    )
    assert output.shape == hidden.shape
    weights = diag["weights"]
    assert weights.shape == (t, n, 2, lookback)
    torch.testing.assert_close(
        weights.sum(-1), torch.ones(t, n, 2), atol=1e-6, rtol=1e-6
    )
    # Before the first real month, left-padding is hard masked and receives
    # no attention mass.
    assert torch.count_nonzero(weights[0, ..., :-1]) == 0
    assert torch.count_nonzero(weights[1, ..., :-2]) == 0


def test_causal_observation_attention_cannot_read_future_hidden_states():
    torch.manual_seed(102)
    module = CausalObservationAttention(8, lookback=4, num_heads=2)
    hidden = torch.randn(8, 2, 8)
    age = torch.rand(8, 2)
    support = torch.rand(8, 2, 3)
    module.eval()
    with torch.no_grad():
        first = module(hidden, age, support)
        changed = hidden.clone()
        changed[4:] += 100.0
        second = module(changed, age, support)
    torch.testing.assert_close(first[:4], second[:4], atol=0, rtol=0)
    assert not torch.equal(first[4:], second[4:])


def test_attention_chunking_is_value_equivalent():
    torch.manual_seed(105)
    full = CausalObservationAttention(8, lookback=4, num_heads=2,
                                      chunk_months=99)
    chunked = CausalObservationAttention(8, lookback=4, num_heads=2,
                                         chunk_months=2)
    chunked.load_state_dict(full.state_dict())
    hidden = torch.randn(9, 3, 8)
    age = torch.rand(9, 3)
    support = torch.rand(9, 3, 3)
    full.eval()
    chunked.eval()
    with torch.no_grad():
        output_full, diag_full = full(hidden, age, support,
                                      return_diagnostics=True)
        output_chunked, diag_chunked = chunked(hidden, age, support,
                                               return_diagnostics=True)
    torch.testing.assert_close(output_full, output_chunked, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(diag_full["weights"], diag_chunked["weights"],
                               atol=1e-6, rtol=1e-6)


def test_attention_residual_zero_initialization_matches_m1_gru():
    torch.manual_seed(103)
    spatial_old = TransportGCNImputer(10, 6, hidden=8, layers=2, dropout=0.0)
    spatial_new = TransportGCNImputer(10, 6, hidden=8, layers=2, dropout=0.0)
    old = ObservationAwareTemporalTransportGCNImputer(
        spatial_old, lookback=4, temporal_hidden=8, chunk_months=3
    )
    new = ObservationAwareAttentionTemporalTransportGCNImputer(
        spatial_new, lookback=4, temporal_hidden=8, chunk_months=3,
        attention_heads=2,
    )
    new.spatial.load_state_dict(old.spatial.state_dict())
    new.temporal.load_state_dict(old.temporal.state_dict())
    old.eval()
    new.eval()
    hidden = torch.randn(8, 3, 8)
    age = torch.rand(8, 3)
    support = torch.rand(8, 3, 3)
    with torch.no_grad():
        baseline = old._temporal_windows(hidden, age, support)
        upgraded = new._temporal_windows(hidden, age, support)
    torch.testing.assert_close(upgraded, baseline, atol=0, rtol=0)
    assert torch.count_nonzero(new.attention_residual.weight) == 0
    assert torch.count_nonzero(new.attention_residual.bias) == 0


def test_attention_branch_supports_backward():
    torch.manual_seed(104)
    spatial = TransportGCNImputer(10, 6, hidden=8, layers=1, dropout=0.0)
    model = ObservationAwareAttentionTemporalTransportGCNImputer(
        spatial, lookback=3, temporal_hidden=8, chunk_months=4,
    )
    hidden = torch.randn(6, 2, 8, requires_grad=True)
    age = torch.rand(6, 2)
    support = torch.rand(6, 2, 3)
    # Open the residual for this gradient contract; a freshly constructed
    # model intentionally blocks this branch until the residual projection
    # learns a nonzero scale.
    with torch.no_grad():
        model.attention_residual.weight.copy_(torch.eye(8))
    result = model._temporal_windows(hidden, age, support)
    result.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.temporal_attention.parameters()
    )
