"""Contracts for sparse causal upstream edge--lag attention."""

import torch

from river_graph.models.river_attention import RiverLagAttention


def _toy_inputs():
    torch.manual_seed(401)
    hidden = torch.randn(7, 4, 8)
    # Every edge is source -> target (upstream to downstream).
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 3]])
    edge_attr = torch.randn(3, 5)
    support = torch.rand(7, 4, 3)
    return hidden, edge_index, edge_attr, support


def test_river_lag_attention_shape_and_target_normalization():
    hidden, edge_index, edge_attr, support = _toy_inputs()
    module = RiverLagAttention(
        8, 5, num_heads=2, lags=(0, 1, 3), dropout=0.0
    )
    module.eval()
    output, diagnostics = module(
        hidden, edge_index, edge_attr, support_seq=support,
        return_diagnostics=True,
    )
    assert output.shape == hidden.shape
    lag_mass = diagnostics["lag_mass"]
    assert lag_mass.shape == (7, 4, 2, 3)
    # The softmax is over edge x lag candidates for each (time, target, head).
    for time in range(hidden.shape[0]):
        for target in range(hidden.shape[1]):
            if target in (1, 3):
                torch.testing.assert_close(
                    lag_mass[time, target].sum(-1),
                    torch.ones(2), atol=1e-6, rtol=1e-6,
                )
            else:
                assert torch.equal(output[time, target], torch.zeros(8))


def test_river_lag_attention_cannot_read_future_source_states():
    hidden, edge_index, edge_attr, support = _toy_inputs()
    module = RiverLagAttention(8, 5, num_heads=2, lags=(0, 1, 3), dropout=0.0)
    module.eval()
    with torch.no_grad():
        first = module(hidden, edge_index, edge_attr, support_seq=support)
        changed = hidden.clone()
        changed[4:, 0] += 100.0
        second = module(changed, edge_index, edge_attr, support_seq=support)
    # Source node 0 can reach target 1 only at the same/previous month.
    torch.testing.assert_close(first[:4, 1], second[:4, 1], atol=0, rtol=0)
    assert not torch.equal(first[4:, 1], second[4:, 1])


def test_river_lag_attention_enforces_upstream_edge_orientation():
    hidden, edge_index, edge_attr, support = _toy_inputs()
    module = RiverLagAttention(8, 5, num_heads=2, lags=(0,), dropout=0.0)
    module.eval()
    with torch.no_grad():
        upstream = module(hidden, edge_index, edge_attr, support_seq=support)
        reversed_edges = edge_index.flip(0)
        reversed_out = module(
            hidden, reversed_edges, edge_attr, support_seq=support
        )
    # The operator never symmetrizes edges: reversing the supplied river
    # direction moves messages to different target nodes.
    assert not torch.equal(upstream, reversed_out)
    assert torch.equal(upstream[:, 0], torch.zeros_like(upstream[:, 0]))
    assert torch.equal(reversed_out[:, 3], torch.zeros_like(reversed_out[:, 3]))


def test_river_lag_attention_only_activates_available_lags():
    hidden, edge_index, edge_attr, support = _toy_inputs()
    module = RiverLagAttention(8, 5, num_heads=2, lags=(0, 1, 3), dropout=0.0)
    module.eval()
    _, diagnostics = module(
        hidden, edge_index, edge_attr, support_seq=support,
        return_diagnostics=True,
    )
    # lag_mass has one slot per requested lag; unavailable early months have
    # no mass in the 3-month bucket and all reported mass is finite.
    assert torch.isfinite(diagnostics["lag_mass"]).all()
    assert torch.count_nonzero(diagnostics["lag_mass"][:3, :, :, 2]) == 0


def test_river_lag_attention_chunking_is_numerically_stable():
    hidden, edge_index, edge_attr, support = _toy_inputs()
    # Chunking changes only the batching strategy, not the candidate set or
    # destination-wise normalization.
    torch.manual_seed(402)
    small = RiverLagAttention(
        8, 5, num_heads=2, lags=(0, 1, 3), dropout=0.0, chunk_months=1
    ).eval()
    large = RiverLagAttention(
        8, 5, num_heads=2, lags=(0, 1, 3), dropout=0.0, chunk_months=16
    ).eval()
    large.load_state_dict(small.state_dict())
    with torch.no_grad():
        out_small, diag_small = small(
            hidden, edge_index, edge_attr, support_seq=support,
            return_diagnostics=True,
        )
        out_large, diag_large = large(
            hidden, edge_index, edge_attr, support_seq=support,
            return_diagnostics=True,
        )
    torch.testing.assert_close(out_small, out_large, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        diag_small["lag_mass"], diag_large["lag_mass"], atol=1e-6, rtol=1e-6
    )
