"""H2X matching: history access, static gates, graph direction and batching."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from river_graph.models.h2x_matched_doc import MatchedInputH2X


def example():
    relation = torch.tensor([[0, 3, 4], [1, 0, 3], [2, 1, 0]])
    return (torch.randn(2, 3, 12, 8), torch.randn(3, 15),
            torch.randn(2, 2), {"relation": relation})


def padded(w):
    return torch.cat([w, torch.zeros_like(w)], -1)


def test_disjoint_batch_matches_released_trunk_per_month():
    torch.manual_seed(7)
    w, e, s, g = example()
    model = MatchedInputH2X(dropout=0.).eval()
    actual = model(padded(w), e, s, g)
    edges = torch.tensor([[0, 1], [1, 2]])
    torch.testing.assert_close(model.directed_edges(g), edges)
    a = torch.cat([e[edges[0]], e[edges[1]]], -1)
    expected = torch.stack([model.trunk(torch.cat([w[i].reshape(3, -1),
                           s[i].expand(3, -1)], -1), edges, a, e)
                           for i in range(2)])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert actual.shape == (2, 3) and torch.isfinite(actual).all()


def test_history_environment_and_directional_gates_receive_gradients():
    torch.manual_seed(8)
    w, e, s, g = example()
    w = padded(w).requires_grad_(True)
    model = MatchedInputH2X(dropout=0.)
    model(w, e, s, g).square().sum().backward()
    assert w.grad[..., 0, :8].abs().sum() > 0
    assert w.grad[..., 8:].eq(0).all()
    for name in ('env_encoder', 'convs'):
        assert sum(p.grad.abs().sum() for p in getattr(model.trunk, name).parameters()) > 0
    for layer in model.trunk.convs:
        for name in ('gate_up', 'gate_down', 'up_lin', 'down_lin'):
            assert sum(p.grad.abs().sum() for p in getattr(layer, name).parameters()) > 0


def test_direction_changes_and_empty_graph_remains_finite():
    torch.manual_seed(9)
    w, e, s, g = example()
    model = MatchedInputH2X(dropout=0.).eval()
    a = model(padded(w), e, s, g)
    b = model(padded(w), e, s, {'relation': g['relation'].T})
    assert not torch.allclose(a, b)
    relation = torch.full((3, 3), 5).fill_diagonal_(0)
    assert torch.isfinite(model(padded(w), e, s, {'relation': relation})).all()


def test_extra_daily_channels_are_refused():
    w, e, s, g = example()
    w = padded(w)
    w[..., 8] = .1
    with pytest.raises(ValueError, match='original monthly covariates only'):
        MatchedInputH2X()(w, e, s, g)


@pytest.mark.skipif(not Path('data/processed/mississippi_graph_graphfix_st357.pt').exists(),
                   reason='local ST357 dataset not available')
def test_frozen_contract_if_available():
    root = Path('experiments/phase4_transfer/doc_h2x_matched_v1')
    if not (root / 'protocol.json').exists():
        pytest.skip('matched H2X protocol not frozen yet')
    from river_graph.experiments.provenance import sha256_file
    protocol = json.loads((root / 'protocol.json').read_text())
    parent = json.loads((Path(protocol['parent']) / 'protocol.json').read_text())
    assert protocol['settings'] == parent['settings']
    assert protocol['joint_primary'] == parent['joint_primary'] == ['mae', 'log1p_mae']
    assert protocol['dataset_sha256'] == parent['dataset_sha256']
    assert protocol['mask_sha256'] == parent['mask_sha256']
    assert len(protocol['tasks']) == 120
    assert sum(t['arm'] == 'h2x_matched' for t in protocol['tasks']) == 24
    assert not protocol['uses_doc_inputs'] and not protocol['uses_daily_features']
    assert not protocol['uses_original_nhd_edge_attributes']
    for name, h in protocol['mask_sha256'].items():
        assert sha256_file(root / 'masks' / f'{name}.npz') == h
