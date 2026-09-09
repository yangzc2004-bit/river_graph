"""Tests for H1.5 stabilization options (shared weights, edge dropout)."""

import torch

from river_graph.models.hydro import DirectedConv, DirectedGCNImputer


def test_shared_weights_parameter_count():
    shared = DirectedGCNImputer(10, hidden=16, layers=2, share_weights=True)
    plain = DirectedGCNImputer(10, hidden=16, layers=2, share_weights=False)
    n_shared = sum(p.numel() for p in shared.parameters())
    n_plain = sum(p.numel() for p in plain.parameters())
    assert n_shared < n_plain


def test_shared_weights_forward_and_direction():
    torch.manual_seed(0)
    model = DirectedGCNImputer(10, hidden=16, layers=2, share_weights=True)
    model.eval()
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with torch.no_grad():
        a = model(x, ei)
        b = model(x, ei.flip(0))
    assert a.shape == (5,)
    assert not torch.allclose(a, b, atol=1e-6)  # direction still matters


def test_edge_dropout_only_in_training():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    conv = DirectedConv(10, 16)
    conv.eval()
    assert torch.equal(conv._edge_dropout(ei, 0.9, training=False), ei)
    conv.train()
    torch.manual_seed(0)
    kept = conv._edge_dropout(ei, 0.9, training=True)
    assert kept.shape[1] < ei.shape[1]  # most edges dropped with p=0.9
