"""Tests for the H2 transport-gated encoder."""

import torch

from river_graph.models.hydro import GatedDirectedConv, TransportGCNImputer


def _toy():
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 0->1->2->3
    ea = torch.randn(3, 6)
    return x, ei, ea


def test_transport_forward_shape():
    x, ei, ea = _toy()
    model = TransportGCNImputer(in_channels=10, edge_dim=6, hidden=16, layers=2)
    assert model(x, ei, ea).shape == (5,)


def test_gates_change_output():
    torch.manual_seed(0)
    x, ei, ea = _toy()
    conv = GatedDirectedConv(10, 16, edge_dim=6)
    conv.eval()
    with torch.no_grad():
        a = conv(x, ei, ea)
        b = conv(x, ei, torch.zeros_like(ea))
    assert not torch.allclose(a, b, atol=1e-6)  # edge features matter


def test_direction_still_matters_with_gates():
    torch.manual_seed(0)
    x, ei, ea = _toy()
    conv = GatedDirectedConv(10, 16, edge_dim=6)
    conv.eval()
    with torch.no_grad():
        a = conv(x, ei, ea)
        b = conv(x, ei.flip(0), ea)
    assert not torch.allclose(a, b, atol=1e-6)


def test_no_edges_safe():
    x, _, ea = _toy()
    conv = GatedDirectedConv(10, 16, edge_dim=6)
    out = conv(x, torch.empty((2, 0), dtype=torch.long), ea[:0])
    assert out.shape == (5, 16) and torch.isfinite(out).all()
