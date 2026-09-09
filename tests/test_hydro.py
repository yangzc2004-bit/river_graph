"""Tests for the directed relational encoder (H1)."""

import torch

from river_graph.models.hydro import DirectedGCNImputer, make_directed_edges


def test_forward_shape():
    model = DirectedGCNImputer(in_channels=10, hidden=16, layers=2)
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert model(x, ei).shape == (5,)


def test_direction_matters():
    # same node features, flipped edges -> different output (unless degenerate)
    torch.manual_seed(0)
    model = DirectedGCNImputer(in_channels=10, hidden=16, layers=2)
    model.eval()
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with torch.no_grad():
        a = model(x, ei)
        b = model(x, ei.flip(0))
    assert not torch.allclose(a, b, atol=1e-6)


def test_separate_relation_weights():
    model = DirectedGCNImputer(in_channels=10, hidden=16, layers=1)
    conv = model.convs[0]
    assert conv.up_conv.lin.weight.shape == conv.down_conv.lin.weight.shape
    assert not torch.equal(conv.up_conv.lin.weight, conv.down_conv.lin.weight)


def test_make_directed_edges_keeps_direction():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    river = make_directed_edges("river", ei, 5)
    assert torch.equal(river, ei)  # NOT symmetrized
    rnd = make_directed_edges("random", ei, 5, seed=42)
    assert rnd.shape == ei.shape
    assert make_directed_edges("none", ei, 5).shape == (2, 0)
