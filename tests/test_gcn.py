"""Tests for the GCN model (tiny synthetic graph, CPU, fast)."""

import torch

from river_graph.models.gcn import GCNImputer, make_edge_index


def test_forward_shape():
    model = GCNImputer(in_channels=10, hidden=16, layers=2)
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    out = model(x, ei)
    assert out.shape == (5,)


def test_make_edge_index_variants():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    n = 5
    river = make_edge_index("river", ei, n)
    assert river.shape[1] == 2 * ei.shape[1]  # symmetrized
    rnd = make_edge_index("random", ei, n, seed=42)
    assert rnd.shape[1] == 2 * ei.shape[1]  # same edge count, symmetrized
    none = make_edge_index("none", ei, n)
    assert none.shape == (2, 0)


def test_random_edge_index_deterministic():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    a = make_edge_index("random", ei, 5, seed=42)
    b = make_edge_index("random", ei, 5, seed=42)
    assert torch.equal(a, b)


def test_random_graph_differs_from_river():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    river = make_edge_index("river", ei, 5)
    rnd = make_edge_index("random", ei, 5, seed=42)
    river_pairs = set(map(tuple, river.t().tolist()))
    rnd_pairs = set(map(tuple, rnd.t().tolist()))
    assert river_pairs != rnd_pairs
