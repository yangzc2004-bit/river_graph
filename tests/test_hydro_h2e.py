"""Tests for the M6 ecological context encoder."""

import torch

from river_graph.models.hydro import TransportGCNImputer


def test_encoder_forward_shape():
    model = TransportGCNImputer(in_channels=10, edge_dim=6, hidden=16,
                                layers=2, env_dim=9, env_emb=16)
    x = torch.randn(5, 10)
    env = torch.randn(5, 9)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    assert model(x, ei, ea, env).shape == (5,)


def test_encoder_embedding_changes_output():
    torch.manual_seed(0)
    model = TransportGCNImputer(in_channels=10, edge_dim=6, hidden=16,
                                layers=2, env_dim=9, env_emb=16)
    model.eval()
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    with torch.no_grad():
        a = model(x, ei, ea, torch.zeros(5, 9))
        b = model(x, ei, ea, torch.randn(5, 9))
    assert not torch.allclose(a, b, atol=1e-6)


def test_encoder_receives_gradients():
    model = TransportGCNImputer(in_channels=10, edge_dim=6, hidden=16,
                                layers=2, env_dim=9, env_emb=16)
    x = torch.randn(5, 10)
    env = torch.randn(5, 9)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    model(x, ei, ea, env).sum().backward()
    grads = [p.grad for p in model.env_encoder.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_without_encoder_unchanged_signature():
    model = TransportGCNImputer(in_channels=10, edge_dim=6, hidden=16)
    x = torch.randn(5, 10)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 6)
    assert model.env_encoder is None
    assert model(x, ei, ea).shape == (5,)
