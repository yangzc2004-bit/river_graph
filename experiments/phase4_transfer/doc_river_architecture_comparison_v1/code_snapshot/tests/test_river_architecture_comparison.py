"""Mechanism and isolation contracts for the matched spatial comparison."""
import numpy as np
import pytest
import torch

from river_graph.models.river_architecture_comparison import (
    ARMS,
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)


def fixture_inputs():
    torch.manual_seed(5)
    windows = torch.randn(2, 4, 12, 8)
    env = torch.randn(4, 15)
    season = torch.randn(2, 2)
    edges = torch.tensor([[0, 1], [1, 2]])
    graph = graph_view(edges, np.array([1, 1, 3, 2]), np.arange(4))
    return windows, env, season, graph


@pytest.mark.parametrize("arm", ARMS)
def test_finite_forward_backward(arm):
    model = RiverArchitectureModel(arm, dropout=0)
    output = model(*fixture_inputs())
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_parameter_count_matches_and_node_reordering_is_equivariant():
    inputs = fixture_inputs()
    permutation = np.array([2, 0, 3, 1])
    counts = []
    for arm in ARMS:
        torch.manual_seed(7)
        model = RiverArchitectureModel(arm, dropout=0).eval()
        counts.append(sum(p.numel() for p in model.parameters()))
        reordered_graph = graph_view(torch.tensor([[0, 1], [1, 2]]),
                                     np.array([1, 1, 3, 2]), permutation)
        result = model(inputs[0][:, permutation], inputs[1][permutation], inputs[2], reordered_graph)
        torch.testing.assert_close(result, model(*inputs)[:, permutation], rtol=1e-5, atol=1e-6)
    assert len(set(counts)) == 1


def test_disconnected_node_cannot_change_gnn_receivers():
    windows, env, season, graph = fixture_inputs()
    changed = windows.clone()
    changed[:, 3] += 100
    for arm in ("local", "directed_gnn", "hierarchical_gnn"):
        model = RiverArchitectureModel(arm, dropout=0).eval()
        torch.testing.assert_close(model(windows, env, season, graph)[:, :3],
                                   model(changed, env, season, graph)[:, :3])


def test_induced_graph_excludes_hidden_role_and_keeps_order_skips():
    edges = np.array([[0, 1, 0], [1, 2, 2]])
    graph = graph_view(edges, np.array([1, 2, 4]), np.array([0, 2]))
    assert graph["edge_count"] == 1
    assert graph["neighbours"][1].tolist() == [1, 0]
    assert graph["relation"][1, 0] == 1
    with pytest.raises(ValueError, match="acyclic"):
        graph_view(np.array([[0, 1], [1, 0]]), np.array([1, 1]), np.arange(2))


def test_windows_are_causal_and_target_labels_are_not_inputs():
    data = {"x": torch.ones(3, 5, 2), "x_mask": torch.ones(3, 5, 2, dtype=torch.bool),
            "regime": torch.ones(3, 13), "static": torch.ones(3, 2),
            "months": [f"2020-{i:02d}" for i in range(1, 6)],
            "feature_channels": ["temperature", "discharge"], "y": torch.ones(3, 5)}
    original = covariate_inputs(data, [0, 1], lookback=3)
    data["x"][2, 4] = 99
    data["y"][:] = 999
    changed = covariate_inputs(data, [0, 1], lookback=3)
    torch.testing.assert_close(original["windows"][:4], changed["windows"][:4])
    assert original["normalization"] == changed["normalization"]
