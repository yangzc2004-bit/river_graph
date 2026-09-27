import torch

from river_graph.experiments.temporal_h2x import _context_features


def test_global_context_excludes_own_value_and_uses_visible_cells_only():
    y = torch.tensor([[1.0], [3.0], [9.0]])
    visible = torch.tensor([[True], [True], [False]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    features = _context_features(y, visible, edge_index, "global")

    assert features.shape == (1, 3, 2)
    # Node 0 sees node 1, node 1 sees node 0, and the hidden node 2 sees both
    # visible values. The own value is excluded from the global mean.
    assert torch.allclose(features[0, :, 0], torch.tensor([3.0, 1.0, 2.0]))
    assert torch.allclose(features[0, :, 1], torch.tensor([0.5, 0.5, 2.0 / 2.0]))


def test_hidden_label_perturbation_does_not_change_context():
    y = torch.tensor([[1.0], [3.0], [9.0]])
    visible = torch.tensor([[True], [True], [False]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    first = _context_features(y, visible, edge_index, "all")
    changed = y.clone()
    changed[2, 0] = 900.0
    second = _context_features(changed, visible, edge_index, "all")
    assert torch.equal(first, second)
