import torch

from river_graph.experiments.temporal_h2x import _context_features, fill_target_channel


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


def test_branching_neighbor_means_use_counts_before_degree_normalization():
    y = torch.tensor([[2.0], [6.0], [10.0], [100.0]])
    visible = torch.tensor([[True], [True], [False], [False]])
    edge = torch.tensor([[0, 1, 3], [2, 2, 2]])
    features = _context_features(y, visible, edge, "all")
    # Node 2 has three upstream nodes, two observed: mean=(2+6)/2=4.
    assert features[0, 2, 2].item() == 4.0
    assert torch.isclose(features[0, 2, 3], torch.tensor(2 / 3))
    downstream = _context_features(y, visible, edge.flip(0), "all")
    assert downstream[0, 2, 4].item() == 4.0
    assert torch.isclose(downstream[0, 2, 5], torch.tensor(2 / 3))


def test_context_is_written_after_visibility_without_overwriting_regime():
    # 10 original inputs, 8 context inputs, 4 hydro regime inputs.
    original = torch.arange(66, dtype=torch.float32).reshape(1, 3, 22)
    y = torch.tensor([[2.0], [6.0], [10.0]])
    visible = torch.tensor([[True], [True], [False]])
    edge = torch.tensor([[0, 1], [2, 2]])
    filled = fill_target_channel(original, y, visible,
                                 context_mode="all", edge_index=edge)
    assert torch.equal(filled[:, :, 10:18], _context_features(y, visible, edge, "all"))
    assert torch.equal(filled[:, :, 18:], original[:, :, 18:])
    assert torch.equal(filled[:, :, :8], original[:, :, :8])
