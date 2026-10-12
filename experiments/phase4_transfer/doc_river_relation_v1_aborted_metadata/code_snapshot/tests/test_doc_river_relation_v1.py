"""Pin flow direction, omitted-role paths, capacity and matched experiment."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.river_architecture_comparison import RiverArchitectureModel
from river_graph.models.river_relation_comparison import (
    ARMS,
    RiverRelationModel,
    attention_allowed,
    relation_graph,
    station_hop_distances,
)


def topology():
    return np.array([[0, 1, 2, 4], [1, 2, 3, 2]]), np.array([1, 1, 2, 2, 1, 1])


def test_closure_restores_missing_intermediate_without_adding_feature_nodes():
    edge, order = topology()
    rows = np.array([0, 2, 3, 4, 5])
    old = relation_graph(edge, order, rows, "fixed_fusion")
    new = relation_graph(edge, order, rows, "path_closure")
    assert old["relation"][1, 0] == 5
    assert new["relation"][1, 0] == 2 and new["relation"][0, 1] == 4
    assert new["station_hops"][1, 0] == 2
    assert new["restored_ancestor_pairs"] == 2
    torch.testing.assert_close(new["rows"], torch.tensor(rows))
    assert new["relation"].shape == (5, 5)
    assert 1 not in new["rows"]


def test_hops_are_shortest_directed_station_counts_and_not_fake_unconnected_distances():
    edge, _ = topology()
    distances = station_hop_distances(edge, 6)
    assert distances[3, 0] == 3 and distances[0, 3] == -1
    assert distances[3, 4] == 2 and distances[5, 0] == -1
    shortcut = np.column_stack([edge, [0, 2]])
    assert station_hop_distances(shortcut, 6)[3, 0] == 2


def test_distance_basis_distinguishes_direction_and_path_length():
    edge, order = topology()
    graph = relation_graph(edge, order, np.arange(6), "path_bias")
    basis = graph["path_basis"]
    assert basis[3, 0, 1] == 1 and basis[0, 3, 2] == 1
    assert basis[3, 0, 4] == 2 and basis[0, 3, 5] == 2
    assert basis[1, 0, 4] == 1
    assert basis[5, 0, 3] == 1 and not basis[5, 0, 4:].any()
    assert not basis.diagonal(dim1=0, dim2=1)[4:].any()


def test_heads_have_distinct_upstream_downstream_and_global_visibility():
    edge, order = topology()
    graph = relation_graph(edge, order, np.arange(6), "direction_heads")
    allowed = attention_allowed(graph, "direction_heads")
    assert allowed.shape == (3, 6, 6)
    assert allowed[0, 2, 0] and not allowed[0, 2, 3]
    assert allowed[1, 2, 3] and not allowed[1, 2, 0]
    assert allowed[2].all()
    assert all(allowed[h].diag().all() for h in range(3))


def test_upstream_model_is_invariant_to_downstream_and_unconnected_covariates():
    edge, order = topology()
    graph = relation_graph(edge, order, np.arange(6), "upstream_only")
    torch.manual_seed(42)
    model = RiverRelationModel("upstream_only", dropout=0).eval()
    windows, env, season = torch.randn(2, 6, 12, 8), torch.randn(6, 15), torch.randn(2, 2)
    output = model(windows, env, season, graph)
    windows[:, [3, 5]] = 500
    env[[3, 5]] = -500
    changed = model(windows, env, season, graph)
    torch.testing.assert_close(output[:, 2], changed[:, 2], atol=0, rtol=0)
    assert torch.isfinite(changed).all()


def test_equal_parameter_tensors_and_seed_initialization_in_all_recipes():
    torch.manual_seed(42)
    reference = RiverArchitectureModel("graph_transformer").state_dict()
    for arm in ARMS:
        torch.manual_seed(42)
        model = RiverRelationModel(arm)
        assert sum(p.numel() for p in model.parameters()) == 12565
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference[name], atol=0, rtol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_every_recipe_forward_and_gradients_are_finite(arm):
    edge, order = topology()
    graph = relation_graph(edge, order, np.arange(6), arm)
    model = RiverRelationModel(arm)
    output = model(torch.randn(2, 6, 12, 8), torch.randn(6, 15), torch.randn(2, 2), graph)
    assert output.shape == (2, 6) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_runner_count_frame_and_original_optimizer_loop(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("run_doc_river_relation_v1")
    assert len(runner.tasks()) == 144
    assert sum(runner.parent(task) is None for task in runner.tasks()) == 105
    task = {"region": "1013", "arm": "path_bias", "seed": 42, "track": "space", "gap_months": 0}
    part = runner.frame({"y": np.array([[1., 3.]])}, task, np.array([0, 1]), [2., 2.])
    assert runner.metrics(part)["mae"] == 1.
    import inspect
    original = importlib.import_module("run_doc_river_architecture_comparison_v1")
    assert inspect.getsource(runner.train_one).replace("RiverRelationModel(arm,", "RiverArchitectureModel(arm,") == inspect.getsource(original.train_one)


@pytest.mark.skipif(not Path("experiments/phase4_transfer/doc_river_relation_v1/protocol.json").exists(),
                    reason="local directed relation protocol has not been frozen")
def test_frozen_relation_masks_endpoints_and_capacity():
    from river_graph.experiments.provenance import sha256_file
    root = Path("experiments/phase4_transfer/doc_river_relation_v1")
    protocol = json.loads((root/"protocol.json").read_text())
    parent = Path("experiments/phase4_transfer/doc_interpolation_v2/masks")
    for name, expected in protocol["mask_sha256"].items():
        assert sha256_file(root/"masks"/f"{name}.npz") == expected == sha256_file(parent/f"{name}.npz")
    assert protocol["joint_primary"] == ["mae", "log1p_mae"]
    assert protocol["new_fits"] == 105 and protocol["reused_fits"] == 39
    assert protocol["prior_paper_endpoints_modified"] is False
