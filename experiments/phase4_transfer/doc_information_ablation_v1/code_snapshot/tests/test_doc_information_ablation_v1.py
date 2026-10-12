"""Guard actual information removal, self-path retention and frozen queries."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.experiments.doc_information_ablation import (
    ARMS,
    ablate_graph,
    ablate_inputs,
    model_arm,
)
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    graph_view,
)


def inputs():
    return {"windows": torch.arange(3*4*12*8, dtype=torch.float32).reshape(3, 4, 12, 8)/100,
            "environment": torch.arange(60, dtype=torch.float32).reshape(4, 15)/10,
            "season": torch.tensor([[0., 1.], [1., 0.], [-1., 0.]]),
            "order": torch.tensor([1, 1, 2, 2])}


def graph():
    return graph_view(np.array([[0, 1, 2], [2, 2, 3]]), np.array([1, 1, 2, 2]), np.arange(4))


def test_environment_removal_and_original_inputs_untouched():
    original = inputs()
    removed = ablate_inputs(original, "no_environment")
    assert torch.count_nonzero(removed["environment"]) == 0
    assert torch.count_nonzero(original["environment"]) > 0
    torch.testing.assert_close(removed["windows"], original["windows"])


def test_time_removal_clears_all_season_mask_and_age_paths():
    removed = ablate_inputs(inputs(), "no_time")
    assert not removed["windows"].any() and not removed["season"].any()
    torch.manual_seed(5)
    model = RiverArchitectureModel("graph_transformer", dropout=0).eval()
    result = model(removed["windows"], removed["environment"], removed["season"], graph())
    torch.testing.assert_close(result[0], result[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(result[1], result[2], atol=1e-6, rtol=0)


def test_current_only_clears_historical_ages_but_preserves_current_values():
    original = inputs()
    removed = ablate_inputs(original, "current_only")
    assert not removed["windows"][:, :, :-1].any()
    assert not removed["windows"][:, :, -1, 4:6].any()
    torch.testing.assert_close(removed["windows"][:, :, -1, :4], original["windows"][:, :, -1, :4])
    torch.testing.assert_close(removed["windows"][:, :, -1, 6:], original["windows"][:, :, -1, 6:])


def test_strict_river_removal_preserves_remaining_environment():
    original = inputs()
    removed = ablate_inputs(original, "no_river")
    assert not removed["environment"][:, :4].any()
    torch.testing.assert_close(removed["environment"][:, 4:], original["environment"][:, 4:])
    assert model_arm("no_river") == "local"
    assert ablate_graph(graph(), "no_river")["edge_count"] == 0


def test_self_only_attention_has_no_other_node_influence():
    torch.manual_seed(7)
    original = ablate_inputs(inputs(), "no_river")
    model = RiverArchitectureModel("local", dropout=0).eval()
    local = ablate_graph(graph(), "no_river")
    result = model(original["windows"], original["environment"], original["season"], local)
    changed = {key: value.clone() for key, value in original.items()}
    changed["windows"][:, 1:] = 500
    changed["environment"][1:, 4:] = -500
    update = model(changed["windows"], changed["environment"], changed["season"], local)
    torch.testing.assert_close(result[:, 0], update[:, 0], atol=0, rtol=0)


def test_relation_removal_keeps_dense_operator_and_generic_self():
    removed = ablate_graph(graph(), "no_relations")
    assert model_arm("no_relations") == "graph_transformer"
    assert torch.equal(removed["relation"].diag(), torch.zeros(4, dtype=torch.long))
    assert torch.all(removed["relation"][~torch.eye(4, dtype=torch.bool)] == 5)
    torch.testing.assert_close(ablate_inputs(inputs(), "no_relations")["environment"], inputs()["environment"])


def test_all_arms_have_identical_parameter_shapes_and_seed_initialization():
    states = []
    for arm in ARMS:
        torch.manual_seed(42)
        states.append(RiverArchitectureModel(model_arm(arm)).state_dict())
    for state in states[1:]:
        assert set(state) == set(states[0])
        for key, value in state.items():
            torch.testing.assert_close(value, states[0][key], atol=0, rtol=0)


def test_runner_uses_frame_metrics_contract(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("run_doc_information_ablation_v1")
    task = {"region": "1013", "arm": "no_time", "seed": 42, "track": "space", "gap_months": 0}
    part = runner.frame({"y": np.array([[1., 3.]])}, task, np.array([0, 1]), [2., 2.])
    assert runner.metrics(part)["mae"] == 1.
    assert len(runner.tasks()) == 141
    assert sum(runner.parent(t) is None for t in runner.tasks()) == 102


@pytest.mark.skipif(not Path("experiments/phase4_transfer/doc_information_ablation_v1/protocol.json").exists(),
                    reason="local information ablation protocol has not been frozen")
def test_frozen_ablation_reuses_exact_parent_masks_and_joint_endpoints():
    from river_graph.experiments.provenance import sha256_file
    root = Path("experiments/phase4_transfer/doc_information_ablation_v1")
    protocol = json.loads((root/"protocol.json").read_text())
    parent = Path("experiments/phase4_transfer/doc_interpolation_v2/masks")
    for name, expected in protocol["mask_sha256"].items():
        assert sha256_file(root/"masks"/f"{name}.npz") == expected == sha256_file(parent/f"{name}.npz")
    assert protocol["joint_primary"] == ["mae", "log1p_mae"]
    assert protocol["new_fits"] == 102 and protocol["reused_fits"] == 39
    assert protocol["prior_paper_endpoints_modified"] is False
