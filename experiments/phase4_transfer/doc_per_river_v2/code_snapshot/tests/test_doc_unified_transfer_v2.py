"""Guard same-model lineage, spatial isolation and receiving-label exclusion."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.river_architecture_comparison import covariate_inputs
from river_graph.models.river_relation_comparison import (
    RiverRelationModel,
    relation_graph,
)
from river_graph.models.unified_river_doc import (
    ARMS,
    UnifiedRiverDOC,
    controlled_graph,
    portable_inputs,
)


def topology():
    return np.array([[0, 1, 2, 4], [1, 2, 3, 2]]), np.array([1, 1, 2, 2, 1, 1])


def test_prediction_frame_satisfies_shared_scoring_contract(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("run_doc_unified_transfer_v2")
    data = {"months": ["2000-01", "2000-02"], "y": np.array([[1., 3.]])}
    task = {"split": 142, "seed": 42, "arm": "direction_heads"}
    part = runner.frame(data, np.array([0, 1]), [2., 2.], task, "source_validation")
    result = runner.metrics(part)
    assert result["mae"] == 1. and result["coverage"] == 1.
    assert result["source_q90_mae"] is None


def test_original_direction_model_and_adapter_are_bitwise_identical():
    edges, order = topology()
    torch.manual_seed(42)
    original = RiverRelationModel("direction_heads", dropout=0).eval()
    torch.manual_seed(42)
    adapter = UnifiedRiverDOC(dropout=0).eval()
    inputs = (torch.randn(2, 6, 12, 8), torch.randn(6, 15), torch.randn(2, 2))
    a = original(*inputs, relation_graph(edges, order, np.arange(6), "direction_heads"))
    b = adapter(*inputs, controlled_graph(edges, order, np.arange(6), "direction_heads"))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for arm in ARMS:
        torch.manual_seed(42)
        model = UnifiedRiverDOC(arm, dropout=0)
        assert sum(p.numel() for p in model.parameters()) == 12565
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_self_only_prediction_ignores_all_other_station_inputs():
    edges, order = topology()
    model = UnifiedRiverDOC("self_only", dropout=0).eval()
    graph = controlled_graph(edges, order, np.arange(6), "self_only")
    windows, env, season = torch.randn(2, 6, 12, 8), torch.randn(6, 15), torch.randn(2, 2)
    first = model(windows, env, season, graph)
    windows[:, [0, 1, 3, 4, 5]] = 500
    env[[0, 1, 3, 4, 5]] = -500
    changed = model(windows, env, season, graph)
    torch.testing.assert_close(first[:, 2], changed[:, 2], rtol=0, atol=0)


def test_permutation_preserves_capacity_and_graph_distribution_without_claiming_full_exclusion():
    edges, order = topology()
    original = controlled_graph(edges, order, np.arange(6), "direction_heads")
    shuffled = controlled_graph(edges, order, np.arange(6), "permuted_river")
    torch.testing.assert_close(original["relation"].flatten().sort().values, shuffled["relation"].flatten().sort().values)
    torch.testing.assert_close(original["allowed"].sum(-1).sort(-1).values,
                               shuffled["allowed"].sum(-1).sort(-1).values)
    again = controlled_graph(edges, order, np.arange(6), "permuted_river")
    torch.testing.assert_close(shuffled["relation"], again["relation"], rtol=0, atol=0)
    assert shuffled["permutation_overlap_pairs"] <= shuffled["permutation_connected_pairs"]
    assert all(shuffled["allowed"][h].diag().all() for h in range(3))


def test_portable_preprocessing_replays_source_without_receiving_statistics_or_doc():
    rng = np.random.default_rng(42)
    x = rng.normal(10, 2, (6, 16, 2))
    x[0, 2, 1], x[4, 7, 0] = -1, np.nan
    regime = rng.normal(size=(6, 13))
    regime[:, 0] = 1
    data = {"x": x, "x_mask": rng.random(x.shape) > .2, "regime": regime,
        "static": rng.normal(size=(6, 2)), "feature_channels": ["temperature", "discharge"],
        "months": np.asarray([f"2000-{i+1:02d}" if i < 12 else f"2001-{i-11:02d}" for i in range(16)])}
    source = covariate_inputs(data, [0, 1, 2])
    replay = portable_inputs(data, source["normalization"])
    for key in ("windows", "environment", "season", "order"):
        torch.testing.assert_close(source[key], replay[key], rtol=0, atol=0)
    data["y"] = np.ones((6, 16))
    with pytest.raises(ValueError, match="label-free"):
        portable_inputs(data, source["normalization"])


@pytest.mark.parametrize("arm", ARMS)
def test_matched_controls_have_finite_forward_and_gradient(arm):
    edges, order = topology()
    model = UnifiedRiverDOC(arm)
    output = model(torch.randn(2, 6, 12, 8), torch.randn(6, 15), torch.randn(2, 2),
        controlled_graph(edges, order, np.arange(6), arm))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not Path("experiments/phase4_transfer/doc_unified_transfer_v2/protocol.json").exists(),
    reason="local unified transfer protocol has not been frozen")
def test_frozen_protocol_user_splits_and_external_inputs():
    from river_graph.experiments.provenance import sha256_file
    root = Path("experiments/phase4_transfer/doc_unified_transfer_v2")
    protocol = json.loads((root/"protocol.json").read_text())
    assert len(protocol["tasks"]) == 36 and protocol["splits"] == [142, 143, 144]
    assert protocol["joint_primary"] == ["mae", "log1p_mae"]
    assert protocol["prior_paper_endpoints_modified"] is False
    for split, expected in protocol["mask_sha256"].items():
        assert sha256_file(root/"masks"/f"split{split}.npz") == expected
    assert sha256_file(root/"external/inputs.npz") == protocol["external_input_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(root/"code_snapshot"/name) == expected
