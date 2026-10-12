"""Transfer recipes preserve lineage and exclude receiving DOC from fitting."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.models.river_architecture_comparison import covariate_inputs
from river_graph.models.transfer_robust_doc import (
    GATED,
    RECIPES,
    TransferRobustDOC,
    basin_weights,
    coordinate_policy,
    huc_labels,
    sensor_dropout,
)
from river_graph.models.unified_river_doc import UnifiedRiverDOC, controlled_graph


def test_mixed_huc12_huc8_and_leading_zeros_come_from_metadata():
    nodes = pd.DataFrame({"site_no": ["06438000", "x", "y"], "huc_cd": ["101301010404", "50200050803", "07080001.0"]})
    labels = huc_labels(nodes, ["06438000", "x", "y"])
    assert labels["huc2"].tolist() == ["10", "05", "07"]
    assert labels["huc4"].tolist() == ["1013", "0502", "0708"]


def test_basin_balancing_preserves_total_weight_and_equalizes_domain_weight():
    counts = np.array([10, 40, 100])
    weights = basin_weights(counts, True)
    assert np.isclose(np.dot(weights, counts), counts.sum())
    np.testing.assert_allclose(weights*counts, np.full(3, counts.sum()/3))
    np.testing.assert_array_equal(basin_weights(counts, False), np.ones(3))


def test_sensor_dropout_preserves_padding_season_temperature_and_recomputes_flow_age():
    windows = torch.ones(2, 4, 12, 8)
    windows[:, :, :3] = 0
    windows[:, :, 3:, 6:8] = torch.tensor([1., 0.])
    changed = sensor_dropout(windows, torch.Generator().manual_seed(42), temperature=0, discharge=1)
    torch.testing.assert_close(changed[..., [0, 2, 4, 6, 7]], windows[..., [0, 2, 4, 6, 7]], atol=0, rtol=0)
    assert not changed[..., 1].any() and not changed[..., 3].any()
    assert not changed[:, :, :3, 5].any() and (changed[:, :, 3:, 5] == 1).all()


def test_coordinate_policy_retains_all_other_features_and_leaves_original_untouched():
    original = {"environment": torch.ones(6, 15)}
    changed = coordinate_policy(original, "balanced_no_coords")
    assert (original["environment"] == 1).all()
    assert not changed["environment"][:, 13:15].any()
    assert (changed["environment"][:, :13] == 1).all()


@pytest.mark.parametrize("recipe", RECIPES)
def test_seeded_core_tensors_match_original_and_gradients_are_finite(recipe):
    torch.manual_seed(42)
    original = UnifiedRiverDOC(dropout=0)
    torch.manual_seed(42)
    model = TransferRobustDOC(recipe, dropout=0)
    assert sum(p.numel() for p in model.parameters()) == (12571 if recipe in GATED else 12565)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    graph = controlled_graph(np.array([[0, 1, 3], [1, 2, 2]]), np.ones(5), np.arange(5), "direction_heads")
    windows = torch.randn(2, 5, 12, 8)
    windows[..., 2:4] = 1
    env, season = torch.randn(5, 15), torch.randn(2, 2)
    prediction = model(windows, env, season, graph)
    assert torch.isfinite(prediction).all()
    if recipe not in GATED:
        torch.testing.assert_close(prediction, original(windows, env, season, graph), atol=0, rtol=0)
    (prediction.square().mean()+model.gate_penalty()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_cross_gate_is_the_original_self_only_path():
    torch.manual_seed(42)
    model = TransferRobustDOC("balanced_gated", dropout=0).eval()
    torch.manual_seed(42)
    local = UnifiedRiverDOC("self_only", dropout=0).eval()
    with torch.no_grad():
        model.cross_gate_logits.fill_(-100)
    graph = controlled_graph(np.array([[0, 1], [1, 2]]), np.ones(4), np.arange(4), "direction_heads")
    self_graph = controlled_graph(np.array([[0, 1], [1, 2]]), np.ones(4), np.arange(4), "self_only")
    windows = torch.randn(2, 4, 12, 8)
    windows[..., 2:4] = 1
    env, season = torch.randn(4, 15), torch.randn(2, 2)
    torch.testing.assert_close(model(windows, env, season, graph), local(windows, env, season, self_graph), rtol=1e-6, atol=1e-6)


def test_actual_training_loop_smoke_and_prediction_frame_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("run_doc_unified_generalization_v1")
    rng = np.random.default_rng(42)
    data = {"y": rng.uniform(1, 5, (8, 8)).astype(np.float32), "x": rng.uniform(1, 20, (8, 8, 2)),
        "x_mask": np.ones((8, 8, 2), bool), "regime": np.ones((8, 13)), "static": rng.normal(size=(8, 2)),
        "feature_channels": ["temperature", "discharge"], "months": [f"2000-{i+1:02d}" for i in range(8)]}
    edges, order = np.array([[0, 2, 4], [1, 3, 5]]), np.ones(8)
    rows = {"train": np.arange(6), "val": np.array([6]), "test": np.array([7])}
    split = {role: np.concatenate([np.arange(row*8, row*8+8) for row in ids]) for role, ids in rows.items()}
    inputs = coordinate_policy(covariate_inputs(data, rows["train"]), "combined")
    graphs = {role: controlled_graph(edges, order, ids, "direction_heads") for role, ids in rows.items()}
    ctx = {"rows": rows, "split": split, "inputs": inputs, "graphs": graphs, "months": 8,
        "source_graphs": [controlled_graph(edges, order, ids, "direction_heads") for ids in ([0, 1], [2, 3], [4, 5])],
        "group_ids": ["A", "B", "C"]}
    settings = {"hidden": 24, "temporal_hidden": 12, "heads": 3, "layers": 2, "dropout": .1,
        "learning_rate": .001, "weight_decay": .0001, "max_epochs": 2, "min_epochs": 1, "patience": 2, "batch_months": 4}
    task = {"recipe": "combined", "seed": 42, "domain": "synthetic"}
    model, info, trace = runner.train_model(task, ctx, data, settings, lambda row: None)
    assert len(trace) == 2 and info["parameters"] == 12571
    values = runner.cell_prediction(model, ctx, "val", info["target_normalization"], 4)
    part = runner.frame(data, split["val"], values, task, "source_validation_unit")
    assert np.isfinite(runner.metrics(part)["mae"])
    assert runner.metrics(part)["coverage"] == 1.
    assert len(runner.tasks()) == 63


@pytest.mark.skipif(not Path("experiments/phase4_transfer/doc_unified_generalization_v1/protocol.json").exists(),
    reason="local source-river-unit transfer optimization has not been frozen")
def test_frozen_optimization_roles_and_unchanged_paper_endpoints():
    from river_graph.experiments.provenance import sha256_file
    root = Path("experiments/phase4_transfer/doc_unified_generalization_v1")
    protocol = json.loads((root/"protocol.json").read_text())
    domains = pd.read_csv(root/"source_domains.csv", dtype={"huc2": str})
    assert len(protocol["tasks"]) == 63 and protocol["joint_primary"] == ["mae", "log1p_mae"]
    assert protocol["prior_paper_endpoints_modified"] is False
    for name, expected in protocol["mask_sha256"].items():
        assert sha256_file(root/"masks"/f"{name}.npz") == expected
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(root/"code_snapshot"/name) == expected
    assert set(domains.huc2) == {"05", "06", "07", "08", "10", "11"}
