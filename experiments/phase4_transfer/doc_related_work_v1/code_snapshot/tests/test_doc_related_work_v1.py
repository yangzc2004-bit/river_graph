"""Official-backbone interface, inductive behavior and experiment contracts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_related_work import (
    OFFICIAL,
    RECIPES,
    THIRD_PARTY,
    build_related_model,
    official_file_hashes,
    saits_inputs,
)

AVAILABLE = all((THIRD_PARTY / record["directory"] / name).exists()
                for record in OFFICIAL.values() for name in record["files"])
AVAILABLE = AVAILABLE and (THIRD_PARTY / "python_deps/einops/__init__.py").exists()
official_available = pytest.mark.skipif(not AVAILABLE, reason="pinned official SAITS/GRIN cache or isolated einops unavailable")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(nodes=4, batch=2):
    generator = torch.Generator().manual_seed(981)
    windows = torch.randn(batch, nodes, 12, 8, generator=generator)
    windows[..., 2:4] = 1.
    environment = torch.randn(nodes, 15, generator=generator)
    adjacency = torch.zeros(nodes, nodes)
    adjacency[torch.arange(1, nodes), torch.arange(nodes - 1)] = 1.
    return windows, environment, {"adjacency": adjacency}


def test_saits_doc_channel_is_always_hidden_and_input_masks_are_retained():
    windows, environment, _ = example()
    windows[..., 2] = 0.
    result = saits_inputs(windows, environment)
    assert result["X"].shape == (8, 12, 24)
    assert not result["X"][..., -1].any()
    assert not result["missing_mask"][..., -1].any()
    assert not result["missing_mask"][..., 0].any()
    assert result["missing_mask"][..., 1].all()
    with pytest.raises(ValueError, match="input contract"):
        saits_inputs(windows[..., :7], environment)


@official_available
@pytest.mark.parametrize("method", ["saits", "grin"])
def test_official_forward_gradient_and_checkpoint_replay(method, tmp_path):
    windows, environment, graph = example()
    torch.manual_seed(9)
    model = build_related_model(method, {"width": 8})
    target = torch.ones(2, 4)
    target[0, 2] = np.nan
    loss = model.objective(windows, environment, target, torch.isfinite(target), graph)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)
    model.eval()
    with torch.no_grad():
        prediction = model(windows, environment, None, graph)
    assert prediction.shape == (2, 4) and torch.isfinite(prediction).all()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = build_related_model(method, {"width": 8})
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(windows, environment, None, graph), prediction, rtol=0, atol=0)


@official_available
def test_saits_labels_only_enter_final_month_loss_targets(monkeypatch):
    windows, environment, graph = example()
    model = build_related_model("saits", {"width": 8})
    captured = {}
    original = model.core.forward

    def capture(inputs, stage):
        captured.update(inputs)
        return original(inputs, stage)

    monkeypatch.setattr(model.core, "forward", capture)
    target = torch.arange(8, dtype=torch.float).reshape(2, 4)
    visible = torch.ones(2, 4, dtype=torch.bool)
    visible[0, 0] = False
    model.objective(windows, environment, target, visible, graph)
    assert not captured["X"][..., -1].any()
    assert not captured["missing_mask"][..., -1].any()
    expected = torch.zeros(8, 12, 24)
    expected[:, -1, -1] = visible.reshape(8)
    torch.testing.assert_close(captured["indicating_mask"], expected)
    torch.testing.assert_close(captured["X_holdout"][:, -1, -1], torch.where(visible, target, 0.).reshape(8))


@official_available
def test_grin_has_no_doc_inputs_or_node_specific_parameters_and_is_equivariant(monkeypatch):
    windows, environment, graph = example()
    model = build_related_model("grin", {"width": 8}).eval()
    assert model.core.emb is None
    assert model.core.fwd_rnn.h0 is None and model.core.bwd_rnn.h0 is None
    captured = []
    original = model.core.forward

    def capture(x, adj, **kwargs):
        captured.append((x, kwargs["mask"], kwargs["cached_support"]))
        return original(x, adj, **kwargs)

    monkeypatch.setattr(model.core, "forward", capture)
    with torch.no_grad():
        baseline = model(windows, environment, None, graph)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = model(windows[:, permutation], environment[permutation], None,
                         {"adjacency": graph["adjacency"][permutation][:, permutation]})
        torch.testing.assert_close(permuted, baseline[:, permutation], rtol=1e-5, atol=1e-6)
        # Same weights accept different river/node counts, then the original graph.
        small_windows, small_env, small_graph = example(nodes=3)
        assert model(small_windows, small_env, None, small_graph).shape == (2, 3)
        torch.testing.assert_close(model(windows, environment, None, graph), baseline, rtol=0, atol=0)
    assert all(not x.any() and not mask.any() and cache is False for x, mask, cache in captured)


def runner_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("doc_related_work_runner", scripts / "run_doc_related_work_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


@official_available
@pytest.mark.parametrize("method", ["saits", "grin"])
def test_target_label_poisoning_does_not_change_training_or_source_selection(method, tmp_path):
    runner = runner_module()
    windows, environment, _ = example(nodes=4, batch=14)
    data = {"y": np.arange(56, dtype=np.float32).reshape(4, 14) / 10 + 1.,
        "site_no": ["a", "b", "v", "t"], "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "months": [f"2020-{m:02d}" for m in range(1, 13)] + ["2021-01", "2021-02"]}
    split = {"train": np.array([0, 1, 2, 14, 15, 16]), "val": np.array([28, 29]), "test": np.array([42, 43])}
    inputs = {"windows": windows, "environment": environment, "order": torch.ones(4)}
    rows = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}
    ctx = {"months": 14, "inputs": inputs, "split": split, "rows": rows,
        "train_y": runner.role_labels(data, split, "train", rows["train"]),
        "graphs": {role: runner.graph_for(data, inputs, rows["train"] if role == "train"
            else np.sort(np.r_[rows["train"], rows[role]])) for role in ("train", "val", "test")}}
    settings = {"learning_rate": .001, "weight_decay": .0001, "max_epochs": 2,
        "min_epochs": 1, "patience": 1, "batch_months": 2, "batch_cells": 3}
    config = {"method": method, "recipe": {"width": 8}, "seed": 42, "settings": settings,
        "runtime_snapshot_hash": "code", "dataset_sha256": "dataset", "mask_sha256": "mask"}
    clean = runner.fit_one(tmp_path / "clean", config, ctx, data, lambda _: None)
    poisoned = {**data, "y": data["y"].copy()}
    poisoned["y"][3] = np.nan
    other = runner.fit_one(tmp_path / "poisoned", config, ctx, poisoned, lambda _: None)
    assert clean["target_normalization"] == other["target_normalization"]
    assert clean["validation_mae"] == other["validation_mae"]
    assert clean["validation_log1p_mae"] == other["validation_log1p_mae"]
    first = runner.load_model(tmp_path / "clean")
    second = runner.load_model(tmp_path / "poisoned")
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert runner.replay_validation(first, method, tmp_path / "clean", ctx, settings, clean["target_normalization"]) == 0.
    with pytest.raises(ValueError, match="config changed"):
        runner.fit_one(tmp_path / "clean", {**config, "seed": 43}, ctx, data, lambda _: None)
    (tmp_path / "clean/checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed package artifact"):
        runner.fit_one(tmp_path / "clean", config, ctx, data, lambda _: None)


DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
ROOT = Path("experiments/phase4_transfer/doc_related_work_v1")


@pytest.mark.skipif(not AVAILABLE or not DATA.exists() or not (ROOT / "protocol.json").exists(),
    reason="local ST357 data, official dependency cache or related-work freeze unavailable")
def test_frozen_official_sources_masks_inputs_and_historical_dispatcher():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["new_fits"] == 60 and protocol["reused_reference_fits"] == 15
    assert protocol["recipes"] == RECIPES
    assert protocol["joint_primary"] == ["native_mae", "log1p_mae"]
    assert protocol["official_sources"] == official_file_hashes()
    assert not protocol["existing_paper_endpoints_modified"]
    assert "previously inspected" in protocol["scope"]
    assert "no DOC context" in protocol["inputs"]
    assert "all regions" in protocol["test_gate"]
    assert sha256_file("scripts/run_ladder.py") == protocol["original_dispatcher_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT / "code_snapshot" / name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected
    months = torch.load(DATA, weights_only=False)["y"].shape[1]
    for region, expected in protocol["mask_sha256"].items():
        path = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks") / f"huc4_{region}.npz"
        assert sha256_file(path) == expected
        with np.load(path) as split:
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                assert not np.intersect1d(split[a] // months, split[b] // months).size
