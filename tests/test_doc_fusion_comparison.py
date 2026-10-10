"""Mechanisms, shared initialization and selection isolation for DOC components."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.doc_fusion_comparison import (
    ENVIRONMENT_CHOICES,
    FUSION_CHOICES,
    TEMPORAL_CHOICES,
    DOCFusionModel,
    TemporalAttention,
)
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    graph_view,
)


def inputs():
    torch.manual_seed(6)
    graph = graph_view(np.array([[0, 1], [1, 2]]), np.array([1, 2, 3, 1]), np.arange(4))
    return torch.randn(2, 4, 12, 8), torch.randn(4, 15), torch.randn(2, 2), graph


SPECS = ([{"environment": value} for value in ENVIRONMENT_CHOICES]
         + [{"temporal": value} for value in TEMPORAL_CHOICES]
         + [{"fusion": value} for value in FUSION_CHOICES])


@pytest.mark.parametrize("spec", SPECS)
def test_finite_shapes_gradients_and_common_initialization(spec):
    torch.manual_seed(42)
    baseline = DOCFusionModel(dropout=0)
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    model = DOCFusionModel(**spec, dropout=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in baseline.state_dict().items():
        if key.startswith(("blocks.", "readout.", "combine.")):
            assert torch.equal(value, model.state_dict()[key])
    result = model(*inputs())
    assert result.shape == (2, 4) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_baseline_reproduces_previous_graph_transformer_exactly():
    torch.manual_seed(42)
    old = RiverArchitectureModel("graph_transformer", dropout=0)
    torch.manual_seed(42)
    new = DOCFusionModel(dropout=0)
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    assert torch.equal(old(*inputs()), new(*inputs()))


def test_information_ablation_does_not_read_removed_values():
    windows, environment, season, graph = inputs()
    model = DOCFusionModel(environment="constant", temporal="current", dropout=0).eval()
    altered = windows.clone()
    altered[:, :, :-1] += 100
    assert torch.equal(model(windows, environment, season, graph),
                       model(altered, environment + 100, season, graph))


def test_conditioning_and_residual_control_start_at_concat_with_equal_extra_capacity():
    fixture = inputs()
    models = []
    for choice in ("concat", "conditioned", "residual_concat", "gated"):
        torch.manual_seed(42)
        model = DOCFusionModel(fusion=choice, dropout=0).eval()
        models.append(model)
        torch.testing.assert_close(model(*fixture), models[0](*fixture), atol=1e-6, rtol=1e-6)
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert counts[1] == counts[2] == counts[0] + 600
    with torch.no_grad():
        models[1].fusion_extra.bias[12:] = .2
    assert not torch.allclose(models[1](*fixture), models[0](*fixture))


@pytest.mark.parametrize("spec", [{"temporal": "transformer"}, {"fusion": "conditioned"},
                                 {"environment": "residual_mlp", "fusion": "gated"}])
def test_node_permutation_equivariance(spec):
    windows, environment, season, graph = inputs()
    model = DOCFusionModel(**spec, dropout=0).eval()
    permutation = np.array([2, 0, 3, 1])
    reordered = graph_view(np.array([[0, 1], [1, 2]]), np.array([1, 2, 3, 1]), permutation)
    torch.testing.assert_close(model(windows[:, permutation], environment[permutation], season, reordered),
                               model(windows, environment, season, graph)[:, permutation],
                               atol=1e-6, rtol=1e-5)


def test_temporal_attention_is_causal_at_intermediate_states():
    model = TemporalAttention(8, 12, 3, 12, 0).eval()
    original = torch.randn(2, 12, 8)
    altered = original.clone()
    altered[:, 6:] += 100
    def states(value):
        return model.block(model.project(value) + model.position, src_mask=model.causal_mask)
    assert torch.equal(states(original)[:, :6], states(altered)[:, :6])


def runner_module():
    script_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, script_dir)
    try:
        spec = importlib.util.spec_from_file_location("doc_fusion_runner",
            Path(script_dir) / "run_doc_fusion_component_comparison_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(script_dir)


def test_validation_only_selection_rejects_incomplete_or_nonfinite_seeds():
    runner = runner_module()
    scores = {"mlp": [2., 2., 2.], "linear": [1., 2., 3.], "residual_mlp": [3., 3., 3.],
              "constant": [.1, .1, .1]}
    assert runner.select_choice(scores, runner.STAGES["environment"]["eligible"]) == "mlp"
    scores["linear"] = [1., 1., 1.]
    assert runner.select_choice(scores, runner.STAGES["environment"]["eligible"]) == "linear"
    with pytest.raises(ValueError, match="three seeds"):
        runner.select_choice({"gru": [1., 1.]}, ("gru",))
    with pytest.raises(ValueError, match="nonfinite"):
        runner.select_choice({"gru": [float("nan")] * 3}, ("gru",))


def test_modified_fit_product_is_rejected(tmp_path):
    runner = runner_module()
    (tmp_path / "checkpoint.pt").write_bytes(b"first")
    runner.write_json(tmp_path / "fit_complete.json", {"config_hash": runner.digest({"seed": 42}),
        "files": {"checkpoint.pt": runner.sha256_file(tmp_path / "checkpoint.pt")}})
    runner.verify_package(tmp_path, "fit_complete.json", {"seed": 42})
    (tmp_path / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed artifact"):
        runner.verify_package(tmp_path, "fit_complete.json", {"seed": 42})
