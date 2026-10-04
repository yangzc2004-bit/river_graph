"""Nonlinear chemical conditioning keeps the fixed native fitting protocol."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from river_graph.models.nonlinear_chemistry_head import NonlinearChemistryHead


def problem(seed=19):
    rng = np.random.default_rng(seed)
    def side(n):
        features = np.zeros((n, 554), dtype=np.float32)
        features[:, :3] = rng.normal(0, .3, (n, 3))
        features[:, 550:552] = rng.uniform(-1, 1, (n, 2))
        features[:, 552:] = 1
        base = np.full(n, 5., dtype=np.float32)
        truth = base + 1.8 * np.tanh(2*features[:, 550]) + .3*features[:, 0]*features[:, 551]
        active = np.ones(n, dtype=bool); active[::13] = False
        return features, base, truth, active
    a, b = side(100), side(43)
    return (*a[:3], *b[:3]), {"source_active": a[3], "validation_active": b[3],
                            "tail_threshold": float(np.quantile(a[2], .9))}


def test_exact_architecture_parameter_count_and_isolated_seed_initialization():
    rng = torch.random.get_rng_state().clone()
    model = NonlinearChemistryHead(seed=42, epochs=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.trainable_parameter_count_ == 1111
    assert model.head.phi.weight.shape == (8, 4)
    assert model.head.output.weight.shape == (1, 1070)
    assert model.head.weight is model.head.output.weight
    same, other = NonlinearChemistryHead(seed=42), NonlinearChemistryHead(seed=43)
    torch.testing.assert_close(model.head.phi.weight, same.head.phi.weight, rtol=0, atol=0)
    assert not torch.equal(model.head.phi.weight, other.head.phi.weight)
    with pytest.raises(ValueError, match="554"):
        NonlinearChemistryHead(n_features=682)


def test_expansion_uses_standardized_hidden_and_hidden_major_order():
    model = NonlinearChemistryHead()
    x = torch.arange(3*554, dtype=torch.float64).reshape(3, 554)/100
    phi = F.silu(model.head.phi(x[:, 550:]))
    expected = torch.cat((x[:, :550], phi, (x[:, :64, None]*phi[:, None, :]).reshape(3, 512)), dim=1)
    torch.testing.assert_close(model.head.expanded_features(x), expected, rtol=0, atol=0)
    np.testing.assert_array_equal(model.head(x).detach(), np.zeros((3, 1)))


def test_zero_projection_then_gradient_reaches_chemistry_basis():
    model = NonlinearChemistryHead()
    x = torch.randn((6, 554), dtype=torch.float64)
    model.head(x).sum().backward()
    assert torch.count_nonzero(model.head.output.weight.grad) > 0
    assert torch.count_nonzero(model.head.phi.weight.grad) == 0
    with torch.no_grad():
        model.head.output.weight[:, 550:558] = .1
    model.head.zero_grad(set_to_none=True)
    model.head(x).square().mean().backward()
    assert torch.isfinite(model.head.phi.weight.grad).all()
    assert torch.count_nonzero(model.head.phi.weight.grad) > 0


def test_epoch_zero_exact_base_fallback_including_inactive_rows():
    data, kwargs = problem()
    model = NonlinearChemistryHead(epochs=0).fit(*data, **kwargs)
    assert model.selected_scale_ == 0 and model.best_epoch_ == 0
    assert model.to_dict()["phi_parameter_distance"] == 0
    np.testing.assert_array_equal(model.predict(data[3], data[4], active=kwargs["validation_active"]), data[4])
    np.testing.assert_array_equal(model.predict_delta(data[3], active=kwargs["validation_active"]), 0)


def test_actual_learning_preserves_frozen_features_and_inactive_predictions():
    data, kwargs = problem()
    source = torch.tensor(data[0], requires_grad=True)
    model = NonlinearChemistryHead(epochs=30, patience=8, learning_rate=.02,
                                   batch_size=50).fit(source, *data[1:], **kwargs)
    assert model.best_epoch_ > 0
    assert model.validation_metrics_["validation_mae"] < .7 * model.baseline_validation_mae_
    assert model.to_dict()["phi_parameter_distance"] > 0
    assert source.grad is None
    np.testing.assert_array_equal(source.detach().numpy(), data[0])
    predicted = model.predict(data[3], data[4], active=kwargs["validation_active"])
    np.testing.assert_array_equal(predicted[~kwargs["validation_active"]], data[4][~kwargs["validation_active"]])
    np.testing.assert_array_equal(model.predict_delta(data[3], active=kwargs["validation_active"])
                                  [~kwargs["validation_active"]], 0)


def test_source_only_normalization_role_and_unseen_row_independence():
    data, kwargs = problem()
    a = NonlinearChemistryHead(epochs=0).fit(*data, **kwargs)
    changed = list(copy.deepcopy(data)); changed[3] *= 9; changed[5] += 100
    b = NonlinearChemistryHead(epochs=0).fit(*changed, **kwargs)
    assert a.to_dict()["normalization"] == b.to_dict()["normalization"]
    np.testing.assert_allclose(a.feature_mean_, data[0].astype(float).mean(0), atol=1e-15)
    with pytest.raises(ValueError, match="source_validation"):
        NonlinearChemistryHead().fit(*data, **kwargs, selection_role="test")
    # Fixed inference is row-local: altering later months/other stations cannot
    # change already-computed rows, and no DOC labels are an inference argument.
    with torch.no_grad():
        a.head.output.weight[:, 550:558] = .1
    a.selected_scale_ = 1
    original = a.predict(data[3], data[4], active=kwargs["validation_active"])
    future = data[3].copy(); future[20:] *= 3
    altered = a.predict(future, data[4], active=kwargs["validation_active"])
    np.testing.assert_array_equal(original[:20], altered[:20])


def test_mode_controls_change_only_chemistry_and_share_architecture():
    data, kwargs = problem()
    modes = {}
    for mode in ["no_aux", "masks", "chemistry"]:
        values = list(copy.deepcopy(data))
        for i in [0, 3]:
            if mode == "no_aux":
                values[i][:, 550:] = 0
            elif mode == "masks":
                values[i][:, 550:552] = 0
        modes[mode] = NonlinearChemistryHead(epochs=0).fit(*values, **kwargs)
    assert all(model.trainable_parameter_count_ == 1111 for model in modes.values())
    for model in modes.values():
        assert model.n_source_active_ == int(kwargs["source_active"].sum())
        assert model.to_dict()["architecture"] == modes["chemistry"].to_dict()["architecture"]


def test_checkpoint_correct_class_roundtrip_and_deterministic_refit():
    data, kwargs = problem()
    model = NonlinearChemistryHead(epochs=4, batch_size=50).fit(*data, **kwargs)
    summary = model.to_dict(); json.dumps(summary, allow_nan=False)
    assert summary["model_class"] == "NonlinearChemistryHead"
    stream = io.BytesIO(); torch.save(model.to_payload(), stream); stream.seek(0)
    restored = NonlinearChemistryHead.from_payload(torch.load(stream, weights_only=True))
    assert restored.to_dict() == summary
    np.testing.assert_array_equal(restored.predict(data[3], data[4], active=kwargs["validation_active"]),
                                  model.predict(data[3], data[4], active=kwargs["validation_active"]))
    restored.fit(*data, **kwargs)
    assert restored.to_dict() == summary
    bad = copy.deepcopy(model.to_payload()); bad["model_class"] = "FrozenNativeFeatureHead"
    with pytest.raises(ValueError, match="unsupported"):
        NonlinearChemistryHead.from_payload(bad)
    bad = copy.deepcopy(model.to_payload()); bad["head_state"]["phi.weight"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        NonlinearChemistryHead.from_payload(bad)
