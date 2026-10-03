"""Native correction contracts on small synthetic station sequences."""
from __future__ import annotations

import copy
import inspect
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from river_graph.models.native_temporal_residual import NativeTemporalResidual


def fixture(*, epochs=3, tail_weight=1, zero_base=False):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        temporal = nn.GRUCell(5, 4).double()
        decay = nn.Linear(4, 4).double()
        with torch.no_grad():
            decay.weight.fill_(.05)
            decay.bias.fill_(.1)
    rng = np.random.default_rng(14)

    def inputs(n):
        return {"encoded": rng.normal(size=(n, 13, 4)),
                "age": rng.uniform(0, 2, (n, 13)),
                "support": rng.uniform(0, 1, (n, 13, 3))}

    source, val = inputs(3), inputs(2)
    source_mask, val_mask = np.ones((3, 13), dtype=bool), np.ones((2, 13), dtype=bool)
    source_mask[:, 2] = False
    val_mask[:, [1, 4, 7]] = False
    base = 0. if zero_base else 3.
    source_base, val_base = np.full((3, 13), base), np.full((2, 13), base)
    source_truth, val_truth = source_base + 1., val_base + 1.
    model = NativeTemporalResidual(
        temporal, decay, epochs=epochs, patience=2, batch_size=8,
        learning_rate=1e-3, head_learning_rate=.05, tail_weight=tail_weight)
    return model, (source, source_base, source_truth, source_mask,
                   val, val_base, val_truth, val_mask), temporal, decay


def fit_model(model, args):
    return model.fit(*args, tail_threshold=4.5, selection_role="source_validation")


def test_zero_head_and_epoch_zero_preserve_context_bitwise():
    model, args, *_ = fixture(epochs=0)
    np.testing.assert_array_equal(model.predict_delta(args[0]), np.zeros_like(args[1]))
    base = args[1].astype(np.float32)
    predicted = model.predict(args[0], base)
    np.testing.assert_array_equal(predicted, base)
    assert predicted.dtype == base.dtype
    fit_model(model, args)
    assert model.best_epoch_ == 0
    assert model.selected_scale_ == 0
    assert model.trace_[0]["validation_scale"] == 0
    np.testing.assert_array_equal(model.predict(args[4], args[5]), args[5])


def test_synthetic_native_error_improves_and_backbone_updates_without_original_mutation():
    model, args, temporal, decay = fixture()
    prior = [copy.deepcopy(module.state_dict()) for module in (temporal, decay)]
    reports = []
    model.fit(*args, tail_threshold=4.5, selection_role="source_validation", progress=reports.append)
    assert model.best_epoch_ > 0 and model.selected_scale_ > 0
    assert model.validation_metrics_["validation_mae"] < .5
    assert model.trace_[1]["temporal_parameter_distance"] > 0
    assert model.trace_[1]["decay_parameter_distance"] > 0
    assert model.trace_[1]["head_parameter_norm"] > 0
    assert reports == model.trace_
    assert model.best_epoch_ == np.argmin([row["validation_mae"] for row in model.trace_])
    for module, before in zip((temporal, decay), prior):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    summary = model.to_dict()
    assert summary["trainable_parameter_count"] == sum(
        parameter.numel() for module in (temporal, decay, model.head) for parameter in module.parameters())
    json.dumps(summary, allow_nan=False)


def test_zero_native_base_has_effective_initial_gradient_and_can_increase():
    model, args, *_ = fixture(zero_base=True)
    prepared = model._prepare_inputs(args[0])
    delta = model._delta_cells(prepared, np.array([0, 1]))
    prediction = model._combine(torch.zeros(2, dtype=torch.float64), delta, 1.)
    (prediction - 1.).abs().mean().backward()
    assert model.head.bias.grad.item() < 0
    fit_model(model, args)
    assert model.selected_scale_ > 0
    prediction = model.predict(args[4], args[5])
    assert np.mean(prediction[args[7]]) > .1
    assert model.validation_metrics_["validation_mae"] < 1


def test_context_fallback_wins_when_source_correction_harms_validation():
    model, args, *_ = fixture(epochs=5)
    args[6][:] = args[5] - 1.
    fit_model(model, args)
    assert model.selected_scale_ == 0 and model.best_epoch_ == 0
    assert model.epochs_run_ == 2
    assert model.to_dict()["head_parameter_norm"] == 0
    np.testing.assert_array_equal(model.predict(args[4], args[5]), args[5])
    assert model.trace_[1]["head_parameter_norm"] > 0  # Attempted training is still recorded.


def test_unselected_truth_and_base_values_do_not_affect_fitting():
    a, original, *_ = fixture(epochs=1, tail_weight=2)
    b, changed, *_ = fixture(epochs=1, tail_weight=2)
    for args, value in ((original, np.nan), (changed, -1e300)):
        args[1][~args[3]] = value
        args[2][~args[3]] = value
        args[5][~args[7]] = value
        args[6][~args[7]] = value
    fit_model(a, original)
    fit_model(b, changed)
    assert a.to_dict() == b.to_dict()
    np.testing.assert_array_equal(a.predict_delta(original[4]), b.predict_delta(original[4]))
    assert list(inspect.signature(a.predict_delta).parameters) == ["full_inputs", "batch_size"]
    with pytest.raises(ValueError, match="source_validation"):
        a.fit(*original, tail_threshold=4.5, selection_role="outer_test")


def test_tail_weight_uses_only_source_truth_and_validation_remains_unweighted():
    model, args, *_ = fixture(epochs=1, tail_weight=2)
    args[2][0, [0, 1]] = [4.5, 9.]  # The threshold tie is included in Q90.
    args[6][0, 0] = 20.
    fit_model(model, args)
    summary = model.to_dict()
    assert summary["n_source_tail_cells"] == 2
    assert summary["source_tail_fraction"] == pytest.approx(2 / args[3].sum())
    initial_score = np.abs(args[5][args[7]] - args[6][args[7]]).mean()
    assert model.trace_[0]["validation_mae"] == pytest.approx(initial_score)
    assert summary["config"]["tail_weight"] == 2


def test_causal_prediction_ignores_future_cached_inputs_with_nonzero_head():
    model, args, *_ = fixture(epochs=0)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[.2, -.3, .1, .4]], dtype=torch.float64))
        model.head.bias.fill_(.07)
    first = model.predict_delta(args[0], batch_size=7)
    changed = {key: array.copy() for key, array in args[0].items()}
    for array in changed.values():
        array[:, 8:] += 100
    second = model.predict_delta(changed, batch_size=7)
    np.testing.assert_array_equal(second[:, :8], first[:, :8])
    assert not np.allclose(second[:, 8:], first[:, 8:])
    np.testing.assert_allclose(model.predict_delta(args[0], batch_size=41), first, atol=1e-14)


def test_trained_torch_payload_roundtrip_preserves_predictions_and_summary():
    model, args, *_ = fixture()
    fit_model(model, args)
    buffer = io.BytesIO()
    torch.save(model.to_payload(), buffer)
    buffer.seek(0)
    restored = NativeTemporalResidual.from_payload(torch.load(buffer, weights_only=True))
    np.testing.assert_array_equal(restored.predict_delta(args[4]), model.predict_delta(args[4]))
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    assert restored.to_dict() == model.to_dict()


def test_nonfinite_and_invalid_shapes_are_rejected():
    model, args, temporal, decay = fixture(epochs=0)
    with pytest.raises(ValueError, match="scale zero"):
        NativeTemporalResidual(temporal, decay, scales=(.5, 1.))
    with pytest.raises(ValueError, match="tail_weight"):
        NativeTemporalResidual(temporal, decay, tail_weight=10)
    args[2][0, 0] = np.inf
    with pytest.raises(ValueError, match="selected native"):
        fit_model(model, args)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        model._combine(torch.ones(1), torch.tensor([float("inf")]), 1.)
