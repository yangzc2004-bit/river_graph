"""Native correction contracts on small synthetic station sequences."""
from __future__ import annotations

import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path

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


def test_optional_extra_head_starts_at_exact_context_and_requires_aligned_features():
    _, args, temporal, decay = fixture(epochs=0)
    model = NativeTemporalResidual(temporal, decay, epochs=0, extra_dim=2)
    for inputs in (args[0], args[4]):
        inputs["extra"] = np.ones((*inputs["age"].shape, 2))
    assert model.head.in_features == 6
    assert torch.count_nonzero(model.head.weight) == 0
    np.testing.assert_array_equal(model.predict_delta(args[0]), np.zeros_like(args[1]))
    np.testing.assert_array_equal(model.predict(args[0], args[1]), args[1])
    fit_model(model, args)
    assert model.to_dict()["config"]["extra_dim"] == 2
    assert model.to_payload()["version"] == 1
    missing = {key: value for key, value in args[0].items() if key != "extra"}
    with pytest.raises(ValueError, match="required"):
        model.predict(missing, args[1])
    malformed = {**args[0], "extra": np.ones((3, 12, 2))}
    with pytest.raises(ValueError, match="extra inputs"):
        model.predict_delta(malformed)
    with pytest.raises(ValueError, match="extra_dim"):
        NativeTemporalResidual(temporal, decay, extra_dim=-1)


def test_current_extra_features_can_explain_signal_missing_from_recurrent_input():
    _, args, temporal, decay = fixture(epochs=0)
    model = NativeTemporalResidual(
        temporal, decay, extra_dim=2, epochs=8, patience=3, batch_size=12,
        learning_rate=1e-3, head_learning_rate=.1)
    rng = np.random.default_rng(704)
    for inputs, truth in ((args[0], args[2]), (args[4], args[6])):
        inputs["encoded"][:] = 0
        numeric = rng.uniform(-1, 1, inputs["age"].shape)
        inputs["extra"] = np.stack([numeric, np.ones_like(numeric)], axis=-1)
        truth[:] = 3 + 1.3 * numeric
    fit_model(model, args)
    assert model.best_epoch_ > 0 and model.selected_scale_ > 0
    assert model.head.weight[0, model.hidden_size] > .5
    assert model.validation_metrics_["validation_mae"] < .2
    full = model.predict_delta(args[4])
    control = copy.deepcopy(args[4])
    control["extra"][..., 0] = 0  # Preserve the freshness flag in column one.
    ablated = model.predict_delta(control)
    np.testing.assert_allclose(
        full - ablated, args[4]["extra"][..., 0] * float(model.head.weight[0, model.hidden_size].detach()),
        atol=1e-13)


def test_extra_is_queried_at_its_own_month_and_does_not_read_labels():
    _, args, temporal, decay = fixture(epochs=0)
    model = NativeTemporalResidual(temporal, decay, epochs=0, extra_dim=2)
    rng = np.random.default_rng(27)
    for inputs in (args[0], args[4]):
        inputs["extra"] = rng.normal(size=(*inputs["age"].shape, 2))
    with torch.no_grad():
        model.head.weight[0, -2:] = torch.tensor([.3, -.2])
    before = model.predict_delta(args[0], batch_size=7)
    changed = copy.deepcopy(args[0])
    changed["extra"][:, 8:] += 100
    after = model.predict_delta(changed, batch_size=7)
    np.testing.assert_array_equal(after[:, :8], before[:, :8])
    assert not np.allclose(after[:, 8:], before[:, 8:])
    changed = copy.deepcopy(args[0])
    changed["extra"][1, 5, 0] += 10
    after = model.predict_delta(changed, batch_size=7)
    unaffected = np.ones_like(after, dtype=bool)
    unaffected[1, 5] = False
    np.testing.assert_array_equal(after[unaffected], before[unaffected])
    assert after[1, 5] - before[1, 5] == pytest.approx(3.)


def test_extra_fit_is_invariant_to_withheld_labels_and_payload_reloads():
    _, original, temporal, decay = fixture(epochs=0)
    rng = np.random.default_rng(27)
    for inputs in (original[0], original[4]):
        inputs["extra"] = rng.normal(size=(*inputs["age"].shape, 2))
    changed = copy.deepcopy(original)
    a = NativeTemporalResidual(temporal, decay, epochs=1, extra_dim=2, head_learning_rate=.05)
    b = NativeTemporalResidual(temporal, decay, epochs=1, extra_dim=2, head_learning_rate=.05)
    original[2][~original[3]] = np.nan
    original[6][~original[7]] = np.nan
    changed[2][~changed[3]] = -1e300
    changed[6][~changed[7]] = np.inf
    fit_model(a, original)
    fit_model(b, changed)
    assert a.to_dict() == b.to_dict()
    np.testing.assert_array_equal(a.predict_delta(original[4]), b.predict_delta(original[4]))
    restored = NativeTemporalResidual.from_payload(a.to_payload())
    assert restored.extra_dim == 2 and restored.to_dict() == a.to_dict()
    np.testing.assert_array_equal(restored.predict(original[4], original[5]),
                                  a.predict(original[4], original[5]))


_OLD_NATIVE_ROOT = Path("experiments/phase4_transfer/doc_tail_residual_v1")
_OLD_NATIVE_CODE = _OLD_NATIVE_ROOT / "code_snapshot/src/river_graph/models/native_temporal_residual.py"
_OLD_NATIVE_WEIGHTS = _OLD_NATIVE_ROOT / "runs/split142_seed42/native_mae.pt"


@pytest.mark.skipif(not (_OLD_NATIVE_CODE.exists() and _OLD_NATIVE_WEIGHTS.exists()),
                    reason="saved original native-residual code and checkpoint are unavailable")
def test_zero_extra_replays_actual_v1_checkpoint_and_preserves_summary_exactly():
    spec = importlib.util.spec_from_file_location("frozen_native_residual_reference", _OLD_NATIVE_CODE)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    payload = torch.load(_OLD_NATIVE_WEIGHTS, map_location="cpu", weights_only=True)
    old = reference_module.NativeTemporalResidual.from_payload(payload)
    new = NativeTemporalResidual.from_payload(payload)
    assert new.extra_dim == 0
    assert new.to_dict() == old.to_dict() == payload["summary"]
    assert "extra_dim" not in new.to_dict()["config"]
    assert "extra_dim" not in new.to_payload()["config"]
    rng = np.random.default_rng(18)
    inputs = {"encoded": rng.normal(size=(2, 17, new.hidden_size)).astype(np.float32),
              "age": rng.uniform(0, 2, (2, 17)).astype(np.float32),
              "support": rng.uniform(0, 1, (2, 17, 3)).astype(np.float32)}
    base = rng.uniform(0, 8, (2, 17))
    np.testing.assert_array_equal(new.predict_delta(inputs, batch_size=7),
                                  old.predict_delta(inputs, batch_size=7))
    np.testing.assert_array_equal(new.predict(inputs, base), old.predict(inputs, base))


def interaction_fixture(*, interaction_indices=(0,), train_memory=False, epochs=12):
    with torch.random.fork_rng():
        temporal = nn.GRUCell(5, 4).double()
        decay = nn.Linear(4, 4).double()
    with torch.no_grad():
        for parameter in (*temporal.parameters(), *decay.parameters()):
            parameter.zero_()
        # A nearly memoryless sign-carrying state gives a clear interaction
        # target; all other hidden coordinates are zero.
        temporal.weight_ih[8, 0] = 1.
        temporal.bias_ih[4:8] = -30.
        decay.bias.fill_(.1)

    def values(n, shift):
        state = np.tile(np.roll([-1., -1., 1., 1.], shift), (n, 4))
        flow = np.tile(np.roll([-1., 1., -1., 1.], shift), (n, 4))
        encoded = np.zeros((n, 16, 4))
        encoded[..., 0] = state
        inputs = {"encoded": encoded, "age": np.ones((n, 16)),
                  "support": np.zeros((n, 16, 3)),
                  "extra": np.stack([flow, np.ones_like(flow)], axis=-1)}
        base = np.full((n, 16), 3.)
        truth = base + np.tanh(state) * flow
        return inputs, base, truth, np.ones((n, 16), dtype=bool)

    model = NativeTemporalResidual(
        temporal, decay, extra_dim=2, interaction_indices=interaction_indices,
        train_memory=train_memory, epochs=epochs, patience=4, batch_size=256,
        head_learning_rate=.2)
    return model, (*values(4, 0), *values(2, 2))


def test_interaction_learns_sign_changing_flow_effect_with_frozen_memory():
    model, args = interaction_fixture()
    additive, additive_args = interaction_fixture(interaction_indices=())
    initial = [copy.deepcopy(module.state_dict()) for module in (model.temporal, model.decay)]
    np.testing.assert_array_equal(model.predict_delta(args[0]), np.zeros_like(args[1]))
    assert model.head.in_features == 4 + 2 + 4
    fit_model(model, args)
    fit_model(additive, additive_args)
    assert model.validation_metrics_["validation_mae"] < .25 * additive.validation_metrics_["validation_mae"]
    assert model.head.weight[0, 6] > .5
    assert model.trainable_parameter_count_ == model.head.weight.numel() + model.head.bias.numel()
    for module, before in zip((model.temporal, model.decay), initial):
        assert not any(parameter.requires_grad for parameter in module.parameters())
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(row["temporal_parameter_distance"] == 0 and row["decay_parameter_distance"] == 0
               for row in model.trace_)
    predicted = model.predict_delta(args[4])
    # Positive flow has opposite correction signs under the two states.
    positive_flow = args[4]["extra"][..., 0] > 0
    high_state = args[4]["encoded"][..., 0] > 0
    assert (predicted[positive_flow & high_state] > 0).all()
    assert (predicted[positive_flow & ~high_state] < 0).all()


def test_zero_numeric_flow_removes_interactions_but_retains_validity_flag_effect():
    model, args = interaction_fixture(epochs=0)
    with torch.no_grad():
        model.head.weight[0, 6:] = torch.tensor([1., 2., 3., 4.])
        model.head.weight[0, 5] = .3
    control = copy.deepcopy(args[0])
    control["extra"][..., 0] = 0
    expected_flags = .3 * control["extra"][..., 1]
    np.testing.assert_allclose(model.predict_delta(control), expected_flags, atol=1e-14)
    assert not np.allclose(model.predict_delta(args[0]), expected_flags)
    control["extra"][..., 1] = 0
    np.testing.assert_array_equal(model.predict_delta(control), 0.)


def test_interaction_future_inputs_cannot_change_earlier_predictions():
    model, args = interaction_fixture(epochs=0)
    with torch.no_grad():
        model.head.weight[0, 6] = 1.
    expected = model.predict_delta(args[0], batch_size=11)
    changed = copy.deepcopy(args[0])
    changed["encoded"][:, 9:] *= -1
    changed["extra"][:, 9:, 0] = .5
    actual = model.predict_delta(changed, batch_size=11)
    np.testing.assert_array_equal(actual[:, :9], expected[:, :9])
    assert not np.allclose(actual[:, 9:], expected[:, 9:])


def test_interaction_payload_preserves_frozen_memory_flags_and_nondefault_config():
    model, args = interaction_fixture(epochs=2)
    fit_model(model, args)
    summary = model.to_dict()
    assert summary["config"]["interaction_indices"] == [0]
    assert summary["config"]["train_memory"] is False
    restored = NativeTemporalResidual.from_payload(model.to_payload())
    assert restored.interaction_indices == (0,) and restored.train_memory is False
    assert restored.to_dict() == summary
    assert not any(p.requires_grad for module in (restored.temporal, restored.decay) for p in module.parameters())
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    default, default_args, *_ = fixture(epochs=0)
    fit_model(default, default_args)
    assert "interaction_indices" not in default.to_dict()["config"]
    assert "train_memory" not in default.to_dict()["config"]


def test_invalid_interaction_definition_and_memory_flag_are_rejected():
    _, _, temporal, decay = fixture(epochs=0)
    for indices in ((0, 0), (2,), (-1,), (True,)):
        with pytest.raises(ValueError, match="interaction"):
            NativeTemporalResidual(temporal, decay, extra_dim=2, interaction_indices=indices)
    with pytest.raises(TypeError, match="train_memory"):
        NativeTemporalResidual(temporal, decay, train_memory="false")
