"""Raw-encoding, visibility and copied-weight contracts for native residuals."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_temporal_adapter import gathered_rolling_states
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.native_temporal_residual import NativeTemporalResidual


def fixture(mode="frozen", *, epochs=2, dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        spatial = TransportGCNImputer(5, 2, hidden=4, layers=2, dropout=.7,
                                     env_dim=2, env_emb=3, edge_direction="upstream").to(dtype=dtype)
        temporal = nn.GRUCell(5, 4).to(dtype=dtype)
        decay = nn.Linear(4, 4).to(dtype=dtype)
        with torch.no_grad():
            for conv in spatial.convs:
                conv.self_lin.weight.fill_(.1)
                conv.self_lin.bias.fill_(.15)
            for parameter in spatial.env_encoder.parameters():
                parameter.fill_(.2)
            decay.weight.fill_(.05)
            decay.bias.fill_(.1)
    rng = np.random.default_rng(14)

    def inputs(n):
        return {"raw": rng.uniform(.1, 1., (n, 14, 5)), "env": rng.uniform(.2, 1., (n, 2)),
                "age": rng.uniform(0, 2, (n, 14)), "support": rng.uniform(0, 1, (n, 14, 3)),
                "extra": rng.uniform(-.2, .2, (n, 14, 30))}

    source, validation = inputs(3), inputs(2)
    source_mask, val_mask = np.ones((3, 14), dtype=bool), np.ones((2, 14), dtype=bool)
    source_mask[:, 2] = False
    val_mask[:, [1, 4, 7]] = False
    source_base, val_base = np.full(source_mask.shape, 3.), np.full(val_mask.shape, 3.)
    model = EncoderNativeResidual(
        spatial, temporal, decay, encoder_mode=mode, encoder_learning_rate=1e-3,
        epochs=epochs, patience=2, batch_size=12, learning_rate=1e-3, head_learning_rate=.04)
    return model, (source, source_base, source_base + 1., source_mask,
                   validation, val_base, val_base + 1., val_mask), spatial, temporal, decay


def fit_model(model, args):
    return model.fit(*args, tail_threshold=4., selection_role="source_validation")


def cached_inputs(model, inputs):
    prepared = model._prepare_inputs(inputs)
    n, t, c = prepared["raw"].shape
    with torch.no_grad():
        encoded = model.spatial.encode_nodes(
            prepared["raw"].reshape(n * t, c), torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 2), dtype=model.dtype),
            prepared["env"][:, None].expand(n, t, -1).reshape(n * t, -1)).reshape(n, t, -1)
    return {"encoded": encoded, **{key: prepared[key] for key in ("age", "support", "extra")}}


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-13), (torch.float32, 2e-6)])
def test_frozen_raw_path_matches_cached_states_padding_and_nonzero_head(dtype, atol):
    model, args, _, temporal, decay = fixture(epochs=0, dtype=dtype)
    cached = cached_inputs(model, args[0])
    cells = np.array([0, 1, 6, 11, 12, 13, 14, 17, 41])
    raw_hidden = model._hidden_cells(model._prepare_inputs(args[0]), cells)
    old_hidden = gathered_rolling_states(temporal, decay, cached, cells // 14, cells % 14, lookback=12)
    torch.testing.assert_close(raw_hidden, old_hidden, rtol=atol, atol=atol)
    old = NativeTemporalResidual(temporal, decay, extra_dim=30, interaction_indices=(0, 2, 4, 28))
    with torch.no_grad():
        model.head.weight.fill_(.03)
        model.head.bias.fill_(.2)
        old.head.load_state_dict(model.head.state_dict())
    np.testing.assert_allclose(model.predict_delta(args[0], batch_size=9),
                               old.predict_delta(cached, batch_size=9), rtol=atol, atol=atol)
    assert not model.spatial.training


def test_frozen_raw_training_matches_cached_protocol_and_has_no_encoder_gradient():
    model, args, spatial, temporal, decay = fixture()
    old = NativeTemporalResidual(temporal, decay, epochs=2, patience=2, batch_size=12,
                                 learning_rate=1e-3, head_learning_rate=.04,
                                 tail_weight=2, extra_dim=30, interaction_indices=(0, 2, 4, 28))
    cached_source, cached_val = cached_inputs(model, args[0]), cached_inputs(model, args[4])
    old_args = (cached_source, *args[1:4], cached_val, *args[5:])
    before = copy.deepcopy(spatial.state_dict())
    fit_model(model, args)
    fit_model(old, old_args)
    assert model.best_epoch_ == old.best_epoch_ and model.selected_scale_ == old.selected_scale_
    np.testing.assert_allclose(model.predict_delta(args[4]), old.predict_delta(cached_val), atol=1e-12)
    assert model.to_dict()["encoder_trainable_parameter_count"] == 0
    assert model.to_dict()["spatial_parameter_distance"] == 0
    for name, parameter in model.spatial.named_parameters():
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    # Deep copies neither change the supplied weights nor the caller's train mode.
    assert spatial.training
    for name, tensor in spatial.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["last_self", "last_self_ecology"])
def test_only_requested_encoder_parameters_receive_gradients_and_updates(mode):
    model, args, spatial, *_ = fixture(mode)
    original = copy.deepcopy(spatial.state_dict())
    with torch.no_grad():
        model.head.weight.fill_(.1)
    delta = model._delta_cells(model._prepare_inputs(args[0]), np.array([0, 5, 13, 40]))
    delta.sum().backward()
    allowed = {"convs.1.self_lin.weight", "convs.1.self_lin.bias"}
    if mode == "last_self_ecology":
        allowed |= {"env_encoder.0.weight", "env_encoder.0.bias", "env_encoder.2.weight", "env_encoder.2.bias"}
    for name, parameter in model.spatial.named_parameters():
        assert parameter.requires_grad == (name in allowed)
        if name in allowed:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad)
        else:
            assert parameter.grad is None
    fit_model(model, args)
    summary = model.to_dict()
    assert model.best_epoch_ > 0 and summary["last_self_parameter_distance"] > 0
    assert (summary["ecology_parameter_distance"] > 0) == (mode == "last_self_ecology")
    assert summary["encoder_trainable_parameter_count"] == sum(
        p.numel() for p in model.spatial.parameters() if p.requires_grad)
    assert summary["trainable_parameter_count"] == sum(
        p.numel() for module in (model.spatial, model.temporal, model.decay, model.head)
        for p in module.parameters() if p.requires_grad)
    for name, value in model.spatial.state_dict().items():
        if name not in allowed:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for name, value in spatial.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert all(not module.training for module in model.spatial.modules())
    assert summary["validation_metrics"]["validation_mae"] < 1.


def test_zero_head_and_harmful_validation_restore_context_and_complete_initial_state():
    model, args, *_ = fixture("last_self_ecology", epochs=4)
    np.testing.assert_array_equal(model.predict_delta(args[0]), np.zeros_like(args[1]))
    base = args[1].astype(np.float32)
    prediction = model.predict(args[0], base)
    np.testing.assert_array_equal(prediction, base)
    assert prediction.dtype == base.dtype
    args[6][:] = 2.  # Source upward correction must lose to epoch-zero validation.
    fit_model(model, args)
    assert model.best_epoch_ == 0 and model.selected_scale_ == 0
    np.testing.assert_array_equal(model.predict(args[4], args[5]), args[5])
    assert model.to_dict()["spatial_parameter_distance"] == 0
    assert model.to_dict()["head_parameter_norm"] == 0
    assert model.trace_[1]["spatial_parameter_distance"] > 0


def test_future_raw_age_support_and_extra_cannot_change_earlier_predictions():
    model, args, *_ = fixture("last_self_ecology", epochs=0)
    with torch.no_grad():
        model.head.weight.fill_(.03)
        model.head.bias.fill_(.07)
    before = model.predict_delta(args[0], batch_size=11)
    changed = copy.deepcopy(args[0])
    for name in ("raw", "age", "support", "extra"):
        changed[name][:, 8:] += 10
    after = model.predict_delta(changed, batch_size=11)
    np.testing.assert_array_equal(after[:, :8], before[:, :8])
    assert not np.allclose(after[:, 8:], before[:, 8:])
    # Static ecology is intentionally shared across months, not estimated from DOC.
    np.testing.assert_array_equal(changed["env"], args[0]["env"])


def test_hidden_label_and_base_values_do_not_affect_fit_or_frozen_normalization():
    a, original, *_ = fixture("last_self_ecology", epochs=1)
    b, changed, *_ = fixture("last_self_ecology", epochs=1)
    for args, value in ((original, np.nan), (changed, -1e300)):
        args[1][~args[3]] = value
        args[2][~args[3]] = value
        args[5][~args[7]] = value
        args[6][~args[7]] = value
    fit_model(a, original)
    fit_model(b, changed)
    assert a.to_dict() == b.to_dict()
    np.testing.assert_array_equal(a.predict_delta(original[4]), b.predict_delta(original[4]))
    with pytest.raises(ValueError, match="source_validation"):
        a.fit(*original, tail_threshold=4., selection_role="outer_test")


@pytest.mark.parametrize("mode", ["frozen", "last_self", "last_self_ecology"])
def test_checkpoint_restores_initial_and_final_weights_predictions_and_summary(mode):
    model, args, *_ = fixture(mode, epochs=1)
    fit_model(model, args)
    buffer = io.BytesIO()
    torch.save(model.to_payload(), buffer)
    buffer.seek(0)
    restored = EncoderNativeResidual.from_payload(torch.load(buffer, weights_only=True))
    assert restored.to_dict() == model.to_dict()
    json.dumps(restored.to_dict(), allow_nan=False)
    np.testing.assert_array_equal(restored.predict_delta(args[4]), model.predict_delta(args[4]))
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    for name, value in restored._initial_spatial_state.items():
        torch.testing.assert_close(value, model._initial_spatial_state[name], rtol=0, atol=0)
    # Refitting after reload starts from the original encoder and all other weights.
    fit_model(restored, args)
    assert restored.to_dict() == model.to_dict()


def test_input_and_architecture_guards_and_no_extra_serialization():
    model, args, spatial, temporal, decay = fixture(epochs=0)
    with pytest.raises(ValueError, match="encoder_mode"):
        EncoderNativeResidual(spatial, temporal, decay, encoder_mode="all")
    with pytest.raises(ValueError, match="baseline"):
        modified = copy.deepcopy(spatial)
        modified.residual = True
        EncoderNativeResidual(modified, temporal, decay)
    bad = copy.deepcopy(args[0])
    bad["env"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite aligned"):
        model.predict_delta(bad)
    bad = copy.deepcopy(args[0])
    del bad["extra"]
    with pytest.raises(ValueError, match="extra inputs"):
        model.predict_delta(bad)
    no_extra = EncoderNativeResidual(spatial, temporal, decay, epochs=0, extra_dim=0,
                                     interaction_indices=())
    fit_model(no_extra, args)
    restored = EncoderNativeResidual.from_payload(no_extra.to_payload())
    assert restored.extra_dim == 0 and restored.interaction_indices == ()
    assert restored.to_dict() == no_extra.to_dict()
