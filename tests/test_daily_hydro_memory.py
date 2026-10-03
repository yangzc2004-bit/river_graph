"""Causal daily-flow injection into the existing encoder/GRU residual lifecycle."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.hydro import TransportGCNImputer


def fixture(mode="off", *, dtype=torch.float64, epochs=2, lookback=4, hidden=4):
    with torch.random.fork_rng():
        torch.manual_seed(18)
        spatial = TransportGCNImputer(5, 2, hidden=hidden, layers=2, dropout=.5,
                                     env_dim=2, env_emb=3, edge_direction="upstream").to(dtype=dtype)
        temporal = nn.GRUCell(hidden + 1, hidden).to(dtype=dtype)
        decay = nn.Linear(4, hidden).to(dtype=dtype)
        with torch.no_grad():
            for conv in spatial.convs:
                conv.self_lin.weight.fill_(.1)
                conv.self_lin.bias.fill_(.15)
            for parameter in spatial.env_encoder.parameters():
                parameter.fill_(.2)
    rng = np.random.default_rng(27)

    def inputs(n):
        return {"raw": rng.uniform(.1, 1., (n, 8, 5)), "env": rng.uniform(.1, 1., (n, 2)),
                "age": rng.uniform(0, 2, (n, 8)), "support": rng.uniform(0, 1, (n, 8, 3)),
                "extra": rng.uniform(-.1, .1, (n, 8, 3)),
                "daily_history": rng.uniform(0, 1, (n, 8, 8))}

    source, validation = inputs(3), inputs(2)
    source_mask, val_mask = np.ones((3, 8), dtype=bool), np.ones((2, 8), dtype=bool)
    source_mask[:, 2] = False
    val_mask[:, 3] = False
    source_base, val_base = np.full((3, 8), 3.), np.full((2, 8), 3.)
    caller_rng = torch.get_rng_state().clone()
    model = EncoderNativeResidual(
        spatial, temporal, decay, encoder_mode="last_self_ecology", encoder_learning_rate=1e-3,
        hydro_sequence_mode=mode, extra_dim=3, interaction_indices=(0,), lookback=lookback,
        epochs=epochs, patience=2, batch_size=6, learning_rate=1e-3, head_learning_rate=.02)
    assert torch.equal(torch.get_rng_state(), caller_rng)
    return model, (source, source_base, source_base + 1., source_mask,
                   validation, val_base, val_base + 1., val_mask)


def fit_model(model, args):
    return model.fit(*args, tail_threshold=4., selection_role="source_validation")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["current_only", "full_history"])
def test_zero_projection_exact_initial_hidden_and_nonzero_head_predictions(mode, dtype):
    old, args = fixture(dtype=dtype, epochs=0)
    new, _ = fixture(mode, dtype=dtype, epochs=0)
    assert new.hydro_projection.bias is None
    assert torch.count_nonzero(new.hydro_projection.weight) == 0
    cells = np.array([0, 1, 3, 6, 7, 8, 12, 23])
    torch.testing.assert_close(old._hidden_cells(old._prepare_inputs(args[0]), cells),
                               new._hidden_cells(new._prepare_inputs(args[0]), cells), rtol=0, atol=0)
    with torch.no_grad():
        old.head.weight.fill_(.05)
        old.head.bias.fill_(.1)
        new.head.load_state_dict(old.head.state_dict())
    np.testing.assert_array_equal(old.predict_delta(args[0], batch_size=7),
                                  new.predict_delta(args[0], batch_size=7))


def test_off_keeps_old_input_and_checkpoint_schema():
    model, args = fixture(epochs=1)
    source_without = {key: value for key, value in args[0].items() if key != "daily_history"}
    args[0]["daily_history"][:] = np.nan  # The off path never reads the unused key.
    np.testing.assert_array_equal(model.predict_delta(args[0]), model.predict_delta(source_without))
    fit_model(model, args)
    payload = model.to_payload()
    assert model.hydro_projection is None
    assert "hydro_sequence_mode" not in payload["config"]
    assert not any("hydro_projection" in key for key in payload)
    assert not any("hydro_projection" in key for key in payload["summary"])
    restored = EncoderNativeResidual.from_payload(payload)
    assert restored.to_dict() == model.to_dict()
    np.testing.assert_array_equal(restored.predict_delta(args[4]), model.predict_delta(args[4]))


LEGACY_SOURCE = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1/code_snapshot/"
                     "src/river_graph/models/encoder_native_residual.py")


@pytest.mark.skipif(not LEGACY_SOURCE.exists(), reason="frozen pre-memory encoder source is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_off_fit_and_payload_exactly_match_frozen_pre_memory_source(dtype):
    spec = importlib.util.spec_from_file_location("pre_memory_encoder", LEGACY_SOURCE)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    current, args = fixture(dtype=dtype)
    previous = legacy.EncoderNativeResidual(
        current.spatial, current.temporal, current.decay, **current._config())
    fit_model(current, args)
    fit_model(previous, args)
    assert current.to_dict() == previous.to_dict()
    a, b = current.to_payload(), previous.to_payload()
    assert a.keys() == b.keys()
    for name in ("spatial", "temporal", "decay", "head", "initial_spatial", "initial_temporal", "initial_decay"):
        assert a[name].keys() == b[name].keys()
        for key in a[name]:
            torch.testing.assert_close(a[name][key], b[name][key], rtol=0, atol=0)
    np.testing.assert_array_equal(current.predict_delta(args[4]), previous.predict_delta(args[4]))
    np.testing.assert_array_equal(current.predict(args[4], args[5]), previous.predict(args[4], args[5]))


def test_projection_has_exactly_512_parameters_for_original_hidden_size():
    model, _ = fixture("full_history", hidden=64, epochs=0)
    old, _ = fixture(hidden=64, epochs=0)
    assert model.hydro_projection.weight.shape == (64, 8)
    assert model.trainable_parameter_count_ - old.trainable_parameter_count_ == 512


def test_current_only_and_full_history_have_distinct_causal_scopes():
    full, args = fixture("full_history", epochs=0)
    current, _ = fixture("current_only", epochs=0)
    with torch.no_grad():
        full.hydro_projection.weight.fill_(.2)
        current.hydro_projection.weight.copy_(full.hydro_projection.weight)
    query = np.array([5])

    def state(model, inputs):
        return model._hidden_cells(model._prepare_inputs(inputs), query)

    baseline_full, baseline_current = state(full, args[0]), state(current, args[0])
    history = copy.deepcopy(args[0]); history["daily_history"][0, 2] += 2
    assert not torch.equal(state(full, history), baseline_full)
    torch.testing.assert_close(state(current, history), baseline_current, rtol=0, atol=0)
    target = copy.deepcopy(args[0]); target["daily_history"][0, 5] += 2
    assert not torch.equal(state(full, target), baseline_full)
    assert not torch.equal(state(current, target), baseline_current)
    outside = copy.deepcopy(args[0])
    outside["daily_history"][0, 6:] += 20  # Future months.
    outside["daily_history"][0, :2] += 20  # Before the four-month lookback.
    outside["daily_history"][1:] += 20  # Other stations.
    torch.testing.assert_close(state(full, outside), baseline_full, rtol=0, atol=0)
    torch.testing.assert_close(state(current, outside), baseline_current, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["current_only", "full_history"])
def test_padding_is_not_projected_or_accumulated(mode):
    long, args = fixture(mode, epochs=0, lookback=12)
    single, _ = fixture(mode, epochs=0, lookback=1)
    with torch.no_grad():
        long.hydro_projection.weight.fill_(.3)
        single.hydro_projection.weight.copy_(long.hydro_projection.weight)
    counts = []
    handle = long.hydro_projection.register_forward_hook(lambda module, inputs, output: counts.append(len(inputs[0])))
    try:
        a = long._hidden_cells(long._prepare_inputs(args[0]), np.array([0]))
    finally:
        handle.remove()
    b = single._hidden_cells(single._prepare_inputs(args[0]), np.array([0]))
    assert counts == [1]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["current_only", "full_history"])
def test_projection_learns_and_checkpoint_reload_refit_are_exact(mode):
    model, args = fixture(mode, epochs=3)
    fit_model(model, args)
    assert model.best_epoch_ > 0
    assert torch.count_nonzero(model.hydro_projection.weight) > 0
    assert model.hydro_projection.weight.grad is not None
    assert torch.isfinite(model.hydro_projection.weight.grad).all()
    summary = model.to_dict()
    assert summary["hydro_projection_parameter_distance"] > 0
    assert summary["hydro_projection_trainable_parameter_count"] == 32
    assert summary["protocol"]["hydro_sequence_learning_rate"] == model.learning_rate
    assert summary["config"]["hydro_sequence_mode"] == mode
    assert summary["hydro_projection_parameter_norm"] == summary["validation_metrics"]["hydro_projection_parameter_norm"]
    buffer = io.BytesIO()
    torch.save(model.to_payload(), buffer)
    buffer.seek(0)
    restored = EncoderNativeResidual.from_payload(torch.load(buffer, weights_only=True))
    assert restored.to_dict() == summary
    json.dumps(summary, allow_nan=False)
    np.testing.assert_array_equal(restored.predict_delta(args[4]), model.predict_delta(args[4]))
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    # A new fit resets the projection together with the rest of the original expert.
    with torch.no_grad():
        restored.hydro_projection.weight.fill_(100.)
    fit_model(restored, args)
    assert restored.to_dict() == summary
    torch.testing.assert_close(restored.hydro_projection.weight, model.hydro_projection.weight, rtol=0, atol=0)


def test_epoch_zero_restores_the_projection_with_all_other_modules():
    model, args = fixture("full_history", epochs=3)
    args[6][:] = 2.  # An upward source correction harms this validation target.
    fit_model(model, args)
    assert model.best_epoch_ == 0 and model.selected_scale_ == 0
    assert model.trace_[1]["hydro_projection_parameter_norm"] > 0
    assert torch.count_nonzero(model.hydro_projection.weight) == 0
    np.testing.assert_array_equal(model.predict(args[4], args[5]), args[5])


def test_active_inputs_finite_backward_and_hidden_label_isolation():
    model, args = fixture("full_history", epochs=1)
    with torch.no_grad():
        model.head.weight.fill_(.05)
    model._delta_cells(model._prepare_inputs(args[0]), np.array([0, 3, 7, 23])).sum().backward()
    assert model.hydro_projection.weight.grad is not None
    assert torch.isfinite(model.hydro_projection.weight.grad).all()
    assert torch.count_nonzero(model.hydro_projection.weight.grad) > 0
    original, same = fixture("full_history", epochs=1)
    for values, mask in ((args[1], args[3]), (args[2], args[3]), (args[5], args[7]), (args[6], args[7])):
        values[~mask] = np.nan
    for values, mask in ((same[1], same[3]), (same[2], same[3]), (same[5], same[7]), (same[6], same[7])):
        values[~mask] = -1e100
    fit_model(model, args)
    fit_model(original, same)
    assert model.to_dict() == original.to_dict()


def test_history_shape_finiteness_mode_and_payload_validation():
    model, args = fixture("full_history", epochs=0)
    without = {key: value for key, value in args[0].items() if key != "daily_history"}
    with pytest.raises(ValueError, match="daily_history is required"):
        model.predict_delta(without)
    for invalid in (np.zeros((3, 8, 7)), np.full((3, 8, 8), np.nan)):
        with pytest.raises(ValueError, match="finite aligned"):
            model.predict_delta({**args[0], "daily_history": invalid})
    with pytest.raises(ValueError, match="hydro_sequence_mode"):
        fixture("future")
    fit_model(model, args)
    missing = copy.deepcopy(model.to_payload()); del missing["hydro_projection"]
    with pytest.raises(ValueError, match="lacks projection"):
        EncoderNativeResidual.from_payload(missing)
    bad = copy.deepcopy(model.to_payload())
    bad["hydro_projection"]["weight"].fill_(torch.nan)
    with pytest.raises(ValueError, match="nonfinite residual weights"):
        EncoderNativeResidual.from_payload(bad)
