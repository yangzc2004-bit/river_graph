"""Matched recurrent clocks preserve features, causal windows and checkpoints."""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.recurrent_clock_residual import ClockNativeResidual


def fixture(clock="legacy", *, parent=False, dtype=torch.float64, epochs=2):
    with torch.random.fork_rng():
        torch.manual_seed(117)
        spatial = TransportGCNImputer(23, 2, hidden=4, layers=2, dropout=.2,
                                     env_dim=2, env_emb=3, edge_direction="upstream").to(dtype)
        temporal, decay = nn.GRUCell(5, 4).to(dtype), nn.Linear(4, 4).to(dtype)
        with torch.no_grad():
            decay.weight.fill_(.08); decay.bias.fill_(.03)
            for conv in spatial.convs:
                conv.self_lin.weight.fill_(.04); conv.self_lin.bias.fill_(.1)
    cls = EncoderNativeResidual if parent else ClockNativeResidual
    kwargs = {} if parent else {"decay_clock": clock}
    model = cls(spatial, temporal, decay, encoder_mode="last_self_ecology", extra_dim=3,
                interaction_indices=(0,), lookback=12, epochs=epochs, patience=2,
                learning_rate=1e-3, head_learning_rate=.02, batch_size=8, **kwargs)
    rng = np.random.default_rng(34)

    def inputs(n):
        age = np.broadcast_to(np.log1p(np.arange(1, 17))/np.log1p(12), (n, 16)).copy()
        raw = rng.uniform(.1, .8, (n, 16, 23))
        raw[..., 3] = rng.integers(0, 2, (n, 16))
        raw[..., 8:10] = 0
        raw[..., -9], raw[..., -8], raw[..., -7] = 0, age, 0
        support = np.zeros((n, 16, 3)); support[..., 1] = rng.uniform(0, 1, (n, 16))
        return {"raw": raw, "env": rng.uniform(0, 1, (n, 2)), "age": age,
                "support": support, "extra": rng.uniform(-.1, .1, (n, 16, 3))}

    source, validation = inputs(3), inputs(2)
    base, valbase = np.full((3, 16), 3.), np.full((2, 16), 3.)
    source_mask, val_mask = np.ones((3, 16), bool), np.ones((2, 16), bool)
    source_mask[:, 2] = False; val_mask[:, 4] = False
    return model, (source, base, base+1, source_mask, validation, valbase, valbase+1, val_mask)


def fit(model, args):
    return model.fit(*args, tail_threshold=4., selection_role="source_validation")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legacy_exact_parent_hidden_fit_weights_and_predictions(dtype):
    old, args = fixture(parent=True, dtype=dtype)
    new, _ = fixture(dtype=dtype)
    cells = np.array([0, 4, 15, 16, 32, 47])
    torch.testing.assert_close(old._hidden_cells(old._prepare_inputs(args[0]), cells),
                               new._hidden_cells(new._prepare_inputs(args[0]), cells), rtol=0, atol=0)
    fit(old, args); fit(new, args)
    assert new.trace_ == old.trace_ and new.selected_scale_ == old.selected_scale_
    assert new.trainable_parameter_count_ == old.trainable_parameter_count_
    for name in ("spatial", "temporal", "decay", "head"):
        for key, value in getattr(old, name).state_dict().items():
            torch.testing.assert_close(value, getattr(new, name).state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(new.predict(args[4], args[5]), old.predict(args[4], args[5]))


def test_unseen_flag_is_appended_m1_channel_not_absolute_column_twelve():
    model, args = fixture("unseen_neutral", epochs=0)
    inputs = args[0]
    inputs["raw"][..., 12] = 1  # Regime hydro value must not be mistaken for the valid flag.
    window = model.clock_window(inputs, [15], include_gamma=True)
    assert not window["clock"].any()
    inputs["raw"][0, 8:, -7] = 1
    changed = model.clock_window(inputs, [15])
    np.testing.assert_array_equal(changed["clock"][0, :4], np.zeros(4))
    np.testing.assert_array_equal(changed["clock"][0, 4:], inputs["age"][0, 8:])
    np.testing.assert_array_equal(inputs["raw"][..., -8], inputs["age"])
    # If every history step is seen, only the clock changes vanish exactly.
    legacy, _ = fixture(epochs=0)
    inputs["raw"][..., -7] = 1
    cells = np.array([0, 7, 15])
    torch.testing.assert_close(model._hidden_cells(model._prepare_inputs(inputs), cells),
                               legacy._hidden_cells(legacy._prepare_inputs(inputs), cells), rtol=0, atol=0)


def test_flow_clock_true_within_window_elapsed_time_and_not_absolute_calendar():
    model, args = fixture("flow_window", epochs=0)
    inputs = args[0]
    inputs["raw"][..., 3] = 0
    inputs["raw"][0, [1, 8, 12], 3] = 1
    window = model.clock_window(inputs, [14], include_gamma=True)
    expected_age = np.array([12, 12, 12, 12, 12, 0, 1, 2, 3, 0, 1, 2])
    np.testing.assert_allclose(window["clock"][0], np.log1p(expected_age)/np.log1p(12), rtol=0, atol=2e-16)
    np.testing.assert_array_equal(window["month_indices"][0], np.arange(3, 15))
    # A visible flow before the window cannot reset the start-of-window clock.
    before = copy.deepcopy(inputs); before["raw"][0, :3, 3] = 0
    np.testing.assert_array_equal(model.clock_window(before, [14])["clock"], window["clock"])
    # A valid zero-flow observation is still an observation.
    inputs["raw"][0, 14, 2:4] = (0, 1)
    assert model.clock_window(inputs, [14])["clock"][0, -1] == 0


@pytest.mark.parametrize("clock", ["legacy", "unseen_neutral", "flow_window"])
def test_future_outside_window_and_other_station_cannot_change_clock_or_hidden(clock):
    model, args = fixture(clock, epochs=0)
    initial = model.clock_window(args[0], [14], include_gamma=True)
    hidden = model._hidden_cells(model._prepare_inputs(args[0]), [14])
    changed = copy.deepcopy(args[0])
    for key in ("raw", "age", "support", "extra"):
        changed[key][0, 15:] += 1
        changed[key][0, :3] += 1
        changed[key][1:] += 1
    for index in (3, -7):
        # Keep explicit validity flags legal while changing the irrelevant cells.
        changed["raw"][0, 15:, index] %= 2
        changed["raw"][0, :3, index] %= 2
        changed["raw"][1:, :, index] %= 2
    revised = model.clock_window(changed, [14], include_gamma=True)
    for key in initial:
        np.testing.assert_array_equal(initial[key], revised[key])
    torch.testing.assert_close(hidden, model._hidden_cells(model._prepare_inputs(changed), [14]), rtol=0, atol=0)


def test_padding_does_not_update_flow_clock_or_hidden_state():
    flow, args = fixture("flow_window", epochs=0)
    old, _ = fixture(epochs=0)
    inputs = args[0]
    inputs["raw"][0, :, 3] = 0
    inputs["raw"][0, 0, 3] = 1
    window = flow.clock_window(inputs, [0, 1])
    assert window["valid"].sum(axis=1).tolist() == [1, 2]
    assert not window["clock"][0].any()
    assert window["clock"][1, -2] == 0
    np.testing.assert_allclose(window["clock"][1, -1], np.log1p(1)/np.log1p(12))
    torch.testing.assert_close(flow._hidden_cells(flow._prepare_inputs(inputs), [0]),
                               old._hidden_cells(old._prepare_inputs(inputs), [0]), rtol=0, atol=0)


@pytest.mark.parametrize("clock", ["legacy", "unseen_neutral", "flow_window"])
def test_fit_backprop_visibility_isolation_and_checkpoint(clock):
    model, args = fixture(clock)
    before = copy.deepcopy(args[0])
    fit(model, args)
    assert model.decay.weight.grad is not None and torch.isfinite(model.decay.weight.grad).all()
    if clock == "unseen_neutral":
        assert not torch.count_nonzero(model.decay.weight.grad[:, 0])
    for key in before:
        np.testing.assert_array_equal(before[key], args[0][key])
    payload = model.to_payload()
    assert payload["model_class"] == "ClockNativeResidual" and payload["config"]["decay_clock"] == clock
    buffer = io.BytesIO(); torch.save(payload, buffer); buffer.seek(0)
    restored = ClockNativeResidual.from_payload(torch.load(buffer, weights_only=True))
    assert restored.to_dict() == model.to_dict()
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    revised = copy.deepcopy(args)
    revised[2][~revised[3]] = np.nan; revised[6][~revised[7]] = np.nan
    revised[0]["query_labels"] = np.full((3, 16), np.nan)
    same, _ = fixture(clock)
    fit(same, revised)
    assert same.trace_ == model.trace_
    np.testing.assert_array_equal(same.predict(args[4], args[5]), model.predict(args[4], args[5]))


def test_old_encoder_checkpoint_imports_as_exact_legacy_and_extra_does_not_define_clock():
    old, args = fixture(parent=True)
    fit(old, args)
    imported = ClockNativeResidual.from_payload(old.to_payload())
    assert imported.decay_clock == "legacy"
    np.testing.assert_array_equal(imported.predict(args[4], args[5]), old.predict(args[4], args[5]))
    flow, _ = fixture("flow_window", epochs=0)
    a = flow.clock_window(args[0], [5, 15])
    changed = copy.deepcopy(args[0]); changed["extra"] += 1e5
    changed["raw"][..., 8] = 1e4  # Clock never reads the target-value channel.
    b = flow.clock_window(changed, [5, 15])
    np.testing.assert_array_equal(a["clock"], b["clock"])


def test_bad_clock_or_nonbinary_visibility_is_rejected():
    with pytest.raises(ValueError, match="decay_clock"):
        fixture("unknown")
    for clock, index in (("unseen_neutral", -7), ("flow_window", 3)):
        model, args = fixture(clock, epochs=0)
        args[0]["raw"][0, 0, index] = .5
        with pytest.raises(ValueError, match="binary"):
            model.clock_window(args[0], [0])
