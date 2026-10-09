"""Detailed ecological inputs preserve initialization and causal native fitting."""
from __future__ import annotations

import copy

import numpy as np
import torch
from test_encoder_native_residual import fixture

from river_graph.models.composition_encoder import expand_ecological_encoder
from river_graph.models.encoder_native_residual import EncoderNativeResidual


def expanded_fixture(epochs=2):
    old, args, *_ = fixture("last_self_ecology", epochs=epochs)
    new = EncoderNativeResidual(expand_ecological_encoder(old.spatial, old.spatial_architecture),
        old.temporal, old.decay, **old._config())
    data = copy.deepcopy(args)
    for inputs in (data[0], data[4]):
        physical = np.linspace(0, 1, len(inputs["env"])*22).reshape(-1, 22)
        inputs["env"] = np.column_stack([inputs["env"], physical])
    return old, new, args, data


def test_extended_encoder_preserves_hidden_states_and_learns_new_columns():
    old, new, args, data = expanded_fixture(epochs=0)
    cells = np.array([0, 3, 13, 14, 27, 41])
    torch.testing.assert_close(old._hidden_cells(old._prepare_inputs(args[0]), cells),
        new._hidden_cells(new._prepare_inputs(data[0]), cells), rtol=1e-13, atol=1e-13)
    for model in (old, new):
        with torch.no_grad():
            model.head.weight.fill_(.03)
            model.head.bias.fill_(.1)
    np.testing.assert_allclose(old.predict_delta(args[0]), new.predict_delta(data[0]), rtol=1e-13, atol=1e-13)
    new._delta_cells(new._prepare_inputs(data[0]), cells).sum().backward()
    gradient = new.spatial.env_encoder[0].weight.grad[:, -22:]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_expanded_model_label_isolation_saved_replay_and_new_station_count():
    _, a, _, args = expanded_fixture()
    _, b, _, changed = expanded_fixture()
    changed[2][~changed[3]] = 1e30
    changed[6][~changed[7]] = -1e30
    for model, data in ((a, args), (b, changed)):
        model.fit(*data, tail_threshold=4., selection_role="source_validation")
    result = a.predict(args[4], args[5])
    np.testing.assert_array_equal(result, b.predict(changed[4], changed[5]))
    saved = EncoderNativeResidual.from_payload(a.to_payload())
    np.testing.assert_array_equal(result, saved.predict(args[4], args[5]))
    single = {key: value[:1] for key, value in args[4].items()}
    np.testing.assert_allclose(saved.predict(single, args[5][:1]), result[:1], rtol=1e-12, atol=1e-12)
    assert saved.spatial_architecture["env_dim"] == 24


def test_expanded_encoder_retains_causal_month_windows():
    _, model, _, args = expanded_fixture(epochs=0)
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    with torch.no_grad():
        model.head.weight.fill_(.02)
    model.selected_scale_ = .5
    changed = copy.deepcopy(args[4])
    for key in ("raw", "age", "support", "extra"):
        changed[key][:, 10:] += .4
    np.testing.assert_array_equal(model.predict(args[4], args[5])[:, :10],
                                  model.predict(changed, args[5])[:, :10])
