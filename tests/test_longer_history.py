"""Longer memory adds past information without changing model capacity or roles."""
from __future__ import annotations

import copy

import numpy as np
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual


def longer(model):
    return EncoderNativeResidual(model.spatial, model.temporal, model.decay,
        **{**model._config(), "lookback": 24})


def test_longer_window_padding_past_influence_and_future_exclusion():
    short, args, *_ = fixture("last_self_ecology", epochs=0)
    long = longer(short)
    assert long.trainable_parameter_count_ == short.trainable_parameter_count_
    for name in ("spatial", "temporal", "decay"):
        for key, value in getattr(short, f"_initial_{name}_state").items():
            torch.testing.assert_close(getattr(long, f"_initial_{name}_state")[key],
                                       value, rtol=0, atol=0)
    inputs = {key: np.concatenate([value, value], axis=1) if key != "env" else value.copy()
              for key, value in args[0].items()}
    early = np.array([0, 1, 5, 11, 28, 31])
    torch.testing.assert_close(short._hidden_cells(short._prepare_inputs(inputs), early),
                               long._hidden_cells(long._prepare_inputs(inputs), early),
                               rtol=0, atol=0)
    current = np.array([24])
    earlier = copy.deepcopy(inputs)
    earlier["raw"][0, 1] += 10.
    old = short._hidden_cells(short._prepare_inputs(inputs), current)
    old_changed = short._hidden_cells(short._prepare_inputs(earlier), current)
    torch.testing.assert_close(old, old_changed, rtol=0, atol=0)
    state = long._hidden_cells(long._prepare_inputs(inputs), current)
    changed = long._hidden_cells(long._prepare_inputs(earlier), current)
    assert (state-changed).abs().max() > 1e-12
    future = copy.deepcopy(inputs)
    for key in ("raw", "age", "support", "extra"):
        future[key][:, 25:] += 100.
    torch.testing.assert_close(state, long._hidden_cells(long._prepare_inputs(future), current),
                               rtol=0, atol=0)
    with torch.no_grad():
        long.head.weight.fill_(.01)
    torch.testing.assert_close(long._delta_cells(long._prepare_inputs(inputs), current),
                               long._delta_cells(long._prepare_inputs(future), current),
                               rtol=0, atol=0)


def test_longer_fit_ignores_hidden_labels_and_saved_model_replays():
    base, args, *_ = fixture("last_self_ecology", epochs=2)
    first, second = longer(base), longer(base)
    altered = list(copy.deepcopy(args))
    altered[2][~altered[3]] = np.nan
    altered[6][~altered[7]] = np.nan
    for model, data in ((first, args), (second, altered)):
        model.fit(*data, tail_threshold=4., selection_role="source_validation")
    assert first.trace_ == second.trace_
    np.testing.assert_array_equal(first.predict(args[4], args[5]),
                                  second.predict(args[4], args[5]))
    assert first.head.weight.abs().sum() > 0
    restored = EncoderNativeResidual.from_payload(first.to_payload())
    assert restored.lookback == 24
    np.testing.assert_array_equal(first.predict(args[4], args[5]),
                                  restored.predict(args[4], args[5]))
