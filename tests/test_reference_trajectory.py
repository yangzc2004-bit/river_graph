"""Reference trajectory integration keeps old behavior and causal information."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.reference_trajectory import (
    fit_reference_scaler,
    reference_history,
)


def reference_fixture(mode, epochs=2):
    old, args, *_ = fixture("last_self_ecology", epochs=epochs)
    model = EncoderNativeResidual(old.spatial, old.temporal, old.decay,
        reference_sequence_mode=mode, **old._config())
    data = copy.deepcopy(args)
    for inputs in (data[0], data[4]):
        inputs["reference_history"] = np.arange(inputs["age"].size).reshape(*inputs["age"].shape, 1)/40
    return old, model, args, data


@pytest.mark.parametrize("mode", ["current_only", "full_history"])
def test_reference_zero_projection_matches_old_path_and_saved_fit(mode):
    old, model, args, data = reference_fixture(mode)
    cells = np.array([0, 1, 7, 14, 40])
    torch.testing.assert_close(old._hidden_cells(old._prepare_inputs(args[0]), cells),
        model._hidden_cells(model._prepare_inputs(data[0]), cells), rtol=0, atol=0)
    assert model.trainable_parameter_count_ == old.trainable_parameter_count_+old.hidden_size
    model.fit(*data, tail_threshold=4., selection_role="source_validation")
    assert model.trace_[-1]["reference_projection_parameter_norm"] > 0
    restored = EncoderNativeResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(model.predict(data[4], data[5]), restored.predict(data[4], data[5]))
    single = {key: values[:1] for key, values in data[4].items()}
    np.testing.assert_allclose(restored.predict(single, data[5][:1]), model.predict(data[4], data[5])[:1], rtol=1e-12, atol=1e-12)


def test_reference_history_only_uses_permitted_past_and_current_steps():
    predictions = {}
    for mode in ("current_only", "full_history"):
        _, model, _, data = reference_fixture(mode, epochs=0)
        with torch.no_grad():
            model.reference_projection.weight.fill_(.1)
            model.head.weight.fill_(.03)
        before = model.predict_delta(data[0])
        future = copy.deepcopy(data[0])
        future["reference_history"][:, 8:] += 20
        np.testing.assert_array_equal(before[:, :8], model.predict_delta(future)[:, :8])
        past = copy.deepcopy(data[0])
        past["reference_history"][:, :6] += 20
        predictions[mode] = (before[:, 6:], model.predict_delta(past)[:, 6:])
    np.testing.assert_array_equal(*predictions["current_only"])
    assert not np.allclose(*predictions["full_history"])


def test_reference_scaling_accepts_only_dense_source_predictions():
    source = np.arange(12).reshape(3, 4)/5
    mean, scale = fit_reference_scaler(source)
    receiving = np.full((7, 4), 100.)
    assert reference_history(receiving, mean, scale).shape == (7, 4, 1)
    assert (mean, scale) == fit_reference_scaler(source)
    with pytest.raises(ValueError, match="dense source"):
        incomplete = source.copy()
        incomplete[0, 1] = np.nan
        fit_reference_scaler(incomplete)
