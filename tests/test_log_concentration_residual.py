"""Relative concentration correction keeps the existing causal residual backbone."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.log_concentration_residual import (
    LogConcentrationEncoderResidual,
)


def log_fixture(epochs=2):
    original, args, *_ = fixture("last_self_ecology", epochs=epochs)
    return LogConcentrationEncoderResidual(original.spatial, original.temporal, original.decay,
                                           **original._config()), args


def test_zero_identity_preserves_concentration_scaled_head_gradient():
    base = torch.tensor([0., .7, 2.4, 27.], dtype=torch.float64)
    delta = torch.zeros_like(base, requires_grad=True)
    prediction = LogConcentrationEncoderResidual._combine(base, delta, 1.)
    torch.testing.assert_close(prediction, base, rtol=0, atol=0)
    prediction.sum().backward()
    torch.testing.assert_close(delta.grad, base+1., rtol=1e-15, atol=1e-15)
    change = torch.full_like(base, .2)
    torch.testing.assert_close(LogConcentrationEncoderResidual._combine(base, change, 1.),
                              (base+1.)*torch.exp(change)-1.)


def test_log_readout_fit_isolation_save_load_and_checkpoint_class():
    a, args = log_fixture()
    b, changed = log_fixture()
    changed[2][~changed[3]] = 1e30
    changed[6][~changed[7]] = -1e30
    for model, data in ((a, args), (b, changed)):
        model.fit(*data, tail_threshold=4., selection_role="source_validation")
    prediction = a.predict(args[4], args[5])
    np.testing.assert_array_equal(prediction, b.predict(changed[4], changed[5]))
    saved = a.to_payload()
    restored = LogConcentrationEncoderResidual.from_payload(saved)
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), prediction)
    assert saved["summary"]["protocol"]["residual_units"] == "log1p DOC"
    with pytest.raises(ValueError):
        EncoderNativeResidual.from_payload(saved)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()


def test_future_inputs_do_not_change_earlier_relative_predictions():
    model, args = log_fixture(epochs=0)
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    with torch.no_grad():
        model.head.weight.fill_(.01)
        model.head.bias.fill_(.01)
    model.selected_scale_ = .5
    altered = copy.deepcopy(args[4])
    for key in ("raw", "age", "support", "extra"):
        altered[key][:, 10:] += .4
    np.testing.assert_array_equal(model.predict(args[4], args[5])[:, :10],
                                  model.predict(altered, args[5])[:, :10])
