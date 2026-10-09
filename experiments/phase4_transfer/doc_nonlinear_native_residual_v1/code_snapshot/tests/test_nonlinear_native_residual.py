"""Nonlinear head retains baseline initialization, isolation and causal windows."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.nonlinear_native_residual import (
    NonlinearNativeResidual,
    ParallelResidualReadout,
)


def nonlinear_fixture(epochs=2):
    original, args, *_ = fixture("last_self_ecology", epochs=epochs)
    model = NonlinearNativeResidual(original.spatial, original.temporal, original.decay,
                                    readout_width=8, **original._config())
    return model, args


def test_parallel_head_preserves_nonzero_linear_output_and_has_gradient():
    linear = torch.nn.Linear(7, 1, dtype=torch.float64)
    x = torch.arange(28, dtype=torch.float64).reshape(4, 7)/7
    head = ParallelResidualReadout(linear, 8, 42)
    torch.testing.assert_close(head(x), linear(x), rtol=0, atol=0)
    head(x).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    assert torch.count_nonzero(head.nonlinear[-1].weight.grad)


def test_nonlinear_fit_isolation_repeatability_and_saved_replay():
    a, args = nonlinear_fixture()
    b, changed = nonlinear_fixture()
    changed[2][~changed[3]] = 1e30
    changed[6][~changed[7]] = -1e30
    for model, data in ((a, args), (b, changed)):
        model.fit(*data, tail_threshold=4., selection_role="source_validation")
    prediction = a.predict(args[4], args[5])
    np.testing.assert_array_equal(prediction, b.predict(changed[4], changed[5]))
    restored = NonlinearNativeResidual.from_payload(a.to_payload())
    np.testing.assert_array_equal(prediction, restored.predict(args[4], args[5]))
    restored.fit(*args, tail_threshold=4., selection_role="source_validation")
    np.testing.assert_array_equal(prediction, restored.predict(args[4], args[5]))
    with pytest.raises(ValueError):
        EncoderNativeResidual.from_payload(a.to_payload())
    assert np.isfinite(prediction).all() and (prediction >= 0).all()


def test_nonlinear_head_keeps_future_inputs_out_of_earlier_predictions():
    model, args = nonlinear_fixture(epochs=0)
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    with torch.no_grad():
        model.head.linear.weight.fill_(.01)
        model.head.nonlinear[-1].weight.fill_(.03)
    model.selected_scale_ = .5
    altered = copy.deepcopy(args[4])
    for key in ("raw", "age", "support", "extra"):
        altered[key][:, 10:] += .4
    np.testing.assert_array_equal(model.predict(args[4], args[5])[:, :10],
                                  model.predict(altered, args[5])[:, :10])
