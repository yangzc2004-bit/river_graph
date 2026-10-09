"""Exact donor-value conversion with the existing native attention architecture."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_current_source_attention import _model

from river_graph.models.ratio_source_attention import RatioSourceAttentionResidual
from river_graph.models.relative_source_attention import RelativeSourceAttentionResidual


def _paired_models():
    current, old, args = _model("fixed_prior", epochs=2)
    models = [cls(current.spatial, current.temporal, current.decay,
                  **current._config(), **current.attention_config)
              for cls in (RelativeSourceAttentionResidual, RatioSourceAttentionResidual)]
    for pos in (0, 4):
        args[pos]["attention_reference"] = args[pos+1].copy()
    return models, old, args


def test_exact_ratio_zero_output_matches_first_order_and_projection_gradient():
    (linear, exact), old, args = _paired_models()
    np.testing.assert_array_equal(exact.predict_delta(args[4]), linear.predict_delta(args[4]))
    np.testing.assert_array_equal(exact.predict_delta(args[4]), old.predict_delta(args[4]))
    cells = torch.tensor([4, 15])
    gradients = []
    for model in (linear, exact):
        delta = model._delta_cells(model._prepare_inputs(args[4]), cells)
        gradients.append(torch.autograd.grad(delta.sum(), model.head.output.weight)[0])
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=1e-15)
    assert exact.trainable_parameter_count_ == linear.trainable_parameter_count_


@pytest.mark.parametrize("weight", [.2, -.2])
def test_source_conversion_uses_exact_inverse_and_keeps_local_native_readout(weight):
    (linear, exact), _, args = _paired_models()
    with torch.no_grad():
        linear.head.output.weight.fill_(weight)
        exact.head.output.weight.fill_(weight)
    inputs = exact._prepare_inputs(args[4])
    cells = torch.tensor([4, 15])
    hidden = exact._hidden_cells(inputs, cells)
    state, _ = exact._attention_cells(inputs, cells, hidden)
    log_delta = exact.head.output(state).squeeze(-1)
    with torch.no_grad():
        exact.head.output.weight.zero_()
    local = exact._delta_cells(inputs, cells)
    with torch.no_grad():
        exact.head.output.weight.fill_(weight)
    actual = exact._delta_cells(inputs, cells)
    torch.testing.assert_close(actual, local+4*torch.expm1(log_delta), rtol=0, atol=0)


def test_exact_ratio_training_reloads_and_preserves_future_input_boundary():
    (_, model), _, args = _paired_models()
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    restored = RatioSourceAttentionResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    diagnostic = model.diagnostics(args[4], np.array([4, 15]))
    replay = restored.diagnostics(args[4], np.array([4, 15]))
    for name in diagnostic:
        np.testing.assert_array_equal(replay[name], diagnostic[name])
    np.testing.assert_array_equal(diagnostic["source_native_correction"],
        4*np.expm1(diagnostic["source_relative_output"]))
    assert restored.to_dict()["protocol"]["local_readout"].startswith("unchanged native")
    future = deepcopy(args[4])
    future["attention_reference"][:, 8:] += 100.
    future["donor_values"][:, 8:, :2] += 100.
    future["donor_values"][~future["donor_valid"]] = 0.
    np.testing.assert_array_equal(model.predict_delta(args[4])[:, :8], model.predict_delta(future)[:, :8])
