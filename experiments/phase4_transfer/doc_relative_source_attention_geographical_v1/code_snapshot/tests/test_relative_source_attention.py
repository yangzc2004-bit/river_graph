"""Relative units, source isolation and compatibility with the retained head."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_current_source_attention import _model
from test_source_innovation_training import _nested_fixture

from river_graph.models.relative_source_attention import (
    RelativeSourceAttentionResidual,
    nested_relative_source_candidates,
    relative_source_residual_grid,
)


def test_log_departures_are_level_relative_and_ignore_unselected_labels():
    base = np.array([[1., 4.], [9., 24.]])
    log_change = .2
    truth = np.expm1(np.log1p(base)+log_change)
    ids, residual = relative_source_residual_grid(truth, np.log1p(base), np.arange(4))
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(residual, log_change, atol=1e-15)
    changed = truth.copy()
    changed[1] = np.nan
    _, first = relative_source_residual_grid(changed, np.log1p(base), np.array([0, 1]))
    np.testing.assert_array_equal(first, residual[:1])


def test_relative_nested_donors_exclude_query_and_receiver_labels():
    args = _nested_fixture()
    hydro = np.zeros((6, 36, 8))
    original, records = nested_relative_source_candidates(*args, hydro)
    changed = deepcopy(args)
    changed[0][:2] += 1000.
    changed[0][6] = np.nan
    altered, _ = nested_relative_source_candidates(*changed, hydro)
    for key in original:
        if key == "donor_hydro_bank":
            np.testing.assert_array_equal(original[key], altered[key])
        else:
            np.testing.assert_array_equal(original[key][:2], altered[key][:2])
    assert all(not set(row["query_station_ids"]) & set(row["library_station_ids"]) for row in records)
    assert not np.isin(original["donor_owner"][:2], [0, 1]).any()
    bad = deepcopy(args)
    bad[-1][(0, 1)]["fitted_stations"].append(0)
    with pytest.raises(ValueError, match="query AND donor"):
        nested_relative_source_candidates(*bad, hydro)


def test_relative_conversion_zero_fusion_and_native_training_replay():
    current, old, args = _model("fixed_prior", epochs=2)
    model = RelativeSourceAttentionResidual(current.spatial, current.temporal, current.decay,
        **current._config(), **current.attention_config)
    for pos in (0, 4):
        args[pos]["attention_reference"] = args[pos+1].copy()
    np.testing.assert_array_equal(old.predict_delta(args[4]), model.predict_delta(args[4]))
    inputs = model._prepare_inputs(args[4])
    cells = torch.tensor([4, 15])
    hidden = model._hidden_cells(inputs, cells)
    original_state, original_weights = current._attention_cells(inputs, cells, hidden)
    state, weights = model._attention_cells(inputs, cells, hidden)
    torch.testing.assert_close(state, original_state*4, rtol=0, atol=0)
    torch.testing.assert_close(weights, original_weights, rtol=0, atol=0)
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    restored = RelativeSourceAttentionResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    assert restored.trainable_parameter_count_ == current.trainable_parameter_count_
    assert restored.to_dict()["protocol"]["source_value_units"] == "dimensionless"
    future = deepcopy(args[4])
    future["attention_reference"][:, 8:] += 100.
    future["donor_values"][:, 8:, :2] += 100.
    future["donor_values"][~future["donor_valid"]] = 0.
    np.testing.assert_array_equal(model.predict_delta(args[4])[:, :8], model.predict_delta(future)[:, :8])
    missing = deepcopy(args[4])
    del missing["attention_reference"]
    with pytest.raises(ValueError, match="frozen receiving reference"):
        model.predict_delta(missing)
