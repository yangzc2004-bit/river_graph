"""Nested source exclusion and an information-only extension of the same GRU."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)


def _nested_fixture():
    months = pd.period_range("2000-01", periods=36, freq="M").astype(str).to_numpy()
    truth = np.arange(7*36.).reshape(7, 36)/100+1
    source = np.arange(6*36)
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    ecology = np.zeros((7, 9))
    ecology[:, 0] = np.arange(7)
    references = {}
    for a in range(3):
        for b in range(a+1, 3):
            hidden = np.sort(np.r_[folds[a], folds[b]])
            cells = source[np.isin(source//36, hidden)]
            references[(a, b)] = {"hidden_stations": hidden.tolist(),
                "fitted_stations": np.setdiff1d(np.arange(6), hidden).tolist(),
                "cells": cells, "pred_z": np.log1p(np.full(len(cells), 1.))}
    return [truth, np.array(list("abcdefg")), months, ecology, source, folds, references]


def test_query_fold_and_receiving_labels_never_enter_nested_library():
    args = _nested_fixture()
    original, records = nested_innovation_inputs(*args)
    changed = deepcopy(args)
    changed[0][6] = np.nan
    altered, _ = nested_innovation_inputs(*changed)
    for key in original:
        np.testing.assert_array_equal(original[key], altered[key])
    changed[0][:2] += 1000.
    altered, _ = nested_innovation_inputs(*changed)
    for key in original:
        np.testing.assert_array_equal(original[key][:2], altered[key][:2])
    assert all(not set(row["query_station_ids"]) & set(row["library_station_ids"]) for row in records)
    leaked = deepcopy(args)
    leaked[-1][(0, 1)]["fitted_stations"].append(0)
    with pytest.raises(ValueError, match="query AND donor"):
        nested_innovation_inputs(*leaked)


def test_information_arms_share_support_and_extend_existing_head():
    args = _nested_fixture()
    parts, _ = nested_innovation_inputs(*args)
    real = innovation_readout_features(parts, "real", 1.)
    historical = innovation_readout_features(parts, "historical", 1.)
    unavailable = innovation_readout_features(parts, "availability", 1.)
    np.testing.assert_array_equal(real[..., 1:], historical[..., 1:])
    np.testing.assert_array_equal(real[..., 1:], unavailable[..., 1:])
    assert np.count_nonzero(unavailable[..., 0]) == 0
    base, inputs, *_ = fixture("last_self_ecology", epochs=2)
    extended = EncoderNativeResidual(base.spatial, base.temporal, base.decay,
        **{**base._config(), "extra_dim": base.extra_dim+3,
           "interaction_indices": (*base.interaction_indices, base.extra_dim)})
    modified = deepcopy(inputs)
    for position in (0, 4):
        shape = modified[position]["extra"].shape[:2]
        extra = np.ones((*shape, 3), dtype=np.float32)
        modified[position]["extra"] = np.concatenate([modified[position]["extra"], extra], -1)
    np.testing.assert_array_equal(base.predict(inputs[4], inputs[5]), extended.predict(modified[4], modified[5]))
    assert extended.trainable_parameter_count_ == base.trainable_parameter_count_+base.hidden_size+3
    extended.fit(*modified, tail_threshold=4., selection_role="source_validation")
    restored = EncoderNativeResidual.from_payload(extended.to_payload())
    np.testing.assert_array_equal(extended.predict(modified[4], modified[5]), restored.predict(modified[4], modified[5]))
    future = deepcopy(modified[4])
    future["extra"][:, 8:, -3:] += 100.
    np.testing.assert_array_equal(extended.predict_delta(modified[4])[:, :8], extended.predict_delta(future)[:, :8])
