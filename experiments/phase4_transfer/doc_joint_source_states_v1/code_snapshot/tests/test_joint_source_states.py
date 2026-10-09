"""Source joint supervision preserves DOC inference and label isolation."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.joint_source_states import (
    SourceChemistryRegularizer,
    normalized_source_targets,
)


def auxiliary(shape):
    rng = np.random.default_rng(19)
    target = rng.normal(size=(*shape, 2))
    valid = rng.random(target.shape) > .3
    return target, valid


def fit(model, args, regularizer=None):
    return model.fit(*args, tail_threshold=4., selection_role="source_validation",
                     source_regularizer=regularizer)


def test_source_scalers_and_shuffle_ignore_unavailable_values():
    target, valid = auxiliary((3, 14))
    changed = target.copy()
    changed[~valid] = np.nan
    before = normalized_source_targets(target, valid)
    after = normalized_source_targets(changed, valid)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    shuffled = normalized_source_targets(changed, valid, shuffle=True)
    for channel in range(2):
        np.testing.assert_array_equal(np.sort(shuffled[0][..., channel][valid[..., channel]]),
                                      np.sort(before[0][..., channel][valid[..., channel]]))
    for a, b in zip(before[1:], shuffled[1:], strict=True):
        np.testing.assert_array_equal(a, b)
    changed[valid] = np.inf
    with pytest.raises(ValueError, match="finite"):
        normalized_source_targets(changed, valid)


def test_zero_auxiliary_weight_matches_ordinary_doc_optimization():
    base, args, *_ = fixture("last_self_ecology", epochs=3)
    joint, _, *_ = fixture("last_self_ecology", epochs=3)
    target, valid = auxiliary(args[1].shape)
    regularizer = SourceChemistryRegularizer(joint, target, valid, weight=0., batch_size=12)
    fit(base, args)
    fit(joint, args, regularizer)
    assert base.best_epoch_ == joint.best_epoch_ and base.selected_scale_ == joint.selected_scale_
    for a, b in zip(base.trace_, joint.trace_, strict=True):
        for key in ("training_loss", "validation_mae", "best_validation_mae"):
            if a[key] is not None:
                np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(base.predict(args[4], args[5]), joint.predict(args[4], args[5]),
                               rtol=0, atol=1e-12)


def test_joint_fit_hidden_labels_causality_and_doc_save_load():
    a, args, *_ = fixture("last_self_ecology", epochs=2)
    b, changed, *_ = fixture("last_self_ecology", epochs=2)
    target, valid = auxiliary(args[1].shape)
    hidden = target.copy()
    hidden[~valid] = -1e300
    changed[2][~changed[3]] = np.nan
    changed[6][~changed[7]] = -1e300
    ra = SourceChemistryRegularizer(a, target, valid, batch_size=12)
    rb = SourceChemistryRegularizer(b, hidden, valid, batch_size=12)
    fit(a, args, ra)
    fit(b, changed, rb)
    prediction = a.predict(args[4], args[5])
    np.testing.assert_array_equal(prediction, b.predict(changed[4], changed[5]))
    restored = EncoderNativeResidual.from_payload(a.to_payload())
    np.testing.assert_array_equal(prediction, restored.predict(args[4], args[5]))
    summary = ra.summary(a, args[0])
    assert summary["receiving_chemistry_used"] is False
    assert summary["sampled_source_cells"] == a.optimizer_steps_*12
    assert summary["union_source_cells"] > np.count_nonzero(args[3] & valid.any(-1))
    # Check the causal operator with an explicitly nonzero DOC readout as well.
    with torch.no_grad():
        restored.head.weight.fill_(.03)
    restored.selected_scale_ = 1.
    earlier = restored.predict(args[4], args[5])
    future = copy.deepcopy(args[4])
    for key in ("raw", "age", "support", "extra"):
        future[key][:, 8:] += 10
    later = restored.predict(future, args[5])
    np.testing.assert_array_equal(earlier[:, :8], later[:, :8])
