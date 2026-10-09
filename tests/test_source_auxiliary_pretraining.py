"""Source auxiliary label isolation and warm-backbone continuation."""
from __future__ import annotations

import copy

import numpy as np
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.source_auxiliary_pretraining import (
    prepare_auxiliary_labels,
    pretrain_source_auxiliary,
    source_auxiliary_targets,
)


def targets():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(3, 14, 2))
    valid = np.ones_like(target, dtype=bool)
    valid[:, 3, 0] = False
    return target, valid, np.array([True, False, False])


def test_receiving_auxiliary_labels_and_masks_are_not_read():
    auxiliary = {name: {"site_no": np.arange(4), "months": np.arange(14),
        "y": np.ones((4, 14)), "y_mask": np.ones((4, 14), dtype=bool)}
        for name in ("ph", "spec_conductance")}
    a = source_auxiliary_targets(auxiliary, np.arange(3), np.arange(4), np.arange(14))
    for dataset in auxiliary.values():
        dataset["y"][3] = np.nan
        dataset["y_mask"][3] = False
    b = source_auxiliary_targets(auxiliary, np.arange(3), np.arange(4), np.arange(14))
    for original, changed in zip(a, b, strict=True):
        np.testing.assert_array_equal(original, changed)


def test_normalization_excludes_held_station_and_shuffle_preserves_fit_labels():
    target, valid, held = targets()
    a = prepare_auxiliary_labels(target, valid, held)
    changed = target.copy()
    changed[held] += 1000
    changed[~valid] = np.nan
    b = prepare_auxiliary_labels(changed, valid, held)
    for i in (1, 2, 3, 4):
        np.testing.assert_array_equal(a[i], b[i])
    np.testing.assert_array_equal(a[0][a[1]], b[0][b[1]])
    shuffled = prepare_auxiliary_labels(target, valid, held, shuffle=True)
    for channel in range(2):
        mask = a[1][..., channel]
        np.testing.assert_array_equal(np.sort(a[0][..., channel][mask]), np.sort(shuffled[0][..., channel][mask]))
    np.testing.assert_array_equal(a[0][a[2]], shuffled[0][shuffled[2]])


def test_auxiliary_hidden_labels_do_not_affect_fit_and_warm_weights_survive_doc_reset():
    old, args, *_ = fixture("last_self_ecology", epochs=0)
    target, valid, held = targets()
    warm, summary, saved = pretrain_source_auxiliary(old, args[0], target, valid, held, epochs=2, batch_size=12)
    changed = target.copy()
    changed[~valid] = np.nan
    repeat, other, _ = pretrain_source_auxiliary(old, args[0], changed, valid, held, epochs=2, batch_size=12)
    assert summary == other
    for name in ("spatial", "temporal", "decay"):
        for key, value in getattr(warm, name).state_dict().items():
            torch.testing.assert_close(value, getattr(repeat, name).state_dict()[key], rtol=0, atol=0)
        for key, value in getattr(old, name).state_dict().items():
            torch.testing.assert_close(value, getattr(old, f"_initial_{name}_state")[key], rtol=0, atol=0)
    new = EncoderNativeResidual(warm.spatial, warm.temporal, warm.decay, **old._config())
    new.fit(*args, tail_threshold=4., selection_role="source_validation")
    for name in ("spatial", "temporal", "decay"):
        for key, value in getattr(new, name).state_dict().items():
            torch.testing.assert_close(value, saved[name][key], rtol=0, atol=0)
    restored = EncoderNativeResidual.from_payload(new.to_payload())
    with torch.no_grad():
        for model in (new, restored):
            model.head.weight.fill_(.02)
            model.selected_scale_ = .5
    prediction = new.predict(args[4], args[5])
    np.testing.assert_array_equal(prediction, restored.predict(args[4], args[5]))
    future = copy.deepcopy(args[4])
    for key in ("raw", "age", "support", "extra"):
        future[key][:, 8:] += 10
    np.testing.assert_array_equal(prediction[:, :8], new.predict(future, args[5])[:, :8])
