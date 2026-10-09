"""Full self-path tuning retains causality and isolates the newly trained scope."""
from __future__ import annotations

import copy

import numpy as np
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual


def full_encoder(base):
    return EncoderNativeResidual(base.spatial, base.temporal, base.decay,
        **{**base._config(), "encoder_mode": "all_self_ecology"})


def test_full_encoder_changes_training_scope_but_preserves_initial_forward():
    base, args, *_ = fixture("last_self_ecology", epochs=2)
    model = full_encoder(base)
    added = sum(p.numel() for conv in base.spatial.convs[:-1] for p in conv.self_lin.parameters())
    assert model.trainable_parameter_count_ == base.trainable_parameter_count_+added
    assert sum(p.numel() for p in model.spatial.parameters()) == sum(p.numel() for p in base.spatial.parameters())
    with torch.no_grad():
        base.head.weight.fill_(.03)
        model.head.load_state_dict(base.head.state_dict())
    np.testing.assert_array_equal(base.predict_delta(args[0]), model.predict_delta(args[0]))
    before = copy.deepcopy(model.spatial.state_dict())
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    assert model.to_dict()["first_self_parameter_distance"] > 0
    for name, parameter in model.spatial.named_parameters():
        active = ".self_lin." in name or name.startswith("env_encoder.")
        assert parameter.requires_grad == active
        if not active:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    restored = EncoderNativeResidual.from_payload(model.to_payload())
    assert restored.encoder_mode == "all_self_ecology"
    assert model.to_dict() == restored.to_dict()
    np.testing.assert_array_equal(model.predict(args[4], args[5]), restored.predict(args[4], args[5]))


def test_full_encoder_hidden_labels_and_future_inputs_remain_isolated():
    base, args, *_ = fixture("last_self_ecology", epochs=2)
    first, second = full_encoder(base), full_encoder(base)
    changed = list(copy.deepcopy(args))
    changed[2][~changed[3]] = np.nan
    changed[6][~changed[7]] = np.nan
    for model, data in ((first, args), (second, changed)):
        model.fit(*data, tail_threshold=4., selection_role="source_validation")
    assert first.to_dict() == second.to_dict()
    inputs = copy.deepcopy(args[4])
    for key in ("raw", "age", "support", "extra"):
        inputs[key][:, 8:] += 100.
    np.testing.assert_array_equal(first.predict_delta(args[4])[:, :8],
                                  first.predict_delta(inputs)[:, :8])
