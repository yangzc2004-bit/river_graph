"""Frozen native heads: compact labels, source normalization and exact routing."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
import torch

from river_graph.models.frozen_native_feature_head import (
    CORRECTION_SCALES,
    FrozenNativeFeatureHead,
)


def problem(seed=8, n=48, v=21):
    rng = np.random.default_rng(seed)
    def side(rows):
        features = rng.normal(size=(rows, 3)).astype(np.float32)
        features[:, 2] = 2
        base = np.full(rows, 5., dtype=np.float64)
        active = np.arange(rows) % 4 != 0
        truth = base + .6 + .4*features[:, 0]-.2*features[:, 1]
        return features, base, truth, active
    a, b = side(n), side(v)
    return (*a[:3], *b[:3]), {"source_active": a[3], "validation_active": b[3], "tail_threshold": 5.8}


def test_zero_initialization_source_only_normalization_and_parameter_count():
    state = torch.random.get_rng_state().clone()
    assert FrozenNativeFeatureHead().trainable_parameter_count_ == 683
    assert torch.equal(torch.random.get_rng_state(), state)
    data, kwargs = problem()
    model = FrozenNativeFeatureHead(3, epochs=0).fit(*data, **kwargs)
    changed = list(copy.deepcopy(data))
    changed[3] = changed[3]*12+50
    changed[5] = changed[5]+100
    other = FrozenNativeFeatureHead(3, epochs=0).fit(*changed, **kwargs)
    assert model.to_dict()["normalization"] == other.to_dict()["normalization"]
    np.testing.assert_allclose(model.feature_mean_, data[0].astype(float).mean(0), atol=1e-15)
    np.testing.assert_allclose(model.feature_raw_std_, data[0].astype(float).std(0), atol=1e-15)
    assert model.feature_scale_[2] == 1
    assert model.best_epoch_ == 0 and model.selected_scale_ == 0
    assert not torch.count_nonzero(model.head.weight) and not torch.count_nonzero(model.head.bias)
    np.testing.assert_array_equal(model.predict_delta(data[3], active=kwargs["validation_active"]), np.zeros(len(data[3])))
    base32 = data[4].astype(np.float32)
    result = model.predict(data[3], base32, active=kwargs["validation_active"])
    assert result.dtype == base32.dtype
    np.testing.assert_array_equal(result, base32)


def test_learning_improves_active_rows_and_never_changes_inactive_native_base():
    data, kwargs = problem()
    source = torch.tensor(data[0], requires_grad=True)
    model = FrozenNativeFeatureHead(3, epochs=45, patience=8, learning_rate=.04, batch_size=12).fit(
        source, *data[1:], **kwargs)
    assert source.grad is None
    assert model.best_epoch_ > 0 and model.selected_scale_ in CORRECTION_SCALES[1:]
    assert model.validation_metrics_["validation_mae"] < .55*model.baseline_validation_mae_
    delta = model.predict_delta(data[3], active=kwargs["validation_active"])
    prediction = model.predict(data[3], data[4], active=kwargs["validation_active"])
    np.testing.assert_array_equal(delta[~kwargs["validation_active"]], 0)
    np.testing.assert_array_equal(prediction[~kwargs["validation_active"]], data[4][~kwargs["validation_active"]])
    assert np.ptp(delta[kwargs["validation_active"]]) > .1
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    np.testing.assert_allclose(abs(prediction-data[5]).mean(), model.validation_metrics_["validation_mae"], atol=1e-14)


def test_tail_threshold_inclusive_and_global_denominator_include_inactive_rows():
    x = np.array([[1., 0.], [2., 1.], [3., 2.]])
    base, y = np.zeros(3), np.array([1., 2., 4.])
    active = np.array([True, False, True])
    model = FrozenNativeFeatureHead(2, epochs=0).fit(x, base, y, x, base, y,
        source_active=active, validation_active=active, tail_threshold=2.)
    summary = model.to_dict()
    assert summary["n_source_tail_cells"] == 2 and summary["n_source_active"] == 2
    assert summary["source_weight_sum"] == 5 and summary["source_weight_mean"] == 5/3
    assert model.trace_[0]["training_loss"] == (1+2*2+2*4)/5
    assert summary["selected_source_weighted_mae"] == 13/5
    assert model.trace_[0]["validation_mae"] == 7/3


def test_zero_baseline_has_upward_gradient_and_inactive_rows_have_none():
    delta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    active = torch.tensor([True, False])
    gated = torch.where(active, delta, torch.zeros_like(delta))
    prediction = FrozenNativeFeatureHead._combine(torch.zeros(2), gated, 1.)
    (prediction-torch.tensor([2., 2.])).abs().sum().backward()
    torch.testing.assert_close(delta.grad, torch.tensor([-1., 0.], dtype=torch.float64), rtol=0, atol=0)
    x, base, truth = np.ones((12, 1)), np.zeros(12), np.ones(12)
    mask = np.ones(12, dtype=bool)
    model = FrozenNativeFeatureHead(1, epochs=8, learning_rate=.1).fit(x, base, truth, x, base, truth,
        source_active=mask, validation_active=mask, tail_threshold=.5)
    assert model.predict(x, base, active=mask).min() > .5


def test_each_batch_uses_one_global_source_weight_denominator(monkeypatch):
    captured = []
    original = torch.nn.utils.clip_grad_norm_
    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured.append(float(parameters[-1].grad))
        return original(parameters, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    x, base, truth = np.zeros((3, 1)), np.zeros(3), np.array([1., 2., 4.])
    active = np.array([True, False, True])
    FrozenNativeFeatureHead(1, epochs=1, batch_size=1, seed=42).fit(x, base, truth, x, base, truth,
        source_active=active, validation_active=active, tail_threshold=2.)
    order = np.random.default_rng(np.random.SeedSequence([42, 1])).permutation(3)
    expected = -np.array([1., 2., 2.]) * active / (5/3)
    np.testing.assert_allclose(captured, expected[order], rtol=0, atol=1e-14)


def test_all_inactive_and_exact_validation_base_keep_epoch_zero_fallback():
    data, kwargs = problem()
    kwargs["source_active"][:] = False
    kwargs["validation_active"][:] = False
    model = FrozenNativeFeatureHead(3, epochs=12, patience=2).fit(*data, **kwargs)
    assert model.best_epoch_ == 0 and model.epochs_run_ == 2 and model.selected_scale_ == 0
    assert not torch.count_nonzero(model.head.weight)
    np.testing.assert_array_equal(model.predict(data[3], data[4], active=kwargs["validation_active"]), data[4])
    assert len(model.predict_delta(data[3][:0], active=np.zeros(0, dtype=bool))) == 0
    assert len(model.predict(data[3][:0], data[4][:0], active=np.zeros(0, dtype=bool))) == 0


def test_inactive_features_do_not_affect_inference_when_head_is_nonzero():
    data, kwargs = problem()
    model = FrozenNativeFeatureHead(3, epochs=8, learning_rate=.05).fit(*data, **kwargs)
    assert model.selected_scale_ > 0
    active = kwargs["validation_active"]
    changed = data[3].copy()
    changed[~active] = 1e6
    np.testing.assert_array_equal(model.predict(changed, data[4], active=active),
                                  model.predict(data[3], data[4], active=active))


def test_checkpoint_ties_use_lower_scale_then_earlier_epoch(monkeypatch):
    data, kwargs = problem()
    outcomes = [(1., 1.), (1., .5), (1., .5)]
    model = FrozenNativeFeatureHead(3, epochs=2)
    def evaluate(*args):
        mae, scale = outcomes.pop(0)
        return {"validation_mae": mae, "selected_scale": scale, "scale_scores": [{"scale": scale, "mae": mae}]}
    monkeypatch.setattr(model, "_evaluate", evaluate)
    model.fit(*data, **kwargs)
    assert model.best_epoch_ == 1 and model.selected_scale_ == .5


def test_payload_and_deterministic_refit_preserve_predictions_and_trace():
    data, kwargs = problem()
    model = FrozenNativeFeatureHead(3, epochs=4, batch_size=13).fit(*data, **kwargs)
    summary = model.to_dict()
    json.dumps(summary, allow_nan=False)
    stream = io.BytesIO()
    torch.save(model.to_payload(), stream)
    stream.seek(0)
    restored = FrozenNativeFeatureHead.from_payload(torch.load(stream, weights_only=True))
    assert restored.to_dict() == summary
    for method in ("predict", "predict_delta"):
        args = (data[3], data[4]) if method == "predict" else (data[3],)
        np.testing.assert_array_equal(getattr(restored, method)(*args, active=kwargs["validation_active"]),
                                      getattr(model, method)(*args, active=kwargs["validation_active"]))
    restored.fit(*data, **kwargs)
    assert restored.to_dict() == summary
    for key, value in model.head.state_dict().items():
        torch.testing.assert_close(value, restored.head.state_dict()[key], rtol=0, atol=0)


def test_role_activity_shape_and_finite_contracts():
    data, kwargs = problem()
    model = FrozenNativeFeatureHead(3, epochs=0)
    with pytest.raises(RuntimeError, match="fit or restore"):
        model.predict(data[3], data[4], active=kwargs["validation_active"])
    with pytest.raises(ValueError, match="source_validation"):
        model.fit(*data, **kwargs, selection_role="test")
    with pytest.raises(ValueError, match="boolean"):
        model.fit(*data, **{**kwargs, "source_active": kwargs["source_active"].astype(int)})
    with pytest.raises(ValueError, match="aligned"):
        model.fit(*data, **{**kwargs, "validation_active": kwargs["validation_active"][:-1]})
    with pytest.raises(ValueError, match="tail_threshold"):
        model.fit(*data, **{**kwargs, "tail_threshold": np.nan})
    broken = list(copy.deepcopy(data)); broken[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(*broken, **kwargs)
    model.fit(*data, **kwargs)
    payload = model.to_payload(); payload["head_state"]["weight"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        FrozenNativeFeatureHead.from_payload(payload)
