"""Conditional residual densities, deterministic medians and honest selection."""
from __future__ import annotations

import copy
import io
import json
import math

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from river_graph.models.distributional_residual_head import (
    CORRECTION_SCALES,
    DistributionalResidualHead,
    gaussian_mixture_median,
    gaussian_residual_nll,
    residual_distribution,
)


def problem(rows=80, val_rows=33, seed=53):
    rng = np.random.default_rng(seed)
    def side(n):
        x = rng.normal(size=(n, 3)).astype(np.float32)
        x[:, 2] = 1
        base = np.full(n, 5., dtype=np.float32)
        truth = np.expm1(np.log1p(base.astype(float)) + .25 + .4*x[:, 0] - .12*x[:, 1])
        return x, base, truth
    return (*side(rows), *side(val_rows))


def test_deterministic_median_balanced_separated_and_identical_gaussians():
    means = [[-20., 20.], [0., 100.], [2., 2.], [-3., 7.]]
    scales = [[1., 1.], [1., 1.], [.1, 20.], [3., 3.]]
    weights = [[.5, .5], [.5, .5], [.02, .98], [.5, .5]]
    actual = gaussian_mixture_median(means, scales, weights).numpy()
    np.testing.assert_allclose(actual, [0., 50., 2., 2.], atol=2e-13, rtol=0)
    single = gaussian_mixture_median([[2.], [-1.]], [[.1], [100.]], [[1.], [1.]])
    np.testing.assert_array_equal(single, [2., -1.])


def test_median_unbalanced_and_pure_component_cases_do_not_use_mixture_mean():
    actual = gaussian_mixture_median(
        [[0., 20.], [0., 20.], [0., 20.], [0., 20.]],
        [[1., 1.]]*4, [[.9, .1], [.1, .9], [1., 0.], [0., 1.]]).numpy()
    expected = [norm.ppf(.5/.9), 20+norm.ppf((.5-.1)/.9), 0., 20.]
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=0)
    assert abs(actual[0] - 2.) > 1
    with pytest.raises(ValueError, match="ordered"):
        gaussian_mixture_median([[2, 1]], [[1, 1]], [[.5, .5]])
    with pytest.raises(ValueError, match="normalized"):
        gaussian_mixture_median([[0, 1]], [[1, 1]], [[.2, .2]])


@pytest.mark.parametrize("components", [1, 2])
def test_nll_matches_independent_normal_density_and_has_finite_gradients(components):
    raw = torch.tensor([[.2, -.5], [-1., 1.]], dtype=torch.float64) if components == 1 else torch.tensor(
        [[.2, -.5, .1, 1., -2.], [-1., 1., -.5, .2, 3.]], dtype=torch.float64)
    raw.requires_grad_(True)
    residual = torch.tensor([.8, -.4], dtype=torch.float64)
    dist = residual_distribution(raw, components)
    independent = -logsumexp(
        np.log(dist["weights"].detach().numpy())
        + norm.logpdf(residual.numpy()[:, None], loc=dist["means"].detach().numpy(),
                      scale=dist["scales"].detach().numpy()), axis=1)
    nll = gaussian_residual_nll(raw, residual, components)
    np.testing.assert_allclose(nll.detach().numpy(), independent, rtol=1e-14, atol=1e-14)
    nll.mean().backward()
    assert torch.isfinite(raw.grad).all() and torch.count_nonzero(raw.grad) > 0
    if components == 2:
        assert (dist["means"][:, 1] >= dist["means"][:, 0]).all()


@pytest.mark.parametrize("components", [1, 2])
def test_initialization_and_normalization_use_source_only(components):
    data = problem()
    a = DistributionalResidualHead(3, components, epochs=0).fit(*data)
    changed = list(copy.deepcopy(data))
    changed[3] = changed[3] * 7 + 40
    changed[5] = changed[5] + 100
    b = DistributionalResidualHead(3, components, epochs=0).fit(*changed)
    np.testing.assert_array_equal(a.feature_mean_, np.asarray(data[0], dtype=float).mean(0))
    np.testing.assert_allclose(a.feature_raw_std_, np.asarray(data[0], dtype=float).std(0), atol=1e-15)
    assert a.feature_scale_[-1] == 1
    assert a.to_dict()["initialization"] == b.to_dict()["initialization"]
    assert a.to_dict()["normalization"] == b.to_dict()["normalization"]
    torch.testing.assert_close(a.head.weight, torch.zeros_like(a.head.weight), atol=0, rtol=0)
    torch.testing.assert_close(a.head.bias, b.head.bias, atol=0, rtol=0)
    r = np.log1p(data[2])-np.log1p(data[1].astype(float))
    expected_sigma = max(r.std(), .05)
    dist = a.predict_distribution(data[0])
    np.testing.assert_allclose(dist["scales"], expected_sigma, atol=1e-15)
    if components == 1:
        np.testing.assert_allclose(dist["means"], r.mean(), atol=1e-15)
    else:
        low, high = np.quantile(r, [.25, .75])
        np.testing.assert_allclose(dist["means"][0], [low, max(high, low+.05)], atol=1e-15)


@pytest.mark.parametrize("components", [1, 2])
def test_actual_feature_learning_improves_native_validation_mae(components):
    data = problem()
    source_features = torch.tensor(data[0], requires_grad=True)
    model = DistributionalResidualHead(3, components, epochs=35, patience=10,
                                       batch_size=40, learning_rate=.02).fit(source_features, *data[1:])
    assert model.best_epoch_ > 0
    assert model.validation_metrics_["validation_mae"] < .6 * model.trace_[0]["validation_mae"]
    assert model.selected_scale_ in CORRECTION_SCALES
    assert source_features.grad is None
    assert model.trainable_parameter_count_ == (3+1)*(2 if components == 1 else 5)
    assert torch.count_nonzero(model.head.weight) > 0
    prediction = model.predict(data[3], data[4])
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    np.testing.assert_allclose(np.mean(abs(prediction-data[5])),
                               model.validation_metrics_["validation_mae"], atol=1e-14)
    assert math.isfinite(model.to_dict()["selected_source_nll"])


def test_exact_base_fallback_epoch_zero_and_nonnegative_inverse():
    x = np.zeros((12, 2), dtype=np.float32)
    base = np.asarray([0., 1., 2., 3.]*3, dtype=np.float32)
    # Source says add DOC but exact validation base is already optimal.
    model = DistributionalResidualHead(2, 2, epochs=4, patience=2).fit(x, base, base+4, x, base, base)
    assert model.selected_scale_ == 0 and model.best_epoch_ == 0
    assert model.epochs_run_ == 2
    actual = model.predict(x, base)
    assert actual.dtype == base.dtype
    np.testing.assert_array_equal(actual, base)
    np.testing.assert_array_equal(DistributionalResidualHead._combine(
        torch.tensor([0., 1.]), torch.tensor([-3., -3.]), 1), [0, 0])
    assert len(model.predict_distribution(x[:0])["median_residual"]) == 0
    assert model.predict(x[:0], base[:0]).shape == (0,)


def test_exact_scale_ties_prefer_lower_scale_then_earlier_epoch(monkeypatch):
    values = [(2., 1.), (2., .5), (2., .5)]
    model = DistributionalResidualHead(3, 1, epochs=2)
    def evaluate(*args):
        mae, scale = values.pop(0)
        return {"validation_mae": mae, "selected_scale": scale, "validation_nll": 1.,
                "scale_scores": [{"scale": scale, "mae": mae}]}
    monkeypatch.setattr(model, "_evaluate", evaluate)
    model.fit(*problem())
    assert model.best_epoch_ == 1 and model.selected_scale_ == .5


@pytest.mark.parametrize("components", [1, 2])
def test_checkpoint_roundtrip_refit_and_inference_need_no_labels(components):
    data = problem(rows=30, val_rows=11)
    model = DistributionalResidualHead(3, components, epochs=3, batch_size=16).fit(*data)
    summary = model.to_dict()
    json.dumps(summary, allow_nan=False)
    stream = io.BytesIO()
    torch.save(model.to_payload(), stream)
    stream.seek(0)
    restored = DistributionalResidualHead.from_payload(torch.load(stream, weights_only=True))
    assert restored.to_dict() == summary
    np.testing.assert_array_equal(restored.predict(data[3], data[4]), model.predict(data[3], data[4]))
    a, b = restored.predict_distribution(data[3]), model.predict_distribution(data[3])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    restored.fit(*data)
    assert restored.to_dict() == summary
    invalid = copy.deepcopy(model.to_payload()); invalid["head_state"]["weight"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        DistributionalResidualHead.from_payload(invalid)


def test_source_role_alignment_finiteness_and_constructor_rng_isolation():
    state = torch.random.get_rng_state().clone()
    model = DistributionalResidualHead(3, 2, epochs=0)
    assert torch.equal(torch.random.get_rng_state(), state)
    data = problem()
    with pytest.raises(ValueError, match="source_validation"):
        model.fit(*data, selection_role="test")
    bad = copy.deepcopy(data); bad[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(*bad)
    with pytest.raises(ValueError, match="aligned"):
        model.fit(data[0], data[1][:-1], *data[2:])
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(data[3], data[4])
    with pytest.raises(ValueError, match="components"):
        DistributionalResidualHead(components=3)
