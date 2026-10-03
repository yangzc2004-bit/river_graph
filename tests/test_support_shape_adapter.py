"""Numerical and information-boundary contracts for frozen shape adaptation."""
from __future__ import annotations

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from river_graph.models.support_shape_adapter import (
    StationTemporalBasis,
    SupportShapeAdapter,
    SupportShapeEpisode,
)


def shape_episode():
    support_cells = np.array([0, 1, 2, 8, 9, 10])
    query_cells = np.array([3, 4, 11, 12])
    support_basis = np.tile(np.array([[-1., 0], [0, 0], [1, 0]]), (2, 1))
    query_basis = np.array([[.5, 0], [2, 0], [.5, 0], [2, 0]])
    residual = np.array([-.3, 0, .3, -.15, 0, .15])
    support_prediction = np.full(6, 3.)
    support_values = np.expm1(np.log1p(support_prediction) + residual)
    coefficient = np.array([.3, .15]) * (2 / 3) / (2 / 3 + .1)
    query_values = np.expm1(np.log(4) + np.repeat(coefficient, 2) * query_basis[:, 0])
    return SupportShapeEpisode(
        k=3, query_cells=query_cells, query_values=query_values,
        query_prediction=np.full(4, 3.), support_cells=support_cells,
        support_values=support_values, support_prediction=support_prediction,
        query_basis=query_basis, support_basis=support_basis,
    )


def adapt_episode(adapter, episode):
    return adapter.adapt(
        episode.query_prediction, episode.query_cells, episode.support_prediction,
        episode.support_cells, episode.support_values, query_basis=episode.query_basis,
        support_basis=episode.support_basis, k=episode.k,
    )


def test_basis_whitens_within_station_dynamics_and_stream_matches():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(5, 12, 4)) + np.arange(5)[:, None, None] * 100
    batch = StationTemporalBasis().fit(features)
    stream = StationTemporalBasis().fit_station_sequences(sequence for sequence in features)
    projected = batch.transform(features)
    assert projected.shape == (5, 12, 2)
    np.testing.assert_allclose(projected.mean(1), 0, atol=1e-13)
    flat = projected.reshape(-1, 2)
    np.testing.assert_allclose(flat.T @ flat / len(flat), np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(stream.transform(features), projected)
    np.testing.assert_allclose(batch.transform(features + 500), projected, atol=1e-12)


def test_basis_station_isolation_and_serialization():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(3, 8, 4))
    target = rng.normal(size=(2, 9, 4))
    basis = StationTemporalBasis().fit(source)
    state = basis.to_dict()
    expected = basis.transform(target)
    target[1] *= 100
    np.testing.assert_array_equal(basis.transform(target)[0], expected[0])
    assert state == basis.to_dict()
    restored = StationTemporalBasis.from_dict(json.loads(json.dumps(state, allow_nan=False)))
    np.testing.assert_array_equal(restored.transform(target), basis.transform(target))
    np.testing.assert_array_equal(basis.transform(target[0]), expected[0])


def test_basis_constant_and_low_rank_edges_are_finite():
    basis = StationTemporalBasis().fit(np.full((2, 10, 3), 17.))
    np.testing.assert_array_equal(basis.transform(np.full((3, 7, 3), 99.)), 0)
    assert np.isfinite(basis.transform(np.arange(21).reshape(7, 3))).all()
    assert (basis.eigenvalues_ == 0).all()
    with pytest.raises(ValueError, match="component"):
        StationTemporalBasis().fit(np.ones((2, 3, 1)))
    with pytest.raises(ValueError, match="finite"):
        StationTemporalBasis().fit(np.full((2, 3, 2), np.nan))
    with pytest.raises(ValueError, match="source_training"):
        StationTemporalBasis().fit(np.ones((2, 3, 2)), source_role="target_test")


def test_k0_preserves_predictions_and_dtype_exactly():
    prediction = np.array([0., .01, 11.4], dtype=np.float32)
    adapter = SupportShapeAdapter(n_months=8)
    output = adapter.adapt(
        prediction, np.array([0, 1, 8]), np.empty(0), np.empty(0, dtype=int),
        np.empty(0), query_basis=np.ones((3, 2)), support_basis=np.empty((0, 2)), k=0,
    )
    np.testing.assert_array_equal(output, prediction)
    assert output.dtype == prediction.dtype
    assert not np.shares_memory(output, prediction)


def test_k1_is_exactly_constant_correction_even_with_nonconstant_basis():
    episode = SupportShapeEpisode(
        k=1, query_cells=np.array([1, 2]), query_values=np.array([3., 7.]),
        query_prediction=np.array([1., 3.]), support_cells=np.array([0]),
        support_values=np.array([7.]), support_prediction=np.array([1.]),
        query_basis=np.array([[-100., 4.], [50., -2.]]), support_basis=np.array([[5., 6.]]),
    )
    shape = SupportShapeAdapter(n_months=8, alpha_values=(.5,), ridge_strengths=(.1,))
    constant = SupportShapeAdapter(n_months=8, alpha_values=(.5,), ridge_strengths=(np.inf,))
    for adapter in (shape, constant):
        adapter.fit([episode], selection_role="source_validation")
    expected = np.maximum(np.expm1(np.log1p(episode.query_prediction)
                                  + .5 * (np.log1p(7.) - np.log1p(1.))), 0)
    np.testing.assert_array_equal(adapt_episode(shape, episode), expected)
    np.testing.assert_array_equal(adapt_episode(shape, episode), adapt_episode(constant, episode))


def test_ridge_matches_mean_squared_objective_and_adapts_shape():
    episode = shape_episode()
    adapter = SupportShapeAdapter(n_months=8, alpha_values=(.5,), ridge_strengths=(.1,))
    adapter.fit([episode], selection_role="source_validation")
    prediction = adapt_episode(adapter, episode)
    np.testing.assert_allclose(prediction, episode.query_values, rtol=1e-12)
    correction = np.log1p(prediction) - np.log1p(episode.query_prediction)
    assert correction[0] != correction[1]


def test_support_changes_cannot_affect_another_station():
    episode = shape_episode()
    adapter = SupportShapeAdapter(n_months=8).fit([episode], selection_role="source_validation")
    expected = adapt_episode(adapter, episode)
    changed_values = episode.support_values.copy()
    changed_values[3:] *= 2
    changed = adapt_episode(adapter, replace(episode, support_values=changed_values))
    np.testing.assert_array_equal(changed[:2], expected[:2])
    assert not np.allclose(changed[2:], expected[2:])


def test_support_and_query_order_invariance():
    episode = shape_episode()
    adapter = SupportShapeAdapter(n_months=8).fit([episode], selection_role="source_validation")
    expected = adapt_episode(adapter, episode)
    sp, qp = np.array([5, 0, 3, 2, 1, 4]), np.array([3, 1, 2, 0])
    reordered = replace(episode,
        query_cells=episode.query_cells[qp], query_values=episode.query_values[qp],
        query_prediction=episode.query_prediction[qp], query_basis=episode.query_basis[qp],
        support_cells=episode.support_cells[sp], support_values=episode.support_values[sp],
        support_prediction=episode.support_prediction[sp], support_basis=episode.support_basis[sp])
    np.testing.assert_allclose(adapt_episode(adapter, reordered), expected[qp], rtol=1e-12)


def test_adapter_serialization_handles_infinity_and_no_query_labels_in_adapt():
    episode = shape_episode()
    adapter = SupportShapeAdapter(n_months=8).fit([episode], selection_role="source_validation")
    state = json.loads(json.dumps(adapter.to_dict(), allow_nan=False))
    assert "infinity" in state["ridge_strengths"]
    restored = SupportShapeAdapter.from_dict(state)
    np.testing.assert_array_equal(adapt_episode(restored, episode), adapt_episode(adapter, episode))
    assert "query_values" not in inspect.signature(adapter.adapt).parameters
    with pytest.raises(ValueError, match="source_validation"):
        adapter.fit([episode], selection_role="target_test")


def test_selection_pools_query_cells_rather_than_episode_means():
    def episode(station, queries, support_y, truth):
        return SupportShapeEpisode(
            k=1, query_cells=np.arange(station * 10 + 1, station * 10 + 1 + queries),
            query_values=np.full(queries, truth), query_prediction=np.ones(queries),
            support_cells=np.array([station * 10]), support_values=np.array([support_y]),
            support_prediction=np.ones(1), query_basis=np.zeros((queries, 2)),
            support_basis=np.zeros((1, 2)))
    tasks = [episode(0, 1, 5., 5.), episode(1, 9, 3., 1.)]
    adapter = SupportShapeAdapter(n_months=10, alpha_values=(0., 1.), ridge_strengths=(np.inf,))
    adapter.fit(tasks, selection_role="source_validation")
    assert adapter.selection_by_k_[1]["alpha"] == 0
    scores = {row["alpha"]: row["mae"] for row in adapter.selection_scores_}
    assert scores[0.] == pytest.approx(.4)
    assert scores[1.] == pytest.approx(1.8)


def test_constant_basis_prefers_constant_control_and_invalid_support_rejected():
    episode = shape_episode()
    constant = replace(episode, query_basis=np.zeros_like(episode.query_basis),
                       support_basis=np.zeros_like(episode.support_basis))
    adapter = SupportShapeAdapter(n_months=8).fit([constant], selection_role="source_validation")
    assert np.isinf(adapter.selection_by_k_[3]["ridge_strength"])
    assert np.isfinite(adapt_episode(adapter, constant)).all()
    with pytest.raises(ValueError, match="disjoint"):
        adapt_episode(adapter, replace(constant, query_cells=np.array([0, 4, 11, 12])))
    with pytest.raises(ValueError, match="exactly K"):
        adapt_episode(adapter, replace(constant, support_cells=np.array([0, 1, 2, 8, 9, 18])))
    with pytest.raises(ValueError, match="finite"):
        adapt_episode(adapter, replace(constant, support_values=np.full(6, np.nan)))
    with pytest.raises(RuntimeError, match="not been selected"):
        adapt_episode(SupportShapeAdapter(n_months=8), constant)


@pytest.mark.parametrize("ridge", [(0,), (-1,), (np.nan,), (-np.inf,)])
def test_invalid_ridge_candidates_rejected(ridge):
    with pytest.raises(ValueError, match="ridge"):
        SupportShapeAdapter(n_months=8, ridge_strengths=ridge)
