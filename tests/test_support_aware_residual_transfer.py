"""Support-aware ecological weighting uses the unchanged station adapter."""

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import (
    SupportShapeAdapter,
    SupportShapeEpisode,
)


def episodes():
    """True support reveals a multiplicative level offset, unlike ecological bias."""
    tasks = []
    for k in (0, 1, 3, 5):
        support_prediction = np.arange(1, k + 1, dtype=float)
        tasks.append(SupportAwareTransferEpisode(
            k=k, query_cells=np.array([5, 6]), query_values=np.array([3., 7.]),
            support_cells=np.arange(k), support_values=2 * (support_prediction + 1) - 1,
            query_context=np.array([1., 3.]), query_temporal=np.array([1., 3.]),
            query_memory=np.full(2, 4.), support_context=support_prediction,
            support_temporal=support_prediction, support_memory=np.full(k, 4.),
            query_basis=np.zeros((2, 2)), support_basis=np.zeros((k, 2)),
        ))
    return tasks


def adapt(model, episode):
    return model.adapt(
        episode.query_context, episode.query_temporal, episode.query_memory, episode.query_cells,
        episode.support_context, episode.support_temporal, episode.support_memory, episode.support_cells,
        episode.support_values, episode.query_basis, episode.support_basis, k=episode.k,
    )


def old_episode(episode):
    return SupportShapeEpisode(
        k=episode.k, query_cells=episode.query_cells, query_values=episode.query_values,
        query_prediction=episode.query_temporal, support_cells=episode.support_cells,
        support_values=episode.support_values, support_prediction=episode.support_temporal,
        query_basis=episode.query_basis, support_basis=episode.support_basis,
    )


def test_informative_support_reduces_ecological_weight_while_k0_stays_locked():
    tasks = episodes()
    model = SupportAwareResidualTransfer(10).fit(tasks, gamma_k0=1)
    assert model.selected_gamma(0) == 1
    np.testing.assert_array_equal(adapt(model, tasks[0]), np.array([5., 7.]))
    assert model.selected_gamma(3) == model.selected_gamma(5) == 0
    np.testing.assert_allclose(adapt(model, tasks[-1]), tasks[-1].query_values, atol=1e-14)
    assert model.selection_by_k_[5]["mae"] < model.selection_by_k_[0]["mae"]
    assert model.selection_by_k_[0]["locked"] is True
    assert model.selection_by_k_[5]["locked"] is False


@pytest.mark.parametrize("ridges", [(float("inf"),), (.1, 1., 10., float("inf"))])
def test_zero_gamma_is_bitwise_existing_interaction_adapter(ridges):
    tasks = episodes()
    model = SupportAwareResidualTransfer(10, ridge_strengths=ridges).fit(tasks, gamma_k0=0)
    original = SupportShapeAdapter(n_months=10, ridge_strengths=ridges).fit(
        [old_episode(task) for task in tasks], selection_role="source_validation",
    )
    assert model.adapters_by_gamma_[0].to_dict() == original.to_dict()
    for task in tasks:
        if model.selected_gamma(task.k) == 0:
            expected = original.adapt(
                task.query_temporal, task.query_cells, task.support_temporal, task.support_cells,
                task.support_values, query_basis=task.query_basis, support_basis=task.support_basis, k=task.k,
            )
            np.testing.assert_array_equal(adapt(model, task), expected)


def test_locked_zero_gamma_preserves_float32_prediction_and_selected_base():
    task = episodes()[0]
    model = SupportAwareResidualTransfer(10).fit([task], gamma_k0=0)
    changed = replace(task, query_temporal=np.array([.1, 10.25], dtype=np.float32))
    prediction = adapt(model, changed)
    np.testing.assert_array_equal(prediction, changed.query_temporal)
    assert prediction.dtype == changed.query_temporal.dtype
    assert not np.shares_memory(prediction, changed.query_temporal)
    np.testing.assert_array_equal(model.selected_base(changed.query_context, changed.query_temporal,
                                                      changed.query_memory, k=0), prediction)


def test_support_labels_change_predictions_but_query_or_unpassed_labels_do_not():
    tasks = episodes()
    model = SupportAwareResidualTransfer(10).fit(tasks, gamma_k0=1)
    task = tasks[-1]
    predicted = adapt(model, task)
    changed_support = adapt(model, replace(task, support_values=2 * task.support_values))
    assert not np.allclose(predicted, changed_support)
    np.testing.assert_array_equal(adapt(model, replace(task, query_values=np.full(2, np.nan))), predicted)
    signature = inspect.signature(model.adapt).parameters
    assert "query_values" not in signature and "future_labels" not in signature


def test_gamma_ties_prefer_zero_then_locked_prior_then_lower_gamma():
    equal = [replace(task, query_memory=np.zeros_like(task.query_memory),
                     support_memory=np.zeros_like(task.support_memory)) for task in episodes()]
    same = SupportAwareResidualTransfer(10).fit(equal, gamma_k0=.5)
    assert same.selected_gamma(0) == .5
    assert same.selected_gamma(5) == 0

    def floor_task(k):
        return SupportAwareTransferEpisode(
            k=k, query_cells=np.array([5]), query_values=np.array([0.]),
            support_cells=np.arange(k), support_values=np.ones(k),
            query_context=np.ones(1), query_temporal=np.ones(1), query_memory=np.full(1, -10.),
            support_context=np.ones(k), support_temporal=np.ones(k), support_memory=np.zeros(k),
            query_basis=np.zeros((1, 2)), support_basis=np.zeros((k, 2)),
        )

    floor_tasks = [floor_task(0), floor_task(1)]
    prior = SupportAwareResidualTransfer(10).fit(floor_tasks, gamma_k0=.5)
    assert prior.selected_gamma(1) == .5
    smaller = SupportAwareResidualTransfer(10).fit(floor_tasks, gamma_k0=0)
    assert smaller.selected_gamma(1) == .25


def test_serialization_replays_every_k_and_keeps_all_candidate_scores():
    tasks = episodes()
    model = SupportAwareResidualTransfer(10).fit(tasks, gamma_k0=.5)
    state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = SupportAwareResidualTransfer.from_dict(state)
    assert restored.to_dict() == model.to_dict()
    assert len(state["gamma_scores"]) == 4 * 4
    assert len(state["adapters_by_gamma"]) == 4
    for task in tasks:
        np.testing.assert_array_equal(adapt(model, task), adapt(restored, task))
        choice = state["selection_by_k"][str(task.k)]
        matching = [score for score in state["gamma_scores"] if score["k"] == task.k
                    and score["gamma"] == choice["gamma"]]
        assert choice["mae"] == matching[0]["mae"]
    state["selection_by_k"]["0"]["gamma"] = 0
    with pytest.raises(ValueError, match="locked K=0"):
        SupportAwareResidualTransfer.from_dict(state)


def test_existing_temporal_shape_basis_remains_active():
    task = episodes()[2]
    sb = np.array([[-1., 0], [0, 0], [1., 0]])
    qb = np.array([[.5, 0], [2, 0]])
    support_residual = np.array([-.3, 0, .3])
    support_prediction = np.full(3, 3.)
    coefficient = .3 * (2 / 3) / (2 / 3 + .1)
    task = replace(task, support_context=support_prediction, support_temporal=support_prediction,
                   support_values=np.expm1(np.log(4) + support_residual),
                   query_context=np.full(2, 3.), query_temporal=np.full(2, 3.),
                   query_values=np.expm1(np.log(4) + coefficient * qb[:, 0]),
                   query_memory=np.zeros(2), support_memory=np.zeros(3), query_basis=qb, support_basis=sb)
    model = SupportAwareResidualTransfer(10).fit([episodes()[0], task], gamma_k0=0)
    assert model.selection_by_k_[3]["ridge_strength"] == .1
    np.testing.assert_allclose(adapt(model, task), task.query_values, rtol=1e-12)


def test_role_missing_k0_signed_memory_and_alignment_guards():
    tasks = episodes()
    with pytest.raises(ValueError, match="source_validation"):
        SupportAwareResidualTransfer(10).fit(tasks, gamma_k0=0, selection_role="target_test")
    with pytest.raises(ValueError, match="K=0"):
        SupportAwareResidualTransfer(10).fit(tasks[1:], gamma_k0=0)
    with pytest.raises(ValueError, match="gamma_k0"):
        SupportAwareResidualTransfer(10).fit(tasks, gamma_k0=.3)
    with pytest.raises(ValueError, match="finite aligned"):
        SupportAwareResidualTransfer(10).fit([replace(tasks[0], query_memory=np.zeros(3))], gamma_k0=0)
    with pytest.raises(ValueError, match="finite aligned"):
        SupportAwareResidualTransfer(10).fit([replace(tasks[0], query_memory=np.full(2, np.nan))], gamma_k0=0)
    model = SupportAwareResidualTransfer(10).fit(tasks, gamma_k0=0)
    with pytest.raises(ValueError, match="disjoint"):
        adapt(model, replace(tasks[-1], support_cells=np.array([0, 1, 2, 3, 5])))
    with pytest.raises(ValueError, match="exactly K"):
        adapt(model, replace(tasks[-1], support_cells=np.array([0, 1, 2, 3, 10])))
    with pytest.raises(ValueError, match="K must"):
        model.selected_gamma(2)
    with pytest.raises(ValueError, match="K must"):
        model.selected_gamma(True)


@pytest.mark.parametrize("settings", [{"n_months": 0}, {"gamma_values": ()},
                                      {"gamma_values": (.25, 1)}, {"gamma_values": (0, np.nan)},
                                      {"ridge_strengths": (0,)}])
def test_invalid_configuration_rejected(settings):
    with pytest.raises(ValueError):
        SupportAwareResidualTransfer(**{"n_months": 10, **settings})
