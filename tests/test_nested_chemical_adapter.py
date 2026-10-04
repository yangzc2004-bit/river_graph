"""Nested chemical shape keeps the complete parent and uses support labels only."""
from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from river_graph.models.nested_chemical_adapter import (
    NestedChemicalAdapter,
    NestedChemicalEpisode,
)


def source_coordinates():
    return np.array([[-2., -1.], [-1., 1.], [0., -1.], [1., 1.], [2., 0.]])


def episode(k, station=0):
    x = np.array([-1., 0., 1.]) if k == 3 else np.array([-1., -.5, 0., .5, 1.])
    if k in (0, 1):
        x = np.zeros(k)
    support_chemical = np.column_stack((x, x ** 2))
    query_chemical = np.array([[.3, .09], [1.5, 2.25]])
    support_prediction = np.full(k, 3.)
    return NestedChemicalEpisode(k=k, query_cells=np.array([8, 9]) + 20 * station,
        query_values=np.expm1(np.log(4) + .35 * query_chemical[:, 0]),
        query_prediction=np.full(2, 3.), support_cells=np.arange(k) + 20 * station,
        support_values=np.expm1(np.log(4) + .35 * x), support_prediction=support_prediction,
        query_chemical=query_chemical, support_chemical=support_chemical,
        query_active=np.ones(2, bool), support_active=np.ones(k, bool))


def tasks(n_stations=3):
    return [episode(k, station) for station in range(n_stations) for k in (3, 5)]


def fitted(episodes=None, **kwargs):
    return NestedChemicalAdapter(20, **kwargs).fit_coordinates(
        source_coordinates(), np.ones(5, bool), source_role="source_training").fit(
        tasks() if episodes is None else episodes, selection_role="source_validation")


def adapt(model, task, *, components=False, **kwargs):
    method = model.adapt_components if components else model.adapt
    return method(task.query_prediction, task.query_cells, task.support_prediction,
        task.support_cells, task.support_values, query_chemical=task.query_chemical,
        support_chemical=task.support_chemical, query_active=task.query_active,
        support_active=task.support_active, k=task.k, **kwargs)


def independent_prediction(model, task, *, ridge, strength):
    qc = (task.query_chemical - model.coordinate_mean_) / model.coordinate_scale_
    sc = (task.support_chemical - model.coordinate_mean_) / model.coordinate_scale_
    active = task.support_active
    centered = sc[active] - sc[active].mean(0)
    residual = np.log1p(task.support_values[active]) - np.log1p(task.support_prediction[active])
    rhs = centered.T @ (residual - residual.mean()) / active.sum()
    coefficient = np.linalg.solve(centered.T @ centered / active.sum() + ridge * np.eye(2), rhs)
    delta = strength * ((qc - sc[active].mean(0)) @ coefficient)
    delta[~task.query_active] = 0
    prediction = task.query_prediction.copy()
    modified = delta != 0
    prediction[modified] = np.maximum(np.expm1(np.log1p(prediction[modified]) + delta[modified]), 0)
    return prediction, delta, coefficient


def test_nested_increment_matches_independent_centered_ridge_calculation():
    model = fitted()
    task = episode(5)
    result = adapt(model, task, components=True, ridge_strength=1., strength=.5)
    prediction, delta, coefficient = independent_prediction(model, task, ridge=1., strength=.5)
    np.testing.assert_allclose(result["y_pred"], prediction, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result["chemical_delta"], delta, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["chemical_coefficients"], np.tile(coefficient, (2, 1)), atol=1e-15)
    np.testing.assert_array_equal(result["chemical_support_count"], [5, 5])
    # The new operator fits the residual remaining AFTER support calibration,
    # rather than using the uncalibrated original head as its response.
    exact_parent = replace(task, support_prediction=task.support_values.copy())
    np.testing.assert_array_equal(adapt(model, exact_parent), task.query_prediction)
    # A residual constant only shifts level; it cannot train a chemical shape.
    constant_residual = replace(task, support_values=2 * (task.support_prediction + 1) - 1)
    np.testing.assert_array_equal(adapt(model, constant_residual), task.query_prediction)


def test_k0_k1_and_zero_strength_preserve_completed_parent_bits_and_dtype():
    model = fitted()
    for k in (0, 1):
        task = replace(episode(k), query_prediction=np.array([.1, 10.25], np.float32))
        actual = adapt(model, task)
        np.testing.assert_array_equal(actual, task.query_prediction)
        assert actual.dtype == task.query_prediction.dtype
        assert not np.shares_memory(actual, task.query_prediction)
        result = adapt(model, task, components=True)
        np.testing.assert_array_equal(result["chemical_support_count"], [k, k])
    for k in (3, 5):
        task = replace(episode(k), query_prediction=np.array([.1, 10.25], np.float32))
        result = adapt(model, task, components=True, ridge_strength=.1, strength=0)
        np.testing.assert_array_equal(result["y_pred"], task.query_prediction)
        assert result["y_pred"].dtype == task.query_prediction.dtype
        np.testing.assert_array_equal(result["chemical_delta"], [0., 0.])
        np.testing.assert_array_equal(result["chemical_support_count"], [k, k])


def test_missing_chemistry_fallback_and_placeholders_are_exact():
    model = fitted()
    task = replace(episode(5), query_active=np.array([False, True]),
                   support_active=np.array([False, True, True, True, True]))
    actual = adapt(model, task)
    assert actual[0] == task.query_prediction[0]
    qc, sc = task.query_chemical.copy(), task.support_chemical.copy()
    qc[~task.query_active] = np.nan
    sc[~task.support_active] = np.nan
    np.testing.assert_array_equal(adapt(model, replace(task, query_chemical=qc, support_chemical=sc)), actual)
    one_active = replace(task, support_active=np.array([True, False, False, False, False]))
    np.testing.assert_array_equal(adapt(model, one_active), task.query_prediction)
    np.testing.assert_array_equal(adapt(model, one_active, components=True)["chemical_support_count"], [1, 1])
    inactive = replace(task, query_active=np.zeros(2, bool))
    np.testing.assert_array_equal(adapt(model, inactive), task.query_prediction)


def test_source_only_normalization_and_degenerate_coordinates():
    coordinates = np.vstack((source_coordinates(), [np.nan, np.nan]))
    active = np.array([True] * 5 + [False])
    model = NestedChemicalAdapter(20).fit_coordinates(coordinates, active, source_role="source_training")
    np.testing.assert_array_equal(model.coordinate_mean_, source_coordinates().mean(0))
    np.testing.assert_array_equal(model.coordinate_std_, source_coordinates().std(0))
    original_mean, original_scale = model.coordinate_mean_.copy(), model.coordinate_scale_.copy()
    model.fit(tasks(), selection_role="source_validation")
    altered = replace(episode(5), query_chemical=episode(5).query_chemical + 100)
    adapt(model, altered)
    np.testing.assert_array_equal(model.coordinate_mean_, original_mean)
    np.testing.assert_array_equal(model.coordinate_scale_, original_scale)
    degenerate = NestedChemicalAdapter(20).fit_coordinates(np.ones((5, 2)), np.ones(5, bool),
        source_role="source_training").fit(tasks(), selection_role="source_validation")
    assert not degenerate.identified_.any()
    assert degenerate.selection_ == {"strength": 0., "ridge_strength": 100.}
    np.testing.assert_array_equal(adapt(degenerate, episode(5)), episode(5).query_prediction)


def test_support_and_row_chemistry_matter_but_query_doc_is_not_an_input():
    model = fitted()
    task = episode(5)
    original = adapt(model, task)
    assert not np.array_equal(original, task.query_prediction)
    changed = replace(task, support_values=task.support_values[::-1])
    assert not np.allclose(adapt(model, changed), original)
    np.testing.assert_array_equal(adapt(model, replace(task, query_values=np.full(2, np.nan))), original)
    # A later query row is not used to center an earlier row or the station.
    coordinates = task.query_chemical.copy()
    coordinates[1] += 5
    result = adapt(model, replace(task, query_chemical=coordinates))
    assert result[0] == original[0]
    assert result[1] != original[1]
    assert "query_values" not in inspect.signature(model.adapt).parameters
    assert "query_values" not in inspect.signature(model.adapt_components).parameters
    assert "query_values" not in inspect.signature(model.fit_coordinates).parameters


def test_shared_rule_uses_k_equal_then_station_equal_scores():
    validation = tasks(2)
    # The two budgets favor different corrections; the operator must make one
    # shared choice rather than winning by choosing a separate K5 fallback.
    validation = [replace(task, query_values=task.query_prediction.copy()) if task.k == 5 else task
                  for task in validation]
    model = fitted(validation)
    state = model.to_dict()
    assert "selection_by_k" not in state
    for score in state["selection_scores"]:
        by_k = {}
        for k in (3, 5):
            by_station = []
            for task in validation:
                if task.k != k:
                    continue
                prediction = adapt(model, task, components=True, ridge_strength=score["ridge_strength"],
                    strength=score["strength"])["y_pred"]
                by_station.append(float(np.abs(prediction - task.query_values).mean()))
            by_k[str(k)] = float(np.mean(by_station))
        assert score["mae_by_k"] == by_k
        assert score["mae"] == np.mean(list(by_k.values()))
    winner = min(state["selection_scores"], key=lambda row: (row["mae"], row["strength"], -row["ridge_strength"]))
    assert state["selection"] == {key: winner[key] for key in ("strength", "ridge_strength")}


def test_station_equal_selection_is_not_dominated_by_repeated_query_rows():
    validation = tasks(2)
    # Repeat station0 rows through multiple episodes. Its station-average loss
    # remains the same, so it cannot acquire extra selection weight.
    balanced = fitted(validation)
    repeated = fitted([*validation, *[task for _ in range(10) for task in validation if task.query_cells[0] < 20]])
    assert balanced.selection_ == repeated.selection_
    for left, right in zip(balanced.selection_scores_, repeated.selection_scores_, strict=True):
        np.testing.assert_allclose(left["mae"], right["mae"], rtol=0, atol=1e-14)
        assert right["n_query_occurrences_by_k"]["3"] > left["n_query_occurrences_by_k"]["3"]


def test_fold_selection_excludes_whole_station_and_replays_fixed_choices():
    validation = tasks(7)
    model = fitted(validation, fold_seed=4342)
    assert len(model.fold_diagnostics_) == 5
    all_held = []
    for fold in model.fold_diagnostics_:
        assert not set(fold["held_stations"]) & set(fold["selection_stations"])
        assert sorted(fold["held_stations"] + fold["selection_stations"]) == list(range(7))
        assert fold["selected_zero"] == (fold["choice"]["strength"] == 0)
        all_held.extend(fold["held_stations"])
        by_k = {}
        for k in (3, 5):
            values = [float(np.abs(adapt(model, task, components=True, **fold["choice"])["y_pred"]
                                   - task.query_values).mean())
                      for task in validation if task.k == k
                      and task.query_cells[0] // 20 in fold["held_stations"]]
            by_k[str(k)] = float(np.mean(values))
        assert fold["held_mae_by_k"] == by_k
        assert fold["held_mae"] == np.mean(list(by_k.values()))
        # Altering one held station's query truth cannot change the rule
        # selected by that fold, while it can change its evaluation score.
        changed = [replace(task, query_values=task.query_values + 100)
                   if task.query_cells[0] // 20 in fold["held_stations"] else task for task in validation]
        other = fitted(changed, fold_seed=4342).fold_diagnostics_[fold["fold"]]
        assert other["choice"] == fold["choice"]
        assert other["held_mae"] != fold["held_mae"]
    assert sorted(all_held) == list(range(7))
    assert "not fully OOF" in model.to_dict()["definition"]["fold_interpretation"]
    repeated = fitted(validation, fold_seed=4342)
    assert repeated.fold_diagnostics_ == model.fold_diagnostics_


def test_json_roundtrip_preserves_state_predictions_and_fold_override():
    model = fitted(fold_seed=77)
    state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = NestedChemicalAdapter.from_dict(state)
    assert restored.to_dict() == state
    for task in [*tasks(), episode(0), episode(1)]:
        np.testing.assert_array_equal(adapt(model, task), adapt(restored, task))
    for fold in state["fold_diagnostics"]:
        np.testing.assert_array_equal(adapt(model, episode(5), components=True, **fold["choice"])["y_pred"],
            adapt(restored, episode(5), components=True, **fold["choice"])["y_pred"])
    invalid = copy.deepcopy(state)
    invalid["selection"]["strength"] = .123
    with pytest.raises(ValueError, match="shared candidate winner"):
        NestedChemicalAdapter.from_dict(invalid)
    invalid = copy.deepcopy(state)
    invalid["coordinate_scale"][0] = 0
    with pytest.raises(ValueError, match="coordinate scale"):
        NestedChemicalAdapter.from_dict(invalid)


def test_role_alignment_selection_and_override_errors_are_explicit():
    with pytest.raises(ValueError, match="source_training"):
        NestedChemicalAdapter(20).fit_coordinates(source_coordinates(), np.ones(5, bool), source_role="test")
    model = fitted()
    with pytest.raises(ValueError, match="source_validation"):
        model.fit(tasks(), selection_role="target_test")
    with pytest.raises(ValueError, match="both K3 and K5"):
        model.fit([episode(3)], selection_role="source_validation")
    with pytest.raises(ValueError, match="query cells must be fixed"):
        model.fit([episode(3), replace(episode(5), query_cells=np.array([9, 10]))], selection_role="source_validation")
    with pytest.raises(ValueError, match="exactly K"):
        adapt(model, replace(episode(5), support_cells=np.array([0, 1, 2, 3, 20])))
    with pytest.raises(ValueError, match="disjoint"):
        adapt(model, replace(episode(5), support_cells=np.array([0, 1, 2, 3, 8])))
    with pytest.raises(ValueError, match="K must"):
        adapt(model, replace(episode(5), k=True))
    with pytest.raises(ValueError, match="finite active"):
        adapt(model, replace(episode(5), query_chemical=np.full((2, 2), np.nan)))
    with pytest.raises(ValueError, match="both ridge_strength"):
        adapt(model, episode(5), components=True, strength=0)
    with pytest.raises(ValueError, match="fixed configured"):
        adapt(model, episode(5), components=True, strength=.1, ridge_strength=.1)
    with pytest.raises(RuntimeError, match="source coordinates"):
        NestedChemicalAdapter(20).fit(tasks(), selection_role="source_validation")


@pytest.mark.parametrize("settings", [{"n_months": 0}, {"ridge_values": (0,)},
    {"ridge_values": (np.inf,)}, {"strength_values": (.5, 1)}, {"strength_values": (0, np.nan)},
    {"selection_folds": 1}, {"fold_seed": -1}, {"scale_floor": 0}])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        NestedChemicalAdapter(**{"n_months": 20, **settings})
