"""Statistical and inference contracts for regularized station-level fusion."""

import json

import numpy as np
import pytest

from river_graph.models.regularized_station_fusion import RegularizedStationFusion


def sample():
    rng = np.random.default_rng(281)
    stations = np.repeat(np.array([f"s{i:02d}" for i in range(10)]), np.arange(2, 12))
    context = np.expm1(rng.uniform(0.2, 2.7, len(stations)))
    temporal = np.expm1(rng.uniform(0.3, 2.4, len(stations)))
    return context, temporal, stations


def fit(context, temporal, truth, stations, **kwargs):
    return RegularizedStationFusion(**kwargs).fit(
        context, temporal, truth, stations, selection_role="source_validation",
    )


def test_exact_context_candidate_can_win_and_predict_bitwise_identity():
    context, temporal, stations = sample()
    model = fit(context, temporal, context, stations)
    assert model.selected_["kind"] == "context"
    assert model.selected_["cv_mae"] == 0.0
    queries = np.array([0.0, 0.1, 123.456789])
    np.testing.assert_array_equal(model.predict(queries, np.ones(3)), queries)
    np.testing.assert_array_equal(model.predict_crossfit(context, temporal, stations), context)


def test_temporal_information_can_be_selected_without_affine_extrapolation():
    context, temporal, stations = sample()
    truth = np.expm1(0.5 * (np.log1p(context) + np.log1p(temporal)))
    model = fit(context, temporal, truth, stations)
    assert model.selected_["kind"] == "convex"
    assert model.selected_["weight"] == 0.5
    np.testing.assert_allclose(model.predict(context, temporal), truth, rtol=1e-13)


def test_station_blocks_and_cell_weighted_cv_are_consistent():
    context, temporal, stations = sample()
    truth = np.expm1(np.log1p(context) + 0.3)
    model = fit(context, temporal, truth, stations)
    held = [station for fold in model.folds_ for station in fold["held_stations"]]
    assert len(held) == len(set(held)) == len(set(stations))
    assert set(held) == set(stations)
    assert len(model.folds_) == 5
    expected = np.abs(model.predict_crossfit(context, temporal, stations) - truth).mean()
    assert model.selected_["cv_mae"] == pytest.approx(expected)
    for candidate in model.cv_results_:
        pooled = sum(row["n_cells"] * row["mae"] for row in candidate["fold_results"]) / len(truth)
        assert candidate["cv_mae"] == pytest.approx(pooled)


def test_ridge_strength_is_invariant_to_replicating_cells():
    context, temporal, stations = sample()
    truth = np.expm1(np.log1p(context) + 0.25)
    first = fit(context, temporal, truth, stations)
    repeat = fit(np.repeat(context, 3), np.repeat(temporal, 3), np.repeat(truth, 3), np.repeat(stations, 3))
    assert first.selected_["kind"] == repeat.selected_["kind"] == "ridge"
    assert first.selected_["strength"] == repeat.selected_["strength"]
    np.testing.assert_allclose(first.coefficients_, repeat.coefficients_, atol=1e-12)
    np.testing.assert_allclose([row["cv_mae"] for row in first.cv_results_],
                               [row["cv_mae"] for row in repeat.cv_results_], atol=1e-12)


def test_held_station_labels_cannot_affect_its_fixed_candidate_coefficients():
    context, temporal, stations = sample()
    truth = np.expm1(np.log1p(context) + 0.25)
    first = fit(context, temporal, truth, stations)
    fold = first.station_to_fold_["s00"]
    held_stations = first.folds_[fold]["held_stations"]
    changed = truth.copy()
    changed[np.isin(stations, held_stations)] += 100.0
    second = fit(context, temporal, changed, stations)
    # Hyperparameter selection can change; each fixed candidate's training fold cannot.
    for before, after in zip(first.cv_results_, second.cv_results_):
        assert before["name"] == after["name"]
        np.testing.assert_array_equal(before["fold_results"][fold]["coefficients"],
                                      after["fold_results"][fold]["coefficients"])


def test_crossfit_support_dates_use_correct_station_fold_and_reject_unknown_station():
    context, temporal, stations = sample()
    model = fit(context, temporal, np.expm1(np.log1p(context) + 0.25), stations)
    query_context = np.array([1.0, 2.0, 3.0])
    query_temporal = np.array([2.0, 1.0, 4.0])
    query_station = np.array(["s00", "s09", "s00"])
    prediction = model.predict_crossfit(query_context, query_temporal, query_station)
    for i, station in enumerate(query_station):
        coefficients = model.crossfit_coefficients_[model.station_to_fold_[station]]
        c, r = np.log1p(query_context[i]), np.log1p(query_temporal[i])
        expected = max(0.0, np.expm1(c + coefficients[0] + coefficients[1] * c + coefficients[2] * (r - c)))
        assert prediction[i] == pytest.approx(expected)
    with pytest.raises(ValueError, match="known source-validation"):
        model.predict_crossfit(query_context[:1], query_temporal[:1], ["never_seen"])


def test_regularization_of_every_coefficient_recovers_context_limit():
    context, temporal, stations = sample()
    truth = np.expm1(np.log1p(context) + 0.25)
    model = fit(context, temporal, truth, stations)
    low = next(row for row in model.cv_results_ if row["strength"] == 0.01)
    high = next(row for row in model.cv_results_ if row["strength"] == 1000.0)
    for low_fold, high_fold in zip(low["fold_results"], high["fold_results"]):
        assert np.linalg.norm(high_fold["coefficients"]) < np.linalg.norm(low_fold["coefficients"])
        assert abs(high_fold["coefficients"][0]) < 0.001


def test_json_roundtrip_preserves_query_and_crossfit_predictions_and_all_scores():
    context, temporal, stations = sample()
    model = fit(context, temporal, np.expm1(np.log1p(context) + 0.25), stations)
    saved = json.loads(json.dumps(model.to_dict(), allow_nan=False, sort_keys=True))
    restored = RegularizedStationFusion.from_dict(saved)
    assert restored.to_dict() == model.to_dict()
    np.testing.assert_array_equal(restored.predict(context, temporal), model.predict(context, temporal))
    np.testing.assert_array_equal(restored.predict_crossfit(context, temporal, stations),
                                  model.predict_crossfit(context, temporal, stations))
    assert len(model.cv_results_) == 27


def test_row_permutation_preserves_station_folds_and_prediction():
    context, temporal, stations = sample()
    truth = np.expm1(np.log1p(context) + 0.25)
    original = fit(context, temporal, truth, stations)
    order = np.random.default_rng(92).permutation(len(context))
    shuffled = fit(context[order], temporal[order], truth[order], stations[order])
    assert shuffled.station_to_fold_ == original.station_to_fold_
    assert shuffled.selected_["name"] == original.selected_["name"]
    np.testing.assert_allclose(shuffled.predict(context, temporal), original.predict(context, temporal), atol=1e-12)


def test_source_validation_role_and_station_population_are_required():
    context, temporal, stations = sample()
    with pytest.raises(ValueError, match="source_validation"):
        RegularizedStationFusion().fit(context, temporal, context, stations, selection_role="test")
    with pytest.raises(ValueError, match="distinct stations"):
        fit(context, temporal, context, np.repeat("only_one", len(context)))
    with pytest.raises(ValueError, match="without missing"):
        fit(context, temporal, context, np.full(len(context), np.nan))


@pytest.mark.parametrize("kwargs", [{"strengths": []}, {"strengths": [0]},
                                    {"strengths": [np.nan]}, {"convex_weights": [2]},
                                    {"convex_weights": []}, {"n_splits": 0}])
def test_invalid_search_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        RegularizedStationFusion(**kwargs)
