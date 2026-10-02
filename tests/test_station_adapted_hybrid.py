"""Contracts for source-fitted fusion and support-only station adaptation."""

import json

import numpy as np
import pytest

from river_graph.models.station_adapted_hybrid import (
    CalibrationEpisode,
    StationAdaptedHybrid,
    station_residual_correction,
)


def test_station_correction_uses_residuals_at_matching_support_months():
    corrected = station_residual_correction(
        np.array([3.0, 7.0, 4.0]), np.array([1, 2, 11]),
        np.array([1.0, 3.0, 9.0]), np.array([0, 3, 10]),
        np.array([3.0, 7.0, 4.0]), n_months=10, alpha=1.0,
    )
    # Station 0 has a factor-two support discrepancy in (1+y); station 1 has 1/2.
    np.testing.assert_allclose(corrected, [7.0, 15.0, 1.5])


def test_support_alignment_is_equivariant_to_joint_permutation():
    args = [np.array([3.0, 7.0, 4.0]), np.array([1, 2, 11])]
    support_prediction = np.array([1.0, 3.0, 9.0])
    support_cells = np.array([0, 3, 10])
    support_values = np.array([3.0, 7.0, 4.0])
    first = station_residual_correction(*args, support_prediction, support_cells,
                                       support_values, n_months=10, alpha=0.5)
    permutation = np.array([2, 0, 1])
    second = station_residual_correction(*args, support_prediction[permutation],
                                        support_cells[permutation], support_values[permutation],
                                        n_months=10, alpha=0.5)
    np.testing.assert_array_equal(first, second)


def test_query_label_perturbation_cannot_change_correction():
    full_labels = np.arange(20.0)
    support = np.array([0, 10])
    query = np.array([1, 2, 11, 12])
    changed_labels = full_labels.copy()
    changed_labels[query] = np.nan
    first = station_residual_correction(np.ones(4), query, np.ones(2), support,
                                       full_labels[support], n_months=10, alpha=0.75)
    second = station_residual_correction(np.ones(4), query, np.ones(2), support,
                                        changed_labels[support], n_months=10, alpha=0.75)
    np.testing.assert_array_equal(first, second)


def test_k_zero_and_zero_shrinkage_are_exact_identity():
    prediction = np.array([0.0, 0.1, 123.456789], dtype=np.float64)
    query = np.array([1, 2, 11])
    empty = np.array([], dtype=np.int64)
    model = StationAdaptedHybrid(n_months=10)
    out = model.adapt(prediction, query, np.array([]), empty, np.array([]), k=0, arm="context")
    np.testing.assert_array_equal(prediction, out)
    assert not np.shares_memory(prediction, out)
    out = station_residual_correction(prediction, query, np.array([1.0]),
                                     np.array([0]), np.array([3.0]), n_months=10, alpha=0.0)
    np.testing.assert_array_equal(prediction, out)


@pytest.mark.parametrize("bad", ["overlap", "duplicate", "misaligned", "negative", "fractional"])
def test_bad_support_inputs_are_rejected(bad):
    query = np.array([1, 2])
    support = np.array([0, 3])
    labels = np.array([2.0, 3.0])
    if bad == "overlap":
        support = np.array([0, 1])
    elif bad == "duplicate":
        support = np.array([0, 0])
    elif bad == "misaligned":
        labels = np.array([2.0])
    elif bad == "negative":
        labels = np.array([-2.0, 3.0])
    elif bad == "fractional":
        support = np.array([0.1, 3.1])
    with pytest.raises(ValueError):
        station_residual_correction(np.ones(2), query, np.ones(2), support, labels,
                                    n_months=10, alpha=0.5)


def test_fusion_can_use_temporal_expert_and_recovers_log_affine_relation():
    context = np.array([0.5, 2.0, 3.0, 5.0, 9.0, 11.0])
    temporal = np.array([6.0, 3.0, 9.0, 1.0, 7.0, 2.0])
    truth = np.expm1(0.1 + 0.35 * np.log1p(context) + 0.65 * np.log1p(temporal))
    model = StationAdaptedHybrid(n_months=10).fit_fusion(
        context, temporal, truth, selection_role="source_validation",
    )
    assert model.fusion_["name"] == "log_affine"
    np.testing.assert_allclose(model.fusion_["coefficients"], [0.1, 0.35, 0.65], atol=1e-12)
    np.testing.assert_allclose(model.predict_components(context, temporal)["hybrid"], truth)
    assert model.fusion_["coefficients"][2] != 0.0


def test_context_selection_is_reported_and_preserves_identity():
    context = np.array([0.1, 2.3, 7.8])
    temporal = np.array([5.0, 0.0, 3.0])
    model = StationAdaptedHybrid(n_months=10).fit_fusion(
        context, temporal, context, selection_role="source_validation",
    )
    assert model.fusion_["name"] == "context"
    np.testing.assert_array_equal(model.predict_components(context, temporal)["hybrid"], context)


def test_geometric_selection_available_without_affine():
    context = np.array([1.0, 7.0, 3.0])
    temporal = np.array([7.0, 1.0, 15.0])
    truth = np.expm1(0.5 * (np.log1p(context) + np.log1p(temporal)))
    model = StationAdaptedHybrid(n_months=10, blend_weights=[0.0, 0.5, 1.0], allow_affine=False)
    model.fit_fusion(context, temporal, truth, selection_role="source_validation")
    assert model.fusion_["name"] == "geometric"
    assert model.fusion_["weight"] == 0.5


def _source_episode():
    return CalibrationEpisode(
        k=1, query_cells=np.array([1, 2, 11, 12]), query_values=np.array([3.0, 7.0, 3.0, 7.0]),
        context_query_prediction=np.array([1.0, 3.0, 1.0, 3.0]),
        temporal_query_prediction=np.array([1.0, 3.0, 1.0, 3.0]),
        support_cells=np.array([0, 10]), support_values=np.array([3.0, 3.0]),
        context_support_prediction=np.array([1.0, 1.0]),
        temporal_support_prediction=np.array([1.0, 1.0]),
    )


def test_source_validation_alone_selects_shrinkage():
    model = StationAdaptedHybrid(n_months=10, allow_affine=False).fit_fusion(
        np.array([1.0, 3.0]), np.array([2.0, 8.0]), np.array([1.0, 3.0]),
        selection_role="source_validation",
    )
    episode = _source_episode()
    model.fit_calibration([episode], selection_role="source_validation")
    assert model.alpha_by_arm_ == {"context": {0: 0.0, 1: 1.0}, "hybrid": {0: 0.0, 1: 1.0}}
    selected_before = model.alpha_by_arm_.copy()
    corrected = model.adapt(episode.context_query_prediction, episode.query_cells,
                            episode.context_support_prediction, episode.support_cells,
                            episode.support_values, k=1, arm="context")
    np.testing.assert_allclose(corrected, episode.query_values)
    # Prediction at a new target changes neither fusion nor source-selected alpha.
    model.predict_components(np.array([100.0]), np.array([0.0]))
    assert model.alpha_by_arm_ == selected_before
    assert model.fusion_selection_role_ == model.calibration_selection_role_ == "source_validation"
    assert {key for key in model.calibration_scores_[0] if key.endswith("_mae")} == {
        "context_mae", "hybrid_mae",
    }


def test_selection_rejects_outer_test_roles_and_refit_invalidates_calibration():
    model = StationAdaptedHybrid(n_months=10)
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="source_validation"):
        model.fit_fusion(values, values, values, selection_role="test")
    model.fit_fusion(values, values, values, selection_role="source_validation")
    with pytest.raises(ValueError, match="source_validation"):
        model.fit_calibration([_source_episode()], selection_role="outer_test")
    model.fit_calibration([_source_episode()], selection_role="source_validation")
    model.fit_fusion(values, values, values, selection_role="source_validation")
    assert model.alpha_by_arm_ == {"context": {0: 0.0}, "hybrid": {0: 0.0}}
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.adapt(values, np.array([1, 11]), values, np.array([0, 10]), values, k=1, arm="context")


def test_exact_support_count_required_for_every_query_station():
    model = StationAdaptedHybrid(n_months=10)
    with pytest.raises(ValueError, match="exactly K"):
        model.adapt(np.ones(2), np.array([1, 11]), np.ones(1), np.array([0]), np.ones(1), k=1, arm="context")
    with pytest.raises(ValueError, match="empty support"):
        model.adapt(np.ones(2), np.array([1, 11]), np.ones(2), np.array([0, 10]), np.ones(2), k=0, arm="context")


def test_each_arm_gets_its_own_best_shrinkage_on_the_same_validation_grid():
    model = StationAdaptedHybrid(n_months=10, allow_affine=False).fit_fusion(
        np.array([5.0, 0.0]), np.array([1.0, 3.0]), np.array([1.0, 3.0]),
        selection_role="source_validation",
    )
    episode = CalibrationEpisode(
        k=1, query_cells=np.array([1, 11]), query_values=np.array([3.0, 3.0]),
        context_query_prediction=np.array([1.0, 1.0]),
        temporal_query_prediction=np.array([3.0, 3.0]),
        support_cells=np.array([0, 10]), support_values=np.array([3.0, 3.0]),
        context_support_prediction=np.array([1.0, 1.0]),
        temporal_support_prediction=np.array([1.0, 1.0]),
    )
    model.fit_calibration([episode], selection_role="source_validation")
    assert model.alpha_by_arm_["context"][1] == 1.0
    assert model.alpha_by_arm_["hybrid"][1] == 0.0
    assert len(model.calibration_scores_) == 5


def test_json_round_trip_preserves_predictions_parameters_and_candidate_scores():
    model = StationAdaptedHybrid(n_months=10).fit_fusion(
        np.array([1.0, 5.0, 7.0]), np.array([5.0, 1.0, 3.0]), np.array([2.0, 2.0, 4.0]),
        selection_role="source_validation",
    )
    episode = _source_episode()
    model.fit_calibration([episode], selection_role="source_validation")
    saved = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = StationAdaptedHybrid.from_dict(saved)
    assert restored.to_dict() == model.to_dict()
    inputs = (episode.context_query_prediction, episode.temporal_query_prediction)
    for arm, prediction in model.predict_components(*inputs).items():
        np.testing.assert_array_equal(restored.predict_components(*inputs)[arm], prediction)
    correction_args = (episode.context_query_prediction, episode.query_cells,
                       episode.context_support_prediction, episode.support_cells, episode.support_values)
    np.testing.assert_array_equal(restored.adapt(*correction_args, k=1, arm="context"),
                                  model.adapt(*correction_args, k=1, arm="context"))
