"""Ecological memory borrows source station errors without target labels."""

import json

import numpy as np
import pytest

import river_graph.models.ecological_residual_transfer as memory_module
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer


def example(*, slope=False):
    n, months = 12, 12
    group = np.array([-1] * 4 + [1] * 4 + [-1, 1, -1, 1])
    regime = np.zeros((n, 13))
    regime[:, 4:] = np.where(group[:, None] < 0, 2.0, 12.0)
    context = np.broadcast_to(np.linspace(5, 15, months), (n, months)).copy()
    u = np.log1p(context)
    u = (u - u[0].mean()) / u[0].std()
    residual = group[:, None] * (1.5 + (u if slope else 0))
    truth = context + residual
    source = np.arange(8 * months)
    validation = np.arange(8 * months, 10 * months)
    return regime, context, truth, source, validation


def fit_case(case, mode="ecological_bias", **kwargs):
    regime, context, truth, source, validation = case
    model = EcologicalResidualTransfer(mode, **kwargs).fit(
        regime, source, context.ravel()[source], truth.ravel()[source], n_months=context.shape[1],
        validation_cells=validation, validation_y=truth.ravel()[validation],
        validation_context=context.ravel()[validation], validation_temporal=context.ravel()[validation],
    )
    return model


def test_ecology_transfers_opposite_station_biases_without_target_labels():
    case = example()
    ecological = fit_case(case, k_grid=(4,))
    global_model = fit_case(case, mode="global_bias", k_grid=(4,))
    _, context, truth, _, _ = case
    assert ecological.selected_["gamma"] > 0
    assert ecological.selected_["validation_mae"] < .05
    assert global_model.selected_["validation_mae"] > 1
    test_pred = ecological.predict(context, context)[10:]
    assert np.abs(test_pred - truth[10:]).mean() < .05
    assert np.all(ecological.predict_delta(context)[10] < 0)
    assert np.all(ecological.predict_delta(context)[11] > 0)


def test_affine_memory_resolves_opposite_ecological_response_slopes():
    case = example(slope=True)
    affine = fit_case(case, mode="ecological_affine", k_grid=(4,))
    bias = fit_case(case, k_grid=(4,))
    _, context, truth, _, _ = case
    assert affine.selected_["validation_mae"] < .3 * bias.selected_["validation_mae"]
    assert affine.coefficients_[10, 1] < 0 < affine.coefficients_[11, 1]
    assert np.abs(affine.predict(context, context)[10:] - truth[10:]).mean() < .1


def test_each_station_has_equal_weight_despite_unequal_record_lengths():
    regime = np.full((4, 13), 5.0)
    months = 20
    source = np.r_[0, np.arange(months, 2 * months)]
    p = np.full(len(source), 6.0)
    y = p + np.r_[1.0, np.full(months, -1.0)]
    model = EcologicalResidualTransfer("global_bias").fit(
        regime, source, p, y, n_months=months, validation_cells=np.array([40]),
        validation_y=np.array([6.0]), validation_context=np.array([6.0]), validation_temporal=np.array([6.0]),
    )
    assert model.normalization_["residual_scale"] == pytest.approx(1.0)
    np.testing.assert_allclose(model.coefficients_[2:], 0.0, atol=1e-12)
    diagnostic = model.station_diagnostics_[2]
    assert diagnostic["donor_source_counts"] == [1, 20]
    assert diagnostic["donor_weights"] == [.5, .5]
    assert diagnostic["effective_count"] == 2


def test_source_only_ecology_and_response_scaling_handles_missing_sentinels():
    first_case = example(slope=True)
    regime, context, truth, source, validation = first_case
    regime[0, 4] = -1
    regime[1, 5] = np.nan
    regime[:8, 6] = -1  # An entirely missing source feature is inactive.
    first = fit_case(first_case, k_grid=(4,))
    changed = regime.copy()
    changed[8:, 4:] = 1e7
    changed[10, 6] = np.inf
    second = fit_case((changed, context, truth, source, validation), k_grid=(4,))
    assert first.normalization_ == second.normalization_
    assert first.ecology_scaler_ == second.ecology_scaler_
    assert first.ecology_scaler_["active"][2] is False
    assert first.ecology_scaler_["median"][2] == 0
    assert first.ecology_scaler_["iqr"][2] == 1
    expected_median = np.median(regime[1:8, 4])
    assert first.ecology_scaler_["median"][0] == expected_median
    # The first four hydraulic columns never enter the ecological similarity.
    ignored = regime.copy()
    ignored[:, :4] = np.inf
    third = fit_case((ignored, context, truth, source, validation), k_grid=(4,))
    assert first.to_dict() == third.to_dict()


def test_source_self_exclusion_and_missing_target_global_fallback():
    case = example(slope=True)
    case[0][10, 4:] = -1
    case[0][11, 4:9] = np.nan  # Four valid features also require fallback.
    local = fit_case(case, mode="ecological_affine", k_grid=(4,), ridge_grid=(.1,))
    global_model = fit_case(case, mode="global_affine", k_grid=(4,), ridge_grid=(.1,))
    for station in range(8):
        diagnostic = local.station_diagnostics_[station]
        assert station not in diagnostic["donor_ids"]
        assert diagnostic["source_self_excluded"] is True
    for station in (10, 11):
        diagnostic = local.station_diagnostics_[station]
        assert diagnostic["fallback"] == "target_insufficient_ecology"
        assert diagnostic["donor_count"] == 8
        assert diagnostic["nearest_distance"] is None
        np.testing.assert_array_equal(local.coefficients_[station], global_model.coefficients_[station])


def test_absent_eligible_ecological_donors_fall_back_to_global():
    case = example()
    case[0][:8, 4:] = -1
    model = fit_case(case, k_grid=(4,))
    assert model.station_diagnostics_[10]["fallback"] == "no_eligible_source_donors"
    assert model.station_diagnostics_[10]["donor_count"] == 8


def test_zero_blend_exact_identity_and_ties_prefer_more_smoothing():
    regime, context, truth, source, validation = example()
    temporal = context.copy()
    temporal.ravel()[validation] = truth.ravel()[validation]
    model = EcologicalResidualTransfer(k_grid=(2, 4), ridge_grid=(.1, 1)).fit(
        regime, source, context.ravel()[source], truth.ravel()[source], n_months=context.shape[1],
        validation_cells=validation, validation_y=truth.ravel()[validation],
        validation_context=context.ravel()[validation], validation_temporal=temporal.ravel()[validation],
    )
    assert model.selected_ == {"k": 4, "ridge": 1.0, "gamma": 0.0, "validation_mae": 0.0}
    assert model.selected_status_ == "prior_temporal_fallback"
    np.testing.assert_array_equal(model.predict(context, temporal), temporal)
    assert model.predict(context, temporal) is not temporal
    assert len(model.validation_trace_) == 16


def test_serialization_preserves_every_prediction_and_omits_label_bank():
    case = example(slope=True)
    model = fit_case(case, mode="ecological_affine", k_grid=(4,))
    saved = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = EcologicalResidualTransfer.from_dict(saved)
    assert restored.to_dict() == model.to_dict()
    assert not {"source_y", "validation_y", "source_oof_native"}.intersection(saved)
    context = case[1]
    np.testing.assert_array_equal(restored.predict_delta(context), model.predict_delta(context))
    np.testing.assert_array_equal(restored.predict(context, context), model.predict(context, context))
    for diagnostic in saved["station_diagnostics"]:
        assert sum(diagnostic["donor_weights"]) == pytest.approx(1.0)
        assert diagnostic["max_weight"] == pytest.approx(1.0 / diagnostic["donor_count"])


def test_permuted_selected_vectors_give_identical_fitted_memory():
    case = example(slope=True)
    baseline = fit_case(case, mode="ecological_affine", k_grid=(4,))
    regime, context, truth, source, validation = case
    changed = fit_case((regime, context, truth, source[::-1], validation[::-1]),
                       mode="ecological_affine", k_grid=(4,))
    assert changed.to_dict() == baseline.to_dict()


def test_only_selected_source_and_validation_labels_are_used():
    case = example(slope=True)
    first = fit_case(case, mode="ecological_affine", k_grid=(4,))
    case[2][10:] = np.nan  # Neither query labels nor a full label matrix is passed.
    second = fit_case(case, mode="ecological_affine", k_grid=(4,))
    assert first.to_dict() == second.to_dict()


def test_line_search_failure_retries_from_zero_and_is_explicitly_recorded(monkeypatch):
    original = memory_module.minimize
    calls = []

    def failed_first(objective, initial, **kwargs):
        calls.append((initial.copy(), kwargs["options"].copy()))
        result = original(objective, initial, **kwargs)
        if len(calls) == 1:
            result.success = False
            result.message = "ABNORMAL: simulated line-search precision termination"
        return result

    monkeypatch.setattr(memory_module, "minimize", failed_first)
    result, attempts = memory_module._solve_profile(lambda x: (float((x[0] - 1)**2), 2 * (x - 1)), 1)
    assert result.success
    assert len(attempts) == 2
    assert attempts[0]["success"] is False
    assert attempts[1]["success"] is True
    assert calls[1][1]["maxls"] > calls[0][1]["maxls"]
    np.testing.assert_array_equal(calls[0][0], np.zeros(1))
    np.testing.assert_array_equal(calls[1][0], np.zeros(1))


def test_failed_retry_never_silently_returns_a_coefficient(monkeypatch):
    original = memory_module.minimize

    def unsuccessful(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success = False
        result.message = "forced failure"
        return result

    monkeypatch.setattr(memory_module, "minimize", unsuccessful)
    with pytest.raises(RuntimeError, match="failed after deterministic retry"):
        memory_module._solve_profile(lambda x: (float((x[0] - 1)**2), 2 * (x - 1)), 1)


def test_role_alignment_station_overlap_and_prediction_shape_guards():
    regime, context, truth, source, validation = example()
    common = {"n_months": context.shape[1], "validation_cells": validation,
              "validation_y": truth.ravel()[validation], "validation_context": context.ravel()[validation],
              "validation_temporal": context.ravel()[validation]}
    with pytest.raises(ValueError, match="source_validation"):
        EcologicalResidualTransfer().fit(regime, source, context.ravel()[source], truth.ravel()[source],
                                        selection_role="test", **common)
    with pytest.raises(ValueError, match="align"):
        EcologicalResidualTransfer().fit(regime, source, context.ravel()[source][:-1], truth.ravel()[source], **common)
    with pytest.raises(ValueError, match="disjoint"):
        EcologicalResidualTransfer().fit(regime, source, context.ravel()[source], truth.ravel()[source],
                                        **{**common, "validation_cells": source[:len(validation)]})
    with pytest.raises(ValueError, match="unique"):
        EcologicalResidualTransfer().fit(regime, np.zeros_like(source), context.ravel()[source],
                                        truth.ravel()[source], **common)
    fitted = fit_case(example(), k_grid=(4,))
    with pytest.raises(ValueError, match="full station-month"):
        fitted.predict_delta(context[:1])


@pytest.mark.parametrize("kwargs", [{"mode": "unknown"}, {"k_grid": (0,)}, {"k_grid": (2.0,)},
                                    {"ridge_grid": (0,)}, {"gamma_grid": (.25, 1)}, {"gamma_grid": (0, 2)}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        EcologicalResidualTransfer(**kwargs)
