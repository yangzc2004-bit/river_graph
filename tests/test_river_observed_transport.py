import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_observed_transport import (
    OPERATORS,
    calibrated_prediction,
    concentration_proxy,
    fit_calibrator,
    nested_predictions,
    receiver_weights,
)


def example():
    rng = np.random.default_rng(4)
    rows = []
    for system in range(5):
        for month in range(8):
            a, b, ap, bp = rng.uniform(1, 8, 4)
            rows.append({"component": system, "target": f"r{system}", "pair_id": f"p{system}",
                         "month_index": month, "doc_a_now": a, "doc_b_now": b,
                         "doc_a_previous": ap, "doc_b_previous": bp, "weight_a": .3,
                         "path_a_km": 20., "path_b_km": 70., "month_sin": np.sin(month),
                         "month_cos": np.cos(month), "year_fraction": 2000.+month/12,
                         "log_basin_area": np.log1p(100+system), "log_discharge": np.log1p(3+month),
                         "temperature": 5.+system+month, "y_true": .3*a+.7*b+.2*system})
    return pd.DataFrame(rows)


def test_same_sources_same_budget_and_zero_delay():
    f = example()
    now = .3*f.doc_a_now+.7*f.doc_b_now
    for operator in OPERATORS[1:]:
        np.testing.assert_array_equal(concentration_proxy(f, operator, 0., 100.), now)
    constant = f.copy()
    constant[["doc_a_now", "doc_b_now", "doc_a_previous", "doc_b_previous"]] = 5.
    for operator in OPERATORS[1:]:
        np.testing.assert_allclose(concentration_proxy(constant, operator, .5, 100.), 5.)


def test_equal_paths_remove_branch_specific_difference():
    f = example()
    f.path_b_km = f.path_a_km
    np.testing.assert_allclose(concentration_proxy(f, "branch_arrival", .5, 100.),
                               concentration_proxy(f, "mean_delay", .5, 100.), atol=1e-14)


def test_branch_difference_requires_path_imbalance_and_different_source_changes():
    f = example()
    w = f.weight_a.to_numpy()
    expected = w*(1-w)*.5*(f.path_a_km-f.path_b_km).to_numpy()/100.*(
        (f.doc_a_previous-f.doc_a_now)-(f.doc_b_previous-f.doc_b_now)).to_numpy()
    difference = concentration_proxy(f, "branch_arrival", .5, 100.)-concentration_proxy(f, "mean_delay", .5, 100.)
    np.testing.assert_allclose(difference, expected, atol=1e-14)
    f.doc_b_previous = f.doc_b_now+(f.doc_a_previous-f.doc_a_now)
    np.testing.assert_allclose(concentration_proxy(f, "branch_arrival", .5, 100.),
                               concentration_proxy(f, "mean_delay", .5, 100.), atol=1e-14)


def test_query_truth_and_future_values_are_not_proxy_inputs():
    f = example()
    changed = f.copy()
    changed.y_true = 10000.
    changed["doc_a_future"] = 1e8
    changed["doc_b_future"] = -1e8
    for operator in OPERATORS[1:]:
        np.testing.assert_array_equal(concentration_proxy(f, operator, .5, 100.),
                                      concentration_proxy(changed, operator, .5, 100.))


def test_receiver_equal_weight_handles_different_month_and_connection_counts():
    f = example()
    extra = f[f.target.eq("r0")].iloc[:3].copy()
    extra.pair_id = "second_pair"
    f = pd.concat([f, extra], ignore_index=True)
    totals = pd.Series(receiver_weights(f)).groupby(f.target).sum()
    np.testing.assert_allclose(totals, 1.)


def test_saved_source_preprocessing_and_coefficients_do_not_refit_on_query():
    f = example()
    train, query = f[f.component.ne(0)].copy(), f[f.component.eq(0)].copy()
    train.loc[train.index[:3], "temperature"] = np.nan
    query.temperature = np.nan
    state = fit_calibrator(train, concentration_proxy(train, "same_month", 0, 100.))
    prediction = calibrated_prediction(query, concentration_proxy(query, "same_month", 0, 100.), state)
    extra = query.copy()
    extra.temperature = 100.
    pooled = pd.concat([query, extra], ignore_index=True)
    np.testing.assert_array_equal(prediction, calibrated_prediction(pooled, concentration_proxy(pooled, "same_month", 0, 100.), state)[:len(query)])
    assert np.isfinite(prediction).all() and (prediction >= 0).all()


def test_held_monitoring_system_truth_does_not_choose_or_fit_its_prediction():
    f = example()
    predictions, _, states = nested_predictions(f, 100.)
    changed = f.copy()
    changed.loc[changed.component.eq(0), "y_true"] *= 1000
    perturbed, _, changed_states = nested_predictions(changed, 100.)
    columns = ["pair_id", "month_index", "operator", "fraction", "y_pred", "q90_threshold"]
    pd.testing.assert_frame_equal(predictions[predictions.component.eq(0)][columns],
                                  perturbed[perturbed.component.eq(0)][columns], check_exact=True)
    assert [s for s in states if s["held_component"] == 0] == [s for s in changed_states if s["held_component"] == 0]
    for state in states:
        assert state["held_component"] not in state["training_components"]


def test_invalid_weights_paths_or_duplicate_calendar_rows_are_rejected():
    f = example()
    f.loc[0, "weight_a"] = 0.
    with pytest.raises(ValueError, match="positive branch"):
        concentration_proxy(f, "same_month", 0., 100.)
    f = example()
    f.loc[0, "path_a_km"] = 101.
    with pytest.raises(ValueError, match="causal paths"):
        concentration_proxy(f, "branch_arrival", .5, 100.)
    with pytest.raises(ValueError, match="calendar month"):
        nested_predictions(pd.concat([example(), example().iloc[:1]], ignore_index=True), 100.)
