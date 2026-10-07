import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_form_flow import (
    assign_states,
    estimable_coefficients,
    flow_references,
    measured_response,
    pair_responses,
    shared_pair_records,
    summarize_pairs,
)


def example(station="a", contrast=.3):
    months = np.arange(72)
    state = np.tile(["low", "middle", "high"], 24)
    date = pd.date_range("2000-01-01", periods=len(months), freq="MS")
    angle = 2*np.pi*date.month/12
    f = pd.DataFrame({"station": station, "month_index": months, "date": date, "calendar_month": date.month,
                      "calendar_year": date.year+(date.month-1)/12, "month_sin": np.sin(angle), "month_cos": np.cos(angle),
                      "flow_state": state, "log_discharge": np.tile([1., 2., 3.], 24),
                      "temperature": np.tile([10., 12., 15.], 24), "temperature_valid": True})
    f["y_true"] = np.expm1(1+.1*np.sin(angle)+contrast*(state == "high")+.1*(state == "middle"))
    return f


def test_thresholds_use_hydro_including_months_without_doc():
    f = example()
    refs = flow_references(f)
    changed = f.copy()
    changed.y_true = 1e8
    pd.testing.assert_frame_equal(refs, flow_references(changed))
    np.testing.assert_array_equal(assign_states(f, refs).flow_state, f.flow_state)
    # A month with no DOC still contributes its flow to the reference.
    extra = f.iloc[:1].copy()
    extra.month_index = 72
    extra.y_true = np.nan
    extra.log_discharge = 100.
    assert flow_references(pd.concat([f, extra])).n_reference_months.iloc[0] == 73
    with pytest.raises(ValueError, match="unique"):
        flow_references(pd.concat([f, f.iloc[:1]]))


def test_constant_flow_does_not_create_artificial_states():
    f = example()
    f.log_discharge = 2.
    assert not flow_references(f).reference_valid.iloc[0]
    assert assign_states(f, flow_references(f)).flow_state.eq("unavailable").all()


def test_known_response_separates_season_from_flow_change():
    f = example(contrast=.3)
    response, summaries = measured_response(f)
    assert response["adjusted_identified"]
    np.testing.assert_allclose(response["adjusted_log_contrast"], .3, atol=1e-12)
    assert sum(s["n_months"] for s in summaries) == len(f)
    assert response["log_flow_contrast"] == 2.
    # Temperature is collinear with the state in this synthetic example.
    temperature_response, _ = measured_response(f, temperature=True)
    assert not temperature_response["adjusted_identified"]
    assert np.isnan(temperature_response["adjusted_log_contrast"])


def test_pairs_compare_only_shared_months_and_difference_of_responses():
    a, b = example("a", .3), example("b", .5)
    a.loc[0, "y_true"] = 1e8
    b = b[b.month_index.gt(0)]
    pairs = pd.DataFrame({"station_a": ["a"], "station_b": ["b"], "class_a": [1], "class_b": [3], "huc4": ["0001"]})
    records = shared_pair_records(pd.concat([a, b]), pairs)
    assert records.month_index.min() == 1 and len(records) == 71
    evidence, ledger, _, _, joint = pair_responses(records, pairs)
    r = evidence.query("population == 'observed_flow' and metric == 'adjusted_log_contrast'").iloc[0]
    np.testing.assert_allclose(r.difference_b_minus_a, .2, atol=1e-12)
    assert ledger.eligible.all() and joint.groupby("flow_state").month_index.min().min() > 0


def test_pair_summary_is_equal_pair_and_single_region_has_no_interval():
    f = pd.DataFrame({"class_a": [1, 1], "class_b": [3, 3], "population": "observed_flow", "metric": "test",
                      "huc4": "0001", "difference_b_minus_a": [1., 3.], "value_a": [2., 4.], "value_b": [3., 7.]})
    summary, _ = summarize_pairs(f, draws=100)
    assert summary.difference_b_minus_a.iloc[0] == 2
    assert summary[["ci_low", "ci_high"]].isna().all().all()
    f.huc4 = ["0001", "0002"]
    summary, _ = summarize_pairs(f, draws=100)
    assert summary.ci_low.iloc[0] >= 1 and summary.ci_high.iloc[0] <= 3


def test_rank_deficiency_is_flagged_per_coefficient():
    x = np.column_stack([np.ones(20), np.arange(20), np.arange(20)])
    coef, rank = estimable_coefficients(x, 2+3*np.arange(20), np.ones(20))
    assert rank == 2
    np.testing.assert_allclose(coef[0], 2)
    assert np.isnan(coef[1:]).all()
