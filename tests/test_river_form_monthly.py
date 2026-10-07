import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_form_monthly import (
    ARMS,
    ENVIRONMENT,
    HYDRO,
    background_prediction,
    crossfit_backgrounds,
    fit_background,
    member_responses,
    paired_records,
    response_evidence,
    response_summary,
    station_weights,
)


def example():
    rng = np.random.default_rng(8)
    rows = []
    for station in range(10):
        for month in range(6+station):
            r = {name: rng.uniform(.1, 1) for name in set(ENVIRONMENT).union(*(set(v) for v in ARMS.values()))}
            r.update(station=f"s{station}", huc4=f"{station//2:04d}", huc2="01", comid=station,
                     cluster=station % 3+1, month_index=month, y_true=rng.uniform(1, 15))
            rows.append(r)
    return pd.DataFrame(rows)


def test_each_source_station_has_total_weight_one():
    f = example()
    totals = pd.Series(station_weights(f)).groupby(f.station).sum()
    np.testing.assert_allclose(totals, 1.)


def test_same_values_and_denominators_give_hand_computed_responses():
    f = pd.DataFrame({"y_true_a": [2., 4., 10., 12.], "environment_log_prediction_a": np.log1p([2., 4., 10., 12.]),
                      "hydro_log_prediction_a": np.log1p([2., 4., 10., 12.]),
                      "environment_high_probability_a": [.1, .2, .4, .5], "hydro_high_probability_a": [.1, .2, .4, .5]})
    r, n = member_responses(f, "a")
    assert r["doc_mean"] == 7. and r["high_doc_fraction"] == .5 and n == 2
    np.testing.assert_allclose(r["doc_sd"], np.std([2, 4, 10, 12], ddof=1))
    assert r["hydro_residual_mean"] == 0 and r["hydro_residual_sd"] == 0
    np.testing.assert_allclose(r["hydro_exceedance"], .2)


def test_source_preprocessing_and_prediction_do_not_read_query_doc_or_morphology():
    f = example()
    train, query = f[f.huc4.ne("0000")].copy(), f[f.huc4.eq("0000")].copy()
    train.loc[train.index[:3], "temperature"] = np.nan
    for probability in (False, True):
        state = fit_background(train, "hydro", probability=probability)
        expected = background_prediction(query, state)
        changed = query.copy()
        changed.y_true = 1e6
        for column in set().union(*[set(v) for v in ARMS.values()])-set(HYDRO):
            changed[column] = 1000.
        np.testing.assert_array_equal(expected, background_prediction(changed, state))
        assert np.isfinite(expected).all()


def test_held_region_labels_do_not_fit_their_background():
    f = example()
    p, states = crossfit_backgrounds(f)
    altered = f.copy()
    altered.loc[altered.huc4.eq("0000"), "y_true"] = 999.
    changed, new_states = crossfit_backgrounds(altered)
    columns = ["station", "month_index", "arm", "log_pred", "high_probability"]
    pd.testing.assert_frame_equal(p[p.huc4.eq("0000")][columns], changed[changed.huc4.eq("0000")][columns], check_exact=True)
    held = [s for s in states if "0000" in s["held_huc4"]]
    assert held == [s for s in new_states if "0000" in s["held_huc4"]]
    assert all(set(s["training_huc4"]).isdisjoint(s["held_huc4"]) for s in states)
    with pytest.raises(ValueError, match="one observation"):
        crossfit_backgrounds(pd.concat([f, f.iloc[:1]]))


def test_single_region_has_no_geographic_interval_and_pairs_have_equal_weight():
    f = pd.DataFrame({"class_a": [1, 1], "class_b": [3, 3], "population": "common_doc", "metric": "doc_mean",
                      "huc4": "0001", "difference_b_minus_a": [1., 3.], "value_a": [2., 2.], "value_b": [3., 5.]})
    summary, _ = response_summary(f, draws=50)
    assert summary.difference_b_minus_a.iloc[0] == 2
    assert summary[["ci_low", "ci_high"]].isna().all().all()


def test_constant_high_value_fit_still_returns_valid_probability():
    f = example()
    f.y_true = 2.
    state = fit_background(f, "environment", probability=True)
    np.testing.assert_array_equal(background_prediction(f, state), 0.)


def test_matched_members_use_identical_calendar_months_and_exclude_unshared_spike():
    rows, predictions = [], []
    for station, first in (("a", 0), ("b", 3)):
        for month in range(first, first+15):
            r = {"station": station, "month_index": month, "date": pd.Timestamp("2000-01-01")+pd.DateOffset(months=month),
                 "calendar_month": month % 12+1, "y_true": 9999. if month == 0 else 5.,
                 "discharge_valid": True, "temperature_valid": True, "log_discharge": 1., "temperature": 10.}
            rows.append(r)
            for arm in ("environment", "hydro"):
                predictions.append({"station": station, "month_index": month, "arm": arm,
                                    "log_pred": np.log1p(5.), "high_probability": .1})
    pairs = pd.DataFrame({"station_a": ["a"], "station_b": ["b"], "class_a": [1], "class_b": [3], "huc4": ["0001"]})
    records, ledger = paired_records(pd.DataFrame(rows), pairs, pd.DataFrame(predictions))
    assert records.month_index.tolist() == list(range(3, 15))
    assert ledger.n_common_months.iloc[0] == 12 and ledger.included.iloc[0]
    assert records.y_true_a.max() == 5. and records.y_true_b.max() == 5.
    evidence = response_evidence(records)
    np.testing.assert_array_equal(evidence.difference_b_minus_a, 0.)
    assert set(evidence.population) == {"common_doc", "complete_hydro"}
