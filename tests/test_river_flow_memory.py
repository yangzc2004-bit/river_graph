"""Scientific invariants of source-only monthly DOC-flow memory diagnostics."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_flow_memory import (
    chronological_prediction,
    design_matrices,
    fit_response,
    flow_month_pairs,
    paired_gains,
)


def example():
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(42)
    q = rng.normal(3, .6, 121)
    angle = 2*np.pi*dates.month/12
    y = 1.8+.2*np.sin(angle)+.3*(q[1:]-3)-.15*(q[:-1]-3)
    return pd.DataFrame({"station": "s", "huc4": "1001", "month": dates,
                         "doc": np.expm1(y), "discharge_cfs": np.exp(q[1:]),
                         "previous_discharge_cfs": np.exp(q[:-1]), "temperature_c": 10+np.sin(angle)})


def test_known_season_adjusted_current_and_previous_response():
    record, _ = fit_response(example())
    assert record["status"] == "included"
    assert record["current_response"] == pytest.approx(.3)
    assert record["previous_response"] == pytest.approx(-.15)


def test_flow_unit_conversion_leaves_predictions_and_responses_equal():
    a = example()
    b = a.copy()
    b[["discharge_cfs", "previous_discharge_cfs"]] *= .0283168466
    assert fit_response(a)[0]["current_response"] == pytest.approx(fit_response(b)[0]["current_response"])
    _, pa = chronological_prediction(a)
    _, pb = chronological_prediction(b)
    np.testing.assert_allclose(pa.prediction, pb.prediction, atol=1e-12)


def test_later_doc_does_not_change_chronological_predictions():
    a = example()
    b = a.copy()
    b.loc[b.month.dt.year >= 2005, "doc"] *= 100
    _, pa = chronological_prediction(a)
    _, pb = chronological_prediction(b)
    np.testing.assert_array_equal(pa.prediction, pb.prediction)
    assert not np.array_equal(pa.doc, pb.doc)


def test_query_hydro_does_not_change_train_centering():
    a = example()
    train, query = a.iloc[:60], a.iloc[60:].copy()
    original = design_matrices(train, query)
    query.discharge_cfs *= 100
    modified = design_matrices(train, query)
    np.testing.assert_array_equal(original[0], modified[0])
    np.testing.assert_array_equal(original[1], modified[1])
    assert not np.array_equal(original[3], modified[3])


def test_previous_flow_uses_calendar_month_not_previous_doc_record():
    data = {"y": np.array([[2., 3., 4., 5.]]), "y_mask": np.ones((1, 4), bool),
            "x": np.array([[[10., 1.], [20., 2.], [30., 3.], [40., 4.]]]),
            "x_mask": np.ones((1, 4, 2), bool), "feature_channels": ["discharge", "temperature"],
            "months": pd.date_range("2000-01-01", periods=4, freq="MS"), "site_no": ["s"]}
    meta = pd.DataFrame({"station": ["s"], "included": [True]})
    a = flow_month_pairs(data, np.array([0, 2]), meta)
    assert np.isnan(a.previous_discharge_cfs.iloc[0])
    assert a.previous_discharge_cfs.iloc[1] == 20
    assert a.pair_usable.tolist() == [False, True]
    changed = data | {"y": np.array([[2., 9999., 4., 9999.]])}
    pd.testing.assert_frame_equal(a, flow_month_pairs(changed, np.array([0, 2]), meta))


def test_collinear_months_not_identifiable():
    s = example()
    s.previous_discharge_cfs = s.discharge_cfs
    assert fit_response(s)[0]["status"] == "current_previous_flow_not_identifiable"


def test_paired_bootstrap_uses_station_not_month_counts():
    records = []
    for station, region in (("a", "1001"), ("b", "1002"), ("c", "1003")):
        for arm, error in (("base", 2.), ("memory", 1.)):
            records.append({"station": station, "huc4": region, "arm": arm, "error": error})
    result = paired_gains(pd.DataFrame(records), [("memory", "base")], draws=100).iloc[0]
    assert result.gain_pct == pytest.approx(50)
    assert result.n_stations == 3
    assert result.ci_low_pct == pytest.approx(50)
