import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_kervidy_observations import (
    flow_selected_windows,
    join_doc_flow,
    occupied_record_coverage,
    prepare_flow,
)


def test_flow_units_and_source_censoring_do_not_clip():
    f = prepare_flow(pd.DataFrame({"timestamp_utc": pd.date_range("2021-01-01", periods=5, freq="15min", tz="UTC"),
                                  "q_dm3_s": [1000., 1238.414, 0, -1, np.nan]}))
    assert f.q_m3_s.iloc[0] == 1.
    assert f.flow_at_reported_cap.tolist() == [False, True, False, False, False]
    assert f.flow_valid.tolist() == [True, True, True, False, False]
    assert f.q_dm3_s.iloc[3] == -1


def test_join_uses_existing_flow_clock_and_two_minute_tolerance():
    f = prepare_flow(pd.DataFrame({"timestamp_utc": ["2021-01-01T00:00Z", "2021-01-01T00:15Z"], "q_dm3_s": [10., 20.]}))
    d = pd.DataFrame({"timestamp_utc": ["2021-01-01T00:00Z", "2021-01-01T00:13Z", "2021-01-01T00:08Z"], "doc_mg_l": [2., 3., 4.]})
    j = join_doc_flow(d, f)
    assert j.flow_match.tolist() == [True, False, True]
    assert j.flow_exact_match.tolist() == [True, False, False]
    assert j.flow_offset_minutes.iloc[2] == 2
    assert j.q_m3_s.iloc[2] == .02
    assert np.isnan(j.q_m3_s.iloc[1])
    with pytest.raises(ValueError, match="Duplicate"):
        join_doc_flow(pd.concat([d, d]), f)


def test_microsecond_storage_does_not_shrink_gaps():
    start = pd.Timestamp("2021-01-01", tz="UTC")
    t = pd.DatetimeIndex([start, start+pd.Timedelta(hours=2)]).as_unit("us")
    x = occupied_record_coverage(t, start, start+pd.Timedelta(hours=3))
    assert x["max_gap_hours"] == 2
    assert x["coverage"] == 2/13


def test_flow_selection_cannot_use_doc_values_and_retains_censored_ties():
    f = prepare_flow(pd.DataFrame({"timestamp_utc": pd.date_range("2021-01-01", periods=4, freq="15min", tz="UTC"),
                                  "q_dm3_s": [10., 1238.414, 1238.414, 50.]}))
    w = flow_selected_windows(f, f.flow_timestamp_utc.min(), f.flow_timestamp_utc.max())
    assert len(w) == 1
    assert w.selected_flow_peak_utc.iloc[0] == f.flow_timestamp_utc.iloc[1]
    assert w.n_equal_maximum_records.iloc[0] == 2
    assert w.reported_flow_max_censored.iloc[0]


def test_extra_source_decimal_digits_at_censoring_limit_are_flagged_not_rewritten():
    f = prepare_flow(pd.DataFrame({"timestamp_utc": ["2023-03-23T21:00Z"],
                                  "q_dm3_s": [1238.41413437507]}))
    assert f.flow_at_reported_cap.iloc[0]
    assert f.q_dm3_s.iloc[0] == 1238.41413437507
