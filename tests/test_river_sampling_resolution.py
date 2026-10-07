"""Sampling metadata, concentration-independent selection and measured flow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_sampling_resolution import (
    attach_sample_flow,
    doc_activities,
    load_station_daily_flow,
    sampling_triplets,
    select_triplet,
    station_cadence,
)


def observations():
    return pd.DataFrame({"site_no": ["a", "a", "a", "b", "r"],
        "event_id": ["a1", "a1", "a2", "b1", "r1"],
        "date": pd.to_datetime(["2000-06-01", "2000-06-01", "2000-06-05", "2000-06-05", "2000-06-05"]),
        "start_time": ["12:00:00", "12:00:00", "12:00:00", "13:00:00", "14:00:00"],
        "time_zone": ["CDT"]*5, "doc": [2., 4., 8., 9., 10.]})


def fixed_input():
    return pd.DataFrame({"pair_id": ["p"], "month_index": [0], "date": pd.to_datetime(["2000-06-01"]),
        "source_a": ["a"], "source_b": ["b"], "target": ["r"]})


def test_replicates_and_timezone_are_sampling_metadata():
    e = doc_activities(observations())
    a = e[e.event_id.eq("a1")].iloc[0]
    assert a.n_results == 2 and a.doc == 3.
    assert a.timestamp_utc == pd.Timestamp("2000-06-01 17:00:00")
    assert len(e) == 4
    cadence, gaps = station_cadence(e)
    assert cadence.set_index("site_no").loc["a", "n_sample_days"] == 2
    assert gaps.gap_days.tolist() == [4.]


def test_conflicting_activity_metadata_is_not_silently_averaged():
    rows = observations()
    rows.loc[1, "start_time"] = "10:00:00"
    with pytest.raises(ValueError, match="conflicting metadata"):
        doc_activities(rows)


def test_missing_time_is_not_imputed():
    rows = observations()
    rows.loc[4, "start_time"] = ""
    e = doc_activities(rows)
    r = e[e.site_no.eq("r")].iloc[0]
    assert not r.timestamp_known and pd.isna(r.timestamp_utc)
    f = sampling_triplets(fixed_input(), e)
    assert f.sample_span_days.iloc[0] == 0
    assert np.isnan(f.sample_span_hours_utc.iloc[0])


def test_date_selection_does_not_read_doc_values():
    e = doc_activities(observations())
    first = sampling_triplets(fixed_input(), e)
    changed = e.copy()
    changed["doc"] = [10000., 100., 0., 99.]
    second = sampling_triplets(fixed_input(), changed)
    cols = [c for c in first if "sample_doc" not in c]
    pd.testing.assert_frame_equal(first[cols], second[cols])
    assert first.sample_event_a.iloc[0] == "a2"
    assert first.sample_span_hours_utc.iloc[0] == 2.


def test_upstream_sample_after_receiver_is_retained_and_flagged():
    rows = observations()
    rows.loc[2, "start_time"] = "16:00:00"
    f = sampling_triplets(fixed_input(), doc_activities(rows))
    assert not f.source_after_receiver_date.iloc[0]
    assert f.source_after_receiver_utc.iloc[0]


def test_selection_ties_use_metadata_and_input_order_does_not_matter():
    e = doc_activities(observations())
    groups = [e[e.site_no.eq(s)] for s in ("a", "b", "r")]
    left = select_triplet(*groups)
    right = select_triplet(*[g.iloc[::-1] for g in groups])
    assert [r["event_id"] for r in left[0]] == [r["event_id"] for r in right[0]]


def test_measured_flow_has_no_gap_filling():
    f = sampling_triplets(fixed_input(), doc_activities(observations()))
    daily = pd.DataFrame({"site_no": ["a", "b"], "date": pd.to_datetime(["2000-06-05"]*2), "discharge_cfs": [0., 7.]})
    product = attach_sample_flow(f, daily)
    assert np.isnan(product.flow_receiver_sample_cfs.iloc[0])
    assert np.isnan(product.flow_a_date_log_change.iloc[0])
    assert product.flow_b_date_log_change.iloc[0] == 0.
    assert not product.all_three_sample_flows_measured.iloc[0]


def test_filtered_daily_reader_keeps_first_series_and_excludes_conflicts(tmp_path):
    (tmp_path/"dv_test.rdb").write_text(
        "agency_cd\tsite_no\tdatetime\tq_00060_00003\tother_00060_00003\n"
        "5s\t15s\t20d\t14n\t14n\n"
        "USGS\ta\t2000-06-01\t10\t999\n"
        "USGS\ta\t2000-06-01\t10\t998\n"
        "USGS\ta\t2000-06-02\t20\t998\n"
        "USGS\ta\t2000-06-02\t30\t998\n"
        "USGS\tunrelated\t2000-06-01\t100\t998\n")
    daily, inventory = load_station_daily_flow(tmp_path, {"a"})
    assert len(daily) == 1 and daily.discharge_cfs.iloc[0] == 10.
    assert inventory["quality_summary"]["conflicting_station_days"] == 1
    assert inventory["files"][0]["ignored_alternative_columns"] == ["other_00060_00003"]
