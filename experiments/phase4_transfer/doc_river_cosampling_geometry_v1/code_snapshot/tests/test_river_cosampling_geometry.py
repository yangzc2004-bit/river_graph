"""Scientific contracts for expanding actual same-day observations."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_cosampling_geometry import (
    pair_sampling_inventory,
    sampling_clock,
    select_cosampled_sources,
    within_month_dates,
)


def test_selection_uses_shared_dates_and_area_not_values():
    dates = pd.date_range("2000-01-01", periods=90)
    f = pd.DataFrame({"station": ["c", "a", "b"], "frontier": True,
        "routed_area_km2": [10., 30., 20.], "reported_area_km2": [10., 30., 20.],
        "doc": [999., -50., 0.]})
    archive = {"a": dates, "b": dates[::2], "c": dates[:10]}
    sources, common, status = select_cosampled_sources(f, archive, dates, receiver_area=100.)
    assert sources == ["a", "b"] and status == "included" and len(common) == 45
    changed = f.iloc[::-1].assign(doc=[-1e9, 1e9, np.nan])
    s, d, _ = select_cosampled_sources(changed, archive, dates, receiver_area=100.)
    assert s == sources
    pd.testing.assert_index_equal(d, common)
    assert select_cosampled_sources(f, archive, dates, receiver_area=20.)[2] == "no_co_sampled_pair"


def test_disjoint_frontier_and_three_sampled_months_required():
    dates = pd.date_range("2001-01-01", periods=20)
    f = pd.DataFrame({"station": ["a", "b", "nested"], "frontier": [True, True, False],
        "routed_area_km2": [1., 1., 999.], "reported_area_km2": [1., 1., 999.]})
    archive = {s: dates for s in f.station}
    assert select_cosampled_sources(f, archive, dates)[0] == []
    sources, _, _ = select_cosampled_sources(f, archive, dates, minimum_months=1)
    assert sources == ["a", "b"]


def test_within_month_selection_does_not_join_calendar_months_across_years():
    dates = pd.to_datetime(["2000-01-01", "2000-01-02", "2001-01-01", "2001-01-02", "2001-01-03"])
    pd.testing.assert_index_equal(within_month_dates(dates), pd.DatetimeIndex(dates[2:]))


def test_clock_missing_and_source_after_receiver_are_separate():
    f = pd.DataFrame({"date": pd.to_datetime(["2000-01-01"]*3+["2000-01-02"]*3),
        "site_no": ["a", "b", "r"]*2, "source_order": [0, 1, -1]*2,
        "timestamp_utc": pd.to_datetime(["2000-01-01 09:00", "2000-01-01 11:00",
            "2000-01-01 10:00", "2000-01-02 09:00", None, "2000-01-02 10:00"])})
    clock = sampling_clock(f)
    assert clock.utc_span_hours.iloc[0] == 2 and clock.source_after_receiver.iloc[0]
    assert np.isnan(clock.utc_span_hours.iloc[1]) and pd.isna(clock.source_after_receiver.iloc[1])
    with pytest.raises(ValueError):
        sampling_clock(pd.concat([f, f.iloc[:1]]))


def test_short_pair_inventory_preserves_missing_dates_not_zero_doc_effects():
    dates = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03", "2000-02-01"])
    f = pd.DataFrame({"station": ["a", "b"], "frontier": True, "comid": [1, 2],
        "routed_area_km2": [10., 20.], "reported_area_km2": [10., 20.]})
    rows = pair_sampling_inventory(f, {"a": dates, "b": dates}, dates)
    assert rows.n_common_days.iloc[0] == 4 and rows.n_within_month_days.iloc[0] == 3
    assert rows.n_common_year_months.iloc[0] == 2 and not rows.same_day_eligible.iloc[0]
