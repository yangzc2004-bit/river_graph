"""Daily descriptors preserve calendar grain, visibility and signed discharge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from river_graph.models.daily_flow_features import (
    availability_only,
    build_daily_flow_features,
    load_first_daily_discharge,
    minimum_valid_days,
    reconcile_daily_discharge,
)


def dataset(sites=("a",), months=("2024-02",)):
    return {"site_no": list(sites), "months": list(months),
            "x_mask": np.ones((len(sites), len(months), 2), dtype=bool)}


def daily(start="2024-02-01", periods=29, values=None, site="a"):
    return pd.DataFrame({"site_no": site, "date": pd.date_range(start, periods=periods),
                         "discharge_cfs": np.arange(periods, dtype=float) if values is None else values})


def test_formula_and_positive_unit_rescaling():
    frame = daily(values=np.arange(1, 30, dtype=float))
    result = build_daily_flow_features(dataset(), frame)["full"][0, 0]
    q10, q90 = np.quantile(frame.discharge_cfs, [.1, .9], method="linear")
    expected = [(q90-q10)/(15+q90-q10), 28/840, 1, 1, 1, 1, 1, 1]
    np.testing.assert_allclose(result, expected, rtol=1e-7)
    frame["discharge_cfs"] *= .028316846592
    np.testing.assert_array_equal(result, build_daily_flow_features(dataset(), frame)["full"][0, 0])


def test_leap_month_threshold_and_invalid_value_zeroing():
    assert minimum_valid_days(29) == (24, 23)
    assert minimum_valid_days(28) == (23, 22)
    frame = daily(periods=24)
    out = build_daily_flow_features(dataset(), frame)["full"][0, 0]
    np.testing.assert_array_equal(out[5:], [1, 1, 1])
    too_short = build_daily_flow_features(dataset(), frame.iloc[:-1])["full"][0, 0]
    np.testing.assert_array_equal(too_short[:3], [0, 0, 0])
    np.testing.assert_array_equal(too_short[5:], [0, 0, 0])
    np.testing.assert_allclose(too_short[3:5], [23/29, 22/28])


def test_gaps_do_not_create_pairs_or_cross_month_edges():
    frame = daily().drop(index=[2, 7, 12, 17, 22])
    # 24 days remain, but 18 actual pairs: width valid, flash/rise invalid.
    out = build_daily_flow_features(dataset(), frame)["full"][0, 0]
    np.testing.assert_array_equal(out[5:], [1, 0, 0])
    np.testing.assert_allclose(out[3:5], [24/29, 18/28])
    crossing = pd.DataFrame({"site_no": ["a", "a"], "date": ["2024-02-29", "2024-03-01"],
                             "discharge_cfs": [1, 100]})
    out = build_daily_flow_features(dataset(months=("2024-02", "2024-03")), crossing)["full"]
    np.testing.assert_array_equal(out[0, :, 4], [0, 0])


@pytest.mark.parametrize("value", [0.0, -10.0, 10.0])
def test_constant_signed_and_zero_flows_remain_valid(value):
    out = build_daily_flow_features(dataset(), daily(values=np.full(29, value)))["full"][0, 0]
    np.testing.assert_array_equal(out, [0, 0, 0, 1, 1, 1, 1, 1])


def test_reverse_flow_is_preserved_in_rising_and_flashiness():
    values = np.tile([-2.0, 2.0], 15)[:29]
    out = build_daily_flow_features(dataset(), daily(values=values))["full"][0, 0]
    np.testing.assert_allclose(out[:3], [4/6, 1, .5])


def test_duplicates_conflicts_and_fixed_magnitude_qc():
    frame = daily()
    duplicated = pd.concat([frame, frame.iloc[[0, 1, 2]]], ignore_index=True)
    np.testing.assert_array_equal(build_daily_flow_features(dataset(), frame)["full"],
                                  build_daily_flow_features(dataset(), duplicated)["full"])
    extra = pd.DataFrame({"site_no": ["a"]*4,
                          "date": ["2024-02-04", "2024-02-05", "2024-02-06", "bad"],
                          "discharge_cfs": [99, 3_000_001, np.nan, 1]})
    clean, summary = reconcile_daily_discharge(pd.concat([duplicated, extra]))
    assert summary["duplicate_excess_rows"] == 4
    assert summary["conflicting_station_days"] == 1
    assert summary["magnitude_excluded_rows"] == 1
    assert summary["nonfinite_value_rows"] == 1
    assert summary["invalid_date_rows"] == 1
    assert pd.Timestamp("2024-02-04") not in set(clean.date)
    # An invalid duplicate does not erase the accepted original value.
    assert pd.Timestamp("2024-02-05") in set(clean.date)


def test_first_series_policy_does_not_fill_from_alternative(tmp_path):
    path = tmp_path / "dv_test.rdb"
    path.write_text("# sample\nagency_cd\tsite_no\tdatetime\t1_00060_00003\t1_00060_00003_cd\t2_00060_00003\n"
                    "5s\t15s\t20d\t14n\t10s\t14n\n"
                    "USGS\ta\t2024-02-01\t2\tA:e\t200\n"
                    "USGS\ta\t2024-02-02\t\t\t300\n")
    frame, inventory = load_first_daily_discharge(tmp_path)
    clean, _ = reconcile_daily_discharge(frame)
    assert len(clean) == 1 and clean.iloc[0].discharge_cfs == 2
    assert inventory["multiple_series_headers"][0]["ignored"] == ["2_00060_00003"]
    assert inventory["raw_cache_files"][0]["qualifier_counts"]["A:e"] == 1


def test_doc_independence_future_flow_and_monthly_visibility():
    class InputsOnly(dict):
        def __getitem__(self, key):
            assert key in {"site_no", "months", "x_mask"}, f"unexpected data access: {key}"
            return super().__getitem__(key)
    ds = InputsOnly(dataset(months=("2024-02", "2024-03")))
    frame = pd.concat([daily(), daily("2024-03-01", 31)])
    before = build_daily_flow_features(ds, frame)["full"]
    frame.loc[frame.date.dt.month == 3, "discharge_cfs"] *= 5
    after = build_daily_flow_features(ds, frame)["full"]
    np.testing.assert_array_equal(before[:, 0], after[:, 0])
    ds["x_mask"][0, 1, 1] = False
    hidden = build_daily_flow_features(ds, frame)["full"]
    np.testing.assert_array_equal(hidden[:, 1], 0)


def test_grid_order_and_matched_availability_ablation():
    ds = dataset(sites=("b", "a"), months=("2024-03", "2024-02"))
    out = build_daily_flow_features(ds, daily())["full"]
    assert out.shape == (2, 2, 8) and out.dtype == np.float32
    assert np.all(out[0] == 0) and np.all(out[:, 0] == 0)
    assert out[1, 1, 5] == 1
    ablated = availability_only(out)
    np.testing.assert_array_equal(ablated[..., :3], 0)
    np.testing.assert_array_equal(ablated[..., 3:], out[..., 3:])
    assert not np.shares_memory(ablated, out)


def test_station_cell_index_does_not_overflow_int16():
    sites = [f"s{i}" for i in range(100)]
    months = pd.date_range("1972-04-01", periods=654, freq="MS").strftime("%Y-%m")
    out = build_daily_flow_features(dataset(sites, months), daily(site="s99"))["full"]
    assert out[99, list(months).index("2024-02"), 5] == 1
    assert out[..., 5].sum() == 1
