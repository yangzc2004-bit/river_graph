from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_junction_layout import partition_mainstem
from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.analysis.river_monitored_arrivals import (
    covered_geometry,
    form_contrasts,
    monitored_frontier,
    receiver_components,
    same_day_activities,
    select_activity_set,
    select_source_set,
    signal_statistics,
)


def small_tree():
    # Two two-reach branches feed a single receiving reach.
    p = SimpleNamespace(comids=np.arange(10, 15), successor=np.array([-1, 0, 0, 1, 2]),
        length=np.ones(5), area=np.array([1., 2., 3., 4., 5.]), distance=np.array([.5, 1.5, 1.5, 2.5, 2.5]))
    return p, np.arange(1, 6)


def test_frontier_and_coverage_count_unique_incremental_area():
    p, h = small_tree()
    gauges = pd.DataFrame({"station": ["a", "b", "c"], "comid": [11, 12, 13]})
    f = monitored_frontier(p, h, gauges)
    assert f.frontier.tolist() == [True, True, False]
    np.testing.assert_array_equal(f.routed_area_km2, [6, 8, 4])
    layout = partition_mainstem(p, h, np.arange(5.))
    meta, area, owner = covered_geometry(p, h, [11, 12], layout)
    np.testing.assert_array_equal(area, [6, 8])
    np.testing.assert_array_equal(owner, [-1, 0, 1, 0, 1])
    assert meta["covered_area_fraction"] == pytest.approx(14/15)
    with pytest.raises(ValueError, match="nested"):
        covered_geometry(p, h, [11, 13], layout)


def test_metadata_only_source_selection_and_fixed_common_months():
    dates = pd.date_range("2001-01-01", periods=48, freq="MS")
    f = pd.DataFrame({"station": ["a", "b", "c"], "frontier": [True]*3,
        "routed_area_km2": [8., 5., 1.], "reported_area_km2": [8., 5., 1.]})
    observed = {s: np.ones(48, bool) for s in f.station}
    observed["c"][2::3] = False
    ids, mask, status = select_source_set(f, observed, np.ones(48, bool), dates, 20)
    assert ids == ["a", "b", "c"] and status == "included"
    np.testing.assert_array_equal(mask, observed["c"])
    ids, _, _ = select_source_set(f, observed, np.ones(48, bool), dates, 9)
    assert ids == ["a", "c"]


def test_coverage_invariant_to_array_order():
    p, h = small_tree()
    first = partition_mainstem(p, h, np.arange(5.))
    a, areas, _ = covered_geometry(p, h, [11, 12], first)
    order = np.array([3, 1, 4, 0, 2])
    reverse = np.argsort(order)
    successor = np.array([reverse[j] if j >= 0 else -1 for j in p.successor[order]])
    q = SimpleNamespace(**{k: getattr(p, k)[order] for k in ("comids", "area", "length", "distance")}, successor=successor)
    second = partition_mainstem(q, h[order], np.arange(5.)[order])
    b, new_area, _ = covered_geometry(q, h[order], [11, 12], second)
    np.testing.assert_allclose(areas, new_area)
    assert a.keys() == b.keys()
    np.testing.assert_allclose(list(a.values()), list(b.values()), atol=1e-12)


def test_complete_overlapping_catchments_define_systems():
    result = receiver_components({"a": {10, 11}, "b": {11, 12}, "c": {20}},
        {"a": {10}, "b": {12}, "c": {20}})
    assert result["a"] == result["b"] != result["c"]


def test_multisource_covariance_identity_and_synchrony():
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    signal = np.random.default_rng(1).normal(size=60)
    a = np.column_stack([10+signal, 20+2*signal, 30+3*signal])
    w = np.array([.2, .3, .5])
    result, frame = signal_statistics(a, a@w, w, dates, shuffles=10)
    assert result["source_coherence"] == pytest.approx(1)
    assert result["asynchronous_buffer_fraction"] == pytest.approx(0, abs=1e-12)
    assert result["outlet_mix_correlation"] == pytest.approx(1)
    assert result["mixture_buffer_fraction"] == pytest.approx(result["amplitude_balance_buffer_fraction"])
    np.testing.assert_allclose(frame.doc_mixture, a@w)
    assert result["real_minus_shuffle_correlation"] > .5


def test_equal_amplitude_opposite_sources_cancel_without_removal():
    dates = pd.date_range("2000-01-01", periods=48, freq="MS")
    signal = np.random.default_rng(2).normal(size=48)
    result, _ = signal_statistics(np.column_stack([10+signal, 10-signal]), 10+signal, [.5, .5], dates, shuffles=0)
    assert result["source_coherence"] == pytest.approx(-1)
    assert result["mixture_buffer_fraction"] == pytest.approx(1)
    assert np.isnan(result["outlet_mix_correlation"])


def activities(site, days):
    return pd.DataFrame({"site_no": site, "date": pd.to_datetime([f"2000-01-{d:02d}" for d in days]),
        "event_id": [f"{site}-{i}" for i in range(len(days))], "doc": np.arange(len(days))+1.})


def test_activity_choice_matches_exhaustive_date_span_and_ignores_doc():
    groups = [activities("a", [1, 8, 15]), activities("b", [3, 9, 18]),
              activities("c", [6, 11]), activities("t", [8, 20])]
    selected, span = select_activity_set(groups)
    expected = min((max(x)-min(x)).days for x in product(*(g.date for g in groups)))
    assert span == expected
    changed = [g.assign(doc=1000-np.arange(len(g))) for g in groups]
    newer, _ = select_activity_set(changed)
    assert [r["event_id"] for r in selected] == [r["event_id"] for r in newer]


def test_gauge_and_source_row_permutations_preserve_selection():
    p, h = small_tree()
    g = pd.DataFrame({"station": ["a", "b"], "comid": [11, 12]})
    f = monitored_frontier(p, h, g)
    alt = monitored_frontier(p, h, g.iloc[::-1])
    pd.testing.assert_frame_equal(f.sort_values("station").reset_index(drop=True), alt.sort_values("station").reset_index(drop=True))


def test_constant_sources_and_empty_conditional_denominators():
    dates = pd.date_range("2000-01-01", periods=48, freq="MS")
    result, _ = signal_statistics(np.full((48, 2), 4.), np.full(48, 5.), [1, 1], dates, shuffles=0)
    assert np.isnan(result["source_coherence"])
    assert np.isnan(result["mixture_buffer_fraction"])


def test_receiving_gauge_cannot_enter_upstream_frontier():
    p, h = small_tree()
    with pytest.raises(ValueError, match="receiving"):
        monitored_frontier(p, h, pd.DataFrame({"station": ["t"], "comid": [10]}))


def test_hidden_cells_do_not_change_source_availability_or_visible_values():
    y = np.arange(180, dtype=float).reshape(3, 60)
    cells = np.flatnonzero(np.tile(np.arange(60) % 4 != 0, 3))
    original = permitted_doc({"y": y}, cells)
    altered = y.copy().ravel()
    altered[np.setdiff1d(np.arange(y.size), cells)] = 1e6
    np.testing.assert_array_equal(original, permitted_doc({"y": altered.reshape(y.shape)}, cells))


def test_daily_selection_uses_metadata_and_does_not_reuse_days():
    rows = []
    for station in ("a", "b", "t"):
        f = activities(station, [1, 1, 8, 15])
        f["timestamp_utc"] = f.date+pd.to_timedelta([9, 8, 9, 9], unit="h")
        f["month"] = pd.Timestamp("2000-01-01")
        rows.append(f)
    raw = pd.concat(rows, ignore_index=True)
    chosen = same_day_activities(raw, ["a", "b", "t"], [pd.Timestamp("2000-01-01")])
    changed = same_day_activities(raw.assign(doc=1e5-raw.doc), ["a", "b", "t"], [pd.Timestamp("2000-01-01")])
    assert chosen.event_id.tolist() == changed.event_id.tolist()
    assert len(chosen) == 9 and not chosen.duplicated(["site_no", "date"]).any()
    assert chosen.loc[chosen.date.eq(pd.Timestamp("2000-01-01")), "event_id"].str.endswith("-1").all()
    assert chosen.loc[chosen.date.eq(pd.Timestamp("2000-01-01")), "doc_daily_mean"].eq(1.5).all()


def test_within_month_adjustment_keeps_covariance_identity_and_removes_month_levels():
    dates = pd.DatetimeIndex([pd.Timestamp(2001, m, d) for m in range(1, 7) for d in (1, 8, 15, 22)])
    rng = np.random.default_rng(7)
    a = np.column_stack([10+dates.month+rng.normal(size=len(dates)),
                         12+2*dates.month+rng.normal(size=len(dates))])
    y = a@np.array([.4, .6])+dates.month
    result, frame = signal_statistics(a, y, [.4, .6], dates, shuffles=10, adjustment="within_month")
    assert result["outlet_mix_correlation"] == pytest.approx(1)
    assert result["mixture_buffer_fraction"] == pytest.approx(
        result["asynchronous_buffer_fraction"]+result["amplitude_balance_buffer_fraction"])
    np.testing.assert_allclose(frame.groupby(frame.date.dt.to_period("M")).receiver_anomaly.mean(), 0, atol=1e-12)


def test_form_contrast_resamples_shared_systems_together():
    f = pd.DataFrame({"cluster": [1, 3, 1, 3, 1, 3], "component": [0, 0, 1, 1, 2, 2],
                      "metric": [0., 2., 5., 7., 20., 22.]})
    r = form_contrasts(f, ["metric"], draws=100).iloc[0]
    np.testing.assert_allclose([r.estimate, r.ci_low, r.ci_high], 2)
