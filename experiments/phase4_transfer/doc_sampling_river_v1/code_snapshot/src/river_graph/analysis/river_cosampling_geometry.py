"""Metadata-only co-sampled source selection and sampling-clock diagnostics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def pair_sampling_inventory(frontier, station_dates, receiver_dates, *, receiver_area=np.nan):
    """Keep every disjoint frontier pair, including too-short sampling records."""
    f = frontier[frontier.frontier & frontier.routed_area_km2.gt(0)].sort_values("station")
    receiver = set(pd.DatetimeIndex(receiver_dates))
    rows = []
    for a, b in combinations(f.itertuples(), 2):
        date = pd.DatetimeIndex(sorted(receiver & set(station_dates[a.station]) & set(station_dates[b.station])))
        dense = within_month_dates(date)
        consistent = (not np.isfinite(receiver_area) or not np.isfinite(a.reported_area_km2+b.reported_area_km2)
                      or a.reported_area_km2+b.reported_area_km2 <= 1.1*receiver_area)
        rows.append({"source_a": a.station, "source_b": b.station,
            "source_a_comid": a.comid, "source_b_comid": b.comid,
            "n_common_days": len(date), "n_common_year_months": date.to_period("M").nunique(),
            "n_within_month_days": len(dense), "n_dense_year_months": dense.to_period("M").nunique(),
            "represented_area_km2": a.routed_area_km2+b.routed_area_km2,
            "official_area_consistent": consistent,
            "same_day_eligible": consistent and len(date) >= 12 and date.to_period("M").nunique() >= 3,
            "within_month_eligible": consistent and len(dense) >= 12 and dense.to_period("M").nunique() >= 3})
    return pd.DataFrame(rows)


def select_cosampled_sources(frontier, station_dates, receiver_dates, *, receiver_area=np.nan,
                             minimum_dates=12, minimum_months=3):
    """Largest-area disjoint pair, then greedy additions, without DOC values."""
    f = frontier[frontier.frontier & frontier.routed_area_km2.gt(0)].copy()
    f = f.sort_values(["routed_area_km2", "station"], ascending=[False, True]).reset_index(drop=True)
    common_receiver = set(pd.DatetimeIndex(receiver_dates))

    def shared(indices):
        common = common_receiver.copy()
        for j in indices:
            common &= set(pd.DatetimeIndex(station_dates[f.station.iloc[j]]))
        dates = pd.DatetimeIndex(sorted(common))
        areas = f.iloc[list(indices)].reported_area_km2.to_numpy(float)
        consistent = (not np.isfinite(receiver_area) or not np.isfinite(areas).all()
                      or areas.sum() <= 1.1*receiver_area)
        ok = (len(dates) >= minimum_dates and dates.to_period("M").nunique() >= minimum_months
              and consistent)
        return dates, ok

    choices = []
    for a, b in combinations(range(len(f)), 2):
        dates, ok = shared((a, b))
        if ok:
            choices.append((-(f.routed_area_km2.iloc[a]+f.routed_area_km2.iloc[b]),
                            -len(dates), f.station.iloc[a], f.station.iloc[b], a, b))
    if not choices:
        reason = "fewer_than_two_frontier_gauges" if len(f) < 2 else "no_co_sampled_pair"
        return [], pd.DatetimeIndex([]), reason
    selected = list(min(choices)[-2:])
    for j in range(len(f)):
        if j not in selected and shared((*selected, j))[1]:
            selected.append(j)
    dates, _ = shared(selected)
    return f.station.iloc[selected].tolist(), dates, "included"


def within_month_dates(dates, *, minimum_per_month=3):
    date = pd.DatetimeIndex(dates)
    counts = pd.Series(date.to_period("M")).value_counts()
    return date[np.asarray(counts.reindex(date.to_period("M"))) >= minimum_per_month]


def sampling_clock(events):
    """One complete date set per group; unknown timestamps remain unknown."""
    rows = []
    for date, g in events.groupby("date", sort=True):
        if g.site_no.duplicated().any() or int(g.source_order.eq(-1).sum()) != 1:
            raise ValueError("one activity per station and one receiver required")
        t = pd.to_datetime(g.timestamp_utc)
        known = bool(t.notna().all())
        receiving = t[g.source_order.eq(-1)].iloc[0]
        after = bool((t[g.source_order.ge(0)] > receiving).any()) if known else None
        rows.append({"date": date, "n_stations": len(g), "all_clocks_known": known,
            "utc_span_hours": (t.max()-t.min()).total_seconds()/3600 if known else np.nan,
            "source_after_receiver": after})
    return pd.DataFrame(rows)
