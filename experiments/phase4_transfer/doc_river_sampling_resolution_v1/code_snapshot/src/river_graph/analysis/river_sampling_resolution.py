"""Actual sampling dates on a fixed, measured tributary/receiver footprint."""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file
from river_graph.models.daily_flow_features import reconcile_daily_discharge

TIME_ZONE_HOURS = {"UTC": 0, "GMT": 0, "EST": -5, "EDT": -4, "CST": -6,
                   "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7}
DAY_CUTS = (0, 1, 3, 7, 29)


def doc_activities(rows):
    """Reconcile result replicates without treating them as sampling events."""
    f = rows.copy()
    needed = ["site_no", "date", "doc", "event_id", "start_time", "time_zone"]
    if f[needed[:3]].isna().any().any() or not np.isfinite(f.doc).all() or (f.doc < 0).any():
        raise ValueError("accepted finite nonnegative DOC and station/date required")
    f["date"] = pd.to_datetime(f.date).dt.normalize()
    f["event_id_missing"] = f.event_id.isna() | f.event_id.eq("")
    f.loc[f.event_id_missing, "event_id"] = [f"missing-id-row-{i}" for i in f.index[f.event_id_missing]]
    for name in ("start_time", "time_zone"):
        f[name] = f[name].fillna("").astype(str).str.strip()
    grouped = f.groupby(["site_no", "event_id"], sort=True)
    if grouped[["date", "start_time", "time_zone"]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("conflicting metadata within a sampling activity")
    events = grouped.agg(date=("date", "first"), doc=("doc", "mean"),
        start_time=("start_time", "first"), time_zone=("time_zone", "first"),
        n_results=("doc", "size"), event_id_missing=("event_id_missing", "max")).reset_index()
    time = pd.to_timedelta(events.start_time.replace("", np.nan), errors="coerce")
    offset = events.time_zone.map(TIME_ZONE_HOURS)
    known = time.notna() & time.ge(pd.Timedelta(0)) & time.lt(pd.Timedelta(days=1)) & offset.notna()
    events["timestamp_known"] = known
    events["timestamp_utc"] = (events.date+time-pd.to_timedelta(offset, unit="h")).where(known)
    events["month"] = events.date.dt.to_period("M").dt.to_timestamp()
    return events


def select_triplet(a, b, receiver):
    """Deterministic metadata-only selection; values are copied after selection."""
    candidates = []
    for rows in product(a.to_dict("records"), b.to_dict("records"), receiver.to_dict("records")):
        dates = [r["date"] for r in rows]
        span = (max(dates)-min(dates)).days
        times = [r["timestamp_utc"] for r in rows]
        known = all(pd.notna(t) for t in times)
        hours = (max(times)-min(times)).total_seconds()/3600 if known else float("inf")
        offset = sum(abs((d-dates[2]).days) for d in dates[:2])
        key = (span, hours, offset, *dates, *[str(r["event_id"]) for r in rows])
        candidates.append((key, rows))
    if not candidates:
        raise ValueError("all three stations need a sampling activity")
    key, selected = min(candidates, key=lambda v: v[0])
    return selected, key[0], key[1] if np.isfinite(key[1]) else np.nan


def sampling_triplets(inputs, events):
    """Keep the fixed connection-month population and label the actual dates."""
    lookup = {k: v for k, v in events.groupby(["site_no", "month"])}
    rows = []
    for r in inputs.sort_values(["pair_id", "month_index"]).itertuples():
        groups = [lookup[(s, pd.Timestamp(r.date))] for s in (r.source_a, r.source_b, r.target)]
        selected, span, hours = select_triplet(*groups)
        row = {"pair_id": r.pair_id, "month_index": r.month_index, "sample_span_days": span,
               "sample_span_hours_utc": hours, "all_timestamps_known": np.isfinite(hours)}
        for role, event, group in zip(("a", "b", "receiver"), selected, groups, strict=True):
            row.update({f"sample_date_{role}": event["date"], f"sample_utc_{role}": event["timestamp_utc"],
                f"sample_event_{role}": event["event_id"], f"sample_doc_{role}": event["doc"],
                f"n_days_{role}": group.date.nunique(), f"n_events_{role}": len(group),
                f"n_results_{role}": group.n_results.sum()})
        row["all_three_dense_month"] = all(g.date.nunique() >= 3 for g in groups)
        row["all_three_one_day"] = all(g.date.nunique() == 1 for g in groups)
        row["source_after_receiver_date"] = any(v["date"] > selected[2]["date"] for v in selected[:2])
        row["source_after_receiver_utc"] = (any(v["timestamp_utc"] > selected[2]["timestamp_utc"] for v in selected[:2])
                                                   if np.isfinite(hours) else None)
        rows.append(row)
    return inputs.merge(pd.DataFrame(rows), on=["pair_id", "month_index"], validate="one_to_one")


def station_cadence(events):
    rows, intervals = [], []
    for site, f in events.groupby("site_no"):
        days = pd.DatetimeIndex(f.date.unique()).sort_values()
        gap = np.diff(days.values)/np.timedelta64(1, "D")
        rows.append({"site_no": site, "n_results": f.n_results.sum(), "n_activities": len(f),
            "n_sample_days": len(days), "n_months": f.month.nunique(), "first_date": days.min(),
            "last_date": days.max(), "median_gap_days": np.median(gap) if len(gap) else np.nan,
            "fraction_gaps_le7": np.mean(gap <= 7) if len(gap) else np.nan,
            "fraction_gaps_le1": np.mean(gap <= 1) if len(gap) else np.nan})
        intervals.extend({"site_no": site, "first_date": start, "next_date": end, "gap_days": delta}
            for start, end, delta in zip(days[:-1], days[1:], gap, strict=True))
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def load_station_daily_flow(cache, sites):
    """Read only relevant stations, using the existing first-series/QC policy."""
    frames, inventory = [], []
    for path in sorted(Path(cache).glob("dv_*.rdb")):
        values, selection, ignored = [], None, []
        with path.open(encoding="utf-8", errors="replace", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if not row or row[0].startswith("#"):
                    continue
                if row[0] == "agency_cd":
                    matches = [i for i, name in enumerate(row) if name.endswith("_00060_00003")]
                    selection = (row.index("site_no"), row.index("datetime"), matches[0]) if matches else None
                    ignored.extend(row[i] for i in matches[1:])
                elif selection is not None and row[0] == "USGS":
                    si, di, vi = selection
                    if len(row) <= max(selection):
                        raise ValueError(f"short daily flow row: {path}")
                    if row[si] in sites:
                        values.append((row[si], row[di], row[vi]))
        frame = pd.DataFrame(values, columns=["site_no", "date", "discharge_cfs"])
        frame["date"] = pd.to_datetime(frame.date, format="mixed", errors="coerce")
        frame["discharge_cfs"] = pd.to_numeric(frame.discharge_cfs, errors="coerce")
        frames.append(frame)
        inventory.append({"path": str(path), "sha256": sha256_file(path), "selected_station_rows": len(frame),
                          "ignored_alternative_columns": ignored})
    if not frames:
        raise FileNotFoundError("cached NWIS daily discharge required")
    daily, qc = reconcile_daily_discharge(pd.concat(frames, ignore_index=True))
    return daily, {"files": inventory, "quality_summary": qc}


def attach_sample_flow(triplets, daily):
    """Measured flow at own/reference local dates; missing data stay missing."""
    if daily.duplicated(["site_no", "date"]).any():
        raise ValueError("one reconciled daily flow per station/day required")
    lookup = daily.set_index(["site_no", "date"]).discharge_cfs
    f = triplets.copy()
    for role, station in (("a", "source_a"), ("b", "source_b"), ("receiver", "target")):
        for when, date in (("sample", f"sample_date_{role}"), ("receiver_date", "sample_date_receiver")):
            ix = pd.MultiIndex.from_arrays([f[station], f[date]])
            f[f"flow_{role}_{when}_cfs"] = lookup.reindex(ix).to_numpy()
        own, ref = f[f"flow_{role}_sample_cfs"], f[f"flow_{role}_receiver_date_cfs"]
        positive = own.gt(0) & ref.gt(0)
        f[f"flow_{role}_date_log_change"] = np.log(own.where(positive)/ref.where(positive))
    names = [f"flow_{r}_sample_cfs" for r in ("a", "b", "receiver")]
    f["all_three_sample_flows_measured"] = f[names].notna().all(axis=1)
    f["all_three_sample_flows_positive"] = f[names].gt(0).all(axis=1)
    return f
