"""Audit optical DOC archives without turning record coverage into field evidence."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def normalize_watershed(value):
    """Retain source identity despite inconsistent storm-table punctuation."""
    if str(value).startswith("Kuparuk"):
        return "Kuparuk River"
    if str(value).startswith("Oksrukuyik"):
        return "Oksrukuyik Creek"
    raise ValueError(f"Unknown watershed: {value}")


def canonical_doc(raw):
    """Coalesce equal values; conflicting timestamp values remain unresolved."""
    frame = raw.copy()
    frame["watershed"] = frame.Watershed.map(normalize_watershed)
    frame["timestamp"] = pd.to_datetime(frame.DateTime_AKDT, format="mixed", errors="raise")
    if not frame.timestamp.dt.year.eq(frame.Year).all():
        raise ValueError("Measurement year disagrees with timestamp")
    frame["doc"] = pd.to_numeric(frame["DOC_mg.L"], errors="raise")
    rows, duplicates = [], []
    for (site, time), g in frame.groupby(["watershed", "timestamp"], sort=True):
        values = g.doc.loc[np.isfinite(g.doc)].unique()
        conflict = len(values) > 1
        row = {"watershed": site, "timestamp": time, "year": time.year,
               "doc_mg_l": float(values[0]) if len(values) == 1 else np.nan,
               "n_source_rows": len(g), "n_distinct_finite_doc": len(values),
               "doc_conflict": conflict,
               "notes": " | ".join(sorted(set(g.Notes.dropna().astype(str)))),
               "monitoring_site": " | ".join(sorted(set(g.Monitoring_Site.astype(str))))}
        rows.append(row)
        if len(g) > 1:
            duplicates.append({**row, "source_row_indices": ",".join(map(str, g.index)),
                               "alternative_doc_values": ",".join(map(str, values))})
    return pd.DataFrame(rows), pd.DataFrame(duplicates)


def longest_exact_sequence(a, b, *, window=10):
    """Longest varying exact ordered sequence; missing cells are removed upstream.

    This is a record-reuse diagnostic, not a correction of the later archive.
    Constant windows are ignored because identical flat readings are ambiguous.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Sequence diagnostic requires finite values")
    index = defaultdict(list)
    for i, value in enumerate(a):
        index[value].append(i)
    best = (0, None, None)
    previous = {}
    for j, value in enumerate(b):
        current = {}
        for i in index.get(value, ()):
            n = previous.get(i-1, 0)+1
            current[i] = n
            if n >= window and n > best[0] and len(set(a[i-n+1:i-n+1+window])) >= 3:
                best = n, i-n+1, j-n+1
        previous = current
    return best


def exact_sequence_audit(series):
    rows = []
    for site, group in series.groupby("watershed"):
        years = {year: g.loc[np.isfinite(g.doc_mg_l)].sort_values("timestamp").reset_index(drop=True)
                 for year, g in group.groupby("year")}
        for year_a, a in years.items():
            for year_b, b in years.items():
                if year_a >= year_b or a.empty or b.empty:
                    continue
                n, i, j = longest_exact_sequence(a.doc_mg_l, b.doc_mg_l)
                row = {"watershed": site, "earlier_year": year_a, "later_year": year_b,
                       "n_exact_shared_later_values": int(np.isin(b.doc_mg_l, a.doc_mg_l).sum()),
                       "n_later_valid": len(b), "longest_exact_sequence_values": n,
                       "requires_source_resolution": n >= 24}
                if n:
                    row.update(earlier_start=a.timestamp.iloc[i], earlier_end=a.timestamp.iloc[i+n-1],
                               later_start=b.timestamp.iloc[j], later_end=b.timestamp.iloc[j+n-1],
                               earlier_index=i, later_index=j,
                               sequence_unique_values=a.doc_mg_l.iloc[i:i+n].nunique())
                rows.append(row)
    return pd.DataFrame(rows)


def interval_coverage(series, start, end, *, minutes=15):
    """Occupied nominal time bins, plus gaps including both event boundaries.

    Binning evaluates coverage only. It does not interpolate or average DOC.
    A faster combined logger grid cannot inflate the fraction above one.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start or minutes <= 0:
        raise ValueError("Positive event duration and nominal cadence required")
    part = series.loc[series.timestamp.between(start, end)].sort_values("timestamp")
    valid = part.loc[np.isfinite(part.doc_mg_l) & (part.doc_mg_l >= 0)]
    elapsed = (end-start).total_seconds()/60
    expected = int(np.floor(elapsed/minutes))+1
    slots = np.floor((valid.timestamp-start).dt.total_seconds().to_numpy()/60/minutes).astype(int)
    occupied = len(np.unique(slots))
    clock = pd.DatetimeIndex([start, *valid.timestamp, end]).sort_values()
    # pandas may store microseconds or nanoseconds; do not assume asi8 units.
    gaps = (clock[1:]-clock[:-1]).total_seconds()/3600
    maximum_gap = float(gaps.max())
    ok = elapsed >= 24*60 and occupied/expected >= .9 and maximum_gap <= 1
    return {"elapsed_hours": elapsed/60, "n_archive_timestamps": len(part),
            "n_valid_doc": len(valid), "n_conflicting_timestamps": int(part.doc_conflict.sum()),
            "expected_nominal_bins": expected, "occupied_nominal_bins": occupied,
            "nominal_doc_coverage": occupied/expected,
            "max_gap_including_boundaries_hours": maximum_gap,
            "doc_coverage_qualified": bool(ok),
            "n_flood_note_timestamps": int(part.notes.str.contains("Major flood", regex=False).sum())}


def year_inventory(series):
    rows = []
    for (site, year), g in series.groupby(["watershed", "year"]):
        good = g.loc[np.isfinite(g.doc_mg_l)].sort_values("timestamp")
        all_dt = g.timestamp.sort_values().diff().dt.total_seconds()/60
        doc_dt = good.timestamp.diff().dt.total_seconds()/60
        rows.append({"watershed": site, "year": year, "n_source_rows": int(g.n_source_rows.sum()),
                     "n_unique_timestamps": len(g), "n_finite_doc": len(good),
                     "n_conflicting_timestamps": int(g.doc_conflict.sum()),
                     "combined_logger_modal_minutes": float(all_dt.mode().iloc[0]),
                     "finite_doc_modal_minutes": float(doc_dt.mode().iloc[0]) if len(good) > 1 else np.nan,
                     "doc_min_mg_l": good.doc_mg_l.min(), "doc_max_mg_l": good.doc_mg_l.max(),
                     "first_doc": good.timestamp.min(), "last_doc": good.timestamp.max(),
                     "monitoring_site": " | ".join(g.monitoring_site.unique())})
    return pd.DataFrame(rows)
