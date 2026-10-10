"""Unit-explicit, unfilled inputs for independent river outlet comparisons."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_clock(frame):
    """Collapse equal measurements; mask conflicting fields, not arbitrary rows."""
    rows, conflicts = [], []
    for clock, group in frame.groupby("clock", sort=True):
        row = {"clock": clock}
        for field in ("q_m3_s", "doc_mg_l"):
            values = group[field].dropna().unique()
            row[field] = values[0] if len(values) == 1 else np.nan
            if len(values) > 1:
                conflicts.append({"clock": clock, "field": field, "n_distinct": len(values)})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(conflicts)


def normalize(frame, case, sample_minutes, area_km2, clock_basis, processing):
    source_count = len(frame)
    duplicate_rows = int(frame.clock.duplicated().sum())
    negative_doc = int(frame.doc_mg_l.lt(0).sum())
    negative_flow = int(frame.q_m3_s.lt(0).sum())
    frame = frame.copy()
    frame.loc[~np.isfinite(frame.doc_mg_l) | frame.doc_mg_l.lt(0), "doc_mg_l"] = np.nan
    frame.loc[~np.isfinite(frame.q_m3_s) | frame.q_m3_s.lt(0), "q_m3_s"] = np.nan
    if frame.clock.isna().any():
        raise ValueError("Invalid source timestamps")
    if duplicate_rows:
        frame, conflicts = canonical_clock(frame)
    else:
        frame = frame.sort_values("clock").reset_index(drop=True)
        conflicts = pd.DataFrame()
    # Legacy pulse operators use *_utc column names. The clocks here retain
    # their source time basis, including naive timestamps if no offset is given.
    # Exported event tables rename these fields to *_clock.
    flow = frame[["clock", "q_m3_s"]].rename(columns={"clock": "flow_timestamp_utc"})
    flow["flow_valid"] = np.isfinite(flow.q_m3_s)
    flow["flow_at_reported_cap"] = False
    doc = frame.loc[np.isfinite(frame.doc_mg_l), ["clock", "doc_mg_l"]].rename(
        columns={"clock": "timestamp_utc"})
    audit = {"case": case, "sample_minutes": sample_minutes, "area_km2": area_km2,
        "clock_basis": clock_basis, "source_processing": processing,
        "source_rows": source_count, "canonical_rows": len(frame), "duplicate_extra_rows": duplicate_rows,
        "conflicting_clock_fields": len(conflicts), "negative_doc_excluded": negative_doc,
        "negative_flow_excluded": negative_flow, "valid_doc_rows": len(doc),
        "valid_flow_rows": int(flow.flow_valid.sum()),
        "paired_valid_rows": int((np.isfinite(frame.q_m3_s) & np.isfinite(frame.doc_mg_l)).sum()),
        "zero_doc_rows_retained": int(frame.doc_mg_l.eq(0).sum()),
        "first_clock": str(frame.clock.min()), "last_clock": str(frame.clock.max()),
        "nonstandard_clock_steps": int(frame.clock.diff().dt.total_seconds().dropna().ne(sample_minutes*60).sum()),
        "median_doc_mg_l": float(doc.doc_mg_l.median())}
    return flow, doc, audit, conflicts


def read_rappbode(path):
    data = pd.read_csv(path)
    frame = pd.DataFrame({"clock": pd.to_datetime(data["Date.time"], format="mixed"),
                          "q_m3_s": data["Q.smooth"]*2.58/86.4,
                          "doc_mg_l": data["DOC.smooth"]})
    return normalize(frame, "Rappbode", 15, 2.58, "native archive clock; offset unspecified",
        "Calibrated UV-VIS DOC; drift/outlier correction; cubic-spline filling of gaps <2 h; 2.5 h moving average")


def read_pangaea_table(path):
    text = Path(path).read_text()
    if "*/" not in text:
        raise ValueError("Missing PANGAEA metadata header")
    return pd.read_csv(StringIO(text.split("*/", 1)[1].strip()), sep="\t")


def read_bouleau(paths):
    data = pd.concat([read_pangaea_table(path) for path in paths], ignore_index=True)
    if set(data.Site) != {"R01"}:
        raise ValueError("Unexpected site in Bouleau outlet archive")
    frame = pd.DataFrame({"clock": pd.to_datetime(data["Date/Time"]),
                          "q_m3_s": data["Q [m**3/h]"]/3600,
                          "doc_mg_l": data["DOC [mg/l]"]})
    result = normalize(frame, "Bouleau", 60, 2.22,
        "native archive clock; Date column labelled ESTERN TIME; no offset assigned",
        "Temperature-corrected fDOM calibrated with lab DOC; observed DOC column only; RF-predicted DOC excluded")
    result[2]["predicted_doc_rows_not_used"] = int(data["DOC [mg/l] (Predicted)"].notna().sum())
    return result


def hourly_observations(flow, doc):
    """Sample existing hourly-clock rows, without averaging or filling gaps."""
    def whole_hour(clock):
        return clock.dt.minute.eq(0) & clock.dt.second.eq(0) & clock.dt.microsecond.eq(0)
    return (flow.loc[whole_hour(flow.flow_timestamp_utc)].reset_index(drop=True),
            doc.loc[whole_hour(doc.timestamp_utc)].reset_index(drop=True))
