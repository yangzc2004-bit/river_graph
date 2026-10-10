"""Clock-explicit joins and flow-selected windows for the Kervidy field case."""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOW_CAP_DM3_S = 1238.414


def prepare_flow(source):
    frame = source.copy()
    frame["flow_timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    if frame.flow_timestamp_utc.duplicated().any():
        raise ValueError("Duplicate source flow timestamps")
    frame["q_dm3_s"] = pd.to_numeric(frame.q_dm3_s, errors="raise")
    frame["flow_valid"] = np.isfinite(frame.q_dm3_s) & frame.q_dm3_s.ge(0)
    frame["q_m3_s"] = frame.q_dm3_s * .001
    # Method PDF reports the cap to 0.001 L/s. Two exported values have extra
    # decimal digits within that rounding interval. Flag them; do not clip.
    frame["flow_at_reported_cap"] = np.isclose(frame.q_dm3_s, FLOW_CAP_DM3_S, rtol=0, atol=.0005)
    return frame.sort_values("flow_timestamp_utc").reset_index(drop=True)


def join_doc_flow(doc, flow, tolerance_minutes=2):
    """Attach an existing flow observation; retain its timestamp and offset."""
    left = doc.copy()
    left["timestamp_utc"] = pd.to_datetime(left.timestamp_utc, utc=True)
    if left.timestamp_utc.duplicated().any():
        raise ValueError("Duplicate source DOC timestamps")
    right = flow.loc[flow.flow_valid, ["flow_timestamp_utc", "q_m3_s", "flow_at_reported_cap"]]
    out = pd.merge_asof(left.sort_values("timestamp_utc"), right.sort_values("flow_timestamp_utc"),
                        left_on="timestamp_utc", right_on="flow_timestamp_utc",
                        direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes))
    out["flow_offset_minutes"] = (out.flow_timestamp_utc-out.timestamp_utc).dt.total_seconds()/60
    out["flow_match"] = out.flow_timestamp_utc.notna()
    out["flow_exact_match"] = out.flow_match & out.flow_offset_minutes.eq(0)
    out["flow_at_reported_cap"] = out.flow_at_reported_cap.eq(True)
    return out.reset_index(drop=True)


def occupied_record_coverage(timestamps, start, end, sample_minutes=15):
    """Count occupied sampling bins, with gaps including both boundaries."""
    if sample_minutes <= 0:
        raise ValueError("Sampling interval must be positive")
    seconds = sample_minutes*60
    t = pd.DatetimeIndex(timestamps).sort_values().unique()
    t = t[(t >= start) & (t <= end)]
    expected = int(np.floor((end-start).total_seconds()/seconds)) + 1
    occupied = len(np.unique(np.floor((t-start).total_seconds()/seconds).astype(int)))
    augmented = pd.DatetimeIndex([start]).append(t).append(pd.DatetimeIndex([end]))
    # Storage may be microseconds rather than nanoseconds; never infer the unit.
    max_gap = float(pd.Series(augmented).diff().dt.total_seconds().max()/3600)
    return {"occupied_bins": occupied, "expected_bins": expected,
            "coverage": occupied/expected, "max_gap_hours": max_gap}


def flow_selected_windows(flow, doc_first, doc_last):
    """Use reported flow only, and keep the earliest timestamp for equal maxima."""
    candidates = flow.loc[flow.flow_valid & flow.q_m3_s.gt(0) &
        flow.flow_timestamp_utc.between(doc_first, doc_last)]
    rows = []
    for year, frame in candidates.groupby(candidates.flow_timestamp_utc.dt.year):
        maximum = frame.q_m3_s.max()
        peaks = frame.loc[frame.q_m3_s.eq(maximum)].sort_values("flow_timestamp_utc")
        peak = peaks.iloc[0]
        rows.append({"year": year, "selected_flow_peak_utc": peak.flow_timestamp_utc,
            "q_peak_m3_s": maximum, "reported_flow_max_censored": bool(peak.flow_at_reported_cap),
            "n_equal_maximum_records": len(peaks),
            "window_start_utc": peak.flow_timestamp_utc-pd.Timedelta(days=3),
            "window_end_utc": peak.flow_timestamp_utc+pd.Timedelta(days=4),
            "partial_observation_year": doc_first.year == year or doc_last.year == year})
    return pd.DataFrame(rows)
