"""Original-point isotope/transport diagnostics for co-released river tracers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

STANDARD_RATIO = 0.0112372


def clock_seconds(time):
    values = np.asarray(time, dtype=float) * 60
    rounded = np.rint(values)
    if not np.all(np.isfinite(values)) or not np.allclose(values, rounded, rtol=0, atol=1e-6):
        raise ValueError("Source clocks are not recorded at whole-second precision")
    return rounded.astype(np.int64)


def atom_fraction(delta):
    values = np.asarray(delta, dtype=float)
    if np.any(values[np.isfinite(values)] < -1000):
        raise ValueError("Isotope delta gives a negative isotope ratio")
    ratio = STANDARD_RATIO * (1 + values / 1000)
    return ratio / (1 + ratio)


def resolve_clock_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain unresolved salt conflicts and average actual carbon replicates."""
    rows = []
    for time, group in frame.groupby("time_min", sort=True):
        salt = group.spc_raw.dropna().unique()
        row = {"time_min": time, "n_original_records": len(group),
               "record_ids": "|".join(group.record_id.astype(str)),
               "n_lab_records": int(group.is_lab_record.sum()),
               "n_lab_measured": int(group.doc_label_raw.notna().sum()),
               "salt_conflict": len(salt) > 1,
               "spc_raw": salt[0] if len(salt) == 1 else np.nan,
               "doc_label_raw": group.doc_label_raw.mean(),
               "doc_total_raw": group.doc_total_raw.mean(),
               "source_notes": " | ".join(group.source_notes.dropna().astype(str).unique())}
        rows.append(row)
    return pd.DataFrame(rows)


def pulse_core_metrics(time, reference, observed, fraction=0.25, stop=300):
    if not 0 < fraction <= 1 or stop <= 0:
        raise ValueError("Invalid pulse core definition")
    time, ref, obs = (np.asarray(x, dtype=float) for x in (time, reference, observed))
    if not (time.shape == ref.shape == obs.shape):
        raise ValueError("Unaligned tracer points")
    valid = np.isfinite(time) & np.isfinite(ref) & (time >= 0) & (time <= stop)
    max_ref = np.max(ref[valid], initial=0)
    core = valid & np.isfinite(obs) & (ref >= fraction * max_ref) & (ref > 0)
    if not core.any():
        return {"n_core_points": 0, "core_slope": np.nan, "core_ratio_median": np.nan,
                "core_first_min": np.nan, "core_last_min": np.nan}
    return {"n_core_points": int(core.sum()),
            "core_slope": float(np.dot(ref[core], obs[core]) / np.dot(ref[core], ref[core])),
            "core_ratio_median": float(np.median(obs[core] / ref[core])),
            "core_first_min": float(time[core].min()), "core_last_min": float(time[core].max())}


def bounded_response(time, values, max_gap=30, stop=300):
    """Integrate only adjacent valid records; no bridging missing clock rows."""
    t, y = (np.asarray(x, dtype=float) for x in (time, values))
    if t.ndim != 1 or t.shape != y.shape or not np.all(np.isfinite(t)):
        raise ValueError("Invalid response clock")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("Response clock must be unique and increasing")
    if max_gap <= 0 or stop <= 0:
        raise ValueError("Invalid integration limits")
    inside = (t >= 0) & (t <= stop)
    t, y = t[inside], y[inside]
    if len(t) < 2:
        return {"n_points": int(np.isfinite(y).sum()), "area": np.nan,
                "centroid_min": np.nan, "duration80_min": np.nan,
                "covered_min": 0.0, "span_min": 0.0, "gap_count": 0,
                "boundary_peak_ratio": np.nan, "sampled_peak": np.nan,
                "sampled_peak_first_min": np.nan, "sampled_peak_last_min": np.nan}
    finite = np.isfinite(y)
    positive = np.maximum(y, 0)
    dt = np.diff(t)
    accepted = finite[:-1] & finite[1:] & (dt <= max_gap)
    left, width = t[:-1][accepted], dt[accepted]
    a, b = positive[:-1][accepted], positive[1:][accepted]
    mass = width * (a + b) / 2
    total = float(mass.sum())
    first_moment = np.sum(left * mass + width**2 * (a / 6 + b / 3))

    def quantile(p):
        if total <= 0:
            return np.nan
        cumulative = np.cumsum(mass)
        index = int(np.searchsorted(cumulative, p * total, side="left"))
        prior = cumulative[index - 1] if index else 0.0
        target = p * total - prior
        partial = lambda x: a[index] * x + (b[index] - a[index]) * x*x / (2*width[index]) - target
        return float(left[index] + brentq(partial, 0, width[index]))

    peak = float(np.nanmax(positive)) if finite.any() else np.nan
    peak_clock = t[finite & np.isclose(positive, peak, rtol=0, atol=1e-12)]
    bounds = positive[[0, -1]]
    boundary = float(np.nanmax(bounds) / peak) if peak > 0 and np.isfinite(bounds).any() else np.nan
    return {"n_points": int(finite.sum()), "area": total,
            "centroid_min": float(first_moment / total) if total > 0 else np.nan,
            "duration80_min": quantile(0.9) - quantile(0.1),
            "covered_min": float(width.sum()), "span_min": float(t[-1] - t[0]),
            "gap_count": int((~accepted).sum()), "boundary_peak_ratio": boundary,
            "sampled_peak": peak,
            "sampled_peak_first_min": float(peak_clock.min()) if len(peak_clock) else np.nan,
            "sampled_peak_last_min": float(peak_clock.max()) if len(peak_clock) else np.nan}


def paired_joint_area(time, reference, observed, max_gap=30, stop=300):
    """Use precisely the same supported segments for both solutes."""
    t, ref, obs = (np.asarray(x, dtype=float) for x in (time, reference, observed))
    if not (t.shape == ref.shape == obs.shape):
        raise ValueError("Unaligned paired response")
    joint = np.isfinite(ref) & np.isfinite(obs)
    ref = np.where(joint, ref, np.nan)
    obs = np.where(joint, obs, np.nan)
    r, o = bounded_response(t, ref, max_gap, stop), bounded_response(t, obs, max_gap, stop)
    return {"joint_reference_area": r["area"], "joint_doc_area": o["area"],
            "joint_doc_reference_fraction": o["area"] / r["area"] if r["area"] > 0 else np.nan,
            "joint_covered_min": r["covered_min"], "joint_gap_count": r["gap_count"]}
