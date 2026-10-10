"""Describe paired hourly waveforms without inventing missing event limbs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def waveform_profile(times, values) -> dict:
    """Sampled peak interval and adjacent-sample half-height crossings.

    The reference is the minimum of this observed segment, not estimated
    baseflow. Multiple lobes and boundary censoring remain explicit.
    """
    clock = pd.DatetimeIndex(times)
    y = np.asarray(values, dtype=float)
    if len(clock) != len(y) or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("At least three finite, aligned hourly observations are required")
    if not (clock[1:] - clock[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Hourly segment must be contiguous; do not bridge gaps")
    x = np.arange(len(y), dtype=float)
    baseline, peak = float(y.min()), float(y.max())
    positions = np.flatnonzero(y == peak)
    first, last = int(positions[0]), int(positions[-1])
    contiguous_peak = len(positions) == last - first + 1
    result = {
        "n_records": len(y), "duration_hours": len(y) - 1,
        "segment_minimum": baseline, "sampled_peak": peak,
        "sampled_range": peak - baseline, "peak_first": clock[first],
        "peak_last": clock[last], "n_tied_peak_samples": len(positions),
        "peak_indices_contiguous": bool(contiguous_peak),
        "peak_at_boundary": bool(first == 0 or last == len(y) - 1),
        "half_height": baseline + .5 * (peak - baseline),
        "n_half_height_lobes": 0, "lobe_first_index": -1,
        "lobe_last_index": -1, "rising_crossing": pd.NaT,
        "falling_crossing": pd.NaT, "width_hours": np.nan,
        "rise_to_first_peak_hours": np.nan,
        "last_peak_to_fall_hours": np.nan,
        "width_status": "flat_segment",
    }
    if peak <= baseline:
        return result
    above = y >= result["half_height"]
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    ends = np.flatnonzero(above & ~np.r_[above[1:], False])
    result["n_half_height_lobes"] = len(starts)
    if not contiguous_peak:
        result["width_status"] = "separated_equal_maxima"
        return result
    k = np.flatnonzero((starts <= first) & (ends >= last))
    if len(k) != 1:
        raise ValueError("Dominant peak does not belong to exactly one lobe")
    left, right = int(starts[k[0]]), int(ends[k[0]])
    result.update(lobe_first_index=left, lobe_last_index=right)
    if left == 0 or right == len(y) - 1:
        result["width_status"] = ("both_limbs_censored" if left == 0 and right == len(y) - 1
                                  else "rising_limb_censored" if left == 0
                                  else "falling_limb_censored")
        return result
    height = result["half_height"]
    rise = x[left - 1] + (height - y[left - 1]) / (y[left] - y[left - 1])
    fall = x[right] + (height - y[right]) / (y[right + 1] - y[right])
    result.update(
        rising_crossing=clock[0] + pd.Timedelta(hours=rise),
        falling_crossing=clock[0] + pd.Timedelta(hours=fall),
        width_hours=float(fall - rise), rise_to_first_peak_hours=float(first - rise),
        last_peak_to_fall_hours=float(fall - last), width_status="observed_crossings",
    )
    return result


def peak_difference(upstream: dict, downstream: dict) -> dict:
    """Downstream-minus-upstream sampled peak bounds, respecting all ties."""
    lower = (downstream["peak_first"] - upstream["peak_last"]).total_seconds() / 3600
    upper = (downstream["peak_last"] - upstream["peak_first"]).total_seconds() / 3600
    usable = (not upstream["peak_at_boundary"] and not downstream["peak_at_boundary"]
              and upstream["peak_indices_contiguous"] and downstream["peak_indices_contiguous"]
              and upstream["sampled_range"] > 0 and downstream["sampled_range"] > 0)
    return {"peak_difference_lower_hours": lower, "peak_difference_upper_hours": upper,
            "interior_peak_pair": bool(usable)}


def optical_quality(profile: dict, values, turbidity, threshold=600.0, source_lab_doc_max=3.88) -> dict:
    """Flag the published regression regime without selecting a replacement peak."""
    y, turb = np.asarray(values, dtype=float), np.asarray(turbidity, dtype=float)
    if len(y) != profile["n_records"] or len(turb) != len(y):
        raise ValueError("Turbidity and waveform must be aligned")
    peaks = y == profile["sampled_peak"]
    peak_turb = turb[peaks]
    unknown = not np.isfinite(peak_turb).all()
    retrieval = bool(np.any(peak_turb > threshold))
    peak_status = ("unknown_turbidity" if unknown else
                   "reported_flow_rainfall_retrieval_regime" if retrieval else
                   "outside_reported_retrieval_regime")
    width_ready = profile["width_status"] == "observed_crossings"
    support = turb[max(0, profile["lobe_first_index"] - 1):profile["lobe_last_index"] + 2] if width_ready else np.array([])
    width_unknown = width_ready and not np.isfinite(support).all()
    width_retrieval = width_ready and bool(np.any(support > threshold))
    return {
        "n_turbidity_gt600": int(np.sum(turb > threshold)),
        "n_turbidity_le40": int(np.sum(turb <= 40)),
        "n_missing_turbidity": int(np.sum(~np.isfinite(turb))),
        "peak_turbidity_min": float(np.min(peak_turb)) if not unknown else np.nan,
        "peak_turbidity_max": float(np.max(peak_turb)) if not unknown else np.nan,
        "peak_quality_status": peak_status,
        "peak_above_source_lab_doc_range": bool(profile["sampled_peak"] > source_lab_doc_max),
        "peak_outside_reported_retrieval_regime": bool(not unknown and not retrieval),
        "lobe_in_reported_retrieval_regime": bool(width_retrieval),
        "optical_only_width_available": bool(width_ready and not width_unknown and not width_retrieval),
        "optical_only_width_hours": profile["width_hours"] if width_ready and not width_unknown and not width_retrieval else np.nan,
    }


def shifted_flow_shape(times, upstream, downstream, offset_hours: int) -> dict:
    """Fit a descriptive scale at a specified sampled peak offset; no lag search."""
    if int(offset_hours) != offset_hours:
        raise ValueError("Flow-shape diagnostic uses an integer-hour sampled offset")
    clock = pd.DatetimeIndex(times)
    a = pd.Series(np.asarray(upstream, dtype=float), index=clock + pd.Timedelta(hours=int(offset_hours)))
    b = pd.Series(np.asarray(downstream, dtype=float), index=clock)
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    result = {"shifted_n_pairs": len(paired), "shape_offset_hours": int(offset_hours),
              "shifted_flow_correlation": np.nan, "shifted_flow_scale": np.nan,
              "shifted_flow_intercept": np.nan, "range_normalized_rmse": np.nan}
    if len(paired) < 3 or paired.a.max() == paired.a.min() or paired.b.max() == paired.b.min():
        return result
    design = np.column_stack([paired.a, np.ones(len(paired))])
    scale, intercept = np.linalg.lstsq(design, paired.b, rcond=None)[0]
    rmse = np.sqrt(np.mean((paired.b.to_numpy() - design @ [scale, intercept]) ** 2))
    result.update(shifted_flow_correlation=float(paired.a.corr(paired.b)),
                  shifted_flow_scale=float(scale), shifted_flow_intercept=float(intercept),
                  range_normalized_rmse=float(rmse / (paired.b.max() - paired.b.min())))
    return result
