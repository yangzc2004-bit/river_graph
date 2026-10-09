"""Bounded, same-station chemical interpolation of support residual remainders.

The caller supplies remainders after its fixed linear support fit. These
functions do not fit that predictor, read query labels, or estimate any target
timeline statistics. Call the interpolation separately for each station.
"""
from __future__ import annotations

import numpy as np


def _active(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be aligned binary availability")
    return array.astype(bool)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def source_chemical_distance_scale(chemical, active, source_station_indices, *,
                                   source_role, max_months_per_station=24):
    """Median positive source within-station Euclidean pair distance.

    At most 24 active months per station are selected at evenly spaced indices
    in chronological active-month order, including both endpoints. The median
    pools each sampled unordered pair once. No distance across two stations is
    used. All-zero/no-pair distances return scale 1 and an explicit fallback.
    """
    if source_role != "source_training":
        raise ValueError("bandwidth definition requires source_training stations")
    chemical = np.asarray(chemical)
    active = np.asarray(active)
    if chemical.ndim != 3 or chemical.shape[-1] != 2 or active.shape != chemical.shape[:2]:
        raise ValueError("chemical/active must be aligned [station,month,2]/[station,month]")
    stations = np.asarray(source_station_indices)
    if (stations.ndim != 1 or not len(stations) or stations.dtype.kind not in "iu"
            or (stations < 0).any() or (stations >= chemical.shape[0]).any()
            or np.unique(stations).size != len(stations)):
        raise ValueError("source_station_indices must be unique valid integer indices")
    if (isinstance(max_months_per_station, (bool, np.bool_))
            or not isinstance(max_months_per_station, (int, np.integer))
            or not 2 <= max_months_per_station <= 24):
        raise ValueError("max_months_per_station must be an integer between 2 and 24")
    stations = np.sort(stations.astype(np.int64))
    # Inspect only explicit source rows, including their availability flags.
    source_active = _active(active[stations], (len(stations), chemical.shape[1]), "source active")
    positive, inventory = [], []
    for i, station in enumerate(stations):
        months = np.flatnonzero(source_active[i])
        if len(months) > max_months_per_station:
            months = months[np.linspace(0, len(months)-1, max_months_per_station, dtype=np.int64)]
        values = np.asarray(chemical[station, months], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("active source chemical coordinates must be finite")
        left, right = np.triu_indices(len(months), k=1)
        distance = np.linalg.norm(values[left]-values[right], axis=1)
        if not np.isfinite(distance).all():
            raise FloatingPointError("nonfinite source pair distance")
        nonzero = distance[distance > 0]
        positive.extend(nonzero.tolist())
        inventory.append({"station_index": int(station), "n_active_months": int(source_active[i].sum()),
            "sampled_month_indices": months.tolist(), "n_sampled_months": len(months),
            "n_pairs": len(distance), "n_positive_pairs": len(nonzero)})
    fallback = not len(positive)
    return {"version": 1, "source_role": source_role, "distance_scale": 1. if fallback else float(np.median(positive)),
        "fallback": fallback, "fallback_reason": "no_positive_within_station_distance" if fallback else None,
        "n_source_stations": len(stations), "n_active_source_stations": sum(row["n_active_months"] > 0 for row in inventory),
        "n_sampled_months": sum(row["n_sampled_months"] for row in inventory),
        "n_pairs": sum(row["n_pairs"] for row in inventory), "n_positive_pairs": len(positive),
        "max_months_per_station": int(max_months_per_station),
        "sampling": "evenly spaced indices in sorted active months, including endpoints; floor integer positions",
        "distance": "Euclidean in frozen 2D chemical coordinates; median of positive within-station unordered pairs",
        "source_station_indices": stations.tolist(), "station_inventory": inventory}


def chemical_residual_kernel(query_chemical, support_chemical, query_active, support_active,
                             support_remainder, *, source_distance_scale, bandwidth_scale=1., eta=1.):
    """Interpolate centered support remainder as an additive log correction.

    Inputs are same-station rows: [Q,2], [K,2], [Q], [K], [K]. Output is [Q].
    Only active supports participate. With fewer than two, or eta=0, correction
    is exactly zero. Inactive query rows are zero; their coordinate placeholders
    and inactive support remainders are never inspected. For eta in [0,1], the
    result is bounded by eta times the active centered remainders' range.
    """
    query, support = np.asarray(query_chemical), np.asarray(support_chemical)
    if query.ndim != 2 or query.shape[1] != 2 or support.ndim != 2 or support.shape[1] != 2:
        raise ValueError("query/support chemical coordinates must have two columns")
    qa = _active(query_active, (len(query),), "query_active")
    sa = _active(support_active, (len(support),), "support_active")
    remainder = np.asarray(support_remainder)
    if remainder.shape != (len(support),):
        raise ValueError("support_remainder must align with support rows")
    scale = _positive(source_distance_scale, "source_distance_scale")
    bandwidth = _positive(bandwidth_scale, "bandwidth_scale") * scale
    if not np.isfinite(bandwidth):
        raise ValueError("combined bandwidth must be finite")
    if isinstance(eta, (bool, np.bool_)) or not np.isscalar(eta) or not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite and in [0,1]")
    correction = np.zeros(len(query), dtype=np.float64)
    if eta == 0 or sa.sum() < 2 or not qa.any():
        return correction
    q, s, r = (np.asarray(array, dtype=np.float64) for array in (query[qa], support[sa], remainder[sa]))
    if not all(np.isfinite(array).all() for array in (q, s, r)):
        raise ValueError("active chemical coordinates and remainders must be finite")
    if np.all(s == s[0]):
        # Uniform weights cannot transfer a centered remainder. Preserve that
        # exact identity rather than returning a rounding-sized correction.
        return correction
    centered = r-r.mean()
    distance2 = np.sum(((q[:, None, :]-s[None, :, :])/bandwidth)**2, axis=-1)
    if not np.isfinite(distance2).all() or not np.isfinite(centered).all():
        raise FloatingPointError("nonfinite kernel distances or centered support remainder")
    logits = -.5*distance2
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    interpolated = float(eta)*(weights @ centered)
    # Enforce mathematical convex-combination bounds against roundoff only.
    correction[qa] = np.clip(interpolated, float(eta)*centered.min(), float(eta)*centered.max())
    return correction
