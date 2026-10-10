"""Exact passive pulse experiments separating arrival differences and spreading."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import exponnorm, norm


def pulse_response(delays, weights, *, sigma, memory=0., offsets=None):
    """Gaussian source pulse and causal unit-gain exponential corridor kernel.

    A memory allocation replaces pure translation by an exponential delay of
    equal mean. Each path's mean remains its prescribed delay. Clocks may be
    offset; paths and the filter always remain causal. Time units are imposed,
    not fitted to DOC. Peak is a continuous mixture maximum, not a sampled DOC
    concentration. Quantile duration concerns positive anomaly mass.
    """
    d, w = np.asarray(delays, float), np.asarray(weights, float)
    o = np.zeros_like(d) if offsets is None else np.asarray(offsets, float)
    if (d.ndim != 1 or not len(d) or w.shape != d.shape or o.shape != d.shape
            or not np.isfinite(np.r_[d, w, o, sigma, memory]).all()
            or (d < 0).any() or (w <= 0).any() or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)
            or sigma <= 0 or memory < 0 or memory > d.min()+1e-10):
        raise ValueError("finite aligned paths, unit positive weights and feasible causal memory required")
    mean_arrivals = d + o
    location = mean_arrivals - memory
    scale = float(sigma * np.sqrt(2*np.pi))

    def pdf(t):
        x = np.asarray(t)[..., None]
        f = (exponnorm.pdf(x, memory/sigma, loc=location, scale=sigma)
             if memory else norm.pdf(x, loc=location, scale=sigma))
        return f @ w * scale

    def cdf(t):
        x = np.asarray(t)[..., None]
        f = (exponnorm.cdf(x, memory/sigma, loc=location, scale=sigma)
             if memory else norm.cdf(x, loc=location, scale=sigma))
        return f @ w

    lo, hi = float(location.min()-8*sigma), float(mean_arrivals.max()+8*sigma+32*memory)
    # Include all source-centred grids so a narrow secondary mode is not lost
    # when paths are very different. Then refine every candidate local maximum.
    centres = np.unique(np.r_[np.linspace(lo, hi, 256),
        (location[:, None] + np.linspace(-5*sigma, 5*sigma+4*memory, 100)).ravel()])
    values = pdf(centres)
    indices = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])) + 1
    peaks = [(float(values.max()), float(centres[np.argmax(values)]))]
    for j in indices:
        result = minimize_scalar(lambda x: -float(pdf(x)), bounds=(centres[j-1], centres[j+1]),
            method="bounded", options={"xatol": 1e-11})
        peaks.append((-float(result.fun), float(result.x)))
    peak, peak_time = max(peaks, key=lambda p: (p[0], -p[1]))
    quantiles = [brentq(lambda t, probability=p: float(cdf(t))-probability, lo, hi, xtol=1e-11) for p in (.1, .9)]
    mean = float(w @ mean_arrivals)
    arrival_variance = float(w @ ((mean_arrivals-mean)**2))
    metrics = {"pulse_peak": peak, "pulse_peak_time": peak_time, "pulse_centroid": mean,
        "pulse_sd": float(np.sqrt(sigma**2+arrival_variance+memory**2)),
        "pulse_central80": quantiles[1]-quantiles[0],
        "input_variance": sigma**2, "arrival_variance": arrival_variance,
        "corridor_variance": memory**2, "anomaly_mass_fraction": 1., "steady_gain": 1.,
        "memory_time": memory, "minimum_translation": float(d.min()-memory)}
    return metrics, pdf, (lo, hi)


def junction_translation(total_paths, common_path):
    """A moved confluence changes a partition, not complete constant-speed paths."""
    d = np.asarray(total_paths, float)
    if (d.ndim != 1 or not len(d) or not np.isfinite(np.r_[d, common_path]).all()
            or common_path < 0 or common_path > d.min()+1e-10):
        raise ValueError("common corridor must fit in every complete path")
    return (d-common_path) + common_path


def source_dates(events):
    """Sampling support, unaffected by chemical magnitudes or row order."""
    if events[["site_no", "event_id"]].duplicated().any():
        raise ValueError("one row per station activity required")
    import pandas as pd

    return {s: pd.DatetimeIndex(g.date.unique()).sort_values() for s, g in events.groupby("site_no")}


def station_coordinate_fields(columns):
    """Use actual WQP3 standardized coordinates, with documented legacy fallback."""
    for lat, lon in (("Location_LatitudeStandardized", "Location_LongitudeStandardized"),
                     ("Location_Latitude", "Location_Longitude")):
        if {lat, lon}.issubset(columns):
            return lat, lon
    raise ValueError("station catalogue lacks supported coordinate columns")
