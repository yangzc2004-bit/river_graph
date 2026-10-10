"""Controlled stationary fluctuations on real tributary arrival paths."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _inputs(delays, weights):
    d, w = np.asarray(delays, float), np.asarray(weights, float)
    if (d.ndim != 1 or len(d) < 2 or w.shape != d.shape
            or not np.isfinite(np.r_[d, w]).all() or (d < 0).any()
            or (w <= 0).any() or w.sum() <= 0 or w@d <= 0):
        raise ValueError("two finite nonnegative paths and positive source shares required")
    return d, w/w.sum()


def source_geometry(reaches):
    """Recover actual cropped gauge paths; count their common suffix once."""
    if (reaches.empty or reaches.target.nunique() != 1
            or reaches.duplicated(["source_station", "comid"]).any()):
        raise ValueError("one receiver and unique source/reach pairs required")
    rows, shared = [], []
    for station, group in reaches.groupby("source_station", sort=True):
        group = group.sort_values("sequence")
        length = group.length_km.to_numpy(float)
        if (not np.isfinite(length).all() or (length < 0).any()
                or group.area_weight.nunique() != 1):
            raise ValueError("finite cropped lengths and one source area share required")
        common = group[group.shared_by_all_sources].copy()
        if common.empty or not np.array_equal(common.sequence, np.arange(common.sequence.min(), len(group))):
            raise ValueError("shared corridor must be a nonempty contiguous suffix")
        shared.append(common[["comid", "length_km", "mapped_waterbody", "waterbody_id"]].reset_index(drop=True))
        rows.append({"source_station": station, "path_km": length.sum(),
            "area_weight": group.area_weight.iloc[0],
            "source_order": int(group.source_order.iloc[0])})
    for common in shared[1:]:
        pd.testing.assert_frame_equal(common, shared[0], check_dtype=False)
    sources = pd.DataFrame(rows).sort_values("source_order").reset_index(drop=True)
    d, w = _inputs(sources.path_km, sources.area_weight)
    common = float(shared[0].length_km.sum())
    if common > d.min()+1e-8:
        raise ValueError("common corridor cannot exceed a complete source path")
    mean = float(w@d)
    sources["area_weight"] = w
    sources["normalized_delay"] = d/mean
    sources["independent_path_km"] = d-common
    return sources, {"path_mean_km": mean, "common_km": common,
        "common_fraction": common/mean,
        "common_tagged_storage_km": float(shared[0].loc[shared[0].mapped_waterbody, "length_km"].sum()),
        "n_sources": len(d), "effective_sources": 1/float(w@w),
        "path_cv": float(np.sqrt(w@((d-mean)**2))/mean)}


def filtered_exponential_covariance(separation, correlation_time, memory_time):
    """Exact covariance after a causal unit-gain exponential common filter.

    Input variance is one. Stable expressions include the equal-timescale limit;
    this is an imposed stationary process, not estimated DOC autocorrelation.
    """
    delta = np.abs(np.asarray(separation, float))
    t, tau = float(correlation_time), float(memory_time)
    if not np.isfinite(np.r_[delta.ravel(), t, tau]).all() or t <= 0 or tau < 0:
        raise ValueError("finite separations, positive correlation time and nonnegative memory required")
    if tau == 0:
        return np.exp(-delta/t)
    ratio = tau/t
    if ratio == 1:
        return .5*(1+delta/t)*np.exp(-delta/t)
    if ratio < 1:
        x = delta/t
        return np.exp(-x)*(-np.expm1(np.log(ratio)-x*(1/ratio-1)))/((1-ratio)*(1+ratio))
    x = delta/tau
    return np.exp(-x)*((ratio-1)-np.expm1(-x*(ratio-1)))/((ratio-1)*(ratio+1))


def stationary_response(delays, weights, *, correlation_time, coherence, memory_time=0.):
    """Same-source instantaneous mixture versus transported stationary variance."""
    d, w = _inputs(delays, weights)
    if not np.isfinite(coherence) or not 0 <= coherence <= 1:
        raise ValueError("source coherence must lie in [0,1]")
    if memory_time < 0 or memory_time > d.min()+1e-8:
        raise ValueError("common dispersion must replace available causal translation")
    difference = d[:, None]-d[None, :]
    raw = filtered_exponential_covariance(difference, correlation_time, 0.)
    filtered = filtered_exponential_covariance(difference, correlation_time, memory_time)
    diagonal = float(filtered_exponential_covariance(0., correlation_time, memory_time))
    square_sum = float(w@w)
    instantaneous = coherence+(1-coherence)*square_sum
    delayed = float(coherence*(w@raw@w)+(1-coherence)*square_sum)
    final = float(coherence*(w@filtered@w)+(1-coherence)*square_sum*diagonal)
    if not 0 < final <= delayed+1e-10 <= instantaneous+2e-10:
        raise ValueError("controlled positive-coherence variance violates passive bounds")
    geometry = .5*np.log(delayed/instantaneous)
    memory = .5*np.log(final/delayed)
    aligned = .5*np.log(diagonal)
    return {"instantaneous_mixture_variance": instantaneous, "delay_only_variance": delayed,
        "outlet_variance": final, "sd_ratio": np.sqrt(final/instantaneous),
        "geometry_sd_ratio": np.sqrt(delayed/instantaneous),
        "aligned_sd_ratio": np.sqrt(diagonal), "log_sd_ratio": geometry+memory,
        "geometry_log_change": geometry, "shared_memory_log_change": memory,
        "aligned_memory_log_change": aligned, "interaction_log_change": memory-aligned,
        "path_mean": float(w@d), "nominal_kernel_mean": float(memory_time),
        "minimum_remaining_translation": float(d.min()-memory_time), "steady_gain": 1.}


def harmonic_response(delays, weights, source_phases, *, period, memory_time=0.):
    """A causal translated common filter, with fixed upstream sinusoidal waves."""
    d, w = _inputs(delays, weights)
    phase = np.asarray(source_phases, float)
    if (phase.shape != d.shape or not np.isfinite(phase).all() or not np.isfinite(period)
            or period <= 0 or not np.isfinite(memory_time) or memory_time < 0
            or memory_time > d.min()+1e-8):
        raise ValueError("finite source phases, positive period and feasible causal memory required")
    omega = 2*np.pi/period
    source = np.exp(1j*phase)
    transfer = np.exp(-1j*omega*(d-memory_time))/(1+1j*omega*memory_time)
    mixture, outlet = complex(w@source), complex(w@(source*transfer))
    amplitude = abs(mixture)
    return {"mixture_complex": mixture, "outlet_complex": outlet,
        "mixture_amplitude": amplitude, "outlet_amplitude": abs(outlet),
        "amplitude_ratio": abs(outlet)/amplitude if amplitude > 1e-8 else np.nan,
        "reference_near_cancellation": amplitude <= 1e-8, "omega": omega,
        "steady_gain": 1., "individual_path_means": d.copy()}
