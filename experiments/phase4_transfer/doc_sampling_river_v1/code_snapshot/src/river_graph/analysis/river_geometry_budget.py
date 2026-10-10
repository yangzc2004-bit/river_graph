"""Separate water routing, DOC pulse redistribution and carbon loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import gamma


def _delay_kernel(delay, dt):
    if delay < 0 or not np.isfinite(delay):
        raise ValueError("Travel time must be finite and causal")
    position = delay / dt
    low = int(np.floor(position))
    kernel = np.zeros(low + 2)
    kernel[low] = 1 - (position - low)
    kernel[low + 1] = position - low
    return kernel


def _shared_kernel(mean, dt, dispersed):
    if not dispersed or mean == 0:
        return _delay_kernel(mean, dt)
    # Bin-integrated gamma probabilities; numerical tails are normalized.
    end = gamma.ppf(1 - 1e-12, a=4, scale=mean / 4)
    boundaries = np.arange(0, end + dt, dt)
    mass = np.diff(gamma.cdf(boundaries, a=4, scale=mean / 4))
    mass /= mass.sum()
    # Probability bins are located at their midpoints, with linear deposition.
    kernel = np.zeros(len(mass) + 1)
    kernel[:-1] += .5 * mass
    kernel[1:] += .5 * mass
    return kernel


def route_water_carbon(paths, weights, *, common=0., path_factor=1.,
                       aligned=False, dispersed=False, reaction_rate=0.,
                       water_amplitude=3., sigma=.15, dt=.005,
                       concentration_amplitude=1., baseline=5.):
    """Route one pulse at each source, preserving water/carbon budgets.

    Lengths and times are scenario units. A shared gamma kernel is an imposed
    transport process, not an inferred hydraulic residence distribution.
    Reaction acts only on the additional DOC pulse, not the baseline pool.
    """
    paths, weights = np.asarray(paths, float), np.asarray(weights, float)
    if (paths.ndim != 1 or paths.shape != weights.shape or not len(paths)
            or not np.isfinite(np.r_[paths, weights]).all()
            or (paths < 0).any() or (weights <= 0).any()
            or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("Causal source paths and positive unit-sum flow shares required")
    parameters = [common, path_factor, reaction_rate, water_amplitude, sigma, dt,
                  concentration_amplitude, baseline]
    if (not np.isfinite(parameters).all() or min(sigma, dt, baseline) <= 0
            or min(common, reaction_rate, water_amplitude, concentration_amplitude) < 0
            or not 0 <= path_factor <= 1):
        raise ValueError("Invalid routing/forcing parameters")
    mean = float(weights @ paths)
    total = mean + path_factor * (paths - mean)
    if common > total.min() + 1e-12:
        raise ValueError("Shared segment cannot exceed a source-to-outlet path")
    branches = np.maximum(total - common, 0)
    offsets = mean - total if aligned else np.zeros_like(paths)
    start = np.floor((min(0., offsets.min()) - 8 * sigma) / dt) * dt
    stop = np.ceil((max(0., offsets.max()) + 8 * sigma) / dt) * dt
    source_time = start + np.arange(round((stop - start) / dt) + 1) * dt
    shared = _shared_kernel(common, dt, dispersed)
    # Survival integrates exp(-k*time) against the same travel distribution.
    reactive_shared = shared * np.exp(-reaction_rate * np.arange(len(shared)) * dt)
    length = len(source_time) + max(len(_delay_kernel(d, dt)) for d in branches) + len(shared) - 2
    routed_water, routed_carbon = np.zeros(length), np.zeros(length)
    input_water, input_carbon = 0., 0.
    for weight, delay, offset in zip(weights, branches, offsets, strict=True):
        pulse = np.exp(-.5 * ((source_time - offset) / sigma) ** 2)
        water = weight * water_amplitude * pulse
        carbon = weight * (1 + water_amplitude * pulse) * concentration_amplitude * pulse
        if dispersed:
            kernel = _delay_kernel(delay, dt)
            q = fftconvolve(fftconvolve(water, kernel), shared)
            j = fftconvolve(fftconvolve(carbon, kernel), reactive_shared)
        else:
            # Compose deterministic distances before discretization. Depositing
            # two fractional shifts separately introduces artificial spreading.
            kernel = _delay_kernel(delay + common, dt)
            q = fftconvolve(water, kernel)
            j = fftconvolve(carbon, kernel) * np.exp(-reaction_rate * common)
        routed_water[:len(q)] += np.maximum(q, 0.)
        routed_carbon[:len(j)] += np.maximum(j, 0.)
        input_water += float(water.sum() * dt)
        input_carbon += float(carbon.sum() * dt)
    time = start + np.arange(length) * dt
    q = 1 + routed_water
    c = baseline + routed_carbon / q
    flux = baseline * q + routed_carbon
    mass = routed_carbon.sum() * dt
    centroid = float(time @ routed_carbon / routed_carbon.sum()) if mass > 0 else np.nan
    sd = float(np.sqrt((time - centroid) ** 2 @ routed_carbon / routed_carbon.sum())) if mass > 0 else np.nan
    peak = int(np.argmax(c))
    half = baseline + .5 * (c[peak] - baseline)
    left = np.flatnonzero(c[:peak + 1] <= half)
    right = np.flatnonzero(c[peak:] <= half)
    width = (time[peak + right[0]] - time[left[-1]]) if len(left) and len(right) else np.nan
    metrics = {
        "concentration_peak_excess": float(c.max() - baseline),
        "concentration_peak_time": float(time[peak]),
        "concentration_flow_peak_lag": float(time[peak] - time[np.argmax(q)]),
        "concentration_half_width": float(width),
        "carbon_excess_centroid": centroid, "carbon_excess_sd": sd,
        "input_extra_carbon": input_carbon, "output_extra_carbon": float(mass),
        "retained_extra_carbon_fraction": float(mass / input_carbon) if input_carbon > 0 else np.nan,
        "input_extra_water": input_water, "output_extra_water": float(routed_water.sum() * dt),
        "mean_path": mean, "path_sd": float(np.sqrt(weights @ (total - mean) ** 2)),
        "shared_kernel_mean": float(np.arange(len(shared)) @ shared * dt),
        "shared_kernel_sd": float(np.sqrt(((np.arange(len(shared)) * dt - np.arange(len(shared)) @ shared * dt) ** 2) @ shared)),
        "common": common, "dt": dt,
    }
    trace = pd.DataFrame({"time": time, "water_flux": q, "concentration": c,
                          "carbon_flux": flux, "carbon_excess_flux": routed_carbon})
    return metrics, trace


def observed_window_budget(frame, start, end, flow_peak, flow_return,
                           concentration_baseline, area_km2):
    """Exact complete hourly endpoint quadrature; mg/L * m3/s = g/s.

    Missing clocks or values invalidate the whole bounded budget. Trapezoids
    summarize adjacent recorded endpoints, with no gap filling/extrapolation.
    """
    if end <= start or area_km2 <= 0 or not np.isfinite(concentration_baseline):
        raise ValueError("A positive interval, area and finite baseline are required")
    selected = frame.loc[frame.clock.between(start, end)].sort_values("clock")
    if selected.clock.duplicated().any():
        raise ValueError("Canonical source timestamps required")
    expected = pd.date_range(start, end, freq="h")
    valid = selected.loc[np.isfinite(selected.q_m3_s) & np.isfinite(selected.doc_mg_l)]
    result = {"expected_hourly_rows": len(expected), "valid_hourly_rows": len(valid),
              "budget_complete": valid.clock.tolist() == expected.tolist(),
              "budget_end_clock": end, "budget_duration_hours": (end - start).total_seconds() / 3600}
    if not result["budget_complete"]:
        return result
    clock = pd.DatetimeIndex(valid.clock)
    # DatetimeIndex storage resolution may be microseconds or nanoseconds.
    seconds = np.diff((clock - clock[0]).total_seconds().to_numpy())
    q, c = valid.q_m3_s.to_numpy(), valid.doc_mg_l.to_numpy()
    j = q * c
    interval_volume = .5 * (q[:-1] + q[1:]) * seconds
    interval_carbon = .5 * (j[:-1] + j[1:]) * seconds / 1000
    positive = np.maximum(c - concentration_baseline, 0) * q
    negative = np.maximum(concentration_baseline - c, 0) * q
    positive_kg = float((.5 * (positive[:-1] + positive[1:]) * seconds).sum() / 1000)
    negative_kg = float((.5 * (negative[:-1] + negative[1:]) * seconds).sum() / 1000)
    volume, carbon = float(interval_volume.sum()), float(interval_carbon.sum())
    after_peak = clock[:-1] >= flow_peak
    after_return = clock[:-1] >= flow_return
    carbon_after = float(interval_carbon[after_peak].sum())
    water_after = float(interval_volume[after_peak].sum())
    if volume <= 0 or carbon <= 0:
        result["budget_complete"] = False
        result["nonpositive_budget"] = True
        return result
    result.update({
        "water_volume_m3": volume, "specific_runoff_mm": volume / (area_km2 * 1000),
        "carbon_kg": carbon, "carbon_yield_kg_km2": carbon / area_km2,
        "constant_concentration_carbon_kg": concentration_baseline * volume / 1000,
        "signed_concentration_extra_kg": positive_kg - negative_kg,
        "positive_concentration_extra_kg": positive_kg, "negative_concentration_extra_kg": negative_kg,
        "positive_extra_fraction_of_export": positive_kg / carbon,
        "flow_weighted_doc_mg_l": carbon * 1000 / volume,
        "carbon_after_flow_peak_fraction": carbon_after / carbon,
        "water_after_flow_peak_fraction": water_after / volume,
        "carbon_minus_water_after_peak_share": carbon_after / carbon - water_after / volume,
        "carbon_after_flow_return_fraction": float(interval_carbon[after_return].sum() / carbon),
        "water_after_flow_return_fraction": float(interval_volume[after_return].sum() / volume),
        "flux_peak_g_s": float(j.max()),
        "flux_peak_minus_flow_peak_hours": (clock[int(np.argmax(j))] - flow_peak).total_seconds() / 3600,
    })
    return result
