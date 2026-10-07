"""Conservative branch mixing and storage responses at fixed mean arrival."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfcx, log_ndtr


def gaussian_storage_response(time, delay, sigma, tau):
    """Gaussian input convolved with a causal, unit-gain exponential kernel.

    The Gaussian input itself extends before its event centre at time zero.
    Causality constrains the routing kernel, not this input's support. A
    piecewise ex-Gaussian expression avoids cancellation for tiny tau.
    """
    t = np.asarray(time, dtype=float)
    if not np.isfinite(t).all() or not np.isfinite([delay, sigma, tau]).all():
        raise ValueError("finite times and response parameters required")
    if delay < 0 or sigma <= 0 or tau < 0:
        raise ValueError("nonnegative causal delay/storage and positive pulse SD required")
    x = (t-delay)/sigma
    if tau == 0:
        return np.exp(-.5*x*x)
    ratio = sigma/tau
    u = x-ratio
    out = np.empty_like(x)
    left = u <= 0
    # exp(r^2/2-x*r) Phi(x-r) = exp(-x^2/2) erfcx((r-x)/sqrt(2))/2.
    out[left] = np.sqrt(np.pi/2)*ratio*np.exp(-.5*x[left]**2)*erfcx(-u[left]/np.sqrt(2))
    out[~left] = np.exp(np.log(np.sqrt(2*np.pi)*ratio)+.5*ratio**2-x[~left]*ratio+log_ndtr(u[~left]))
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError("invalid storage response")
    return out


def controlled_responses(branch_a, branch_b, common, weight, *, sigma=.15,
                         fractions=(0., .25, .5, 1.), dt=.0025, keep_curves=True):
    """Cross actual/equal branches with fixed common-trunk storage fractions."""
    if not np.isfinite([branch_a, branch_b, common, weight, sigma, dt]).all():
        raise ValueError("finite geometry and forcing required")
    if min(branch_a, branch_b, sigma, dt) <= 0 or common < 0 or not 0 < weight < 1:
        raise ValueError("positive branches/shares and nonnegative common trunk required")
    f = np.asarray(fractions, dtype=float)
    if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or ((f < 0) | (f > 1)).any() or len(np.unique(f)) != len(f):
        raise ValueError("distinct storage fractions between zero and one required")
    weights = np.array([weight, 1-weight])
    mean = np.dot(weights, [branch_a, branch_b])+common
    branches = np.array([branch_a, branch_b])/mean
    c = common/mean
    branch_mean = np.dot(weights, branches)
    # Identical absolute evaluation grid across all branch and storage scenarios.
    low = min(branches.min(), branch_mean)+c-c*f.max()-9*sigma
    high = max(branches.max(), branch_mean)+c+9*sigma+30*c*f.max()
    time = np.arange(np.floor(low/dt), np.ceil(high/dt)+1)*dt
    rows, curves = [], []
    for condition, b in (("actual", branches), ("equal", np.full(2, branch_mean))):
        branch_variance = np.dot(weights, (b-branch_mean)**2)
        for fraction in f:
            tau = fraction*c
            deterministic_delays = b+c-tau
            response = sum(w*gaussian_storage_response(time, d, sigma, tau)
                           for w, d in zip(weights, deterministic_delays, strict=True))
            area = np.trapezoid(response, time)
            density = response/area
            centroid = np.trapezoid(time*density, time)
            sd = np.sqrt(np.trapezoid((time-centroid)**2*density, time))
            cdf = cumulative_trapezoid(density, time, initial=0)
            quantiles = np.interp([.1, .5, .9], cdf, time)
            peak_index = int(response.argmax())
            rows.append({"branch_condition": condition, "storage_fraction": float(fraction),
                "input_sd": sigma, "common_mean_budget": c, "storage_tau": tau,
                "common_translation": c-tau, "branch_variance": branch_variance,
                "storage_variance": tau**2, "pulse_peak": float(response[peak_index]),
                "peak_time": float(time[peak_index]), "pulse_centroid": centroid,
                "analytic_centroid": 1., "pulse_sd": sd,
                "analytic_sd": np.sqrt(sigma**2+branch_variance+tau**2),
                "t10": quantiles[0], "t50": quantiles[1], "t90": quantiles[2],
                "duration_80": quantiles[2]-quantiles[0],
                "anomaly_area_fraction": area/(np.sqrt(2*np.pi)*sigma),
                "steady_gain": 1., "dt": dt})
            if keep_curves:
                curves.append(pd.DataFrame({"branch_condition": condition, "storage_fraction": float(fraction),
                    "input_sd": sigma, "relative_time": time, "outlet_anomaly": response}))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def structural_contrasts(scenarios):
    """Branch/storage contrasts matched by pair, forcing and mean arrival."""
    rows = []
    for (pair, sigma), f in scenarios.groupby(["pair_id", "input_sd"], sort=True):
        ref = f.set_index(["branch_condition", "storage_fraction"])
        a0, e0 = ref.loc[("actual", 0.)], ref.loc[("equal", 0.)]
        for fraction in sorted(f.storage_fraction.unique()):
            actual, equal = ref.loc[("actual", fraction)], ref.loc[("equal", fraction)]
            row = {"pair_id": pair, "input_sd": sigma, "storage_fraction": fraction}
            for metric in ("pulse_peak", "pulse_sd", "duration_80", "peak_time"):
                row[f"branch_{metric}_difference"] = actual[metric]-equal[metric]
                row[f"storage_{metric}_difference"] = actual[metric]-a0[metric]
                row[f"interaction_{metric}_difference"] = (actual[metric]-equal[metric])-(a0[metric]-e0[metric])
            row["branch_peak_reduction_pct"] = 100*(1-actual.pulse_peak/equal.pulse_peak)
            row["storage_peak_reduction_pct"] = 100*(1-actual.pulse_peak/a0.pulse_peak)
            row["combined_peak_reduction_pct"] = 100*(1-actual.pulse_peak/e0.pulse_peak)
            row["storage_duration_change_pct"] = 100*(actual.duration_80/a0.duration_80-1)
            row["storage_sd_change_pct"] = 100*(actual.pulse_sd/a0.pulse_sd-1)
            row["branch_variance"] = actual.branch_variance
            row["storage_variance"] = actual.storage_variance
            added = actual.branch_variance+actual.storage_variance
            row["storage_share_of_added_variance"] = actual.storage_variance/added if added > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
