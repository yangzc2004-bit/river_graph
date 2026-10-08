"""Measured source-water participation on fixed mapped river paths."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from river_graph.analysis.river_joint_campaigns import seasonal_design

HYDRO_METRICS = ("effective_sources", "effective_fraction", "path_mean_km", "path_sd_km",
                 "path_sd_reference", "common_fraction", "source_receiver_flow_share")


def participation(paths, weights, *, reference_mean, common_km):
    d, w = np.asarray(paths, float), np.asarray(weights, float)
    if w.ndim == 1:
        w = w[None, :]
    if (d.ndim != 1 or len(d) < 2 or w.shape[1] != len(d)
            or not np.isfinite(np.r_[d, w.ravel(), reference_mean, common_km]).all()
            or (w < 0).any() or (w.sum(axis=1) <= 0).any()
            or reference_mean <= 0 or common_km < 0 or common_km > d.min()+1e-8):
        raise ValueError("Aligned positive shares, fixed paths and feasible common corridor required")
    w = w/w.sum(axis=1, keepdims=True)
    mean = w@d
    sd = np.sqrt(np.sum(w*(d[None, :]-mean[:, None])**2, axis=1))
    effective = 1/np.sum(w*w, axis=1)
    return pd.DataFrame({"effective_sources": effective, "effective_fraction": effective/len(d),
        "path_mean_km": mean, "path_sd_km": sd, "path_sd_reference": sd/reference_mean,
        "common_fraction": common_km/mean})


def hydro_contrasts(frame):
    dates = pd.to_datetime(frame.date)
    low, high = frame.flow_state.eq("low"), frame.flow_state.eq("high")
    if len(frame) < 24 or dates.dt.year.nunique() < 3 or min(low.sum(), high.sum()) < 6:
        raise ValueError("Insufficient complete positive-flow state observations")
    seasonal = seasonal_design(dates)
    design = np.column_stack((seasonal[:, 0], low, high, seasonal[:, 1:]))
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Flow-state and calendar design is not identifiable")
    result = {"n_months": len(frame), "n_years": dates.dt.year.nunique(),
              "n_low": int(low.sum()), "n_high": int(high.sum())}
    for metric in HYDRO_METRICS:
        values = frame[metric].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Hydrological participation must be observed and finite")
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        result[metric+"_raw_change"] = float(values[high].mean()-values[low].mean())
        result[metric+"_adjusted_change"] = float(beta[2]-beta[1])
        result[metric+"_low"] = float(values[low].mean())
        result[metric+"_high"] = float(values[high].mean())
    return result


def multibranch_potential(weights, sd, correlation):
    w, s, c = np.asarray(weights, float), np.asarray(sd, float), np.asarray(correlation, float)
    if (w.shape != s.shape or c.shape != (len(w), len(w)) or len(w) < 2
            or not np.isfinite(np.r_[w, s, c.ravel()]).all() or (w < 0).any()
            or (s <= 0).any() or not np.isclose(w.sum(), 1)
            or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1)):
        raise ValueError("Finite aligned source SDs, correlation and unit shares required")
    covariance = s[:, None]*c*s[None, :]
    if np.linalg.eigvalsh(c).min() < -1e-8:
        raise ValueError("Correlation matrix must be positive semidefinite")
    return float(100*(1-(w@covariance@w)/(w@(s*s))))


def mixing_decomposition(low, high):
    """Three-group all-order analytic attribution; no input is a causal intervention."""
    names = ("correlation", "sd", "weights")
    values = {}
    for size in range(4):
        for group in combinations(names, size):
            parameters = {k: high[k] if k in group else low[k] for k in names}
            values[frozenset(group)] = multibranch_potential(
                parameters["weights"], parameters["sd"], parameters["correlation"])
    result = {}
    for name in names:
        total = 0.
        other = [k for k in names if k != name]
        for size in range(3):
            factor = factorial(size)*factorial(2-size)/factorial(3)
            for group in combinations(other, size):
                key = frozenset(group)
                total += factor*(values[key | {name}]-values[key])
        result[name+"_contribution_pp"] = total
    result["mixing_potential_change_pp"] = values[frozenset(names)]-values[frozenset()]
    np.testing.assert_allclose(sum(result[k+"_contribution_pp"] for k in names),
                               result["mixing_potential_change_pp"], atol=1e-9)
    return result


def observed_mixing(sources, outlet, weights, dates, states):
    """All series share one projection; varying shares act before projection."""
    a, y, w = np.asarray(sources, float), np.asarray(outlet, float), np.asarray(weights, float)
    states = np.asarray(states)
    if (a.ndim != 2 or w.shape != a.shape or len(y) != len(a) or len(states) != len(y)
            or not np.isfinite(np.r_[a.ravel(), y, w.ravel()]).all()):
        raise ValueError("Aligned measured DOC and flow shares required")
    low, high = states == "low", states == "high"
    if len(y) < 24 or pd.DatetimeIndex(dates).year.nunique() < 3 or min(low.sum(), high.sum()) < 5:
        raise ValueError("Insufficient observed DOC state support")
    w = w/w.sum(axis=1, keepdims=True)
    design = seasonal_design(pd.Series(dates))
    if np.linalg.matrix_rank(design) != 4:
        raise ValueError("DOC calendar design is not identifiable")
    raw = np.column_stack((a, np.sum(a*w, axis=1), y))
    adjusted = raw-design@np.linalg.lstsq(design, raw, rcond=None)[0]
    params, summaries = {}, {}
    for state, keep in (("low", low), ("high", high)):
        source = adjusted[keep, :-2]
        params[state] = {"weights": w[keep].mean(axis=0), "sd": source.std(axis=0, ddof=1),
                         "correlation": np.corrcoef(source, rowvar=False)}
        p = params[state]
        mixture_sd, outlet_sd = adjusted[keep, -2:].std(axis=0, ddof=1)
        if min(mixture_sd, outlet_sd) <= 1e-12:
            raise ValueError("DOC fluctuation ratio is not identifiable")
        summaries[state] = {"n": int(keep.sum()), "mixture_sd": mixture_sd, "outlet_sd": outlet_sd,
            "ratio": outlet_sd/mixture_sd, "potential": multibranch_potential(p["weights"], p["sd"], p["correlation"])}
    result = {"n_doc_months": len(y)}
    for state, metrics in summaries.items():
        result.update({state+"_"+k: v for k, v in metrics.items()})
    result.update(mixing_decomposition(params["low"], params["high"]))
    result["log_ratio_change"] = float(np.log(summaries["high"]["ratio"]/summaries["low"]["ratio"]))
    return result
