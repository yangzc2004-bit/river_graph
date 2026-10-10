"""Actual source coordination and receiving excursions on common dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.analysis.river_monitored_arrivals import _project


def coordination_statistics(sources, receiver, weights, dates, *, adjustment="calendar_year",
                            quantile=.75, projected=False, minimum_group=5):
    """Separate input covariance from an independently measured receiving signal.

    The receiving reference sums source SDs, without source cross-covariance.
    Peak frequencies are descriptive within the supplied, identical sample.
    """
    a, y, w = np.asarray(sources, float), np.asarray(receiver, float), np.asarray(weights, float)
    date = pd.DatetimeIndex(dates)
    if (a.ndim != 2 or a.shape != (len(y), len(w)) or len(w) < 2 or len(y) < 12
            or len(date) != len(y) or date.has_duplicates or date.hasnans
            or not np.isfinite(np.r_[a.ravel(), y, w]).all() or (w <= 0).any()
            or not .5 < quantile < 1 or minimum_group < 1):
        raise ValueError("finite common dates, two sources and positive shares required")
    w = w/w.sum()
    values = np.column_stack([a, y])
    if not projected:
        if (values < 0).any():
            raise ValueError("native DOC must be nonnegative")
        if adjustment == "raw":
            values = values-values.mean(axis=0)
        else:
            values = _project(values, date, adjustment)
    values = values-values.mean(axis=0)
    s, r = values[:, :-1], values[:, -1]
    covariance = s.T@s/len(y)
    variance = np.diag(covariance)
    sd = np.sqrt(np.maximum(variance, 0))
    mixture = s@w
    independent = float((w*w)@variance)
    perfect = float((w@sd)**2)
    actual, receiving = float(np.mean(mixture**2)), float(np.mean(r**2))
    coherence_den = perfect-independent
    coherence = (actual-independent)/coherence_den if coherence_den > 1e-12 else np.nan
    if np.isfinite(coherence) and not -1-1e-9 <= coherence <= 1+1e-9:
        raise ValueError("pair-correlation average must lie in [-1,1]")
    correlation = float(mixture@r/len(y)/np.sqrt(actual*receiving)) if min(actual, receiving) > 1e-12 else np.nan
    high = s > np.quantile(s, quantile, axis=0)
    receiver_high = r > np.quantile(r, quantile)
    count = high.sum(axis=1)
    coincident, solo = count >= 2, count == 1
    n_coincident, n_solo = int(coincident.sum()), int(solo.sum())
    coincident_high, solo_high = int((coincident & receiver_high).sum()), int((solo & receiver_high).sum())
    pc = coincident_high/n_coincident if n_coincident else np.nan
    ps = solo_high/n_solo if n_solo else np.nan
    eligible = min(n_coincident, n_solo) >= minimum_group
    stats = {"n_dates": len(y), "n_sources": len(w), "source_coherence": coherence,
        "source_perfect_sd": np.sqrt(perfect), "source_independent_sd": np.sqrt(independent),
        "mixture_sd": np.sqrt(actual), "receiver_sd": np.sqrt(receiving),
        "outlet_sync_log_sd_ratio": .5*np.log(receiving/perfect) if min(receiving, perfect) > 1e-12 else np.nan,
        "outlet_mix_log_sd_ratio": .5*np.log(receiving/actual) if min(receiving, actual) > 1e-12 else np.nan,
        "outlet_mix_correlation": np.clip(correlation, -1, 1),
        "excursion_quantile": quantile, "n_coincident": n_coincident, "n_solo": n_solo,
        "n_receiver_high_coincident": coincident_high, "n_receiver_high_solo": solo_high,
        "receiver_high_given_coincident": pc, "receiver_high_given_solo": ps,
        "peak_comparison_eligible": eligible,
        "peak_risk_difference": pc-ps if eligible else np.nan,
        "coincident_date_fraction": n_coincident/len(y)}
    trace = pd.DataFrame({"date": date, "doc_receiver": y, "doc_mixture": a@w,
        "receiver_anomaly": r, "mixture_anomaly": mixture,
        "n_sources_high": count, "source_high_area_share": high@w,
        "receiver_high": receiver_high, "coincident_sources": coincident, "solo_source": solo})
    for i in range(len(w)):
        trace[f"source_{i}_doc"] = a[:, i]
        trace[f"source_{i}_anomaly"] = s[:, i]
    return stats, trace


def within_receiver_association(frame, *, x="source_coherence", y="outlet_sync_log_sd_ratio",
                                draws=5000):
    """Receiver-centred relationship, equal total weight per repeated network."""
    f = frame.dropna(subset=[x, y, "target", "component"]).copy()
    counts = f.groupby("target").target.transform("size")
    f = f[counts >= 2].copy()
    if f.empty:
        return {"n_receivers": 0, "n_periods": 0, "n_components": 0,
                "correlation": np.nan, "slope": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "valid_draws": 0, "omitted_min": np.nan, "omitted_max": np.nan}, f
    for name in (x, y):
        f[name+"_within"] = f[name]-f.groupby("target")[name].transform("mean")
    weight = 1/f.groupby("target").target.transform("size").to_numpy(float)
    a, b = f[x+"_within"].to_numpy(float), f[y+"_within"].to_numpy(float)
    _, codes = np.unique(f.component, return_inverse=True)
    n = codes.max()+1
    xy, xx, yy = weight*a*b, weight*a*a, weight*b*b
    def corr(weights):
        den = np.sqrt((weights@xx)*(weights@yy))
        return np.divide(weights@xy, den, out=np.full(weights.shape[0], np.nan), where=den > 1e-12)
    point = corr(np.ones((1, len(f))))[0]
    slope = xy.sum()/xx.sum() if xx.sum() > 1e-12 else np.nan
    boot_weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, codes]
    finite = corr(boot_weights)
    finite = finite[np.isfinite(finite)]
    lo, hi = np.quantile(finite, [.025, .975]) if n >= 2 and len(finite) else (np.nan, np.nan)
    omitted = [corr((codes != c)[None, :].astype(float))[0] for c in range(n)]
    omitted = np.asarray(omitted)[np.isfinite(omitted)]
    return {"n_receivers": f.target.nunique(), "n_periods": len(f), "n_components": int(n),
        "correlation": point, "slope": slope, "ci_low": lo, "ci_high": hi,
        "valid_draws": len(finite) if n >= 2 else 0,
        "omitted_min": omitted.min() if len(omitted) else np.nan,
        "omitted_max": omitted.max() if len(omitted) else np.nan}, f


def excursion_block_interval(trace, blocks, *, draws=5000, minimum_group=5):
    """Paired excursion frequency difference, resampling complete time blocks.

    Excursion thresholds are fixed at their recorded full-case empirical values.
    This interval describes one case; it is not a population of river forms.
    """
    t = trace.copy()
    if len(blocks) != len(t):
        raise ValueError("aligned blocks required")
    _, codes = np.unique(blocks, return_inverse=True)
    n = int(codes.max()+1)
    c, s, high = (t[name].to_numpy(bool) for name in ("coincident_sources", "solo_source", "receiver_high"))
    group = np.column_stack([np.bincount(codes, weights=values, minlength=n)
        for values in (c, s, c & high, s & high)])
    weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)
    totals = weights@group
    eligible = np.minimum(totals[:, 0], totals[:, 1]) >= minimum_group
    risk = np.divide(totals[:, 2], totals[:, 0], out=np.full(draws, np.nan), where=totals[:, 0] > 0)
    risk -= np.divide(totals[:, 3], totals[:, 1], out=np.full(draws, np.nan), where=totals[:, 1] > 0)
    risk = risk[eligible & np.isfinite(risk)]
    lo, hi = np.quantile(risk, [.025, .975]) if n >= 2 and len(risk) else (np.nan, np.nan)
    return {"peak_ci_low": lo, "peak_ci_high": hi, "peak_valid_draws": len(risk) if n >= 2 else 0,
            "n_time_blocks": n}
