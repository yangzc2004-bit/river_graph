"""Morphology-centred diagnostics of observed DOC signal transmission."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.analysis.river_mechanisms import correlation, detrend

FOCAL_TERMS = (
    "log_drainage_density", "mainstem_share", "log_route_mean_scaled",
    "route_distance_cv", "mainstem_sinuosity", "log_basin_aspect",
    "log_network_axis_ratio",
)
MIX_CONTROLS = (
    "log_polygon_area", "source_drainage_coverage", "log_junction_km", "log_n_months",
)
PATH_CONTROLS = (
    "log_polygon_area", "log_source_area", "log_path_km", "added_drainage_fraction",
    "storage_fraction", "wetland_change_pct", "forest_change_pct", "log_n_months",
)


def log_sd_ratio(numerator_variance, denominator_variance):
    if not np.isfinite(numerator_variance+denominator_variance):
        return np.nan
    if numerator_variance <= 1e-12 or denominator_variance <= 1e-12:
        return np.nan
    return .5*np.log(numerator_variance/denominator_variance)


def anomaly_diagnostics(frame, *, mixing=False, hydro=None):
    """Calendar-adjusted log DOC on one exact common observational population."""
    columns = ("doc_a", "doc_b", "doc_target", "mixture_doc") if mixing else ("doc_a", "doc_target")
    if len(frame) < 12 or frame.date.duplicated().any():
        raise ValueError("at least twelve unique common dates required")
    values = frame[list(columns)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("finite nonnegative common DOC required")
    if hydro is not None and mixing:
        raise ValueError("hydro adjustment is defined for two-station paths")
    if hydro is None:
        residuals = [detrend(np.log1p(frame[c]), frame.date) for c in columns]
    else:
        if len(hydro) != 2:
            raise ValueError("source and target hydro arrays required")
        residuals = [detrend(np.log1p(frame[c]), frame.date, h)
                     for c, h in zip(columns, hydro, strict=True)]
    a, target = residuals[0], residuals[2 if mixing else 1]
    source_variance = np.var(a)
    if mixing:
        b, mixture = residuals[1], residuals[3]
        source_variance = .5*(source_variance+np.var(b))
        pearson = np.corrcoef(a, b)[0, 1] if min(np.std(a), np.std(b)) > 1e-6 else np.nan
        return {
            "log_anomaly_sd_ratio": log_sd_ratio(np.var(target), source_variance),
            "log_mixture_sd_ratio": log_sd_ratio(np.var(mixture), source_variance),
            "log_downstream_mixture_sd_ratio": log_sd_ratio(np.var(target), np.var(mixture)),
            "source_anomaly_pearson": pearson,
            "signal_rho": correlation(mixture, target),
            "mean_log_departure": np.mean(np.log1p(frame.doc_target)-np.log1p(frame.mixture_doc)),
            "n_months": len(frame),
        }
    return {
        "log_anomaly_sd_ratio": log_sd_ratio(np.var(target), source_variance),
        "signal_rho": correlation(a, target),
        "mean_log_departure": np.mean(np.log1p(frame.doc_target)-np.log1p(frame.doc_a)),
        "n_months": len(frame),
    }


def aggregate_receivers(frame, metrics):
    """One vote per receiving station, after averaging its observed connections."""
    if frame.empty:
        return pd.DataFrame(columns=["target", "huc4", "component", *metrics, "n_connections"])
    if frame.groupby("target")[["huc4", "component"]].nunique().gt(1).any().any():
        raise ValueError("receiver context must be consistent across connections")
    averaged = frame.groupby("target", as_index=False)[list(metrics)].mean()
    context = frame.groupby("target", as_index=False).agg(
        huc4=("huc4", "first"), component=("component", "first"), n_connections=("target", "size"))
    return context.merge(averaged, on="target", validate="one_to_one")


def cluster_mean(frame, metric, unit, *, draws=5000, contrast=False):
    """Receiver mean, or broad-minus-elongated; shared blocks sampled jointly."""
    s = frame.dropna(subset=[metric, unit]).copy()
    if contrast:
        s = s[s.cluster.isin([1, 3])].copy()
    if s.empty:
        return None
    values = s[metric].to_numpy(float)
    _, index = np.unique(s[unit], return_inverse=True)
    n = int(index.max()+1)
    weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, index]
    if contrast:
        first, second = s.cluster.eq(1).to_numpy(), s.cluster.eq(3).to_numpy()
        if not first.any() or not second.any():
            return None
        estimate = values[second].mean()-values[first].mean()
        def mean_in(mask):
            den = weights[:, mask].sum(axis=1)
            return np.divide(weights[:, mask]@values[mask], den,
                             out=np.full(draws, np.nan), where=den > 0)
        boot = mean_in(second)-mean_in(first)
        identifiable = min(s[first][unit].nunique(), s[second][unit].nunique()) >= 2
    else:
        estimate = values.mean()
        boot = weights@values/weights.sum(axis=1)
        identifiable = n >= 2
    finite = boot[np.isfinite(boot)]
    lo, hi = np.quantile(finite, [.025, .975]) if identifiable and len(finite) else (np.nan, np.nan)
    return {"metric": metric, "estimate": estimate, "ci_low": lo, "ci_high": hi,
            "unit": unit, "n_receivers": len(s), "n_blocks": n,
            "n_huc4": s.huc4.nunique(), "n_components": s.component.nunique(),
            "interval_status": "estimable" if identifiable else "not_estimable_single_block",
            "valid_bootstrap_draws": len(finite) if identifiable else 0}


def conditional_association(frame, outcome, focal, controls, *, unit="component", draws=5000):
    """Single morphology term plus fixed background, whole-block bootstrap.

    Nuisance imputation and scaling are fixed at the observed receiver population.
    Rank-deficient replicates are not turned into minimum-norm discoveries.
    """
    s = frame.dropna(subset=[outcome, focal, unit]).copy()
    raw = s[list(controls)].to_numpy(float)
    names, columns = [], []
    for j, name in enumerate(controls):
        valid = np.isfinite(raw[:, j])
        if not valid.any():
            continue
        values = np.where(valid, raw[:, j], np.median(raw[valid, j]))
        if np.std(values) > 1e-9:
            columns.append((values-values.mean())/values.std())
            names.append(name)
        if not valid.all():
            columns.append((~valid).astype(float))
            names.append(name+"_missing")
    values = s[focal].to_numpy(float)
    scale = np.std(values)
    if not len(s) or scale <= 1e-9:
        return None, pd.DataFrame()
    design = np.column_stack([np.ones(len(s)), *columns, (values-values.mean())/scale])
    y = s[outcome].to_numpy(float)
    p = design.shape[1]
    if len(s) <= p or np.linalg.matrix_rank(design) != p:
        return None, pd.DataFrame()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    orthogonal, _ = np.linalg.qr(design, mode="reduced")
    leverage = np.sum(orthogonal**2, axis=1)
    _, index = np.unique(s[unit], return_inverse=True)
    n = int(index.max()+1)
    counts = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)
    boot = []
    for count in counts:
        root = np.sqrt(count[index])
        d = design*root[:, None]
        if np.linalg.matrix_rank(d) == p:
            boot.append(np.linalg.lstsq(d, y*root, rcond=None)[0][-1])
    lo, hi = np.quantile(boot, [.025, .975]) if n >= 2 and boot else (np.nan, np.nan)
    leave = []
    for block in np.unique(s[unit]):
        keep = s[unit].ne(block).to_numpy()
        if np.linalg.matrix_rank(design[keep]) == p:
            coefficient = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0][-1]
            leave.append({"omitted_block": str(block), "estimate": coefficient})
    detail = pd.DataFrame(leave)
    result = {
        "outcome": outcome, "focal": focal, "estimate_per_receiver_sd": beta[-1],
        "ci_low": lo, "ci_high": hi, "unit": unit, "n_receivers": len(s),
        "n_blocks": n, "n_huc4": s.huc4.nunique(), "n_components": s.component.nunique(),
        "focal_sd": scale, "design_columns": p, "condition": np.linalg.cond(design),
        "max_leverage": float(leverage.max()),
        "max_leverage_receiver": str(s.target.iloc[np.argmax(leverage)]) if "target" in s else "",
        "valid_bootstrap_draws": len(boot), "requested_bootstrap_draws": draws,
        "nuisance_columns": ";".join(names),
        "loco_min": detail.estimate.min() if len(detail) else np.nan,
        "loco_max": detail.estimate.max() if len(detail) else np.nan,
        "loco_same_sign_fraction": np.mean(np.sign(detail.estimate) == np.sign(beta[-1])) if len(detail) else np.nan,
    }
    return result, detail
