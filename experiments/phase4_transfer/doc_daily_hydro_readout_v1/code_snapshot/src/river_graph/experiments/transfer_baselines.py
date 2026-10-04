"""Deterministic, no-training baselines for the cross-basin transfer route.

The functions in this module take an explicit source ``fit_mask`` and an
explicit target support set.  They never need query labels.  Keeping the
prediction layer separate from the runner makes the visibility boundary easy
to test with label perturbations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

METHODS = (
    "climatology",
    "eco_month_climatology",
    "local_mean",
    "mean_bias",
    "analytic_blend",
)


def _month_numbers(months) -> np.ndarray:
    """Return calendar month numbers (1--12) from ISO month strings."""
    out = np.asarray([int(str(value)[5:7]) for value in months], dtype=int)
    if out.ndim != 1 or np.any((out < 1) | (out > 12)):
        raise ValueError("invalid calendar month values")
    return out


def source_month_climatology(
    y: np.ndarray, fit_mask: np.ndarray, months
) -> np.ndarray:
    """Predict each cell from source-train labels in its calendar month.

    The fallback is the source-train global mean.  Both the month means and
    the fallback are calculated only from ``fit_mask``.
    """
    y = np.asarray(y, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if y.ndim != 2 or fit.shape != y.shape:
        raise ValueError("y and fit_mask must be aligned N x T arrays")
    if not np.isfinite(y[fit]).all() or not fit.any():
        raise ValueError("fit labels must be finite and nonempty")
    month_num = _month_numbers(months)
    fallback = float(y[fit].mean())
    pred = np.full_like(y, fallback, dtype=float)
    for month in range(1, 13):
        cols = np.flatnonzero(month_num == month)
        if not len(cols):
            continue
        chunk = y[:, cols]
        chunk_fit = fit[:, cols]
        values = chunk[chunk_fit]
        pred[:, cols] = float(values.mean()) if len(values) else fallback
    return pred


def ecology_month_climatology(
    y: np.ndarray,
    fit_mask: np.ndarray,
    months,
    regime: np.ndarray,
) -> np.ndarray:
    """Source-only month and ecological-regime climatology.

    ``regime`` is a covariate, not a target.  The dominant regime column is
    used as a fixed categorical grouping.  Empty month/regime cells fall back
    to the corresponding source month mean, then to the source global mean.
    This is intentionally a transparent aggregation baseline rather than a
    learned ecological model.
    """
    y = np.asarray(y, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    regime = np.asarray(regime, dtype=float)
    if y.ndim != 2 or fit.shape != y.shape:
        raise ValueError("y and fit_mask must be aligned N x T arrays")
    if regime.ndim != 2 or regime.shape[0] != y.shape[0] or not np.isfinite(regime).all():
        raise ValueError("regime must be a finite N x R array")
    base = source_month_climatology(y, fit, months)
    month_num = _month_numbers(months)
    group = np.argmax(regime, axis=1)
    fallback = float(y[fit].mean())
    pred = base.copy()
    for month in range(1, 13):
        cols = np.flatnonzero(month_num == month)
        if not len(cols):
            continue
        month_fit = fit[:, cols]
        month_values = y[:, cols]
        month_mean = float(month_values[month_fit].mean()) if month_fit.any() else fallback
        for category in np.unique(group):
            rows = np.flatnonzero(group == category)
            if not len(rows):
                continue
            sub = month_values[np.ix_(rows, np.arange(len(cols)))]
            sub_fit = month_fit[np.ix_(rows, np.arange(len(cols)))]
            values = sub[sub_fit]
            value = float(values.mean()) if len(values) else month_mean
            pred[np.ix_(rows, cols)] = value
    return pred


def support_predictions(
    base: np.ndarray,
    target_y: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    hops: Callable[[int, int], int | None],
) -> dict[str, dict[int, float]]:
    """Build the four support-aware predictions without reading query labels.

    ``target_y`` is expected to contain values only at ``support`` cells; all
    other entries are ignored by construction.  The analytic blend is the
    existing fixed residual/local blend and therefore remains a diagnostic
    baseline rather than a trained model.
    """
    base = np.asarray(base, dtype=float)
    target_y = np.asarray(target_y, dtype=float)
    support = np.asarray(support, dtype=int)
    query = np.asarray(query, dtype=int)
    if base.shape != target_y.shape or base.ndim != 2:
        raise ValueError("base and target_y must be aligned N x T arrays")
    if np.intersect1d(support, query).size:
        raise ValueError("support and query cells overlap")
    flat_base = base.ravel()
    flat_y = target_y.ravel()
    base_query = {int(q): float(flat_base[q]) for q in query}
    if not len(support):
        return {
            "local_mean": base_query.copy(),
            "mean_bias": base_query.copy(),
            "analytic_blend": base_query.copy(),
        }
    if not np.isfinite(flat_y[support]).all():
        raise ValueError("support labels must be finite")
    support_values = flat_y[support]
    local = float(support_values.mean())
    bias = float(np.mean(support_values - flat_base[support]))
    local_pred = {int(q): local for q in query}
    bias_pred = {int(q): float(flat_base[q] + bias) for q in query}
    spread = float(support_values.std() + 1e-6)
    rel_spread = spread / (abs(local) + 1.0)
    blend = {}
    t = base.shape[1]
    for q in query:
        qrow = int(q) // t
        distances = [hops(int(s) // t, qrow) for s in support]
        finite = [float(h) for h in distances if h is not None]
        mean_h = float(np.mean(finite)) if finite else 5.0
        gate = float(np.exp(-2.5 * rel_spread - 0.15 * max(0.0, mean_h - 1.0)))
        blend[int(q)] = float((1.0 - gate) * flat_base[q] + gate * local)
    return {
        "local_mean": local_pred,
        "mean_bias": bias_pred,
        "analytic_blend": blend,
    }


def all_baseline_predictions(
    base: np.ndarray,
    eco_base: np.ndarray,
    target_y: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    hops: Callable[[int, int], int | None],
) -> dict[str, dict[int, float]]:
    """Return all frozen baseline predictions keyed by method and cell."""
    if np.asarray(base).shape != np.asarray(eco_base).shape:
        raise ValueError("base and eco_base shapes differ")
    support_preds = support_predictions(base, target_y, support, query, hops)
    return {
        "climatology": {int(q): float(np.asarray(base).ravel()[q]) for q in query},
        "eco_month_climatology": {
            int(q): float(np.asarray(eco_base).ravel()[q]) for q in query
        },
        **support_preds,
    }
