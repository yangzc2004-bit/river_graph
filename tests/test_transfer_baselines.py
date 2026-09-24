"""Contracts for the deterministic Stage-2B cross-basin baselines."""

from __future__ import annotations

import numpy as np

from river_graph.experiments.transfer_baselines import (
    all_baseline_predictions,
    ecology_month_climatology,
    source_month_climatology,
    support_predictions,
)


def _toy():
    y = np.arange(4 * 6, dtype=float).reshape(4, 6)
    fit = np.zeros_like(y, dtype=bool)
    fit[:2, :4] = True
    months = ["2000-01", "2000-02", "2001-01", "2001-02", "2002-01", "2002-02"]
    return y, fit, months


def test_source_climatology_uses_fit_labels_only():
    y, fit, months = _toy()
    original = source_month_climatology(y, fit, months)
    perturbed_hidden = y.copy()
    perturbed_hidden[~fit] += 10_000
    np.testing.assert_array_equal(original, source_month_climatology(perturbed_hidden, fit, months))
    perturbed_fit = y.copy()
    perturbed_fit[0, 0] += 1
    assert not np.array_equal(original, source_month_climatology(perturbed_fit, fit, months))


def test_support_predictions_ignore_query_labels_and_k0_falls_back_to_base():
    y, fit, months = _toy()
    base = source_month_climatology(y, fit, months)
    support = np.array([0, 1], dtype=int)
    query = np.array([4, 5], dtype=int)
    visible = np.zeros_like(y)
    visible.ravel()[support] = y.ravel()[support]
    hops = lambda a, b: abs(a - b)
    first = support_predictions(base, visible, support, query, hops)
    changed_query = visible.copy()
    changed_query.ravel()[query] = 1_000_000
    second = support_predictions(base, changed_query, support, query, hops)
    assert first == second
    zero = all_baseline_predictions(base, base, visible, np.array([], dtype=int), query, hops)
    for method in ("climatology", "eco_month_climatology", "local_mean", "mean_bias", "analytic_blend"):
        assert zero[method] == {int(q): float(base.ravel()[q]) for q in query}


def test_analytic_blend_is_finite_and_ecology_group_is_source_only():
    y, fit, months = _toy()
    regime = np.zeros((4, 3), dtype=float)
    regime[:2, 0] = 1
    regime[2:, 1] = 1
    eco = ecology_month_climatology(y, fit, months, regime)
    assert np.isfinite(eco).all()
    support = np.array([0, 1, 2], dtype=int)
    query = np.array([4, 5], dtype=int)
    base = source_month_climatology(y, fit, months)
    visible = np.zeros_like(y)
    visible.ravel()[support] = y.ravel()[support]
    result = support_predictions(base, visible, support, query, lambda a, b: abs(a - b))
    assert all(np.isfinite(v) for v in result["analytic_blend"].values())
