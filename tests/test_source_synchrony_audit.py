"""Source-only extraction and temporal alignment for the diagnostic audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_doc_source_synchrony_v1 import (
    pair_correlation,
    seasonal_shuffle,
    source_residuals,
)


def test_source_residual_extraction_does_not_read_unselected_labels():
    truth = np.arange(72.).reshape(3, 24)
    oof = np.log1p(np.ones_like(truth))
    train = np.array([24, 25, 30, 50, 60])
    ids, expected = source_residuals(truth, oof, train)
    keep = np.zeros(truth.size, dtype=bool)
    keep[train] = True
    changed_truth, changed_oof = truth.copy(), oof.copy()
    changed_truth.ravel()[~keep] = np.nan
    changed_oof.ravel()[~keep] = np.inf
    actual_ids, actual = source_residuals(changed_truth, changed_oof, train)
    np.testing.assert_array_equal(ids, [1, 2])
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).sum() == len(train)


def test_calendar_shuffle_and_lag_preserve_support_without_wrapping():
    values = np.arange(48., dtype=float).reshape(2, 24)
    values[0, 3] = np.nan
    calendar = np.tile(np.arange(1, 13), 2)
    shuffled = seasonal_shuffle(values, calendar, 42)
    np.testing.assert_array_equal(np.isfinite(shuffled), np.isfinite(values))
    for row in range(2):
        for month in range(1, 13):
            slots = (calendar == month) & np.isfinite(values[row])
            np.testing.assert_array_equal(np.sort(values[row, slots]), np.sort(shuffled[row, slots]))
    signal = np.arange(24., dtype=float)
    corr, count, status = pair_correlation(signal, signal, lag=3)
    assert abs(corr-1.) < 1e-14 and count == 21 and status == "available"
    corr, count, status = pair_correlation(signal, signal, lag=20)
    assert np.isnan(corr) and count == 4 and status == "insufficient_overlap"
    corr, count, status = pair_correlation(signal, np.ones(24))
    assert np.isnan(corr) and count == 24 and status == "constant_series"
