"""Leakage-safe station residual transfer for spatial extrapolation pilots.

The transfer source is restricted to station-blocked OOF residuals on the
training stations.  Corrections are deliberately simple: same-month directed
upstream residuals and feature-space station analogues.  This module contains
no model fitting and is also used by the pilot tests.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def source_residuals(y_z: np.ndarray, context_oof_z: np.ndarray,
                     train_cells: Iterable[int]) -> np.ndarray:
    """Return an ``(N,T)`` residual array with non-training cells masked."""
    y_z, context_oof_z = np.asarray(y_z, float), np.asarray(context_oof_z, float)
    if y_z.shape != context_oof_z.shape or y_z.ndim != 2:
        raise ValueError("aligned 2-D target and OOF arrays required")
    out = np.full(y_z.shape, np.nan, dtype=float)
    cells = np.asarray(list(train_cells), dtype=int)
    if len(cells) and (cells.min() < 0 or cells.max() >= y_z.size):
        raise ValueError("train cell outside target grid")
    out.ravel()[cells] = y_z.ravel()[cells] - context_oof_z.ravel()[cells]
    if len(cells) and not np.isfinite(out.ravel()[cells]).all():
        raise ValueError("nonfinite source residual")
    return out


def upstream_residual_transfer(residual: np.ndarray, edge_index: np.ndarray,
                               query_cells: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Mean same-month source residual from directed incoming river neighbours.

    ``edge_index[0] -> edge_index[1]`` is the source-to-target orientation used
    by the project graph.  The returned support count is useful for stratified
    diagnostics.  Queries with no train source receive exactly zero.
    """
    residual = np.asarray(residual, float)
    edges = np.asarray(edge_index, dtype=int)
    if residual.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("invalid residual/edge arrays")
    n, t = residual.shape
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge endpoint outside residual grid")
    incoming: list[list[int]] = [[] for _ in range(n)]
    for source, target in edges.T:
        incoming[target].append(int(source))
    cells = np.asarray(list(query_cells), dtype=int)
    correction = np.zeros(len(cells), dtype=float)
    support = np.zeros(len(cells), dtype=int)
    for k, cell in enumerate(cells):
        station, month = divmod(int(cell), t)
        values = [residual[src, month] for src in incoming[station]
                  if np.isfinite(residual[src, month])]
        if values:
            correction[k] = float(np.mean(values))
            support[k] = len(values)
    return correction, support


def _standardize(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, float)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("finite 2-D feature matrix required")
    center = source.mean(0)
    scale = source.std(0)
    scale[scale < 1e-8] = 1.0
    return (source - center) / scale, center, scale


def analog_residual_transfer(source_features: np.ndarray, source_residual: np.ndarray,
                             query_features: np.ndarray, k: int = 10,
                             source_month: np.ndarray | None = None,
                             query_month: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance mean residual from the nearest source feature rows.

    ``source_month``/``query_month`` pair optionally restricts neighbours to
    the same month; when supplied, a
    query first searches source rows from that month and falls back to all
    source rows when the month has no available source.  Source rows are never
    inferred from query labels.
    """
    source_features, query_features = np.asarray(source_features, float), np.asarray(query_features, float)
    source_residual = np.asarray(source_residual, float)
    if source_features.ndim != 2 or query_features.ndim != 2 or source_features.shape[1] != query_features.shape[1]:
        raise ValueError("source/query feature dimensions differ")
    if len(source_features) != len(source_residual) or not len(source_features):
        raise ValueError("source rows and residuals must be nonempty and aligned")
    if k < 1:
        raise ValueError("k must be positive")
    source_z, center, scale = _standardize(source_features)
    query_z = (query_features - center) / scale
    if not np.isfinite(query_z).all() or not np.isfinite(source_residual).all():
        raise ValueError("nonfinite analogue inputs")
    source_month = None if source_month is None else np.asarray(source_month, dtype=int)
    query_month = None if query_month is None else np.asarray(query_month, dtype=int)
    if (source_month is None) != (query_month is None):
        raise ValueError("source/query month must be provided together")
    if source_month is not None and (len(source_month) != len(source_features) or
                                     len(query_month) != len(query_features)):
        raise ValueError("month length mismatch")
    correction = np.zeros(len(query_features), dtype=float)
    support = np.zeros(len(query_features), dtype=int)
    all_idx = np.arange(len(source_z))
    for i, row in enumerate(query_z):
        candidates = all_idx if source_month is None else all_idx[source_month == query_month[i]]
        if len(candidates) == 0:
            candidates = all_idx
        distances = np.linalg.norm(source_z[candidates] - row, axis=1)
        take = candidates[np.argsort(distances)[: min(k, len(candidates))]]
        d = np.linalg.norm(source_z[take] - row, axis=1)
        weights = 1.0 / np.maximum(d, 1e-6)
        correction[i] = float(np.sum(weights * source_residual[take]) / np.sum(weights))
        support[i] = len(take)
    return correction, support


def apply_transformed_correction(base_z: np.ndarray, correction: np.ndarray,
                                 *, alpha: float, inverse) -> np.ndarray:
    """Apply a correction in target-transform space and invert to observations."""
    base_z, correction = np.asarray(base_z, float), np.asarray(correction, float)
    if base_z.shape != correction.shape:
        raise ValueError("base/correction shape mismatch")
    return np.asarray(inverse(base_z + alpha * correction), float)
