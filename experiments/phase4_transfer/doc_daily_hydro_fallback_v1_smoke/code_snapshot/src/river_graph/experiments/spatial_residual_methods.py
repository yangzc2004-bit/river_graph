"""Small, fixed candidate set for target-station residual calibration."""
from __future__ import annotations

import numpy as np


def huber_location(values: np.ndarray, *, delta: float = 1.345) -> float:
    """Robust location estimate with a fixed Huber tuning constant."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Huber input must be non-empty and finite")
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)) / 0.67448975)
    if scale <= 1e-12:
        return float(np.mean(values))
    for _ in range(30):
        standardized = (values - center) / scale
        weights = np.minimum(1.0, delta / np.maximum(np.abs(standardized), 1e-12))
        updated = float(np.sum(weights * values) / np.sum(weights))
        if abs(updated - center) < 1e-10:
            break
        center = updated
    return center


def _ridge_from_zero(
    design: np.ndarray,
    target: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Solve a ridge system with a zero coefficient prior."""
    gram = design.T @ design
    penalty = regularization * np.eye(design.shape[1], dtype=np.float64)
    rhs = design.T @ target
    return np.linalg.solve(gram + penalty, rhs)


def station_residual_adjustment(
    method: str,
    support_z: np.ndarray,
    support_residual: np.ndarray,
    query_z: np.ndarray,
    support_month: np.ndarray,
    query_month: np.ndarray,
    *,
    alpha: float = 1.0,
    regularization: float = 10.0,
) -> np.ndarray:
    """Estimate query residuals from support residuals only.

    Support features and support residuals are the only target-labelled
    quantities consumed by this function. The returned vector is in log1p
    units and has one value per query cell.
    """
    support_z = np.asarray(support_z, dtype=np.float64)
    support_residual = np.asarray(support_residual, dtype=np.float64)
    query_z = np.asarray(query_z, dtype=np.float64)
    support_month = np.asarray(support_month, dtype=np.int64)
    query_month = np.asarray(query_month, dtype=np.int64)
    if support_z.shape != support_residual.shape:
        raise ValueError("support features and residuals must be aligned")
    if not np.isfinite(support_z).all() or not np.isfinite(support_residual).all():
        raise ValueError("support inputs must be finite")
    if not np.isfinite(query_z).all() or not np.isfinite(alpha):
        raise ValueError("query features and alpha must be finite")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if method in {"mean", "median", "huber"}:
        if method == "mean":
            location = float(np.mean(support_residual))
        elif method == "median":
            location = float(np.median(support_residual))
        else:
            location = huber_location(support_residual)
        return np.full(query_z.shape, alpha * location, dtype=np.float64)
    if regularization <= 0:
        raise ValueError("regularization must be positive for structured methods")
    if method == "affine":
        center = float(np.mean(support_z))
        design = np.column_stack([
            np.ones(len(support_z)),
            support_z - center,
        ])
        query_design = np.column_stack([
            np.ones(len(query_z)),
            query_z - center,
        ])
    elif method == "seasonal":
        support_phase = 2 * np.pi * (support_month % 12) / 12.0
        query_phase = 2 * np.pi * (query_month % 12) / 12.0
        design = np.column_stack([
            np.ones(len(support_z)),
            np.sin(support_phase),
            np.cos(support_phase),
        ])
        query_design = np.column_stack([
            np.ones(len(query_z)),
            np.sin(query_phase),
            np.cos(query_phase),
        ])
    else:
        raise ValueError(f"unknown residual method: {method}")
    coefficients = _ridge_from_zero(
        design, support_residual, float(regularization),
    )
    return alpha * (query_design @ coefficients)


def candidate_specs() -> list[dict[str, float | str | None]]:
    """Return the fixed, small candidate set used for internal selection."""
    specs: list[dict[str, float | str | None]] = []
    for method in ("mean", "median", "huber"):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            specs.append({
                "method": method,
                "alpha": alpha,
                "regularization": None,
            })
    for method in ("affine", "seasonal"):
        for regularization in (1.0, 10.0, 100.0):
            specs.append({
                "method": method,
                "alpha": 1.0,
                "regularization": regularization,
            })
    return specs


def spec_label(spec: dict[str, float | str | None]) -> str:
    method = str(spec["method"])
    if spec["regularization"] is None:
        return f"{method}_a{float(spec['alpha']):g}"
    return f"{method}_l{float(spec['regularization']):g}"


__all__ = [
    "candidate_specs",
    "huber_location",
    "spec_label",
    "station_residual_adjustment",
]
