"""Small, auditable gates for ecology-conditioned support fusion."""

from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = (
    "analyte",
    "k",
    "support_count",
    "support_month_span",
    "support_site_span",
    "ecological_novelty",
    "h2x_baseline_disagreement",
    "source_support_residual",
)


def clip_gate_weight(weight: np.ndarray | float) -> np.ndarray:
    """Return finite H2X weights in the closed unit interval."""

    values = np.asarray(weight, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("gate weights must be finite")
    return np.clip(values, 0.0, 1.0)


def blend_predictions(
    h2x_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    h2x_weight: np.ndarray | float,
) -> np.ndarray:
    """Blend predictions without changing their native measurement units."""

    h2x = np.asarray(h2x_prediction, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    if h2x.shape != baseline.shape:
        raise ValueError("H2X and baseline predictions must have the same shape")
    if not np.isfinite(h2x).all() or not np.isfinite(baseline).all():
        raise ValueError("predictions must be finite")
    weight = clip_gate_weight(h2x_weight)
    return weight * h2x + (1.0 - weight) * baseline


def fixed_weight_grid(step: float = 0.05) -> np.ndarray:
    """Return the frozen transparent candidate grid for source calibration."""

    if not 0 < step <= 1:
        raise ValueError("step must lie in (0, 1]")
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    return np.round(grid, 10)

