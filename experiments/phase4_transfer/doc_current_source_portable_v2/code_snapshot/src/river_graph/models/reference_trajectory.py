"""Source-only scaling of cross-fitted environmental DOC trajectories."""
from __future__ import annotations

import numpy as np


def fit_reference_scaler(source_log_prediction):
    source = np.asarray(source_log_prediction, dtype=np.float64)
    if source.ndim != 2 or not source.size or not np.isfinite(source).all():
        raise ValueError("dense source station-month reference must be finite")
    return float(source.mean()), max(float(source.std()), 1e-6)


def reference_history(log_prediction, mean, scale):
    prediction = np.asarray(log_prediction, dtype=np.float64)
    if (prediction.ndim != 2 or not prediction.size or not np.isfinite(prediction).all()
            or not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("reference history requires finite predictions and a positive source scale")
    return ((prediction-mean)/scale)[..., None]
