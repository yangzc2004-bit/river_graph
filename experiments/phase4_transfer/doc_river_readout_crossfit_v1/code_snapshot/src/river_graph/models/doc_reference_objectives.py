"""Matched environmental references for native-concentration DOC error."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

ARMS = {
    "native_target_trees": "native",
    "log_l1_boosting": "log1p",
    "native_l1_boosting": "native",
}


def make_reference(arm, retained_forest, seed):
    """Change the tree target alone; compare fixed L1 boosting on two scales."""
    if arm not in ARMS:
        raise ValueError(f"unknown reference arm: {arm}")
    if arm == "native_target_trees":
        return clone(retained_forest).set_params(n_jobs=2)
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=.05,
        max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1., early_stopping=False, random_state=seed)


def reference_target(values, arm):
    """Only the caller's source fit labels enter the fitted objective."""
    if arm not in ARMS:
        raise ValueError(f"unknown reference arm: {arm}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("reference fit requires finite nonnegative source DOC")
    return np.log1p(values) if ARMS[arm] == "log1p" else values.copy()


def predict_reference(model, inputs, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown reference arm: {arm}")
    prediction = model.predict(inputs)
    if ARMS[arm] == "log1p":
        prediction = np.expm1(prediction)
    prediction = np.maximum(0., prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite environmental reference prediction")
    return prediction
