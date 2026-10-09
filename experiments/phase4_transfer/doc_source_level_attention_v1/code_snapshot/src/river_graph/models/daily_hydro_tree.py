"""Fixed-parameter ExtraTrees probes for current versus historical daily flow.

These are independent comparators. They do not replace the frozen context
forest or generate a new out-of-fold residual base for a neural model.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.unified_spatial_protocol import (
    support_query_cells,
    validate_unified_spatial_split,
)
from river_graph.models.daily_flow_features import FEATURE_NAMES
from river_graph.models.kgml_local_transport import (
    build_rf_features,
    rf_feature_names,
    target_values,
)

MODES = ("current", "history")
TRAIN_VISIBILITY = ("train",)
LOOKBACK = 12


def _lookback(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("lookback must be a positive integer")
    return int(value)


def daily_tree_feature_names(lookback=LOOKBACK):
    """Names for unchanged context columns then chronological nine-column slots."""
    lookback = _lookback(lookback)
    return [*rf_feature_names(True), *[
        f"lag_{lag}_{name}"
        for lag in range(lookback - 1, -1, -1)
        for name in (*FEATURE_NAMES, "history_valid")]]


def _inputs(dataset, split, daily_full, lookback):
    lookback = _lookback(lookback)
    validate_unified_spatial_split(dataset["y_mask"], split)
    shape = tuple(np.asarray(dataset["y_mask"]).shape)
    if tuple(np.asarray(dataset["y"]).shape) != shape:
        raise ValueError("DOC labels and observation mask must align")
    months = np.asarray(dataset["months"], dtype="datetime64[M]")
    if (months.shape != (shape[1],) or np.isnat(months).any()
            or (len(months) > 1 and not np.all(np.diff(months.astype(np.int64)) == 1))):
        raise ValueError("daily history requires consecutive chronological calendar months")
    daily = np.asarray(daily_full)
    if (daily.shape != (*shape, len(FEATURE_NAMES)) or not np.issubdtype(daily.dtype, np.number)
            or np.iscomplexobj(daily) or not np.isfinite(daily).all()
            or ((daily < 0) | (daily > 1)).any() or not np.isin(daily[..., 5:8], (0, 1)).all()):
        raise ValueError("daily features must be finite bounded aligned eight-channel inputs")
    hydro_mask = np.asarray(dataset["x_mask"])
    if (hydro_mask.ndim != 3 or hydro_mask.shape[:2] != shape or hydro_mask.shape[2] < 2
            or not np.isin(hydro_mask[..., 1], (0, 1)).all()):
        raise ValueError("monthly discharge visibility must align with the feature grid")
    if np.any(daily[~hydro_mask[..., 1].astype(bool)] != 0):
        raise ValueError("daily features must retain the frozen monthly discharge footprint")
    train = np.asarray(split["train"], dtype=np.int64)
    selected_y = np.asarray(dataset["y"]).ravel()[train]
    if not np.isfinite(selected_y).all() or (selected_y < 0).any():
        raise ValueError("source DOC labels must be finite and nonnegative")
    # Structural isolation also avoids ever transforming hidden target values.
    source_y = np.zeros(shape, dtype=np.asarray(dataset["y"]).dtype)
    source_y.ravel()[train] = selected_y
    source_dataset = {**dataset, "y": source_y}
    base = build_rf_features(source_dataset, split, TRAIN_VISIBILITY,
                             target_transform="log1p", include_network=True)
    if base.shape != (np.prod(shape), len(rf_feature_names(True))):
        raise ValueError("the selected context feature definition changed")
    history = np.zeros((*shape, lookback, len(FEATURE_NAMES) + 1), dtype=np.float32)
    daily = daily.astype(np.float32, copy=False)
    for slot, lag in enumerate(range(lookback - 1, -1, -1)):
        if lag < shape[1]:
            history[:, lag:, slot, :8] = daily[:, :shape[1]-lag]
            history[:, lag:, slot, 8] = 1
    return base, history, source_dataset, train


def _matrix(base, history, mode):
    if mode not in MODES:
        raise ValueError("tree daily mode must be current or history")
    block = history
    if mode == "current":
        block = history.copy()
        block[:, :, :-1, :8] = 0
    return np.concatenate([base, block.reshape(len(base), -1)], axis=-1)


def build_daily_tree_features(dataset, split, daily_full, *, mode="history", lookback=LOOKBACK):
    """Return matched float32 [N*T,147] features at the default 12-month window.

    Both modes use all 39 selected-context input columns unchanged. Each of the
    twelve chronological slots adds eight daily descriptors plus a valid-date
    bit. ``current`` zeros only the eleven older descriptor slots; its valid-
    date bits match ``history``. No future month or hidden DOC enters a feature.
    """
    base, history, _, _ = _inputs(dataset, split, daily_full, lookback)
    return _matrix(base, history, mode)


def fit_daily_tree_probes(expert, dataset, split, daily_full, *, n_jobs=2,
                          lookback=LOOKBACK, progress=None):
    """Fit two clones of the selected context ExtraTrees; return native products.

    Returns ``pred_native`` and ``models`` dictionaries keyed by current/history,
    and JSON-compatible ``records`` describing fixed parameters, features and
    source-validation K0 query MAE. Validation scores are descriptive: they do
    not choose a probe or tune its parameters. Models can be saved with joblib;
    replay uses :func:`build_daily_tree_features` and nonnegative ``expm1``.

    Only n_jobs can override a copied forest parameter. There is no source OOF
    refit because these probes are standalone baselines, not neural base models.
    """
    selected = expert.context_forest
    if not isinstance(selected, ExtraTreesRegressor) or not hasattr(selected, "estimators_"):
        raise ValueError("expert.context_forest must be a fitted selected ExtraTreesRegressor")
    if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0):
        raise ValueError("n_jobs must be a nonzero integer or None")
    lookback = _lookback(lookback)
    base, history, source_dataset, train = _inputs(dataset, split, daily_full, lookback)
    shape = np.asarray(dataset["y"]).shape
    if selected.n_features_in_ != base.shape[1]:
        raise ValueError("selected forest must use the unmodified context feature set")
    z = target_values(source_dataset, "log1p").ravel()[train]
    _, validation_query = support_query_cells(split, target_role="val", k=0, n_months=shape[1])
    validation_y = np.asarray(dataset["y"], dtype=float).ravel()[validation_query]
    if not np.isfinite(validation_y).all() or (validation_y < 0).any():
        raise ValueError("validation diagnostic labels must be finite and nonnegative")
    predictions, models, records = {}, {}, {}
    names = daily_tree_feature_names(lookback)
    for mode in MODES:
        features = _matrix(base, history, mode)
        model = clone(selected).set_params(n_jobs=n_jobs)
        model.fit(features[train], z)
        prediction = np.maximum(0, np.expm1(model.predict(features))).reshape(shape)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite daily tree prediction")
        record = {"mode": mode, "lookback": lookback, "feature_names": names,
                  "n_features": len(names), "n_source_cells": len(train),
                  "source_station_ids": np.unique(train // shape[1]).tolist(),
                  "context_name": getattr(expert, "context_name", None),
                  "selected_parent_parameters": selected.get_params(deep=False),
                  "forest_parameters": model.get_params(deep=False),
                  "target_transform": "log1p", "visible_roles": list(TRAIN_VISIBILITY),
                  "inference_roles": list(TRAIN_VISIBILITY), "hyperparameter_search": False,
                  "replaces_neural_context": False, "source_oof_fitting": False,
                  "validation_role": "source_validation; diagnostic only, no selection",
                  "validation_query_cells": len(validation_query),
                  "validation_mae_native": float(np.abs(prediction.ravel()[validation_query]-validation_y).mean()),
                  "history_layout": "chronological oldest-to-current; eight daily channels then history_valid",
                  "current_ablation": f"older {lookback-1} daily slots zero; all history_valid bits retained"}
        predictions[mode], models[mode], records[mode] = prediction, model, record
        if progress is not None:
            progress({"stage": "daily_tree_probe", "mode": mode,
                      "n_source_cells": len(train), "n_features": len(names)})
        del features
    return {"pred_native": predictions, "models": models, "records": records}
