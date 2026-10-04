"""Station-held-out context predictions for source adaptation episodes.

These forests refit the selected context expert's configuration on each source
fold. Hiding labels only in a frozen forest's inputs would not make its source
predictions out of fold, because those labels already fitted its trees.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
    target_values,
)


def fit_context_oof(model, dataset, split, n_jobs=2, progress=None) -> dict:
    """Fit five context clones and return source-cell OOF log1p predictions.

    ``model.context_forest`` supplies all estimator parameters, including its
    selected leaf size, feature sampling and random state. Only parallelism is
    overridden. The saved ``model.rf.folds`` define the five station folds;
    every label at each held station is hidden when constructing that fold's
    features, and only remaining source training cells fit its forest.

    Returns ``pred_z`` [station, month], with finite predictions at every train
    cell and NaN elsewhere, and JSON-compatible ``fold_records``. Forests are
    released after each fold rather than retained as five large model copies.
    ``progress``, when supplied, receives one small dictionary per finished fold.
    The selected configuration remains fixed; this function does not repeat its
    source-validation model selection or use validation/test labels for fitting.
    """
    if not isinstance(model.context_forest, ExtraTreesRegressor):
        raise TypeError("the selected context expert must be an ExtraTreesRegressor")
    if not hasattr(model.context_forest, "estimators_"):
        raise ValueError("the selected context expert must already be fitted")
    shape = tuple(dataset["y"].shape)
    if len(shape) != 2 or not all(shape):
        raise ValueError("DOC labels must have a nonempty station-month shape")
    n, months = shape
    train = np.asarray(split["train"])
    if (train.ndim != 1 or not np.issubdtype(train.dtype, np.integer)
            or not len(train) or (train < 0).any() or (train >= n * months).any()
            or len(np.unique(train)) != len(train)):
        raise ValueError("train must contain unique valid integer cell identities")
    train = train.astype(np.int64, copy=False)
    observed = np.asarray(dataset["y_mask"], dtype=bool)
    if observed.shape != shape or not observed.ravel()[train].all():
        raise ValueError("all training cells must be observed DOC labels")
    source_stations = np.unique(train // months)
    folds = [np.asarray(stations) for stations in model.rf.folds]
    if (len(folds) != 5 or any(stations.ndim != 1 or not len(stations)
                             or not np.issubdtype(stations.dtype, np.integer)
                             for stations in folds)):
        raise ValueError("exactly five nonempty integer station folds are required")
    if not np.array_equal(np.sort(np.concatenate(folds)), source_stations):
        raise ValueError("saved folds must partition the source stations exactly once")
    z = target_values(dataset, "log1p").ravel()
    if not np.isfinite(z[train]).all():
        raise ValueError("source training labels must have finite log1p values")

    prediction = np.full(shape, np.nan, dtype=np.float64)
    records = []
    for fold_id, held_stations in enumerate(folds):
        held_stations = held_stations.astype(np.int64, copy=False)
        view = fold_split(split, held_stations, months)
        fit_cells = np.asarray(view["train"], dtype=np.int64)
        held_cells = train[np.isin(train // months, held_stations)]
        features = build_rf_features(dataset, view, FIT_ROLES,
                                     target_transform="log1p", include_network=True)
        forest = clone(model.context_forest).set_params(n_jobs=n_jobs)
        forest.fit(features[fit_cells], z[fit_cells])
        held_prediction = forest.predict(features[held_cells])
        if not np.isfinite(held_prediction).all():
            raise ValueError(f"nonfinite context prediction in source fold {fold_id}")
        prediction.ravel()[held_cells] = held_prediction
        record = {
            "fold": fold_id,
            "held_station_ids": held_stations.tolist(),
            "training_station_ids": np.unique(fit_cells // months).tolist(),
            "n_train_cells": len(fit_cells),
            "n_oof_cells": len(held_cells),
            "n_features": features.shape[1],
            "context_name": getattr(model, "context_name", None),
            "forest_params": forest.get_params(deep=False),
            "target_transform": "log1p",
            "visible_roles": list(FIT_ROLES),
        }
        records.append(record)
        if progress is not None:
            progress({"stage": "context_oof_fold", "fold": fold_id,
                      "n_train_cells": len(fit_cells), "n_oof_cells": len(held_cells),
                      "n_features": features.shape[1]})
        del forest, features
    if not np.isfinite(prediction.ravel()[train]).all():
        raise ValueError("source context OOF prediction coverage is incomplete")
    return {"pred_z": prediction, "fold_records": records}
