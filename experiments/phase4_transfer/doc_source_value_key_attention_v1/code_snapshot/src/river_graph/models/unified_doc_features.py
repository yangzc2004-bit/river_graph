"""Frozen GRU and tree features for local DOC support adaptation.

Source feature views hide every DOC label at the represented source station's
fold. The existing expert weights and normalization statistics remain fixed:
these are label-hidden input views, not out-of-fold fitted expert models.
"""

from __future__ import annotations

import numpy as np
import torch

from river_graph.models.graph_upgrade import ObservationAwareTemporalTransportGCNImputer
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
)


def _check_residual(residual):
    if (residual.builder.mechanism != "m1" or residual.temporal_operator != "gru"
            or type(residual.model) is not ObservationAwareTemporalTransportGCNImputer):
        raise ValueError("hidden extraction supports the standard M1 observation-aware GRU only")
    if residual.model.history_ablation == "hydro_only":
        raise ValueError("hydro-only history is not supported by this extraction path")


def extract_gru_hidden(residual, split, *, verify_head=False):
    """Return frozen states as float32 [station, month, hidden].

    This reproduces the current M1 forward path exactly, including its source
    visibility, directed support features, observation-age decay and rolling
    windows. ``verify_head`` independently compares applying the existing head
    to these states against the usual full residual forward pass.
    """
    _check_residual(residual)
    model = residual.model
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            x, age = residual.input_view(split, FIT_ROLES)
            encoded = model.encode_months(x, model._edge_index, model._edge_attr, residual.inputs.env_raw)
            support = torch.stack([x[..., 9], x[..., -2], x[..., -1]], dim=-1)
            hidden = model._memory_states(encoded, age, support)
            if verify_head:
                reconstructed = model.spatial.predict_from_hidden(hidden).T
                expected = residual.delta_tensor(x, age)
                torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
            result = hidden.permute(1, 0, 2).contiguous().cpu().numpy().copy()
    finally:
        model.train(was_training)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite GRU representation")
    return result.astype(np.float32, copy=False)


def tree_prediction_features(forest, x, *, row_batch_size=8192):
    """Return [row, tree] log1p predictions without changing any fitted tree.

    The context forest was fitted to log1p(DOC), so its individual trees already
    produce transformed outputs. No inverse transform, target labels or refit
    is used here. Float32 keeps full-cohort feature memory bounded.
    """
    if (isinstance(row_batch_size, bool) or not isinstance(row_batch_size, (int, np.integer))
            or row_batch_size < 1):
        raise ValueError("row_batch_size must be a positive integer")
    if not hasattr(forest, "estimators_") or not len(forest.estimators_):
        raise ValueError("tree feature extraction needs an already fitted forest")
    if getattr(forest, "n_outputs_", 1) != 1:
        raise ValueError("tree feature extraction supports a single target only")
    features = np.asarray(x, dtype=np.float32)
    if (features.ndim != 2 or features.shape[1] != forest.n_features_in_
            or not np.isfinite(features).all()):
        raise ValueError("forest inputs must be finite and match the fitted feature dimension")
    result = np.empty((len(features), len(forest.estimators_)), dtype=np.float32)
    for start in range(0, len(features), row_batch_size):
        stop = min(start + row_batch_size, len(features))
        batch = np.ascontiguousarray(features[start:stop])
        for column, tree in enumerate(forest.estimators_):
            result[start:stop, column] = tree.predict(batch, check_input=False)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite per-tree representation")
    return result


def extract_basis_inputs(model, dataset, split, progress=None, *, verify_head=False,
                         row_batch_size=8192):
    """Collect frozen GRU and forest representations for basis adaptation.

    Returns ``source_gru`` [Ns,T,H], ``source_tree`` [Ns,T,n_trees],
    ``full_gru`` [N,T,H], ``full_tree`` [N,T,n_trees], and sorted integer
    ``source_station_ids`` [Ns] matching the source array's first dimension.
    Full-cohort states use train/context visibility. Each source station is
    represented under its saved training fold's hidden-label view. Target labels
    do not fit, normalize, or select a representation in this extraction step.

    ``progress`` receives small dictionaries, one per completed extraction
    stage. Expert weights, datasets, split roles and forest objects are unchanged.
    """
    _check_residual(model.residual)
    n, months = dataset["y"].shape
    if tuple(model.residual.inputs.y_model.shape) != (n, months):
        raise ValueError("dataset dimensions do not match the frozen residual model")
    train = np.asarray(split["train"], dtype=np.int64)
    source_ids = np.unique(train // months)
    folds = [np.asarray(stations, dtype=np.int64) for stations in model.rf.folds]
    if not folds or any(stations.ndim != 1 or not len(stations) for stations in folds):
        raise ValueError("saved source station folds must be nonempty one-dimensional arrays")
    combined = np.concatenate(folds)
    if not np.array_equal(np.sort(combined), source_ids):
        raise ValueError("saved station folds must partition the source stations exactly once")

    def report(stage, **details):
        if progress is not None:
            progress({"stage": stage, **details})

    full_gru = extract_gru_hidden(model.residual, split, verify_head=verify_head)
    report("full_gru", shape=list(full_gru.shape))
    context_x = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    full_tree = tree_prediction_features(model.context_forest, context_x, row_batch_size=row_batch_size)
    full_tree = full_tree.reshape(n, months, -1)
    del context_x
    report("full_tree", shape=list(full_tree.shape))

    source_gru = np.empty((len(source_ids), months, full_gru.shape[-1]), dtype=np.float32)
    source_tree = np.empty((len(source_ids), months, full_tree.shape[-1]), dtype=np.float32)
    for fold_id, stations in enumerate(folds):
        view = fold_split(split, stations, months)
        hidden = extract_gru_hidden(model.residual, view, verify_head=verify_head)
        positions = np.searchsorted(source_ids, stations)
        source_gru[positions] = hidden[stations]
        del hidden
        context_x = build_rf_features(dataset, view, FIT_ROLES, target_transform="log1p", include_network=True)
        selected_x = context_x.reshape(n, months, -1)[stations].reshape(len(stations) * months, -1)
        # Release full-view inputs before allocating this fold's per-tree output.
        del context_x
        source_tree[positions] = tree_prediction_features(
            model.context_forest, selected_x, row_batch_size=row_batch_size,
        ).reshape(len(stations), months, -1)
        del selected_x
        report("source_fold", fold=fold_id, stations=stations.tolist())
    return {"source_gru": source_gru, "source_tree": source_tree,
            "full_gru": full_gru, "full_tree": full_tree, "source_station_ids": source_ids}
