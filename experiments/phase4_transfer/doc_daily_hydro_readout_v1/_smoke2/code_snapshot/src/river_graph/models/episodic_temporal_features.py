"""Frozen spatial inputs for episodic tuning of the existing DOC memory.

Source station views hide that station fold's DOC labels, while keeping the
previously fitted spatial encoder and its normalization statistics fixed.
These are input caches, not independently fitted out-of-fold neural models.
"""

from __future__ import annotations

import numpy as np
import torch

from river_graph.models.graph_upgrade import ObservationAwareTemporalTransportGCNImputer
from river_graph.models.kgml_local_transport import FIT_ROLES, fold_split


def _check_model(residual):
    if (residual.builder.mechanism != "m1" or residual.temporal_operator != "gru"
            or type(residual.model) is not ObservationAwareTemporalTransportGCNImputer):
        raise ValueError("temporal input extraction supports the standard M1 GRU only")
    if residual.model.history_ablation != "none":
        raise ValueError("temporal input extraction requires no history ablation")
    if residual.edge_set != "empty" or residual.model._edge_index.numel():
        raise ValueError("temporal input extraction requires the no-message residual")


def _extract_view(residual, split):
    """Called only while the full model is in evaluation/inference mode."""
    model = residual.model
    x, age = residual.input_view(split, FIT_ROLES)
    encoded = model.encode_months(x, model._edge_index, model._edge_attr, residual.inputs.env_raw)
    support = torch.stack([x[..., 9], x[..., -2], x[..., -1]], dim=-1)
    arrays = {
        "encoded": encoded.permute(1, 0, 2).contiguous().cpu().numpy().copy(),
        "age": age.T.contiguous().cpu().numpy().copy(),
        "support": support.permute(1, 0, 2).contiguous().cpu().numpy().copy(),
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("nonfinite frozen temporal input")
    return arrays


def extract_temporal_inputs(model, dataset, split, progress=None) -> dict:
    """Cache exact spatial outputs and GRU-D covariates, without fitting.

    Returns ``full_encoded`` [N,T,H], ``full_age`` [N,T], ``full_support``
    [N,T,3] and the corresponding ``source_*`` arrays with Ns source stations.
    ``source_station_ids`` is sorted and indexes the source arrays' first axis.
    H is the frozen encoder width (64 in the current production model).

    Full-cohort features use FIT_ROLES visibility. Each source station's cached
    inputs use its saved RF station fold's hidden-label view. The support axis
    is local visibility, upstream support, downstream support, exactly as in
    the existing GRU forward path; KGML already zeros its downstream channel.

    Transpose the first two axes back to [time, station, ...] before replaying
    ``model.residual.model._memory_states(encoded, age, support)``. Padding and
    rolling-window semantics remain inside that existing temporal method.
    Every module's original training flag is restored, including on failure.
    ``progress`` receives a dictionary after the full view and each source fold.
    """
    residual = model.residual
    _check_model(residual)
    shape = tuple(dataset["y"].shape)
    if len(shape) != 2 or tuple(residual.inputs.y_model.shape) != shape:
        raise ValueError("dataset dimensions do not match the frozen residual model")
    n, months = shape
    train = np.asarray(split["train"])
    if (train.ndim != 1 or not len(train) or not np.issubdtype(train.dtype, np.integer)
            or (train < 0).any() or (train >= n * months).any()
            or len(np.unique(train)) != len(train)):
        raise ValueError("source train cells must have unique valid integer identities")
    source_ids = np.unique(train // months).astype(np.int64)
    folds = [np.asarray(stations) for stations in model.rf.folds]
    if not folds or any(stations.ndim != 1 or not len(stations)
                        or not np.issubdtype(stations.dtype, np.integer) for stations in folds):
        raise ValueError("saved station folds must be nonempty integer arrays")
    if not np.array_equal(np.sort(np.concatenate(folds)), source_ids):
        raise ValueError("saved station folds must partition source stations exactly once")

    def report(stage, **details):
        if progress is not None:
            progress({"stage": stage, **details})

    training_states = [(module, module.training) for module in residual.model.modules()]
    residual.model.eval()
    try:
        with torch.inference_mode():
            full = _extract_view(residual, split)
            report("full_temporal_inputs", shape=list(full["encoded"].shape))
            source = {
                name: np.empty((len(source_ids), *array.shape[1:]), dtype=array.dtype)
                for name, array in full.items()
            }
            for fold_id, stations in enumerate(folds):
                stations = stations.astype(np.int64, copy=False)
                view = _extract_view(residual, fold_split(split, stations, months))
                positions = np.searchsorted(source_ids, stations)
                for name, array in view.items():
                    source[name][positions] = array[stations]
                report("source_fold", fold=fold_id, stations=stations.tolist())
                del view
    finally:
        # Restore mixed parent/child modes rather than recursively overwriting
        # intentionally frozen child modules with the parent's original mode.
        for module, training in training_states:
            module.training = training
    return {**{f"full_{name}": array for name, array in full.items()},
            **{f"source_{name}": array for name, array in source.items()},
            "source_station_ids": source_ids}
