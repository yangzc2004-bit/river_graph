"""Fixed-visibility raw inputs for adapting the existing DOC spatial encoder.

The saved expert's input preprocessing and ecological normalization are reused
unchanged. Source stations use their saved RF fold's hidden-DOC view; these are
input views, not independently fitted out-of-fold neural models.
"""

from __future__ import annotations

import numpy as np
import torch

from river_graph.models.episodic_temporal_features import _check_model
from river_graph.models.kgml_local_transport import FIT_ROLES, fold_split


def _raw_view(residual, split):
    x, age = residual.input_view(split, FIT_ROLES)
    support = torch.stack([x[..., 9], x[..., -2], x[..., -1]], dim=-1)
    arrays = {
        "raw": x.permute(1, 0, 2).contiguous().cpu().numpy().copy(),
        "age": age.T.contiguous().cpu().numpy().copy(),
        "support": support.permute(1, 0, 2).contiguous().cpu().numpy().copy(),
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("nonfinite raw temporal input")
    return arrays


def extract_raw_temporal_inputs(expert, dataset, split, progress=None) -> dict:
    """Return raw full-cohort and station-fold-hidden source input caches.

    ``full_raw`` has shape [N,T,C], ``full_age`` [N,T], and ``full_support``
    [N,T,3]. Corresponding ``source_*`` arrays have Ns stations indexed by
    sorted ``source_station_ids``. ``env`` [N,E] and ``source_env`` [Ns,E]
    copy the expert's existing ``inputs.env_raw`` exactly; its historical
    normalization is neither re-estimated nor replaced with source-only fits.

    Raw and support channels exactly match the existing M1 no-message GRU.
    The three support channels are local visibility, upstream support and
    downstream support (the latter is already zeroed by the KGML input view).
    Source views hide every DOC observation at the held station fold. Full
    views use FIT_ROLES. No target labels or normalization statistics are fit
    here, and every module's original training flag is restored on return or
    failure. ``progress`` receives full-view and source-fold dictionaries.
    """
    residual = expert.residual
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
    folds = [np.asarray(stations) for stations in expert.rf.folds]
    if not folds or any(stations.ndim != 1 or not len(stations)
                        or not np.issubdtype(stations.dtype, np.integer) for stations in folds):
        raise ValueError("saved station folds must be nonempty integer arrays")
    if not np.array_equal(np.sort(np.concatenate(folds)), source_ids):
        raise ValueError("saved station folds must partition source stations exactly once")
    env_raw = residual.inputs.env_raw
    if env_raw is None or env_raw.ndim != 2 or env_raw.shape[0] != n:
        raise ValueError("raw temporal extraction requires the saved station ecology input")

    def report(stage, **details):
        if progress is not None:
            progress({"stage": stage, **details})

    training_states = [(module, module.training) for module in residual.model.modules()]
    residual.model.eval()
    try:
        with torch.inference_mode():
            env = env_raw.detach().cpu().numpy().copy()
            if not np.isfinite(env).all():
                raise ValueError("nonfinite saved ecological input")
            full = _raw_view(residual, split)
            report("full_raw_temporal_inputs", shape=list(full["raw"].shape))
            source = {
                name: np.empty((len(source_ids), *array.shape[1:]), dtype=array.dtype)
                for name, array in full.items()
            }
            for fold_id, stations in enumerate(folds):
                stations = stations.astype(np.int64, copy=False)
                view = _raw_view(residual, fold_split(split, stations, months))
                positions = np.searchsorted(source_ids, stations)
                for name, array in view.items():
                    source[name][positions] = array[stations]
                report("source_fold", fold=fold_id, stations=stations.tolist())
                del view
    finally:
        for module, training in training_states:
            module.training = training
    return {**{f"full_{name}": array for name, array in full.items()},
            **{f"source_{name}": array for name, array in source.items()},
            "env": env, "source_env": env[source_ids].copy(),
            "source_station_ids": source_ids}
