"""Fresh retained native DOC and ecological stages for station confirmation.

This helper refits exactly two neural branches: the 30-epoch interaction
initializer used to select the ecological profile, and the 120-epoch retained
native encoder. It reads no target query labels and fits no chemistry/tree
models. Full 550-dimensional head features remain streamed by the downstream
chemical stage; only observed source/validation views are saved here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_distribution_head_v1 import observed_features
from run_unified_doc_spatial import bind_files, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import (
    support_query_cells,
    validate_unified_spatial_split,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features

INTERACTION_INDICES = (0, 2, 4, 28, 30, 31, 32)


def _array_identity(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _state_identity(module):
    return {name: _array_identity(value.detach().cpu().numpy())
            for name, value in module.state_dict().items()}


def _inputs_identity(inputs):
    return {name: _array_identity(value) for name, value in inputs.items()}


def _emit(progress, stage, row=None):
    if progress is not None:
        progress({"stage": stage, **(row or {})})


def _fixed_config(path, config):
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Native-stage inputs/settings changed: {path}")
    else:
        write_json(path, config)


def _fit_cached(directory, name, model, arrays, threshold, config, progress):
    completion = f"{name}_complete.json"
    if (directory / completion).exists():
        verify_files(directory, completion, config)
        loaded = type(model).from_payload(torch.load(directory / f"{name}.pt",
                                                     map_location="cpu", weights_only=False))
        if loaded.to_dict() != json.loads((directory / f"{name}.json").read_text()):
            raise ValueError(f"Checkpoint summary differs for {name}")
        _emit(progress, name, {"status": "reused verified checkpoint"})
        return loaded
    model.fit(*arrays, tail_threshold=threshold, selection_role="source_validation",
              progress=lambda row: _emit(progress, name, row))
    torch.save(model.to_payload(), directory / f"{name}.pt")
    write_json(directory / f"{name}.json", model.to_dict())
    pd.DataFrame(model.to_dict()["trace"]).to_csv(directory / f"{name}_trace.csv", index=False)
    bind_files(directory, completion,
               [directory / f"{name}{suffix}" for suffix in (".pt", ".json", "_trace.csv")], config)
    return model


def _combine(base, delta, scale):
    if scale == 0:
        return base.copy()
    result = np.maximum(0, base + scale * delta)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite fresh native prediction")
    return result


def fit_native_and_ecology(run, dataset, split, seed, initial, *, daily,
                           daily_metadata, smoke=False, progress=None,
                           runtime_snapshot_hash=None):
    """Fit/load retained native states on fresh roles and return chemical inputs.

    ``initial`` is the fresh ``fit_initial_and_basis`` result, including expert,
    flat native context, source-only ``oof_z`` and cached temporal input views.
    The caller validates alignment/identity of the fixed eight-channel daily
    pack. ``smoke`` reduces both neural caps to one epoch, in a separate root.

    Returned ``source_base`` is source OOF forest plus the source-trained neural
    correction; it is not a fully cross-fitted neural prediction. All flat cells
    use station-major order. ``source_features`` and ``validation_features`` are
    actual selected-head vectors of width550, using historical 512-cell feature
    extraction batches; full native prediction uses the historical 2048 batch.
    ``full_inputs`` and ``source_inputs`` are raw encoder dictionaries, allowing
    downstream full-grid feature streaming without allocating [N*T,550].
    """
    started = time.monotonic()
    directory = Path(run) / "native"
    directory.mkdir(parents=True, exist_ok=True)
    validate_unified_spatial_split(dataset["y_mask"], split)
    shape = tuple(np.asarray(dataset["y_mask"]).shape)
    n, months = shape
    train = np.sort(np.asarray(split["train"], dtype=np.int64))
    _, val_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    val_cells = np.sort(val_cells)
    source_ids, val_ids = np.unique(train // months), np.unique(val_cells // months)
    if not len(train) or not len(val_cells):
        raise ValueError("Native fitting requires source and fixed validation query cells")
    labels = np.asarray(dataset["y"], dtype=np.float64)
    if labels.shape != shape:
        raise ValueError("DOC values and mask must align")
    train_y, val_y = labels.ravel()[train], labels.ravel()[val_cells]
    if (not np.isfinite(train_y).all() or not np.isfinite(val_y).all()
            or (train_y < 0).any() or (val_y < 0).any()):
        raise ValueError("Source/validation DOC must be finite and nonnegative")
    # Neither unreserved validation support labels nor test labels are supplied
    # to the two native fits. The feature view itself receives train DOC only.
    fit_y = np.zeros(shape, dtype=np.float64)
    fit_y.ravel()[train], fit_y.ravel()[val_cells] = train_y, val_y
    input_y = np.zeros_like(labels)
    input_y.ravel()[train] = train_y
    input_dataset = {**dataset, "y": input_y}
    train_mask, val_mask = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    train_mask.ravel()[train], val_mask.ravel()[val_cells] = True, True
    threshold = float(np.quantile(train_y, .9))
    expert = initial["expert"]
    context = np.asarray(initial["context"], dtype=np.float64).reshape(-1)
    oof = np.asarray(initial["oof_z"], dtype=np.float64)
    if (context.shape != (n * months,) or not np.isfinite(context).all()
            or (context < 0).any() or oof.shape != shape
            or not np.isfinite(oof[train_mask]).all() or not np.isnan(oof[~train_mask]).all()):
        raise ValueError("Fresh context/OOF predictions must align and OOF cover exactly train")
    source_oof = np.maximum(0, np.expm1(oof.ravel()[train]))
    source_base_grid = np.full(shape, np.nan)
    source_base_grid.ravel()[train] = source_oof
    daily = np.asarray(daily)
    if (daily.shape != (*shape, 8) or daily.dtype != np.float32
            or not np.isfinite(daily).all() or ((daily < 0) | (daily > 1)).any()
            or not np.isin(daily[..., 5:8], (0, 1)).all()
            or daily_metadata["value_feature_indices"] != [0, 1, 2]
            or daily_metadata["availability_feature_indices"] != [3, 4, 5, 6, 7]):
        raise ValueError("Daily pack must preserve the fixed eight-channel definition")
    if np.any(daily[~np.asarray(dataset["x_mask"])[..., 1].astype(bool)] != 0):
        raise ValueError("Daily features must retain the monthly discharge footprint")
    flow = build_causal_flow_features(input_dataset)
    if tuple(flow["value_feature_indices"]) != (0, 2, 4):
        raise ValueError("Interaction initializer flow definition changed")
    extra = build_regime_head_features(dataset["regime"], train, source_oof,
        context.reshape(shape), flow["full"], n_months=months)
    raw = extract_raw_temporal_inputs(expert, input_dataset, split)
    features = initial["features"]
    for values in (raw["source_station_ids"], features["source_station_ids"],
                   extra["source_station_ids"], initial["source_station_ids"]):
        np.testing.assert_array_equal(source_ids, values)
    source_inputs = {key: raw[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    source_inputs["extra"] = np.concatenate((extra["source_extra"], daily[source_ids]), axis=-1)
    full_inputs = {key: raw[f"full_{key}"] for key in ("raw", "age", "support")}
    full_inputs.update(env=raw["env"], extra=np.concatenate((extra["full_extra"], daily), axis=-1))
    validation_inputs = {key: value[val_ids] for key, value in full_inputs.items()}
    encoded_source = {key: features[f"source_{key}"] for key in ("encoded", "age", "support")}
    encoded_source["extra"] = flow["full"][source_ids]
    encoded_full = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
    encoded_full["extra"] = flow["full"]
    encoded_val = {key: value[val_ids] for key, value in encoded_full.items()}
    def arrays(source, validation):
        return (source, source_base_grid[source_ids], fit_y[source_ids], train_mask[source_ids],
                validation, context.reshape(shape)[val_ids], fit_y[val_ids], val_mask[val_ids])

    shared = {"seed": int(seed), "lookback": 12, "patience": 5, "batch_size": 512,
              "learning_rate": 1e-4, "head_learning_rate": 1e-3, "tail_weight": 2,
              "scales": [0, .25, .5, 1], "train_memory": True}
    interaction_config = {**shared, "epochs": 1 if smoke else 30,
                          "extra_dim": 10, "interaction_indices": [0, 2, 4]}
    off_config = {**shared, "epochs": 1 if smoke else 120, "extra_dim": 38,
                  "interaction_indices": list(INTERACTION_INDICES),
                  "encoder_mode": "last_self_ecology", "encoder_learning_rate": 1e-5}
    feature_definition = {key: value for key, value in extra.items()
                          if key not in ("source_extra", "full_extra", "source_station_ids")}
    feature_definition.update({"source_station_ids": source_ids.tolist(),
        "feature_names": [*extra["feature_names"], *daily_metadata["feature_names"]],
        "daily_feature_names": daily_metadata["feature_names"],
        "daily_value_feature_indices": [30, 31, 32],
        "daily_availability_feature_indices": [33, 34, 35, 36, 37],
        "daily_feature_policy": daily_metadata["policy"], "extra_dim": 38,
        "combined_interaction_indices": list(INTERACTION_INDICES)})
    snapshot = runtime_code_snapshot()
    snapshot[str(Path(__file__).resolve().relative_to(Path.cwd()))] = sha256_file(__file__)
    snapshot["scripts/run_doc_distribution_head_v1.py"] = sha256_file("scripts/run_doc_distribution_head_v1.py")
    config = {"stage": "fresh_native_and_ecological", "seed": int(seed), "smoke": bool(smoke),
        "runtime_snapshot_hash": runtime_snapshot_hash, "execution_sources": snapshot,
        "torch_threads": torch.get_num_threads(), "shape": list(shape),
        "source_cells": _array_identity(train), "validation_cells": _array_identity(val_cells),
        "source_labels": _array_identity(train_y), "validation_labels": _array_identity(val_y),
        "source_oof_native": _array_identity(source_oof), "context": _array_identity(context),
        "source_inputs": _inputs_identity(source_inputs), "full_inputs": _inputs_identity(full_inputs),
        "encoded_source_inputs": _inputs_identity(encoded_source),
        "encoded_full_inputs": _inputs_identity(encoded_full), "daily_metadata": daily_metadata,
        "ecological_regime": _array_identity(np.asarray(dataset["regime"])),
        "initial_spatial": _state_identity(expert.residual.model.spatial),
        "initial_temporal": _state_identity(expert.residual.model.temporal),
        "initial_decay": _state_identity(expert.residual.model.decay),
        "interaction_tuned": interaction_config, "off": off_config,
        "q90_threshold_train": threshold, "inference_roles": ["train"],
        "selection_role": "source_validation", "ecological_mode": "ecological_affine",
        "profile_initializer": "interaction_tuned", "profile_k_grid": [20, 40, 80],
        "profile_ridge_grid": [.1, 1], "profile_gamma_grid": [0, .25, .5, 1],
        "source_neural_status": "source-trained; forest baseline is station-blocked OOF",
        "full_prediction_batch_size": 2048, "head_feature_batch_size": 512}
    _fixed_config(directory / "config.json", config)
    write_json(directory / "feature_definition.json", feature_definition)
    write_json(directory / "flow_definition.json", {key: value for key, value in flow.items() if key != "full"})
    interaction = _fit_cached(directory, "interaction_tuned", NativeTemporalResidual(
        expert.residual.model.temporal, expert.residual.model.decay, **interaction_config),
        arrays(encoded_source, encoded_val), threshold, config, progress)
    interaction_delta = interaction.predict_delta(encoded_full).ravel()
    interaction_point = _combine(context, interaction_delta, interaction.selected_scale_)
    if (directory / "ecological_complete.json").exists():
        verify_files(directory, "ecological_complete.json", config)
        profile = EcologicalResidualTransfer.from_dict(json.loads((directory / "ecological_affine.json").read_text()))
    else:
        _emit(progress, "ecological_affine", {"status": "fitting fresh source profiles"})
        profile = EcologicalResidualTransfer(mode="ecological_affine").fit(
            dataset["regime"], train, source_oof, train_y, n_months=months,
            validation_cells=val_cells, validation_y=val_y, validation_context=context[val_cells],
            validation_temporal=interaction_point[val_cells], selection_role="source_validation")
        write_json(directory / "ecological_affine.json", profile.to_dict())
        bind_files(directory, "ecological_complete.json", [directory / "ecological_affine.json"], config)
    memory = profile.predict_delta(context.reshape(shape)).ravel()
    off = _fit_cached(directory, "off", EncoderNativeResidual(expert.residual.model.spatial,
        expert.residual.model.temporal, expert.residual.model.decay, **off_config),
        arrays(source_inputs, validation_inputs), threshold, config, progress)
    point_delta = off.predict_delta(full_inputs).ravel()
    point = _combine(context, point_delta, off.selected_scale_)
    local_cells = np.flatnonzero(train_mask[source_ids])
    source_cells = source_ids[local_cells // months] * months + local_cells % months
    np.testing.assert_array_equal(source_cells, train)
    source_features, source_delta = observed_features(off, source_inputs, local_cells)
    validation_features, _ = observed_features(off, full_inputs, val_cells)
    source_base = _combine(source_oof, source_delta, off.selected_scale_)
    if (source_features.shape != (len(train), 550)
            or validation_features.shape != (len(val_cells), 550)):
        raise ValueError("Retained native head must expose exactly550 features")
    components = {"context": context, "point": point, "point_delta": point_delta,
                  "memory": memory, "interaction_tuned": interaction_point,
                  "interaction_delta": interaction_delta}
    compact = {"source_features": source_features, "source_base": source_base,
               "source_delta": source_delta, "source_oof_native": source_oof,
               "source_cells": source_cells, "source_local_cells": local_cells,
               "source_station_ids": source_ids, "validation_features": validation_features,
               "validation_cells": val_cells, "validation_base": point[val_cells]}
    for values in (*components.values(), *compact.values()):
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite native stage artifact")
    if (directory / "complete.json").exists():
        verify_files(directory, "complete.json", config)
        for filename, expected in (("native_components.npz", components), ("source_views.npz", compact)):
            with np.load(directory / filename, allow_pickle=False) as saved:
                for key, values in expected.items():
                    np.testing.assert_array_equal(saved[key], values)
    else:
        np.savez(directory / "native_components.npz", **components)
        np.savez(directory / "source_views.npz", **compact)
        write_json(directory / "timing.json", {"elapsed_seconds": time.monotonic() - started})
        files = [path for path in directory.iterdir() if path.is_file() and path.name != "complete.json"]
        bind_files(directory, "complete.json", sorted(files), config)
    _emit(progress, "native_complete", {"elapsed_seconds": time.monotonic() - started,
                                       "selected_off_scale": off.selected_scale_})
    return {"native_dir": directory, "model": off, "profile": profile,
            "source_inputs": source_inputs, "full_inputs": full_inputs,
            "daily": daily, "daily_metadata": daily_metadata, "q90_threshold_train": threshold,
            "legacy_basis": initial["legacy_basis"], **components, **compact}
