"""Refit the retained initial DOC expert, source OOF forest and legacy basis.

This helper is deliberately not a historical experiment dispatcher. It fits
only the branches used by the accepted chemical reconstruction recipe. No
pretrained model, target-label prediction product or prior fitted normalizer is
loaded. The driver owns the new station partition and final evaluation.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import (
    bind_files,
    covariate_diagnostics,
    digest,
    verify_files,
    write_json,
)

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.episodic_station_adapter import EpisodicStationProjector
from river_graph.models.episodic_station_data import fit_context_oof
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.kgml_local_transport import fold_split
from river_graph.models.unified_doc import UnifiedDOCReconstructor
from river_graph.models.unified_doc_features import extract_gru_hidden


def retained_initial_recipe(*, smoke=False):
    """Executed branch settings; smoke only reduces epochs and tree count."""
    return {
        "initial": {"n_estimators": 20 if smoke else 300, "max_epochs": 1 if smoke else 20,
                    "patience": 5, "hidden": 64, "chunk_months": 64},
        "initial_structure": {"base_variant": "local", "edge_set": "empty", "edge_direction": "upstream",
            "temporal_operator": "gru", "lookback": 12, "dropout": .1, "lr": .001,
            "forest_backend": "extra_trees", "forest_min_samples_leaf": 4, "forest_max_features": 1.},
        "context_candidates": ["et_leaf4", "et_leaf2", "et_leaf1", "et_sqrt"],
        "context_oof_folds": 5,
        "projector": {"n_components": 16, "output_dim": 2, "epochs": 1 if smoke else 100,
                      "patience": 15, "learning_rate": .01, "batch_size": 32},
        "memory": {"lookback": 12, "epochs": 1 if smoke else 30, "patience": 5,
                   "learning_rate": 1e-4, "batch_size": 8, "anchor_count": 32, "scale_floor": 1e-4},
        "basis_training": {"k": [3, 5], "ridge": [1., 10.], "alpha": 1.,
            "source_weighting": "equal station", "selection": "pooled source-validation query native MAE",
            "validation_mask": "all split.val cells; support/query reservation inside existing learner",
            "source_neural_status": "source-trained weights, source-fold-hidden inputs; not OOF-fitted neural experts"},
    }


def _content_hash(value):
    """Hash in-memory supplied inputs, including array dtype/shape and order."""
    hasher = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            hasher.update(f"array:{item.dtype}:{item.shape}:".encode())
            if item.dtype.hasobject:
                add(item.tolist())
            else:
                hasher.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            hasher.update(b"dict:")
            for key in sorted(item):
                add(str(key)); add(item[key])
        elif isinstance(item, (tuple, list)):
            hasher.update(f"sequence:{len(item)}:".encode())
            for entry in item:
                add(entry)
        else:
            hasher.update(json.dumps(item, sort_keys=True, default=str, allow_nan=False).encode()+b";")

    add(value)
    return hasher.hexdigest()


def _stage(directory, name, common, upstream=None):
    directory.mkdir(parents=True, exist_ok=True)
    config = {"stage": name, **common, "upstream": upstream or {}}
    path = directory / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Refusing to reuse different initial/basis stage inputs: {directory}")
    else:
        write_json(path, config)
    complete = (directory / "complete.json").is_file()
    if complete:
        verify_files(directory, "complete.json", config)
    return config, complete


def _finish(directory, config, names, started):
    write_json(directory / "timing.json", {"elapsed_seconds": time.monotonic()-started})
    bind_files(directory, "complete.json", [directory / name for name in ("config.json", *names, "timing.json")], config)


def _safe_role_arrays(dataset, split):
    shape = tuple(dataset["y"].shape)
    truth = np.full(shape, np.nan, dtype=np.float64)
    masks = {}
    for role in ("train", "val"):
        cells = np.asarray(split[role], dtype=np.int64)
        truth.ravel()[cells] = np.asarray(dataset["y"]).ravel()[cells]
        mask = np.zeros(shape, dtype=bool)
        mask.ravel()[cells] = True
        masks[role] = mask
    if np.intersect1d(split["train"], split["val"]).size or not np.isnan(truth.ravel()[split["test"]]).all():
        raise ValueError("Source, validation and target label roles must be disjoint")
    return truth, masks


def _full_grid(expert, dataset, split, seed):
    components = expert.predict_components()
    n, months = dataset["y"].shape
    cells = np.arange(n*months)
    roles = np.full(len(cells), "unobserved", dtype=object)
    for role, selected in split.items():
        roles[selected] = role
    novelty, upstream = covariate_diagnostics(dataset, split)
    full = pd.DataFrame({"cell": cells, "station": np.asarray(dataset["site_no"], str)[cells//months],
        "month": np.asarray(dataset["months"], str)[cells % months], "visibility_role": roles,
        "input_roles": "train", "analyte": "doc", "seed": seed,
        "ecological_novelty": novelty[cells//months], "upstream_support": upstream.ravel()})
    for name, values in components.items():
        full[name] = np.asarray(values).ravel()
    if not np.isfinite(full.select_dtypes(include="number")).all().all():
        raise FloatingPointError("Nonfinite initial full-grid components")
    return full


def _gru_projector_features(expert, dataset, split, progress):
    """Keep old full-cohort fold batching while omitting unused tree features."""
    months = dataset["y"].shape[1]
    source_ids = np.unique(np.asarray(split["train"])//months)
    folds = [np.asarray(stations, dtype=np.int64) for stations in expert.rf.folds]
    if not np.array_equal(np.sort(np.concatenate(folds)), source_ids):
        raise ValueError("Initial forest folds must partition source stations")
    full = extract_gru_hidden(expert.residual, split, verify_head=True)
    source = np.empty((len(source_ids), *full.shape[1:]), dtype=full.dtype)
    for fold_id, stations in enumerate(folds):
        hidden = extract_gru_hidden(expert.residual, fold_split(split, stations, months), verify_head=True)
        source[np.searchsorted(source_ids, stations)] = hidden[stations]
        progress("projector_feature_fold", fold=fold_id)
    return source, full, source_ids


def projector_readout(projector):
    """Retained v3 whitened-PCA to v4 two-coordinate readout, without refit."""
    basis = projector.basis_
    weights = basis.components_.T / np.sqrt(np.maximum(basis.eigenvalues_, basis.eigenvalue_floor))
    weights[:, ~projector.active_modes_] = 0
    return weights @ projector.projection_.T


def fit_initial_and_basis(run, dataset, split, seed, *, n_jobs=4, torch_threads=2, smoke=False):
    """Fit/restore fresh initial, matched OOF, and retained GRU support stages.

    Return ``expert`` (UnifiedDOCReconstructor), flat native ``context`` and
    ``temporal``, ``full_grid`` descriptors/components, ``oof_z[N,T]`` (finite
    only at source train cells), ``features`` from extract_temporal_inputs,
    flat ``legacy_basis[N*T,2]``, sorted ``source_station_ids``, ``memory`` and
    stage paths. Native raw input/extra38 feature construction belongs to the
    downstream native helper. No historical fitted artifact is loaded.
    """
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch_threads)
    recipe = retained_initial_recipe(smoke=smoke)
    parent_config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else None
    common = {"seed": int(seed), "n_jobs": n_jobs, "torch_threads": torch_threads, "smoke": bool(smoke),
        "recipe": recipe, "dataset_content_hash": _content_hash(dataset), "split_content_hash": _content_hash(split),
        "helper_hash": sha256_file(__file__), "library_snapshot_hash": digest(runtime_code_snapshot()),
        "driver_config_hash": digest(parent_config) if parent_config is not None else None}

    def progress(stage, **record):
        print(json.dumps({"run": run.name, "stage": stage, **record}, allow_nan=False), flush=True)

    initial = run / "initial"
    config, completed = _stage(initial, "initial_expert", common)
    started = time.monotonic()
    if not completed:
        model_files = ["forests.joblib", "context.joblib", "residual.pt", "model.json", "adapter.json"]
        if (initial / "fitted_model.meta.json").exists():
            verify_files(initial, "fitted_model.meta.json", config)
            expert = UnifiedDOCReconstructor.load(initial, dataset, split)
        else:
            rf = None
            if (initial / "forest_cache.meta.json").exists():
                verify_files(initial, "forest_cache.meta.json", config)
                rf = joblib.load(initial / "forests.joblib")

            def save_forests(forests):
                joblib.dump(forests, initial / "forests.joblib", compress=3)
                bind_files(initial, "forest_cache.meta.json", [initial / "forests.joblib"], config)

            expert = UnifiedDOCReconstructor(seed=seed, n_jobs=n_jobs, **recipe["initial"],
                epoch_callback=lambda row: progress("initial_temporal", **row))
            expert.fit(dataset, split, rf=rf, forest_callback=save_forests)
            expert.save(initial)
            bind_files(initial, "forest_cache.meta.json", [initial / "forests.joblib"], config)
            bind_files(initial, "fitted_model.meta.json", [initial / name for name in model_files], config)
        full = _full_grid(expert, dataset, split, seed)
        full.to_parquet(initial / "full_grid.parquet", index=False)
        write_json(initial / "full_grid.meta.json", {"config_hash": digest(config),
            "dataset_content_hash": common["dataset_content_hash"], "split_content_hash": common["split_content_hash"],
            "prediction_sha256": sha256_file(initial / "full_grid.parquet"),
            "model_files": {name: sha256_file(initial / name) for name in model_files},
            "rows": len(full), "inference_roles": ["train"], "label_export": "none"})
        _finish(initial, config, [*model_files, "fitted_model.meta.json", "full_grid.parquet", "full_grid.meta.json"], started)
    else:
        expert = UnifiedDOCReconstructor.load(initial, dataset, split)
        full = pd.read_parquet(initial / "full_grid.parquet")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    source_ids = np.unique(np.asarray(split["train"])//months)
    validation_ids = np.unique(np.asarray(split["val"])//months)
    context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
    truth, masks = _safe_role_arrays(dataset, split)

    oof_dir = run / "source_oof"
    config, completed = _stage(oof_dir, "matched_context_oof", common,
        {"initial_completion_hash": sha256_file(initial / "complete.json")})
    started = time.monotonic()
    if not completed:
        oof = fit_context_oof(expert, dataset, split, n_jobs=n_jobs,
                              progress=lambda row: progress("selected_context_oof", detail=row))
        np.savez_compressed(oof_dir / "source_oof.npz", pred_z=oof["pred_z"])
        write_json(oof_dir / "folds.json", oof["fold_records"])
        _finish(oof_dir, config, ["source_oof.npz", "folds.json"], started)
    with np.load(oof_dir / "source_oof.npz", allow_pickle=False) as saved:
        oof_z = saved["pred_z"].copy()
    if (oof_z.shape != shape or not np.isfinite(oof_z[masks["train"]]).all()
            or not np.isnan(oof_z[~masks["train"]]).all()):
        raise ValueError("Matched context OOF predictions must cover source train cells only")

    basis_dir = run / "basis"
    inputs_dir = basis_dir / "inputs"
    upstream = {"initial_completion_hash": sha256_file(initial / "complete.json"),
                "oof_completion_hash": sha256_file(oof_dir / "complete.json")}
    config, completed = _stage(inputs_dir, "frozen_temporal_inputs", common, upstream)
    started = time.monotonic()
    if not completed:
        features = extract_temporal_inputs(expert, dataset, split,
                    progress=lambda row: progress("temporal_inputs", detail=row))
        np.savez_compressed(inputs_dir / "temporal_inputs.npz", **features)
        _finish(inputs_dir, config, ["temporal_inputs.npz"], started)
    else:
        with np.load(inputs_dir / "temporal_inputs.npz", allow_pickle=False) as saved:
            features = {name: saved[name].copy() for name in saved.files}
    np.testing.assert_array_equal(source_ids, features["source_station_ids"])

    projector_dir = basis_dir / "projector"
    config, completed = _stage(projector_dir, "gru_station_projector", common, upstream)
    started = time.monotonic()
    if not completed:
        source_gru, full_gru, feature_ids = _gru_projector_features(expert, dataset, split, progress)
        np.testing.assert_array_equal(feature_ids, source_ids)
        projector = EpisodicStationProjector(seed=seed, **recipe["projector"]).fit(
            source_gru, oof_z[source_ids], truth[source_ids], masks["train"][source_ids],
            full_gru[validation_ids], context.reshape(shape)[validation_ids], truth[validation_ids], masks["val"][validation_ids],
            selection_role="source_validation", progress=lambda row: progress("legacy_projector", **row))
        write_json(projector_dir / "projector.json", projector.to_dict())
        pd.DataFrame(projector.trace_).to_csv(projector_dir / "trace.csv", index=False)
        readout = projector_readout(projector)
        np.savez_compressed(projector_dir / "readout.npz", readout=readout)
        restored = EpisodicStationProjector.from_dict(projector.to_dict())
        np.testing.assert_array_equal(projector.transform(full_gru[:2]), restored.transform(full_gru[:2]))
        _finish(projector_dir, config, ["projector.json", "trace.csv", "readout.npz"], started)
        del source_gru, full_gru, projector, restored
    with np.load(projector_dir / "readout.npz", allow_pickle=False) as saved:
        readout = saved["readout"].copy()

    memory_dir = basis_dir / "memory"
    config, completed = _stage(memory_dir, "episodic_temporal_support_basis", common,
        {**upstream, "input_completion_hash": sha256_file(inputs_dir / "complete.json"),
         "projector_completion_hash": sha256_file(projector_dir / "complete.json")})
    started = time.monotonic()
    full_inputs = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
    if not completed:
        source = {key: features[f"source_{key}"] for key in full_inputs}
        validation = {key: value[validation_ids] for key, value in full_inputs.items()}
        memory = EpisodicTemporalAdapter(expert.residual.model.temporal, expert.residual.model.decay,
                                        readout, seed=seed, **recipe["memory"])
        memory.fit(source, oof_z[source_ids], truth[source_ids], masks["train"][source_ids],
            validation, context.reshape(shape)[validation_ids], truth[validation_ids], masks["val"][validation_ids],
            selection_role="source_validation", progress=lambda row: progress("legacy_memory", **row))
        torch.save(memory.to_payload(), memory_dir / "memory.pt")
        write_json(memory_dir / "memory.json", memory.to_dict())
        pd.DataFrame(memory.trace_).to_csv(memory_dir / "trace.csv", index=False)
        legacy = memory.transform(full_inputs).reshape(-1, 2)
        np.savez_compressed(memory_dir / "representations.npz", gru_tuned_anchor=legacy, source_station_ids=source_ids)
        write_json(memory_dir / "normalization.json", memory.normalization_stats(full_inputs))
        restored = EpisodicTemporalAdapter.from_payload(torch.load(memory_dir / "memory.pt", weights_only=False))
        small = {key: value[:2] for key, value in full_inputs.items()}
        np.testing.assert_array_equal(memory.transform(small), restored.transform(small))
        _finish(memory_dir, config, ["memory.pt", "memory.json", "trace.csv", "representations.npz", "normalization.json"], started)
    else:
        memory = EpisodicTemporalAdapter.from_payload(torch.load(memory_dir / "memory.pt", weights_only=False))
        with np.load(memory_dir / "representations.npz", allow_pickle=False) as saved:
            legacy = saved["gru_tuned_anchor"].copy()
            np.testing.assert_array_equal(saved["source_station_ids"], source_ids)
    if legacy.shape != (np.prod(shape), 2) or not np.isfinite(legacy).all():
        raise ValueError("Legacy support basis must cover the complete station/month grid")
    return {"expert": expert, "context": context, "temporal": temporal, "full_grid": full,
        "oof_z": oof_z, "features": features, "legacy_basis": legacy, "source_station_ids": source_ids,
        "memory": memory, "initial_dir": initial, "oof_dir": oof_dir, "basis_dir": basis_dir,
        "memory_dir": memory_dir, "stage_completion_files": [directory / "complete.json" for directory in
            (initial, oof_dir, inputs_dir, projector_dir, memory_dir)]}
