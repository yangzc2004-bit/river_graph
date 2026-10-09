"""Fit the frozen DOC recipe on supplied source train/validation stations.

This source-only trainer reuses the exact geographical helpers and budgets.
New-site or external labels are not accepted by the fitting interface.
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_geographical_confirmation_v1 import (
    MODELS,
    fit_memory,
    fit_native,
    select_tree,
)
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_station_data import fit_context_oof
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
    station_folds,
)
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.station_adapted_hybrid import station_residual_correction
from river_graph.models.unified_doc import UnifiedDOCReconstructor


def validate_source_roles(dataset, split):
    """Require station-disjoint source fitting roles and no external/test cells."""
    observed = np.asarray(dataset["y_mask"], bool)
    months = observed.shape[1]
    for role in ("train", "val", "test", "context"):
        cells = np.asarray(split[role])
        if (cells.ndim != 1 or cells.dtype.kind not in "iu" or (cells < 0).any()
                or (cells >= observed.size).any() or len(np.unique(cells)) != len(cells)
                or not observed.ravel()[cells].all()):
            raise ValueError("source role cells must be unique valid observed identities")
    if len(split["test"]) or len(split["context"]):
        raise ValueError("source-only fit accepts train/val roles; test/context must be empty")
    train_sites = np.unique(split["train"]//months)
    val_sites = np.unique(split["val"]//months)
    if len(train_sites) < 5 or not len(val_sites) or np.intersect1d(train_sites, val_sites).size:
        raise ValueError("source training and validation stations must be disjoint and nonempty")
    for role in ("train", "val"):
        labels = np.asarray(dataset["y"]).ravel()[split[role]]
        if not np.isfinite(labels).all() or (labels < 0).any():
            raise ValueError("source DOC labels must be finite and nonnegative")
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    if not len(query):
        raise ValueError("source validation needs stations with at least six DOC observations")


def source_support_policies(bases, dataset, split):
    """Select each procedure's support shrinkage on source validation only."""
    months = dataset["y"].shape[1]
    truth = np.asarray(dataset["y"]).ravel()
    states = {}
    for name, grid in bases.items():
        flat = grid.ravel()
        states[name] = {}
        for k in (0, 1, 3, 5):
            support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
            candidates = []
            for alpha in ((0.,) if k == 0 else (0., .25, .5, .75, 1.)):
                pred = station_residual_correction(flat[query], query, flat[support], support,
                    truth[support], n_months=months, alpha=alpha)
                candidates.append({"alpha": alpha, "validation_mae": float(np.abs(pred-truth[query]).mean())})
            states[name][str(k)] = {"selected": min(candidates, key=lambda v: (v["validation_mae"], v["alpha"])),
                                   "candidates": candidates}
    return states


def fit_source_recipe(*, dataset_path, mask_path, daily_path, run, seed, runtime,
                      progress_path=None):
    """Train one saved source package with no independent-target scoring."""
    dataset_path, mask_path, daily_path, run = map(Path, (dataset_path, mask_path, daily_path, run))
    run.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = torch.load(dataset_path, weights_only=False, map_location="cpu")
    with np.load(mask_path, allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    validate_source_roles(source, split)
    shape = tuple(source["y"].shape)
    months = shape[1]
    threshold = float(np.quantile(np.asarray(source["y"]).ravel()[split["train"]], .9))
    config = {"experiment": "doc_portable_source_fit_v1", "seed": seed, "split_seed": None,
        "dataset_path": str(dataset_path), "dataset_hash": sha256_file(dataset_path),
        "mask_path": str(mask_path), "mask_hash": sha256_file(mask_path),
        "daily_path": str(daily_path), "daily_hash": sha256_file(daily_path),
        "runtime_snapshot_hash": runtime, "started_at": datetime.now(timezone.utc).isoformat(),
        "q90_threshold_train": threshold, "models": list(MODELS),
        "source_backbone_epochs": 20, "current_native_epochs": 120, "candidate_native_epochs": 30,
        "patience": 5, "selection_role": "source_validation",
        "evaluation_role": "source_validation_only", "information_condition": "no target water-quality inputs",
        "candidate_choice": "fixed integrated recipe; no external model selection"}
    if (run/"config.json").exists():
        old = json.loads((run/"config.json").read_text())
        if any(old[k] != config[k] for k in ("runtime_snapshot_hash", "dataset_hash", "mask_hash", "daily_hash", "seed")):
            raise ValueError("saved source fit changed; preserve it and use a new version")
        config = old
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return run
    dataset = strip_auxiliary_water(source)
    dataset["y"] = development_labels(dataset, split)
    del source
    with np.load(daily_path, allow_pickle=False) as saved:
        daily = saved["full"].copy()
    if daily.shape != (*shape, 8) or not np.isfinite(daily).all():
        raise ValueError("source daily-hydro input must be finite [station,month,8]")

    def progress(stage, row):
        entry = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        if progress_path is not None:
            write_json(Path(progress_path), entry)
        print(json.dumps(entry), flush=True)

    # Fresh source fit; all neural and forest settings match the fixed recipe.
    backbone_dir = run / "backbone"
    if (run / "backbone_complete.json").exists():
        verify_files(run, "backbone_complete.json", config)
        expert = UnifiedDOCReconstructor.load(backbone_dir, dataset, split)
    else:
        expert = UnifiedDOCReconstructor(seed=seed, n_jobs=2, max_epochs=20,
            epoch_callback=lambda row: progress("source_backbone", row)).fit(dataset, split)
        expert.save(backbone_dir)
        bind_files(run, "backbone_complete.json", list(backbone_dir.iterdir()), config)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    folds = station_folds(split["train"], months, seed)
    train, val = split["train"], split["val"]
    labels = np.asarray(dataset["y"]).ravel()
    # Retain the current model's 39-feature environmental context predictor.
    x39 = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    legacy_context = np.maximum(0, np.expm1(expert.context_forest.predict(x39))).reshape(shape)
    if (run / "trees_complete.json").exists():
        verify_files(run, "trees_complete.json", config)
        hidden_tree = joblib.load(run / "station_hidden_trees.joblib")
        matched_tree = joblib.load(run / "matched_daily_trees.joblib")
        with np.load(run / "oof.npz", allow_pickle=False) as saved:
            old_oof, new_oof = saved["old"].copy(), saved["new"].copy()
    else:
        old_fit = fit_context_oof(expert, dataset, split, progress=lambda row: progress("current_oof", row))
        old_oof = old_fit["pred_z"]
        x47 = np.column_stack([x39, daily.reshape(-1, 8)])
        matched_tree, matched_scores = select_tree(x47[train], x47, labels[train], labels[val], val, seed=seed)
        hidden_training, hidden_inference = station_hidden_tree_inputs(dataset, split, folds, daily)
        hidden_tree, hidden_scores = select_tree(hidden_training, hidden_inference, labels[train], labels[val], val, seed=seed)
        new_oof = np.full(shape, np.nan)
        records = []
        for fold_index, held_stations in enumerate(folds):
            outer = fold_split(split, held_stations, months)
            fit, inference = station_hidden_tree_inputs(dataset, outer,
                station_folds(outer["train"], months, seed), daily)
            tree = clone(hidden_tree).fit(fit, np.log1p(labels[outer["train"]]))
            selected = train[np.isin(train//months, held_stations)]
            new_oof.ravel()[selected] = tree.predict(inference[selected])
            records.append({"fold": fold_index, "hidden_stations": held_stations.tolist(),
                            "fitted_stations": np.unique(outer["train"]//months).tolist()})
            progress("station_hidden_oof", {"fold": fold_index, "query_cells": len(selected)})
            del fit, inference, tree
        joblib.dump(hidden_tree, run / "station_hidden_trees.joblib", compress=3)
        joblib.dump(matched_tree, run / "matched_daily_trees.joblib", compress=3)
        np.savez_compressed(run / "oof.npz", old=old_oof, new=new_oof)
        write_json(run / "oof_records.json", {"current": old_fit["fold_records"], "station_hidden": records})
        write_json(run / "tree_selection.json", {"matched": matched_scores, "station_hidden": hidden_scores})
        bind_files(run, "trees_complete.json", [run / name for name in ("station_hidden_trees.joblib",
            "matched_daily_trees.joblib", "oof.npz", "oof_records.json", "tree_selection.json")], config)
        del hidden_training, hidden_inference, x47
    inference47 = np.column_stack([x39, daily.reshape(-1, 8)])
    new_context = np.maximum(0, np.expm1(hidden_tree.predict(inference47))).reshape(shape)
    matched_context = np.maximum(0, np.expm1(matched_tree.predict(inference47))).reshape(shape)
    del inference47, x39, hidden_tree, matched_tree
    if (run / "current_complete.json").exists():
        verify_files(run, "current_complete.json", config)
        current_native = EncoderNativeResidual.from_payload(torch.load(run / "current_native.pt", weights_only=False))
        with np.load(run / "current_predictions.npz", allow_pickle=False) as saved:
            current = saved["integrated"].copy()
    else:
        current_native, base, _, normalization, _ = fit_native(dataset, split, features, daily,
            legacy_context, old_oof, expert.residual.model, seed=seed, epochs=120, threshold=threshold,
            progress=lambda row: progress("current_native", row))
        memory = fit_memory(dataset, split, old_oof, legacy_context, base, months)
        current = memory.predict(legacy_context, base)
        torch.save(current_native.to_payload(), run / "current_native.pt")
        write_json(run / "current_native.json", current_native.to_dict())
        write_json(run / "current_memory.json", memory.to_dict())
        write_json(run / "current_readout_normalization.json", normalization)
        np.savez_compressed(run / "current_predictions.npz", integrated=current)
        bind_files(run, "current_complete.json", [run / name for name in ("current_native.pt", "current_native.json",
            "current_memory.json", "current_readout_normalization.json", "current_predictions.npz")], config)
    candidate, native, _delta, normalization, full_inputs = fit_native(dataset, split, features, daily,
        new_context, new_oof, current_native, seed=seed, epochs=30, threshold=threshold,
        progress=lambda row: progress("candidate_native", row))
    candidate_memory = fit_memory(dataset, split, new_oof, new_context, native, months)
    integrated = candidate_memory.predict(new_context, native)
    torch.save(candidate.to_payload(), run / "native.pt")
    write_json(run / "native.json", candidate.to_dict())
    write_json(run / "memory.json", candidate_memory.to_dict())
    write_json(run / "readout_normalization.json", normalization)
    bases = {"current_model": current, "matched_daily_trees": matched_context,
             "station_hidden_trees": new_context, "unmonitored_residual": native, "unmonitored_integrated": integrated}
    bases = {"current_model": current, "matched_daily_trees": matched_context,
             "station_hidden_trees": new_context, "unmonitored_residual": native,
             "unmonitored_integrated": integrated}
    adapters = source_support_policies(bases, dataset, split)
    write_json(run/"support_adapters.json", adapters)
    cells = split["val"]
    names, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    rows = [pd.DataFrame({"cell": cells, "station": names[cells//months], "month": dates[cells % months],
        "analyte": "doc", "y_pred": grid.ravel()[cells], "y_true": np.asarray(dataset["y"]).ravel()[cells],
        "model_name": name, "seed": seed, "k": 0, "visibility_role": "source_validation"})
        for name, grid in bases.items()]
    pd.concat(rows, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    np.savez_compressed(run/"components.npz", environment=new_context, native=native,
        integrated=integrated, current=current, matched_trees=matched_context)
    ids = np.unique(cells//months)
    np.savez_compressed(run/"validation_inputs.npz", **{key: value[ids] for key, value in full_inputs.items()},
                        context=new_context[ids], cell=cells)
    files = ["config.json", "native.pt", "native.json", "memory.json", "readout_normalization.json",
        "components.npz", "validation_inputs.npz", "support_adapters.json", "backbone_complete.json",
        "trees_complete.json", "current_complete.json"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"models": len(MODELS), "validation_cells": len(cells), "best_epoch": candidate.best_epoch_})
    del features, expert, dataset, candidate, current_native, full_inputs
    gc.collect()
    return run
