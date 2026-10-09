"""Refit the DOC station-hidden residual recipe on its original temporal tasks."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import INTERACTION_INDICES, load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, DATASET, select_tree
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    strip_auxiliary_water,
    validate_zero_observation_view,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_station_data import fit_context_oof
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
    station_folds,
)
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.station_adapted_hybrid import StationAdaptedHybrid
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_temporal_compatibility_v1")
MASKS = ("e2a_strict", "e2b_partial")
MODELS = ("original_hybrid", "context_trees", "matched_daily_trees", "station_hidden_trees",
          "current_native", "current_fusion", "upgraded_native", "upgraded_fusion")


def temporal_label_view(dataset, split):
    result = strip_auxiliary_water(dataset)
    labels = np.full(dataset["y"].shape, np.nan, dtype=np.float64)
    for role in ("train", "val", "context"):
        labels.ravel()[split[role]] = np.asarray(dataset["y"]).ravel()[split[role]]
    result["y"] = labels
    return result


def temporal_arrays(features, extra, daily, dataset, split, oof, context):
    """Station-hidden source training; legitimate local-history temporal inference."""
    ids, months = features["source_station_ids"], dataset["y"].shape[1]
    receiving = np.unique(split["val"]//months)
    source = {key: features[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    source["extra"] = np.concatenate([extra["source_extra"], daily[ids]], axis=-1)
    validate_zero_observation_view(source, np.arange(len(ids)))
    full = {key: features[f"full_{key}"] for key in ("raw", "age", "support")}
    full["env"] = features["env"]
    full["extra"] = np.concatenate([extra["full_extra"], daily], axis=-1)
    validation = {key: value[receiving] for key, value in full.items()}
    masks = {}
    for role in ("train", "val"):
        masks[role] = np.zeros(dataset["y"].shape, bool)
        masks[role].ravel()[split[role]] = True
    return full, (source, np.maximum(0, np.expm1(oof[ids])), dataset["y"][ids], masks["train"][ids],
        validation, context[receiving], dataset["y"][receiving], masks["val"][receiving])


def native_fit(dataset, split, features, daily, context, oof, backbone, *, seed, epochs, threshold, progress):
    months = dataset["y"].shape[1]
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=months)
    inputs, arrays = temporal_arrays(features, extra, daily, dataset, split, oof, context)
    model = EncoderNativeResidual(backbone.spatial, backbone.temporal, backbone.decay,
        seed=seed, epochs=epochs, patience=5, extra_dim=38, tail_weight=2,
        interaction_indices=INTERACTION_INDICES, encoder_mode="last_self_ecology", encoder_learning_rate=1e-5)
    model.fit(*arrays, tail_threshold=threshold, selection_role="source_validation", progress=progress)
    return model, model.predict(inputs, context)


def temporal_fusion(context, native, labels, split):
    selected = split["val"]
    model = StationAdaptedHybrid(n_months=context.shape[1]).fit_fusion(
        context.ravel()[selected], native.ravel()[selected], labels.ravel()[selected], selection_role="source_validation")
    return model, model.predict_components(context.ravel(), native.ravel())["hybrid"].reshape(context.shape)


def run_one(root, mask, seed, runtime):
    started = time.monotonic()
    run = root/"runs"/f"{mask}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    mask_path = Path("experiments/masks_stcore_v1")/f"{mask}.npz"
    truth_dataset = torch.load(DATASET, weights_only=False, map_location="cpu")
    with np.load(mask_path, allow_pickle=False) as saved:
        split = {role: saved[role].copy() if role in saved.files else np.empty(0, dtype=np.int64)
                 for role in ("train", "val", "test", "context")}
    truth = np.asarray(truth_dataset["y"], float).copy()
    threshold = float(np.quantile(truth.ravel()[split["train"]], .9))
    config = {"experiment": "doc_temporal_compatibility_v1", "mask": mask, "seed": seed,
        "dataset_path": str(DATASET), "dataset_hash": sha256_file(DATASET),
        "mask_path": str(mask_path), "mask_hash": sha256_file(mask_path),
        "runtime_snapshot_hash": runtime, "started_at": datetime.now(timezone.utc).isoformat(),
        "q90_threshold_train": threshold, "models": list(MODELS), "source_backbone_epochs": 20,
        "current_native_epochs": 120, "candidate_native_epochs": 30, "patience": 5,
        "inference_roles": list(FIT_ROLES), "selection_role": "source_validation",
        "query_policy": "exact existing temporal test cells; no K support added",
        "memory": "disabled: source/validation stations overlap", "scope": "retrospective temporal compatibility"}
    if (run/"config.json").exists():
        old = json.loads((run/"config.json").read_text())
        if old["runtime_snapshot_hash"] != runtime or old["mask_hash"] != config["mask_hash"]:
            raise ValueError("temporal execution changed; retain it and use a new version")
        config = old
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return

    def progress(stage, row):
        record = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    dataset = temporal_label_view(truth_dataset, split)
    del truth_dataset
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], truth.shape)
    backbone = run/"backbone"
    if (run/"backbone_complete.json").exists():
        verify_files(run, "backbone_complete.json", config)
        expert = UnifiedDOCReconstructor.load(backbone, dataset, split)
    else:
        expert = UnifiedDOCReconstructor(seed=seed, n_jobs=2, max_epochs=20,
            epoch_callback=lambda row: progress("backbone", row)).fit(dataset, split)
        expert.save(backbone)
        bind_files(run, "backbone_complete.json", list(backbone.iterdir()), config)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    folds, months = station_folds(split["train"], truth.shape[1], seed), truth.shape[1]
    train, val = split["train"], split["val"]
    x = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    context = np.maximum(0, np.expm1(expert.context_forest.predict(x))).reshape(truth.shape)
    if (run/"trees_complete.json").exists():
        verify_files(run, "trees_complete.json", config)
        matched = joblib.load(run/"matched.joblib")
        hidden = joblib.load(run/"hidden.joblib")
        with np.load(run/"oof.npz", allow_pickle=False) as saved:
            old_oof, new_oof = saved["old"].copy(), saved["new"].copy()
    else:
        old_oof = fit_context_oof(expert, dataset, split, progress=lambda row: progress("context_oof", row))["pred_z"]
        x47 = np.column_stack([x, daily.reshape(-1, 8)])
        labels = dataset["y"].ravel()
        matched, ordinary_scores = select_tree(x47[train], x47, labels[train], labels[val], val, seed=seed)
        fit, inference = station_hidden_tree_inputs(dataset, split, folds, daily)
        hidden, hidden_scores = select_tree(fit, inference, labels[train], labels[val], val, seed=seed)
        new_oof = np.full(truth.shape, np.nan)
        for fold_id, stations in enumerate(folds):
            outer = fold_split(split, stations, months)
            fit, inference = station_hidden_tree_inputs(dataset, outer, station_folds(outer["train"], months, seed), daily)
            forest = clone(hidden).fit(fit, np.log1p(labels[outer["train"]]))
            held = train[np.isin(train//months, stations)]
            new_oof.ravel()[held] = forest.predict(inference[held])
            progress("station_hidden_oof", {"fold": fold_id, "cells": len(held)})
        np.savez_compressed(run/"oof.npz", old=old_oof, new=new_oof)
        joblib.dump(matched, run/"matched.joblib", compress=3)
        joblib.dump(hidden, run/"hidden.joblib", compress=3)
        write_json(run/"tree_selection.json", {"ordinary": ordinary_scores, "station_hidden": hidden_scores})
        bind_files(run, "trees_complete.json", [run/name for name in ("oof.npz", "matched.joblib", "hidden.joblib", "tree_selection.json")], config)
        del fit, inference, forest, x47
    x47 = np.column_stack([x, daily.reshape(-1, 8)])
    new_context = np.maximum(0, np.expm1(hidden.predict(x47))).reshape(truth.shape)
    ordinary_context = np.maximum(0, np.expm1(matched.predict(x47))).reshape(truth.shape)
    if (run/"current_complete.json").exists():
        verify_files(run, "current_complete.json", config)
        current = EncoderNativeResidual.from_payload(torch.load(run/"current.pt", weights_only=False, map_location="cpu"))
        with np.load(run/"current_predictions.npz", allow_pickle=False) as saved:
            current_prediction = saved["prediction"].copy()
    else:
        current, current_prediction = native_fit(dataset, split, features, daily, context, old_oof,
            expert.residual.model, seed=seed, epochs=120, threshold=threshold,
            progress=lambda row: progress("current_native", row))
        torch.save(current.to_payload(), run/"current.pt")
        np.savez_compressed(run/"current_predictions.npz", prediction=current_prediction)
        bind_files(run, "current_complete.json", [run/"current.pt", run/"current_predictions.npz"], config)
    candidate, prediction = native_fit(dataset, split, features, daily, new_context, new_oof, current,
        seed=seed, epochs=30, threshold=threshold, progress=lambda row: progress("candidate_native", row))
    current_fusion, current_combined = temporal_fusion(context, current_prediction, dataset["y"], split)
    upgraded_fusion, upgraded_combined = temporal_fusion(new_context, prediction, dataset["y"], split)
    torch.save(candidate.to_payload(), run/"candidate.pt")
    write_json(run/"fusion.json", {"current": current_fusion.to_dict(), "upgraded": upgraded_fusion.to_dict()})
    grids = dict(zip(MODELS, (expert.predict_components()["hybrid_pred"], context, ordinary_context,
        new_context, current_prediction, current_combined, prediction, upgraded_combined), strict=True))
    cells = split["test"]
    rows = [pd.DataFrame({"cell": cells, "station": np.asarray(dataset["site_no"], str)[cells//months],
        "month": np.asarray(dataset["months"], str)[cells % months], "analyte": "doc", "mask": mask,
        "model_name": name, "seed": seed, "y_pred": grid.ravel()[cells], "y_true": truth.ravel()[cells],
        "visibility_role": "test", "k": 0}) for name, grid in grids.items()]
    pd.concat(rows, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    np.savez_compressed(run/"components.npz", **grids)
    files = ["config.json", "candidate.pt", "fusion.json", "components.npz", "backbone_complete.json",
             "trees_complete.json", "current_complete.json"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"models": len(MODELS), "test_cells": len(cells)})
    del expert, features, matched, hidden, current, candidate, dataset, x, x47, grids
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", __file__, "scripts/run_doc_geographical_confirmation_v1.py",
            "scripts/run_doc_daily_hydro_residual_v1.py", "scripts/run_doc_unmonitored_trees_v1.py",
            "scripts/run_unified_doc_spatial.py", "scripts/run_doc_tail_residual_v1.py", str(ROOT/"study_plan.md")):
        key = str(Path(name).relative_to(Path.cwd())) if Path(name).is_absolute() else str(name)
        snapshot[key] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("saved temporal execution changed")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            destination = args.root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    for mask in args.masks:
        for seed in args.seeds:
            run_one(args.root, mask, seed, digest(snapshot))


if __name__ == "__main__":
    main()
