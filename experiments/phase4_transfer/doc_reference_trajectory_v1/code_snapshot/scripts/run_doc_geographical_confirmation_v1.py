"""Refit DOC recipes on whole-HUC4 source regions and score fixed new regions."""
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
from run_doc_daily_hydro_residual_v1 import (
    INTERACTION_INDICES,
    load_daily_pack,
)
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.experiments.unmonitored_doc import (
    HUC4_BLOCKS,
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
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
from river_graph.models.station_adapted_hybrid import station_residual_correction
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_geographical_confirmation_v1")
TASKS = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DAILY_ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
MODELS = ("current_model", "matched_daily_trees", "station_hidden_trees",
          "unmonitored_residual", "unmonitored_integrated")


def select_tree(training, inference, train_y, val_y, val, *, seed):
    scores, selected, best = [], None, np.inf
    for name, leaf, maximum in (("leaf4", 4, 1.), ("leaf2", 2, 1.), ("leaf1", 1, 1.), ("sqrt", 1, "sqrt")):
        tree = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=leaf,
            max_features=maximum, n_jobs=2, random_state=seed).fit(training, np.log1p(train_y))
        prediction = np.maximum(0, np.expm1(tree.predict(inference[val])))
        score = float(np.abs(prediction-val_y).mean())
        scores.append({"candidate": name, "validation_mae": score})
        if score < best:
            selected, best = tree, score
    return selected, scores


def fit_memory(dataset, split, oof, context, temporal, months):
    train, val = split["train"], split["val"]
    return EcologicalResidualTransfer().fit(np.asarray(dataset["regime"]), train,
        np.maximum(0, np.expm1(oof.ravel()[train])), np.asarray(dataset["y"]).ravel()[train],
        n_months=months, validation_cells=val, validation_y=np.asarray(dataset["y"]).ravel()[val],
        validation_context=context.ravel()[val], validation_temporal=temporal.ravel()[val],
        selection_role="source_validation")


def fit_native(dataset, split, features, daily, context, oof, backbone, *, seed, epochs, threshold, progress):
    months = dataset["y"].shape[1]
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=months)
    _, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    model = EncoderNativeResidual(backbone.spatial, backbone.temporal, backbone.decay,
        seed=seed, epochs=epochs, patience=5, extra_dim=38, tail_weight=2,
        interaction_indices=INTERACTION_INDICES, encoder_mode="last_self_ecology",
        encoder_learning_rate=1e-5).fit(*arrays, tail_threshold=threshold,
        selection_role="source_validation", progress=progress)
    delta = model.predict_delta(full_inputs)
    prediction = np.maximum(0, context + model.selected_scale_ * delta)
    return model, prediction, delta, extra["normalization"], full_inputs


def support_curves(bases, dataset, split, truth):
    """Select identical simple support adapters on validation, then apply to test.

    Target support never enters the backbone. The five reserved cells are
    excluded from every curve query, including K0. Primary K0 uses all test
    observations separately. These curves describe retrospective calibration.
    """
    months = dataset["y"].shape[1]
    rows, states = [], {}
    names, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    for model_name, grid in bases.items():
        flat = grid.ravel()
        states[model_name] = {}
        for k in (0, 1, 3, 5):
            vs, vq = support_query_cells(split, target_role="val", k=k, n_months=months)
            candidates = []
            for alpha in ((0.,) if k == 0 else (0., .25, .5, .75, 1.)):
                prediction = station_residual_correction(flat[vq], vq, flat[vs], vs,
                    np.asarray(dataset["y"]).ravel()[vs], n_months=months, alpha=alpha)
                candidates.append({"alpha": alpha, "validation_mae": float(
                    np.abs(prediction-np.asarray(dataset["y"]).ravel()[vq]).mean())})
            chosen = min(candidates, key=lambda value: (value["validation_mae"], value["alpha"]))
            states[model_name][str(k)] = {"selected": chosen, "candidates": candidates}
            support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
            prediction = station_residual_correction(flat[query], query, flat[support], support,
                truth.ravel()[support], n_months=months, alpha=chosen["alpha"])
            rows.append(pd.DataFrame({"cell": query, "station": names[query//months],
                "month": dates[query % months], "model_name": model_name, "k": k,
                "y_pred": prediction, "y_true": truth.ravel()[query], "visibility_role": "test",
                "support_count": k, "support_alpha": chosen["alpha"]}))
    return pd.concat(rows, ignore_index=True), states


def run_one(root, huc4, seed, runtime):
    started = time.monotonic()
    run = root / "runs" / f"huc4_{huc4}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    mask_path = TASKS / "geographical_masks" / f"huc4_{huc4}.npz"
    source = torch.load(DATASET, weights_only=False, map_location="cpu")
    truth = np.asarray(source["y"], dtype=np.float64).copy()
    with np.load(mask_path, allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    threshold = float(np.quantile(truth.ravel()[split["train"]], .9))
    config = {"experiment": "doc_geographical_confirmation_v1", "target_huc4": huc4,
        "split_seed": int(huc4), "seed": seed, "dataset_path": str(DATASET), "dataset_hash": sha256_file(DATASET),
        "mask_path": str(mask_path), "mask_hash": sha256_file(mask_path), "runtime_snapshot_hash": runtime,
        "started_at": datetime.now(timezone.utc).isoformat(), "q90_threshold_train": threshold,
        "models": list(MODELS), "source_backbone_epochs": 20, "current_native_epochs": 120,
        "candidate_native_epochs": 30, "patience": 5, "information_condition": "no target DOC/pH/conductance",
        "query_policy": "primary K0 all valid test cells; separate fixed-query support curve",
        "source_experience": "station-blocked OOF; same ecological memory family in both full models",
        "support_adapter": "matched validation-selected log1p mean residual; support not fed to the backbone",
        "selection_role": "source_validation", "evaluation_role": "whole HUC4 test",
        "candidate_choice": "fixed integrated station-hidden recipe from source-validation development",
        "historical_scope": "ST357 retrospective geographical replication, not independent external validation"}
    if (run / "config.json").exists():
        existing = json.loads((run / "config.json").read_text())
        if any(existing[k] != config[k] for k in ("runtime_snapshot_hash", "dataset_hash", "mask_hash")):
            raise ValueError("geographical execution changed; use a new version")
        config = existing
    else:
        write_json(run / "config.json", config)
    if (run / "complete.json").exists():
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified completed geographical run", flush=True)
        return
    dataset = strip_auxiliary_water(source)
    dataset["y"] = development_labels(dataset, split)
    del source
    shape = truth.shape
    months = shape[1]
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], shape)

    def progress(stage, row):
        entry = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root / "progress.json", entry)
        print(json.dumps(entry), flush=True)

    # Fresh geographic fit: no checkpoint trained on this test/validation region
    # is reused. The old recipe supplies architecture and budgets only.
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
    candidate, native, delta, normalization, full_inputs = fit_native(dataset, split, features, daily,
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
    # Scoring and explicitly designated support access begin only after every
    # point predictor and source-validation choice have been saved.
    curves, adapters = support_curves(bases, dataset, split, truth)
    curves["split_seed"], curves["seed"], curves["target_huc4"] = int(huc4), seed, huc4
    curves.to_parquet(run / "support_curves.parquet", index=False)
    write_json(run / "support_adapters.json", adapters)
    names, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    cells = split["test"]
    rows = [pd.DataFrame({"cell": cells, "station": names[cells//months], "month": dates[cells % months],
        "analyte": "doc", "y_pred": grid.ravel()[cells], "y_true": truth.ravel()[cells], "model_name": name,
        "visibility_role": "test", "k": 0, "split_seed": int(huc4), "seed": seed, "target_huc4": huc4,
        "evaluation_population": "all valid target DOC cells"}) for name, grid in bases.items()]
    pd.concat(rows, ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
    np.savez_compressed(run / "components.npz", environment=new_context, local_temporal_delta=delta,
        native=native, integrated=integrated, current=current, matched_trees=matched_context)
    test_ids = np.unique(cells//months)
    np.savez_compressed(run / "test_inputs.npz", **{k: v[test_ids] for k, v in full_inputs.items()},
                        context=new_context[test_ids], cell=cells)
    files = ["config.json", "native.pt", "native.json", "memory.json", "readout_normalization.json",
        "components.npz", "test_inputs.npz", "support_adapters.json", "backbone_complete.json",
        "trees_complete.json", "current_complete.json"]
    for name in ("predictions.parquet", "support_curves.parquet"):
        bind_product(run, name, config, runtime, files)
    all_files = [run / name for name in (*files, "predictions.parquet", "predictions.meta.json",
                                       "support_curves.parquet", "support_curves.meta.json")]
    bind_files(run, "complete.json", all_files, config)
    progress("complete", {"models": len(MODELS), "query_cells": len(cells), "best_epoch": candidate.best_epoch_})
    del features, expert, dataset, candidate, current_native, full_inputs
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--huc4", nargs="+", default=list(HUC4_BLOCKS), choices=HUC4_BLOCKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_geographical_confirmation_v1.py",
        "scripts/run_doc_unmonitored_trees_v1.py", "scripts/run_doc_source_retrieval_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve existing geographical execution after code changes")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for huc4 in args.huc4:
        for seed in args.seeds:
            run_one(args.root, huc4, seed, digest(snapshot))


if __name__ == "__main__":
    main()
