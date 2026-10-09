"""Refit the existing DOC ecology/GRU residual on station-OOF leaf medians."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import fit_memory
from run_doc_leaf_distribution_reference_v1 import ROOT as REFERENCE_ROOT
from run_doc_reference_trajectory_v1 import ROOT as FOLD_ROOT
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import fold_split, station_folds
from river_graph.models.leaf_distribution_reference import LeafDistributionReference
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_leaf_median_residual_v1")


def fold_distribution(dataset, split, stations, daily, forest, seed):
    """A previously verified complement-fit forest gets only complement labels."""
    months = dataset["y"].shape[1]
    outer = fold_split(split, stations, months)
    training, inference = station_hidden_tree_inputs(dataset, outer,
        station_folds(outer["train"], months, seed), daily)
    model = LeafDistributionReference().fit(forest, training,
        np.asarray(dataset["y"]).ravel()[outer["train"]])
    held = split["train"][np.isin(split["train"]//months, stations)]
    return model, held, inference[held], outer


def source_medians(run, prior, dataset, split, daily, config, progress):
    prior_config = json.loads((prior/"config.json").read_text())
    verify_files(prior, "complete.json", prior_config)
    for key in ("dataset_hash", "mask_hash", "seed", "split_seed"):
        if config[key] != prior_config[key]:
            raise ValueError("nested cached forests belong to different source roles")
    shape = tuple(dataset["y"].shape)
    oof = np.full(shape, np.nan)
    files, records = [], []
    for index, stations in enumerate(station_folds(split["train"], shape[1], config["seed"])):
        prefix = f"median_fold{index}"
        stage = f"{prefix}_complete.json"
        cached_prefix = f"reference_fold{index}"
        verify_files(prior, f"{cached_prefix}_complete.json", prior_config)
        old_record = json.loads((prior/f"{cached_prefix}.json").read_text())
        outer = fold_split(split, stations, shape[1])
        np.testing.assert_array_equal(old_record["hidden_stations"], stations)
        np.testing.assert_array_equal(old_record["fitted_stations"], np.unique(outer["train"]//shape[1]))
        if not old_record["held_labels_removed_before_training_features"]:
            raise ValueError("cached OOF forest was not whole-station hidden")
        held = split["train"][np.isin(split["train"]//shape[1], stations)]
        if (run/stage).exists():
            verify_files(run, stage, config)
            with np.load(run/f"{prefix}.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["cell"], held)
                values = saved["pred_z"].copy()
        else:
            forest = joblib.load(prior/f"{cached_prefix}.joblib")
            with threadpool_limits(limits=2):
                model, ids, inputs, _ = fold_distribution(dataset, split, stations, daily, forest, config["seed"])
                np.testing.assert_array_equal(ids, held)
                values = np.log1p(model.predict(inputs))
            joblib.dump(model, run/f"{prefix}.joblib", compress=3)
            np.savez_compressed(run/f"{prefix}.npz", cell=held, pred_z=values, features=inputs)
            write_json(run/f"{prefix}.json", {"hidden_stations": stations.tolist(),
                "fitted_stations": np.unique(outer["train"]//shape[1]).tolist(),
                "fitted_cells_sha256": digest(outer["train"].tolist()),
                "cached_forest": str(prior/f"{cached_prefix}.joblib"),
                "cached_forest_sha256": sha256_file(prior/f"{cached_prefix}.joblib"),
                "cached_fold_complete_sha256": sha256_file(prior/f"{cached_prefix}_complete.json"),
                "new_forest_fit": False, "source_distribution_excludes_held_fold": True})
            bind_files(run, stage, [run/f"{prefix}{suffix}" for suffix in (".npz", ".joblib", ".json")], config)
            del model, forest, inputs
        oof.ravel()[held] = values
        records.append(json.loads((run/f"{prefix}.json").read_text()))
        files += [f"{prefix}{suffix}" for suffix in (".npz", ".joblib", ".json")]+[stage]
        progress("median_station_oof", {"fold": index, "held_cells": len(held)})
    observed = np.zeros(oof.size, bool)
    observed[split["train"]] = True
    if not np.isfinite(oof.ravel()[observed]).all() or not np.isnan(oof.ravel()[~observed]).all():
        raise ValueError("median OOF predictions must cover exactly source-training cells")
    np.savez_compressed(run/"source_oof.npz", pred_z=oof)
    write_json(run/"oof_records.json", records)
    files += ["source_oof.npz", "oof_records.json"]
    return oof, files


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    reference_run, fold_run = (base/"runs"/parent.name for base in (REFERENCE_ROOT, FOLD_ROOT))
    reference_config = json.loads((reference_run/"config.json").read_text())
    verify_files(reference_run, "complete.json", reference_config)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "q90_threshold_train", "split_seed", "seed")}, "experiment": root.name,
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "reference_run": str(reference_run), "reference_complete_hash": sha256_file(reference_run/"complete.json"),
        "fold_run": str(fold_run), "fold_complete_hash": sha256_file(fold_run/"complete.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*old["models"], "leaf_median_reference", "leaf_median_residual", "leaf_median_integrated"],
        "epochs": 30, "patience": 5, "tail_weight": 2, "loss": "native mg/L MAE",
        "main_change": "fixed conditional median reference and its station-OOF residual; existing neural architecture",
        "selection_role": "source_validation", "evaluation_role": "source_validation_development",
        "initialization": "retained initial spatial/GRU/decay state; zero existing scalar head",
        "cached_forests_reused": True, "oof_cache_storage_transform": "log1p", "quantile": .5}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if any(saved[key] != config[key] for key in ("runtime_snapshot_hash", "parent_completion_hash",
                "reference_complete_hash", "fold_complete_hash")):
            raise ValueError("saved median neural residual execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return

    def progress(stage, values):
        record = {"run": run.name, "stage": stage, **values, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("median residual source inputs changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    shape = dataset["y"].shape
    tree_config = json.loads((Path(old["parent_run"])/"config.json").read_text())
    ancestor = json.loads((Path(tree_config["parent_run"])/"config.json").read_text())
    monthly = Path(ancestor["parent_run"])
    monthly_config = json.loads((monthly/"config.json").read_text())
    daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], shape)
    expert = UnifiedDOCReconstructor.load(Path(monthly_config["source_run"]), dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    _, inference = station_hidden_tree_inputs(dataset, split, station_folds(split["train"], shape[1], seed), daily)
    reference = joblib.load(reference_run/"distribution.joblib")
    with threadpool_limits(limits=2):
        context = reference.predict(inference).reshape(shape)
    del reference, inference
    oof, fold_files = source_medians(run, fold_run, dataset, split, daily, config, progress)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.expm1(oof.ravel()[split["train"]]), context,
        build_causal_flow_features(dataset)["full"], n_months=shape[1])
    _, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    for name in ("spatial", "temporal", "decay"):
        getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
    model = EncoderNativeResidual(retained.spatial, retained.temporal, retained.decay, **retained._config())
    if (run/"native_complete.json").exists():
        verify_files(run, "native_complete.json", config)
        model = EncoderNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    else:
        model.fit(*arrays, tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
            progress=lambda row: progress("median_native_residual", row))
        torch.save(model.to_payload(), run/"native.pt")
        write_json(run/"native.json", model.to_dict())
        bind_files(run, "native_complete.json", [run/"native.pt", run/"native.json"], config)
    val = split["val"]
    ids = np.unique(val//shape[1])
    val_inputs = {key: value[ids] for key, value in full_inputs.items()}
    prediction = model.predict(val_inputs, context[ids])
    native = prediction[np.searchsorted(ids, val//shape[1]), val % shape[1]]
    native_grid = context.copy()
    native_grid[ids] = prediction
    memory = fit_memory(dataset, split, oof, context, native_grid, shape[1])
    integrated = memory.predict(context, native_grid).ravel()[val]
    previous = pd.read_parquet(parent/"predictions.parquet")
    template = previous[previous.model_name.eq("unmonitored_residual")].copy()
    np.testing.assert_array_equal(template.cell, val)
    frames = [previous]
    for name, values in (("leaf_median_reference", context.ravel()[val]),
                          ("leaf_median_residual", native), ("leaf_median_integrated", integrated)):
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, values
        frames.append(frame)
    write_json(run/"memory.json", memory.to_dict())
    write_json(run/"readout_normalization.json", extra["normalization"])
    np.savez_compressed(run/"validation_inputs.npz", **val_inputs, context=context[ids], cell=val)
    files = ["config.json", "native.pt", "native.json", "native_complete.json", "memory.json",
        "readout_normalization.json", "validation_inputs.npz", *fold_files]
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"best_epoch": model.best_epoch_, "selected_scale": model.selected_scale_})
    del model, retained, expert, dataset, features, arrays, full_inputs, memory
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_leaf_median_residual_v1.py",
        "scripts/analyze_doc_leaf_median_residual_v1.py", "scripts/plot_doc_source_development_v1.py",
        "scripts/run_doc_unmonitored_trees_v1.py", "scripts/run_doc_geographical_confirmation_v1.py",
        "scripts/run_doc_daily_hydro_residual_v1.py", "scripts/run_doc_source_retrieval_v1.py",
        "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py",
        "tests/test_leaf_median_oof.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve saved median neural execution after code changes")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == "__main__":
    main()
