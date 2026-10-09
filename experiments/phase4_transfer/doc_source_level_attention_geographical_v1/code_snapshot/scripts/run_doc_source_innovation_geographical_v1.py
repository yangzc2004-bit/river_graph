"""Replicate the source-selected innovation readout on saved HUC4 roles."""
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
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, fit_memory, support_curves
from run_doc_geographical_confirmation_v1 import MODELS as PRIOR_MODELS
from run_doc_geographical_confirmation_v1 import ROOT as PARENT
from run_doc_source_innovation_learning_v1 import pair_references
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    HUC4_BLOCKS,
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_source_innovation_geographical_v1")
SOURCE_DECISION = Path("experiments/phase4_transfer/doc_source_innovation_learning_v1/research_decision.md")
ARMS = ("innovation_real_native", "innovation_real_integrated", "innovation_real_trees")
MODELS = (*PRIOR_MODELS, *ARMS)
INPUT_FIELDS = ("raw", "env", "age", "support", "extra")


def geographic_inputs(parent, dataset, split, daily, oof, context):
    """Reconstruct the retained source views and check unchanged test inputs."""
    expert = UnifiedDOCReconstructor.load(parent/"backbone", dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0., np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=context.shape[1])
    source, full, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    ids = np.unique(split["test"]//context.shape[1])
    with np.load(parent/"test_inputs.npz", allow_pickle=False) as saved:
        for key, value in full.items():
            np.testing.assert_array_equal(value[ids], saved[key])
        np.testing.assert_array_equal(context[ids], saved["context"])
    del expert, features
    return source, full, arrays, extra["normalization"]


def run_one(root, huc4, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"huc4_{huc4}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    for stage in ("complete.json", "backbone_complete.json", "trees_complete.json", "current_complete.json"):
        verify_files(parent, stage, old)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "target_huc4", "split_seed", "seed", "q90_threshold_train")}, "experiment": ROOT.name,
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "source_decision": str(SOURCE_DECISION), "source_decision_hash": sha256_file(SOURCE_DECISION),
        "runtime_snapshot_hash": runtime, "started_at": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS), "epochs": 30, "patience": 5, "tail_weight": 2,
        "extra_features": 3, "extra_trainable_parameters": 67, "innovation_scale": 1.,
        "main_change": "source-selected same-month innovation readout in retained ecology/GRU",
        "selection_role": "source_validation", "evaluation_role": "retrospective whole-HUC4 replication",
        "target_information": "receiving DOC/pH/conductance absent",
        "nested_reference_exclusion": "query AND donor fold; ten newly fitted pair references",
        "forest_settings": "retained regional source-selected strong tree; no new forest search",
        "support_adapter": old["support_adapter"], "query_policy": old["query_policy"],
        "historical_scope": "already evaluated ST357 roles; not independent external validation"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        for key in ("runtime_snapshot_hash", "parent_completion_hash", "source_decision_hash"):
            if saved[key] != config[key]:
                raise ValueError("saved geographical source-information execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return

    def progress(stage, row):
        entry = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", entry)
        print(json.dumps(entry), flush=True)

    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("geographical inputs changed")
    original = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    shape = np.asarray(original["y"]).shape
    months = shape[1]
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(original)
    dataset["y"] = development_labels(dataset, split)
    del original
    train, val, test = (split[role] for role in ("train", "val", "test"))
    ids, test_ids = np.unique(train//months), np.unique(test//months)
    if np.intersect1d(ids, np.unique(np.r_[val, test]//months)).size:
        raise ValueError("source and receiving station roles overlap")
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], shape)
    with np.load(parent/"components.npz", allow_pickle=False) as saved:
        context = saved["environment"].copy()
    with np.load(parent/"oof.npz", allow_pickle=False) as saved:
        oof = saved["new"].copy()
    source_inputs, full_inputs, arrays, normalization = geographic_inputs(parent, dataset, split, daily, oof, context)
    forest = joblib.load(parent/"station_hidden_trees.joblib")
    folds = station_folds(train, months, seed)
    references, files = pair_references(run, dataset, split, daily, forest, folds, config, progress)
    names, calendar = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    source_parts, records = nested_innovation_inputs(dataset["y"], names, calendar,
        full_inputs["env"], train, folds, references)
    write_json(run/"source_library_roles.json", records)
    source_ids, source_residual = source_residual_grid(dataset["y"], oof, train)
    np.testing.assert_array_equal(source_ids, ids)
    library = SourceDOCInnovationLibrary().fit(names[ids], calendar, full_inputs["env"][ids], source_residual)
    library.save(run/"source_library.npz")
    full_parts = library.predict_components(names, full_inputs["env"], calendar)
    source_features = innovation_readout_features(source_parts, "real", 1.)
    full_features = innovation_readout_features(full_parts, "real", 1.)
    source_view = {**source_inputs, "extra": np.concatenate([source_inputs["extra"], source_features], -1)}
    full_view = {**full_inputs, "extra": np.concatenate([full_inputs["extra"], full_features], -1)}
    val_ids = np.unique(val//months)
    validation_view = {key: value[val_ids] for key, value in full_view.items()}
    fit_arrays = (source_view, *arrays[1:4], validation_view, *arrays[5:])
    np.savez_compressed(run/"innovation_inputs.npz", source_ids=ids, source_features=source_features,
        full_features=full_features, support_count=full_parts["support_count"], weight_mass=full_parts["weight_mass"])
    write_json(run/"readout_normalization.json", normalization)
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    for name in ("spatial", "temporal", "decay"):
        getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
    settings = {**retained._config(), "extra_dim": retained.extra_dim+3,
        "interaction_indices": (*retained.interaction_indices, retained.extra_dim), "epochs": 30, "patience": 5}
    if (run/"native_complete.json").exists():
        verify_files(run, "native_complete.json", config)
        model = EncoderNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    else:
        model = EncoderNativeResidual(retained.spatial, retained.temporal, retained.decay, **settings)
        model.fit(*fit_arrays, tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
            progress=lambda row: progress("innovation_real_native", row))
        torch.save(model.to_payload(), run/"native.pt")
        write_json(run/"native.json", model.to_dict())
        bind_files(run, "native_complete.json", [run/"native.pt", run/"native.json"], config)
    native = model.predict(full_view, context)
    memory = fit_memory(dataset, split, oof, context, native, months)
    integrated = memory.predict(context, native)
    write_json(run/"memory.json", memory.to_dict())
    training, inference = station_hidden_tree_inputs(dataset, split, folds, daily)
    source_local = np.searchsorted(ids, train//months), train % months
    tree_training = np.column_stack([training, source_features[source_local]])
    tree_inference = np.column_stack([inference, full_features.reshape(-1, 3)])
    if (run/"tree_complete.json").exists():
        verify_files(run, "tree_complete.json", config)
        enriched = joblib.load(run/"enriched_tree.joblib")
    else:
        enriched = clone(forest).set_params(n_jobs=2).fit(tree_training,
            np.log1p(np.asarray(dataset["y"]).ravel()[train]))
        joblib.dump(enriched, run/"enriched_tree.joblib", compress=3)
        bind_files(run, "tree_complete.json", [run/"enriched_tree.joblib"], config)
    tree_grid = np.maximum(0., np.expm1(enriched.predict(tree_inference))).reshape(shape)
    np.savez_compressed(run/"tree_test_inputs.npz", features=tree_inference[test], cells=test)
    np.savez_compressed(run/"components.npz", environment=context, native=native, integrated=integrated,
        enriched_trees=tree_grid)
    np.savez_compressed(run/"test_inputs.npz", **{key: value[test_ids] for key, value in full_view.items()},
        context=context[test_ids], cell=test)
    files += ["config.json", "source_library.npz", "source_library_roles.json", "innovation_inputs.npz",
        "reference_label_contract.json", "readout_normalization.json", "native.pt", "native.json",
        "native_complete.json", "memory.json", "enriched_tree.joblib", "tree_complete.json",
        "tree_test_inputs.npz", "components.npz", "test_inputs.npz"]
    # Persist label-free point states before scoring or designated support access.
    bind_files(run, "point_complete.json", [run/name for name in files], config)
    truth = np.asarray(torch.load(config["dataset_path"], weights_only=False, map_location="cpu")["y"], float)
    bases = dict(zip(ARMS, (native, integrated, tree_grid), strict=True))
    curves, adapters = support_curves(bases, dataset, split, truth)
    curves["split_seed"], curves["seed"], curves["target_huc4"] = int(huc4), seed, huc4
    prior_curves = pd.read_parquet(parent/"support_curves.parquet")
    pd.concat([prior_curves, curves], ignore_index=True).to_parquet(run/"support_curves.parquet", index=False)
    write_json(run/"support_adapters.json", adapters)
    prior = pd.read_parquet(parent/"predictions.parquet")
    template = prior[prior.model_name.eq("unmonitored_integrated")].copy()
    np.testing.assert_array_equal(template.cell, test)
    np.testing.assert_array_equal(template.y_true, truth.ravel()[test])
    frames = [prior.copy()]
    for name, grid in bases.items():
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, grid.ravel()[test]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    files += ["point_complete.json", "support_adapters.json"]
    for name in ("predictions.parquet", "support_curves.parquet"):
        bind_product(run, name, config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json",
        "support_curves.parquet", "support_curves.meta.json")], config)
    progress("complete", {"query_cells": len(test), "new_neural_fits": 1, "pair_reference_fits": 10,
        "new_enriched_tree_fits": 1, "best_epoch": model.best_epoch_})
    del dataset, source_inputs, full_inputs, model, memory, forest, enriched, training, inference
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--huc4", nargs="+", choices=HUC4_BLOCKS, default=list(HUC4_BLOCKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_innovation_geographical_v1.py",
        "scripts/verify_doc_source_innovation_geographical_v1.py", "scripts/run_doc_source_innovation_learning_v1.py",
        "scripts/run_doc_geographical_confirmation_v1.py", "scripts/run_doc_source_retrieval_v1.py",
        "scripts/run_doc_unmonitored_trees_v1.py", "scripts/run_unified_doc_spatial.py", str(SOURCE_DECISION),
        str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve executed version; new fitting code requires a new directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for huc4 in args.huc4:
        for seed in args.seeds:
            run_one(args.root, huc4, seed, digest(snapshot))


if __name__ == "__main__":
    main()
