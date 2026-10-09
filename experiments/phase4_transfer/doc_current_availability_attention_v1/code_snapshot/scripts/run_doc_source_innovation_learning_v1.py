"""Learn sparse source innovations in the existing ecology/GRU DOC residual."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_joint_source_states_v1 import ROOT as SOURCE_INPUTS
from run_doc_source_innovation_transfer_v1 import ROOT as PROBE
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import fold_split, station_folds
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)

ROOT = Path("experiments/phase4_transfer/doc_source_innovation_learning_v1")
MODES = ("real", "historical", "availability")
ARMS = tuple(f"innovation_{mode}_{kind}" for mode in MODES for kind in ("native", "integrated"))
TREE_ARMS = ("innovation_real_trees", "innovation_historical_trees")
INPUT_FIELDS = ("raw", "env", "age", "support", "extra")


def pair_references(run, dataset, split, daily, forest, folds, config, progress):
    """Cache ten forests which each omit BOTH source station folds."""
    references, files = {}, []
    months = dataset["y"].shape[1]
    source = split["train"]
    for a, b in combinations(range(len(folds)), 2):
        prefix = f"pair_reference_{a}_{b}"
        stage = f"{prefix}_complete.json"
        hidden = np.sort(np.r_[folds[a], folds[b]])
        selected = source[np.isin(source//months, hidden)]
        outer = fold_split(split, hidden, months)
        if (run/stage).exists():
            verify_files(run, stage, config)
        else:
            training, inference = station_hidden_tree_inputs(dataset, outer,
                station_folds(outer["train"], months, config["seed"]), daily)
            if (a, b) == (0, 1):
                # Validate the actual cohort boundary before the first fit.
                forbidden = np.unique(np.r_[hidden, split["val"]//months, split["test"]//months])
                changed = {**dataset, "y": np.asarray(dataset["y"]).copy()}
                changed["y"][forbidden] = 12345.
                check_training, check_inference = station_hidden_tree_inputs(changed, outer,
                    station_folds(outer["train"], months, config["seed"]), daily)
                np.testing.assert_array_equal(check_training, training)
                np.testing.assert_array_equal(check_inference, inference)
                np.testing.assert_array_equal(np.asarray(changed["y"]).ravel()[outer["train"]],
                    np.asarray(dataset["y"]).ravel()[outer["train"]])
                write_json(run/"reference_label_contract.json", {"query_and_donor_pair": [a, b],
                    "perturbed_station_ids": forbidden.tolist(), "training_and_inference_inputs_unchanged": True,
                    "fitted_label_vector_unchanged": True, "executed_before_pair_fitting": True})
                del changed, check_training, check_inference
            model = clone(forest).set_params(n_jobs=2)
            model.fit(training, np.log1p(np.asarray(dataset["y"]).ravel()[outer["train"]]))
            prediction = model.predict(inference[selected])
            np.savez_compressed(run/f"{prefix}.npz", cells=selected, pred_z=prediction,
                features=inference[selected])
            joblib.dump(model, run/f"{prefix}.joblib", compress=3)
            write_json(run/f"{prefix}.json", {"folds": [a, b], "hidden_stations": hidden.tolist(),
                "fitted_stations": np.unique(outer["train"]//months).tolist(),
                "hidden_labels_removed_before_features_and_fit": True})
            bind_files(run, stage, [run/f"{prefix}{suffix}" for suffix in (".npz", ".joblib", ".json")], config)
            del model, training, inference
            gc.collect()
        with np.load(run/f"{prefix}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["cells"], selected)
            references[(a, b)] = {**json.loads((run/f"{prefix}.json").read_text()),
                "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
        files += [f"{prefix}{suffix}" for suffix in (".npz", ".joblib", ".json")]+[stage]
        progress("pair_reference", {"query_donor_folds": [a, b], "held_cells": len(selected)})
    return references, files


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    source, probe = SOURCE_INPUTS/"runs"/parent.name, PROBE/"runs"/parent.name
    for origin in (source, probe):
        verify_files(origin, "complete.json", json.loads((origin/"config.json").read_text()))
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
              "q90_threshold_train", "split_seed", "seed")}, "experiment": "doc_source_innovation_learning_v1",
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "source_inputs_run": str(source), "source_inputs_hash": sha256_file(source/"joint_source_inputs.npz"),
        "probe_run": str(probe), "probe_completion_hash": sha256_file(probe/"complete.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*old["models"], *ARMS, *TREE_ARMS], "epochs": 30, "patience": 5,
        "main_change": "same-month source innovation/support enters existing GRU residual readout",
        "selection_role": "source_validation", "evaluation_role": "source_validation_only",
        "nested_reference_exclusion": "query fold AND donor fold; ten new pair forests",
        "target_information": "receiving DOC/pH/conductance absent; source-only library",
        "new_features": 3, "extra_trainable_neural_parameters": 67,
        "tail_weight": 2, "forest_settings": "retained strong tree, no new hyperparameter selection"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        for key in ("runtime_snapshot_hash", "parent_completion_hash", "source_inputs_hash", "probe_completion_hash"):
            if saved[key] != config[key]:
                raise ValueError("saved innovation-learning execution changed")
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
            raise ValueError("source data or station roles changed")
    dataset = strip_auxiliary_water(torch.load(config["dataset_path"], weights_only=False, map_location="cpu"))
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset["y"] = development_labels(dataset, split)
    shape = dataset["y"].shape
    months = shape[1]
    train, val = split["train"], split["val"]
    ids, validation_ids = np.unique(train//months), np.unique(val//months)
    if np.intersect1d(ids, np.unique(np.r_[val, split["test"]]//months)).size:
        raise ValueError("source and receiving station roles overlap")
    if not np.isin(split["context"]//months, ids).all():
        raise ValueError("unmonitored receivers have context DOC")
    with np.load(source/"joint_source_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_station_ids"], ids)
        source_inputs = {key: saved[key].copy() for key in INPUT_FIELDS}
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], val)
        validation_inputs = {key: saved[key].copy() for key in INPUT_FIELDS}
        validation_base = saved["context"].copy()
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    tree_run = Path(old["parent_run"])
    tree_config = json.loads((tree_run/"config.json").read_text())
    ancestor = json.loads((Path(tree_config["parent_run"])/"config.json").read_text())
    monthly = Path(ancestor["parent_run"])
    daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], shape)
    forest = joblib.load(tree_run/"station_hidden_trees.joblib")
    folds = station_folds(train, months, seed)
    references, files = pair_references(run, dataset, split, daily, forest, folds, config, progress)
    ecology = np.full((shape[0], 9), np.nan)
    ecology[ids] = source_inputs["env"]
    names, calendar = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    source_parts, records = nested_innovation_inputs(dataset["y"], names, calendar, ecology, train, folds, references)
    write_json(run/"source_library_roles.json", records)
    library = SourceDOCInnovationLibrary.load(probe/"source_library.npz")
    np.testing.assert_array_equal(library.source_names_, names[ids])
    validation_parts = library.predict_components(names[validation_ids], validation_inputs["env"], calendar)
    local_source = np.searchsorted(ids, train//months), train % months
    scale = 1.
    write_json(run/"innovation_scale.json", {"scale": scale, "fit_role": "none",
        "statistic": "fixed native unit1mg/L; no cross-bank label-dependent scaling",
        "features": ["innovation/scale", "matched_count/20", "weight_mass/(1+weight_mass)"]})
    source_features = {mode: innovation_readout_features(source_parts, mode, scale) for mode in MODES}
    validation_features = {mode: innovation_readout_features(validation_parts, mode, scale) for mode in MODES}
    np.savez_compressed(run/"innovation_inputs.npz", source_ids=ids, validation_ids=validation_ids, cells=val,
        **{f"source_{mode}": values for mode, values in source_features.items()},
        **{f"validation_{mode}": values for mode, values in validation_features.items()})
    source_mask, validation_mask = np.zeros(shape, bool), np.zeros(shape, bool)
    source_mask.ravel()[train] = True
    _, val_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    validation_mask.ravel()[val_query] = True
    source_base = np.maximum(0., np.expm1(oof[ids]))
    source_truth, validation_truth = np.asarray(dataset["y"])[ids], np.asarray(dataset["y"])[validation_ids]
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    for name in ("spatial", "temporal", "decay"):
        getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
    previous = pd.read_parquet(parent/"predictions.parquet")
    frames = [previous.copy()]
    template = previous[previous.model_name.eq("unmonitored_residual")].copy()
    np.testing.assert_array_equal(template.cell, val)
    local_val = np.searchsorted(validation_ids, val//months), val % months
    full_base = np.zeros(shape)
    full_base[validation_ids] = validation_base
    settings = {**retained._config(), "extra_dim": retained.extra_dim+3,
        "interaction_indices": (*retained.interaction_indices, retained.extra_dim), "epochs": 30, "patience": 5}
    for mode in MODES:
        prefix = f"innovation_{mode}"
        stage = f"{prefix}_complete.json"
        source_view = {**source_inputs, "extra": np.concatenate([source_inputs["extra"], source_features[mode]], -1)}
        val_view = {**validation_inputs, "extra": np.concatenate([validation_inputs["extra"], validation_features[mode]], -1)}
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = EncoderNativeResidual.from_payload(torch.load(run/f"{prefix}.pt", weights_only=False, map_location="cpu"))
        else:
            model = EncoderNativeResidual(retained.spatial, retained.temporal, retained.decay, **settings)
            model.fit(source_view, source_base, source_truth, source_mask[ids], val_view, validation_base,
                validation_truth, validation_mask[validation_ids], tail_threshold=config["q90_threshold_train"],
                selection_role="source_validation", progress=lambda row, arm=prefix: progress(arm, row))
            torch.save(model.to_payload(), run/f"{prefix}.pt")
            write_json(run/f"{prefix}.json", model.to_dict())
            bind_files(run, stage, [run/f"{prefix}.pt", run/f"{prefix}.json"], config)
        prediction = model.predict(val_view, validation_base)
        memory = EcologicalResidualTransfer().fit(np.asarray(dataset["regime"]), train,
            np.maximum(0., np.expm1(oof.ravel()[train])), np.asarray(dataset["y"]).ravel()[train],
            n_months=months, validation_cells=val, validation_y=np.asarray(dataset["y"]).ravel()[val],
            validation_context=validation_base[local_val], validation_temporal=prediction[local_val],
            selection_role="source_validation")
        full_native = full_base.copy()
        full_native[validation_ids] = prediction
        integrated = memory.predict(full_base, full_native).ravel()[val]
        write_json(run/f"{prefix}_memory.json", memory.to_dict())
        for kind, values in (("native", prediction[local_val]), ("integrated", integrated)):
            frame = template.copy()
            frame["model_name"], frame["y_pred"] = f"{prefix}_{kind}", values
            frame["source_support_count"] = validation_parts["support_count"][local_val]
            frame["source_weight_mass"] = validation_parts["weight_mass"][local_val]
            frames.append(frame)
        files += [f"{prefix}.pt", f"{prefix}.json", stage, f"{prefix}_memory.json"]
        del model, memory, source_view, val_view
        gc.collect()
    training, inference = station_hidden_tree_inputs(dataset, split, folds, daily)
    np.savez_compressed(run/"tree_validation_inputs.npz", features=inference[val], cells=val)
    for mode in ("real", "historical"):
        prefix = f"innovation_{mode}_trees"
        stage = f"{prefix}_complete.json"
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = joblib.load(run/f"{prefix}.joblib")
        else:
            model = clone(forest).set_params(n_jobs=2)
            model.fit(np.column_stack([training, source_features[mode][local_source]]),
                np.log1p(np.asarray(dataset["y"]).ravel()[train]))
            joblib.dump(model, run/f"{prefix}.joblib", compress=3)
            bind_files(run, stage, [run/f"{prefix}.joblib"], config)
        prediction = np.maximum(0., np.expm1(model.predict(
            np.column_stack([inference[val], validation_features[mode][local_val]]))))
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = prefix, prediction
        frames.append(frame)
        files += [f"{prefix}.joblib", stage]
        progress(prefix, {"validation_mae": float(np.abs(prediction-template.y_true).mean())})
        del model
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    files += ["config.json", "source_library_roles.json", "innovation_scale.json", "innovation_inputs.npz",
              "tree_validation_inputs.npz", "reference_label_contract.json"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"models": len(config["models"]), "new_neural_fits": 3, "pair_reference_fits": 10})
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_innovation_learning_v1.py",
                 "scripts/analyze_doc_source_innovation_learning_v1.py", "tests/test_source_innovation_training.py",
                 "scripts/run_doc_unmonitored_trees_v1.py", "scripts/run_doc_source_innovation_transfer_v1.py",
                 "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve the executed version; new fitting code requires a new directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            destination = args.root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == "__main__":
    main()
