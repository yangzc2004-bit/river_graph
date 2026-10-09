"""Fit detailed composition in the retained DOC ecological/GRU encoder."""
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
from audit_doc_ecological_composition_v1 import ATTRIBUTES, NODES
from audit_doc_ecological_composition_v1 import ROOT as AUDIT_ROOT
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import fit_memory
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.composition_encoder import expand_ecological_encoder
from river_graph.models.ecological_composition import composition_inputs
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_composition_encoder_v1")
ARMS = (("aggregate_encoder", "aggregate_control"), ("composition_encoder", "detailed"))


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
              "q90_threshold_train", "split_seed", "seed")}, "experiment": "doc_composition_encoder_v1",
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*old["models"], *(f"{name}_{kind}" for name, _ in ARMS for kind in ("residual", "integrated"))],
        "nodes_path": str(NODES), "nodes_hash": sha256_file(NODES),
        "attributes_path": str(ATTRIBUTES), "attributes_hash": sha256_file(ATTRIBUTES),
        "data_audit_hash": sha256_file(AUDIT_ROOT/"data_audit.json"),
        "epochs": 30, "patience": 5, "tail_weight": 2, "loss": "native mg/L MAE",
        "residual_units": "native mg/L", "added_ecological_inputs": 22, "initialization": "parent initial encoder/GRU/decay; zero scalar head",
        "main_change": "detailed physical composition in the existing ecology encoder; matched aggregate control",
        "selection_role": "source_validation", "evaluation_role": "source_validation_only"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if saved["runtime_snapshot_hash"] != runtime or saved["parent_completion_hash"] != config["parent_completion_hash"]:
            raise ValueError("saved composition encoder execution changed")
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

    if sha256_file(config["dataset_path"]) != config["dataset_hash"] or sha256_file(config["mask_path"]) != config["mask_hash"]:
        raise ValueError("source inputs or station roles changed")
    for kind in ("nodes", "attributes"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("ecological source inputs changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    shape = dataset["y"].shape
    composition = composition_inputs(dataset["site_no"],
        pd.read_csv(NODES, dtype={"site_no": str, "comid": str}),
        pd.read_csv(ATTRIBUTES, dtype={"comid": str}))
    tree_run = Path(old["parent_run"])
    tree_config = json.loads((tree_run/"config.json").read_text())
    ancestor = json.loads((Path(tree_config["parent_run"])/"config.json").read_text())
    monthly = Path(ancestor["parent_run"])
    monthly_config = json.loads((monthly/"config.json").read_text())
    daily, _, _ = load_daily_pack(monthly.parent.parent, config["dataset_hash"], shape)
    expert = UnifiedDOCReconstructor.load(Path(monthly_config["source_run"]), dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    forest = joblib.load(tree_run/"station_hidden_trees.joblib")
    _, inference = station_hidden_tree_inputs(dataset, split, station_folds(split["train"], shape[1], seed), daily)
    context = np.maximum(0., np.expm1(forest.predict(inference))).reshape(shape)
    del inference, forest
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0., np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=shape[1])
    _, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    val = split["val"]
    ids = np.unique(val//shape[1])
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        for key, value in full_inputs.items():
            np.testing.assert_array_equal(value[ids], saved[key])
        np.testing.assert_allclose(context[ids], saved["context"], rtol=1e-12, atol=1e-12)
        context[ids] = saved["context"]
    # Use the cached base for validation selection and saved inference alike.
    arrays = (*arrays[:5], context[ids], *arrays[6:])
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    for name in ("spatial", "temporal", "decay"):
        getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
    previous = pd.read_parquet(parent/"predictions.parquet")
    frames = [previous.copy()]
    template = previous[previous.model_name.eq("unmonitored_residual")].copy()
    np.testing.assert_array_equal(template.cell, val)
    source_ids = np.unique(split["train"]//shape[1])
    files = ["config.json"]
    for name, key in ARMS:
        physical = composition[key]
        source_inputs = {field: value.copy() for field, value in arrays[0].items()}
        val_inputs = {field: value.copy() for field, value in arrays[4].items()}
        source_inputs["env"] = np.column_stack([source_inputs["env"], physical[source_ids]])
        val_inputs["env"] = np.column_stack([val_inputs["env"], physical[ids]])
        model = EncoderNativeResidual(expand_ecological_encoder(retained.spatial, retained.spatial_architecture),
            retained.temporal, retained.decay, **retained._config())
        expanded_args = (source_inputs, *arrays[1:4], val_inputs, *arrays[5:])
        stage = f"{name}_complete.json"
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = EncoderNativeResidual.from_payload(torch.load(run/f"{name}.pt", weights_only=False, map_location="cpu"))
        else:
            model.fit(*expanded_args, tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                      progress=lambda row, stage_name=name: progress(stage_name, row))
            torch.save(model.to_payload(), run/f"{name}.pt")
            write_json(run/f"{name}.json", model.to_dict())
            bind_files(run, stage, [run/f"{name}.pt", run/f"{name}.json"], config)
        prediction = model.predict(val_inputs, context[ids])
        selected = prediction[np.searchsorted(ids, val//shape[1]), val % shape[1]]
        native_grid = context.copy()
        native_grid[ids] = prediction
        memory = fit_memory(dataset, split, oof, context, native_grid, shape[1])
        integrated = memory.predict(context, native_grid).ravel()[val]
        for kind, values in (("residual", selected), ("integrated", integrated)):
            frame = template.copy()
            frame["model_name"], frame["y_pred"] = f"{name}_{kind}", values
            frames.append(frame)
        write_json(run/f"{name}_memory.json", memory.to_dict())
        np.savez_compressed(run/f"{name}_validation_inputs.npz", **val_inputs, context=context[ids], cell=val)
        files += [f"{name}.pt", f"{name}.json", stage, f"{name}_memory.json", f"{name}_validation_inputs.npz"]
        del model, source_inputs, val_inputs, expanded_args, memory
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"models": len(config["models"]), "ecological_input_dimension": retained.spatial_architecture["env_dim"]+22})
    del retained, expert, dataset, features, arrays, full_inputs
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_composition_encoder_v1.py",
                 "scripts/analyze_doc_composition_encoder_v1.py", "tests/test_composition_encoder.py",
                 "scripts/audit_doc_ecological_composition_v1.py",
                 "scripts/run_doc_geographical_confirmation_v1.py",
                 "scripts/run_doc_source_retrieval_v1.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_unified_doc_spatial.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("retain completed execution; use a new version if training code changes")
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
