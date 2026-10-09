"""Check current-source attention on the two saved temporal DOC tasks.

Reuse source-fitted temporal references and their saved initialization. Fit one
fixed attention residual per task/seed; temporal test labels remain score-only.
"""
from __future__ import annotations

import argparse
import copy
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
from run_doc_current_source_portable_v2_r1 import source_attention_reference
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT
from run_doc_tail_residual_v1 import bind_product
from run_doc_temporal_compatibility_v1 import (
    ROOT as PARENT,
)
from run_doc_temporal_compatibility_v1 import (
    temporal_arrays,
    temporal_fusion,
    temporal_label_view,
)
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
    station_folds,
)
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
    nested_current_available_candidates,
)
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)
from river_graph.models.source_level_attention import source_level_input_view
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_current_source_temporal_v3")
MASKS = ("e2a_strict", "e2b_partial")
MODELS = ("available_native", "available_fusion")


def hidden_pair_label_probe(dataset, split, hidden, months):
    """Perturb held stations and val/test CELLS, not all overlapping stations."""
    changed = {**dataset, "y": np.asarray(dataset["y"]).copy()}
    changed["y"][hidden] = 12345.
    for role in ("val", "test"):
        changed["y"].ravel()[split[role]] = 12345.
    outer = fold_split(split, hidden, months)
    np.testing.assert_array_equal(changed["y"].ravel()[outer["train"]],
        np.asarray(dataset["y"]).ravel()[outer["train"]])
    return changed


def pair_references(run, data, split, daily, forest, folds, config, progress):
    references, files = {}, []
    months, source = data["y"].shape[1], split["train"]
    for a, b in combinations(range(5), 2):
        prefix = f"pair_reference_{a}_{b}"
        stage = f"{prefix}_complete.json"
        hidden = np.sort(np.r_[folds[a], folds[b]])
        selected = source[np.isin(source//months, hidden)]
        outer = fold_split(split, hidden, months)
        if (run/stage).exists():
            verify_files(run, stage, config)
        else:
            inner = station_folds(outer["train"], months, config["seed"])
            fit, inference = station_hidden_tree_inputs(data, outer, inner, daily)
            if (a, b) == (0, 1):
                changed = hidden_pair_label_probe(data, split, hidden, months)
                check_fit, check_inference = station_hidden_tree_inputs(changed, outer, inner, daily)
                np.testing.assert_array_equal(fit, check_fit)
                np.testing.assert_array_equal(inference, check_inference)
                write_json(run/"label_contract.json", {"pair_folds": [a, b],
                    "held_station_labels_and_all_val_test_cells": "inputs and training labels unchanged",
                    "temporal_station_overlap": "validation/test cells hidden individually"})
                del changed, check_fit, check_inference
            model = clone(forest).set_params(n_jobs=2).fit(fit, np.log1p(data["y"].ravel()[outer["train"]]))
            prediction = model.predict(inference[selected])
            np.savez_compressed(run/f"{prefix}.npz", cells=selected, pred_z=prediction)
            joblib.dump(model, run/f"{prefix}.joblib", compress=3)
            write_json(run/f"{prefix}.json", {"folds": [a, b], "hidden_stations": hidden.tolist(),
                "fitted_stations": np.unique(outer["train"]//months).tolist()})
            bind_files(run, stage, [run/f"{prefix}{suffix}" for suffix in (".npz", ".joblib", ".json")], config)
            del model, fit, inference
            gc.collect()
        with np.load(run/f"{prefix}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["cells"], selected)
            references[(a, b)] = {**json.loads((run/f"{prefix}.json").read_text()),
                "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
        files += [f"{prefix}{suffix}" for suffix in (".npz", ".joblib", ".json")]+[stage]
        progress("pair_reference", {"folds": [a, b]})
    return references, files


def run_one(root, mask, seed, runtime):
    run, parent = root/"runs"/f"{mask}_seed{seed}", PARENT/"runs"/f"{mask}_seed{seed}"
    if (run/"complete.json").exists():
        config = json.loads((run/"config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("preserve temporal attention execution version")
        verify_files(run, "complete.json", config)
        return
    started = time.monotonic()
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "q90_threshold_train", "mask", "seed")}, "experiment": ROOT.name,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "models": [*old["models"], *MODELS], "epochs": 30, "patience": 5,
        "candidate_count": 20, "attention_heads": 2, "attention_dimensions": 32, "lookback": 12,
        "current_source_requires_previous_observation": False, "source_library_roles": ["train"],
        "inference_roles": list(FIT_ROLES), "selection_role": "validation only",
        "memory": "disabled: train/val stations overlap", "fusion": "same retained validation-selected family",
        "scope": "retrospective temporal compatibility; temporal test already inspected for preceding recipe"}
    run.mkdir(parents=True, exist_ok=True)
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if digest({k: v for k, v in saved.items() if k != "started_at"}) != digest(
                {k: v for k, v in config.items() if k != "started_at"}):
            raise ValueError("saved temporal configuration changed")
        config = saved
    else:
        write_json(run/"config.json", config)

    def progress(stage, row):
        record = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    original = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() if role in saved.files else np.empty(0, np.int64)
                 for role in ("train", "val", "test", "context")}
    data = temporal_label_view(original, split)
    del original
    shape, t = data["y"].shape, data["y"].shape[1]
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], shape)
    expert = UnifiedDOCReconstructor.load(parent/"backbone", data, split)
    raw = extract_raw_temporal_inputs(expert, data, split)
    hidden = joblib.load(parent/"hidden.joblib")
    x39 = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    context = np.maximum(0, np.expm1(hidden.predict(np.column_stack([x39, daily.reshape(-1, 8)])))).reshape(shape)
    with np.load(parent/"oof.npz", allow_pickle=False) as saved:
        oof = saved["new"].copy()
    extra = build_regime_head_features(data["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(data)["full"], n_months=t)
    full, arrays = temporal_arrays(raw, extra, daily, data, split, oof, context)
    source = arrays[0]
    ids, val_ids = np.unique(split["train"]//t), np.unique(split["val"]//t)
    folds = station_folds(split["train"], t, seed)
    references, files = pair_references(run, data, split, daily, hidden, folds, config, progress)
    parts, aggregate_roles = nested_innovation_inputs(data["y"], data["site_no"], data["months"],
        full["env"], split["train"], folds, references)
    candidates, roles = nested_current_available_candidates(data["y"], data["site_no"], data["months"],
        full["env"], split["train"], folds, references, source["extra"][..., 30:38])
    native_ids, residual = source_residual_grid(data["y"], oof, split["train"])
    relative_ids, relative_residual = relative_source_residual_grid(data["y"], oof, split["train"])
    np.testing.assert_array_equal(ids, native_ids)
    np.testing.assert_array_equal(ids, relative_ids)
    names = np.asarray(data["site_no"], str)
    aggregate = SourceDOCInnovationLibrary().fit(names[ids], data["months"], full["env"][ids], residual)
    relative = SourceDOCInnovationLibrary().fit(names[ids], data["months"], full["env"][ids], relative_residual)
    full_parts = aggregate.predict_components(names, full["env"], data["months"])
    full_candidates = current_available_candidates(relative, names, full["env"], data["months"], names[ids],
        source["extra"][..., 30:38])
    source_view = source_level_input_view({**source, "attention_reference": source_attention_reference(arrays[1], arrays[3]),
        "extra": np.concatenate([source["extra"], innovation_readout_features(parts, "real", 1.)], -1)}, candidates, "real")
    full_view = source_level_input_view({**full, "attention_reference": context,
        "extra": np.concatenate([full["extra"], innovation_readout_features(full_parts, "real", 1.)], -1)}, full_candidates, "real")
    val_view = {key: value if key == "donor_hydro_bank" else value[val_ids] for key, value in full_view.items()}
    np.savez_compressed(run/"inputs.npz", **{f"source_{k}": v for k, v in source_view.items()},
        **{f"full_{k}": v for k, v in full_view.items()}, source_reference=arrays[1], source_truth=arrays[2],
        source_loss=arrays[3], val_reference=arrays[5], val_truth=arrays[6], val_loss=arrays[7], val_ids=val_ids)
    write_json(run/"source_roles.json", {"aggregate": aggregate_roles, "individual": roles})
    if (run/"neural_complete.json").exists():
        verify_files(run, "neural_complete.json", config)
        model = AvailableSourceAttentionResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    else:
        retained = EncoderNativeResidual.from_payload(torch.load(parent/"candidate.pt", weights_only=False, map_location="cpu"))
        for name in ("spatial", "temporal", "decay"):
            getattr(retained, name).load_state_dict(copy.deepcopy(getattr(retained, f"_initial_{name}_state")))
        settings = {**retained._config(), "extra_dim": 41, "interaction_indices": (*retained.interaction_indices, 38),
            "epochs": 30, "patience": 5}
        model = AvailableSourceAttentionResidual(retained.spatial, retained.temporal, retained.decay, **settings)
        model.fit(source_view, *arrays[1:4], val_view, *arrays[5:], tail_threshold=config["q90_threshold_train"],
            selection_role="validation", progress=lambda row: progress("attention", row))
        torch.save(model.to_payload(), run/"native.pt")
        write_json(run/"native.json", model.to_dict())
        bind_files(run, "neural_complete.json", [run/"native.pt", run/"native.json"], config)
    native = model.predict(full_view, context)
    fusion, final = temporal_fusion(context, native, data["y"], split)
    write_json(run/"fusion.json", fusion.to_dict())
    np.savez_compressed(run/"components.npz", environment=context, available_native=native, available_fusion=final)
    files += ["config.json", "label_contract.json", "inputs.npz", "source_roles.json", "native.pt", "native.json",
        "neural_complete.json", "fusion.json", "components.npz"]
    bind_files(run, "point_complete.json", [run/name for name in files], config)
    previous = pd.read_parquet(parent/"predictions.parquet")
    template = previous[previous.model_name.eq("upgraded_fusion")].copy()
    np.testing.assert_array_equal(template.cell, split["test"])
    frames = [previous]
    for name, values in zip(MODELS, (native, final), strict=True):
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, values.ravel()[split["test"]]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    files += ["point_complete.json"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"new_pair_reference_fits": 10, "new_neural_fits": 1, "test_cells": len(template)})
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", __file__, "scripts/run_doc_current_source_portable_v2_r1.py",
        "scripts/run_doc_temporal_compatibility_v1.py", "scripts/run_unified_doc_spatial.py",
        "scripts/run_doc_daily_hydro_residual_v1.py", "scripts/run_doc_geographical_confirmation_v1.py",
        "scripts/run_doc_unmonitored_trees_v1.py", "tests/test_current_source_temporal_v3.py",
        str(ROOT/"study_plan.md")):
        key = str(Path(name).relative_to(Path.cwd())) if Path(name).is_absolute() else str(name)
        snapshot[key] = sha256_file(key)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("changed temporal fitting code requires a new version")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for mask in args.masks:
        for seed in args.seeds:
            run_one(args.root, mask, seed, digest(snapshot))


if __name__ == "__main__":
    main()
