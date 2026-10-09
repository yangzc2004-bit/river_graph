"""Replicate the fixed current-source attention and its two matched controls."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_current_source_attention_v1 import ARMS, MODES
from run_doc_current_source_attention_v1 import ROOT as SOURCE
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, fit_memory, support_curves
from run_doc_geographical_confirmation_v1 import ROOT as PARENT
from run_doc_source_innovation_geographical_v1 import ROOT as PRECEDING
from run_doc_source_innovation_geographical_v1 import geographic_inputs
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    HUC4_BLOCKS,
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.current_source_attention import CurrentSourceAttentionResidual
from river_graph.models.current_source_candidates import (
    attention_input_view,
    nested_source_attention_candidates,
    source_attention_candidates,
)
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)

ROOT = Path("experiments/phase4_transfer/doc_current_source_attention_geographical_v1")
SOURCE_DECISION = SOURCE/"research_decision.md"


def subset_view(view, station_ids):
    return {key: value if key == "donor_hydro_bank" else value[station_ids] for key, value in view.items()}


def prepare(huc4, seed):
    parent, previous = (root/"runs"/f"huc4_{huc4}_seed{seed}" for root in (PARENT, PRECEDING))
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    previous_config = json.loads((previous/"config.json").read_text())
    verify_files(previous, "complete.json", previous_config)
    for kind in ("dataset", "mask"):
        if sha256_file(old[f"{kind}_path"]) != old[f"{kind}_hash"]:
            raise ValueError("geographical dataset/roles changed")
    dataset = strip_auxiliary_water(torch.load(old["dataset_path"], weights_only=False, map_location="cpu"))
    with np.load(old["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset["y"] = development_labels(dataset, split)
    shape = dataset["y"].shape
    t = shape[1]
    ids, val_ids = np.unique(split["train"]//t), np.unique(split["val"]//t)
    if np.intersect1d(ids, np.unique(np.r_[split["val"], split["test"]]//t)).size:
        raise ValueError("source stations overlap geographical receivers")
    from run_doc_daily_hydro_residual_v1 import load_daily_pack
    daily, _, _ = load_daily_pack(DAILY_ROOT, old["dataset_hash"], shape)
    with np.load(parent/"components.npz", allow_pickle=False) as saved:
        context = saved["environment"].copy()
    with np.load(parent/"oof.npz", allow_pickle=False) as saved:
        oof = saved["new"].copy()
    source_inputs, full_inputs, arrays, normalization = geographic_inputs(parent, dataset, split, daily, oof, context)
    np.testing.assert_array_equal(source_inputs["extra"][..., 30:38], daily[ids])
    folds, references = station_folds(split["train"], t, seed), {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = previous/f"pair_reference_{a}_{b}"
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads(prefix.with_suffix(".json").read_text()),
                                  "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
    source_candidates, roles = nested_source_attention_candidates(dataset["y"], dataset["site_no"],
        dataset["months"], full_inputs["env"], split["train"], folds, references, daily[ids])
    library = SourceDOCInnovationLibrary.load(previous/"source_library.npz")
    np.testing.assert_array_equal(library.source_names_, np.asarray(dataset["site_no"], str)[ids])
    full_candidates = source_attention_candidates(library, dataset["site_no"], full_inputs["env"],
        dataset["months"], library.source_names_, daily[ids])
    source_parts, old_roles = nested_innovation_inputs(dataset["y"], dataset["site_no"], dataset["months"],
        full_inputs["env"], split["train"], folds, references)
    if roles != old_roles:
        # The older role record also reports source-cell counts, but all
        # station populations and exclusions must agree exactly.
        for actual, expected in zip(roles, old_roles, strict=True):
            for key in actual:
                if actual[key] != expected[key]:
                    raise ValueError("nested query/source roles changed")
    full_parts = library.predict_components(dataset["site_no"], full_inputs["env"], dataset["months"])
    features = {mode: (innovation_readout_features(source_parts, mode, 1.),
                       innovation_readout_features(full_parts, mode, 1.)) for mode in ("real", "historical")}
    with np.load(previous/"innovation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_features"], features["real"][0])
        np.testing.assert_array_equal(saved["full_features"], features["real"][1])
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    for name in ("spatial", "temporal", "decay"):
        getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
    settings = {**retained._config(), "extra_dim": retained.extra_dim+3,
                "interaction_indices": (*retained.interaction_indices, retained.extra_dim), "epochs": 30, "patience": 5}
    return {"parent": parent, "previous": previous, "old": old, "data": dataset, "split": split,
            "source_ids": ids, "validation_ids": val_ids, "source_inputs": source_inputs, "full_inputs": full_inputs,
            "source_candidates": source_candidates, "full_candidates": full_candidates, "source_roles": roles,
            "features": features, "arrays": arrays, "context": context, "oof": oof,
            "normalization": normalization, "retained": retained, "settings": settings}


def model_views(prepared, mode):
    source_features, full_features = prepared["features"][mode]
    source = {**prepared["source_inputs"], "extra": np.concatenate([prepared["source_inputs"]["extra"], source_features], -1)}
    full = {**prepared["full_inputs"], "extra": np.concatenate([prepared["full_inputs"]["extra"], full_features], -1)}
    return (attention_input_view(source, prepared["source_candidates"], mode),
            attention_input_view(full, prepared["full_candidates"], mode))


def run_one(root, huc4, seed, runtime):
    run = root/"runs"/f"huc4_{huc4}_seed{seed}"
    if (run/"complete.json").exists():
        config = json.loads((run/"config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("completed geographical attention version changed")
        verify_files(run, "complete.json", config)
        return
    started = time.monotonic()
    prepared = prepare(huc4, seed)
    old, dataset, split = prepared["old"], prepared["data"], prepared["split"]
    run.mkdir(parents=True, exist_ok=True)
    previous_config = json.loads((prepared["previous"]/"config.json").read_text())
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "target_huc4", "split_seed", "seed", "q90_threshold_train", "support_adapter", "query_policy")},
        "experiment": ROOT.name, "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime_snapshot_hash": runtime, "models": [*previous_config["models"], *ARMS],
        "source_decision": str(SOURCE_DECISION), "source_decision_hash": sha256_file(SOURCE_DECISION),
        "epochs": 30, "patience": 5, "attention_heads": 2, "attention_dimensions": 32,
        "new_forest_fits": 0, "reused_pair_references": 10, "selection_role": "source_validation",
        "evaluation_role": "retrospective ST357 geographical replication; not external validation"}
    for name in ("parent", "previous"):
        config[f"{name}_run"] = str(prepared[name])
        config[f"{name}_completion_hash"] = sha256_file(prepared[name]/"complete.json")
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if digest({k: v for k, v in saved.items() if k != "started_at"}) != digest(
                {k: v for k, v in config.items() if k != "started_at"}):
            raise ValueError("saved geographical attention execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)

    def progress(stage, row):
        record = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    files = ["config.json", "source_library_roles.json", "attention_candidates.npz", "readout_normalization.json"]
    write_json(run/"source_library_roles.json", prepared["source_roles"])
    write_json(run/"readout_normalization.json", prepared["normalization"])
    np.savez_compressed(run/"attention_candidates.npz",
        **{f"source_{k}": v for k, v in prepared["source_candidates"].items()},
        **{f"full_{k}": v for k, v in prepared["full_candidates"].items()})
    bases, components, diagnostics = {}, {"environment": prepared["context"]}, {}
    t = dataset["y"].shape[1]
    for prefix, (source_mode, attention_mode) in MODES.items():
        source_view, full_view = model_views(prepared, source_mode)
        validation_view = subset_view(full_view, prepared["validation_ids"])
        stage = f"{prefix}_complete.json"
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = CurrentSourceAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt", weights_only=False, map_location="cpu"))
        else:
            retained = prepared["retained"]
            model = CurrentSourceAttentionResidual(retained.spatial, retained.temporal, retained.decay,
                attention_mode=attention_mode, **prepared["settings"])
            arrays = prepared["arrays"]
            model.fit(source_view, *arrays[1:4], validation_view, *arrays[5:],
                tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                progress=lambda row, arm=prefix: progress(arm, row))
            torch.save(model.to_payload(), run/f"{prefix}.pt")
            write_json(run/f"{prefix}.json", model.to_dict())
            bind_files(run, stage, [run/f"{prefix}.pt", run/f"{prefix}.json"], config)
        native = model.predict(full_view, prepared["context"])
        memory = fit_memory(dataset, split, prepared["oof"], prepared["context"], native, t)
        integrated = memory.predict(prepared["context"], native)
        write_json(run/f"{prefix}_memory.json", memory.to_dict())
        bases[f"{prefix}_native"], bases[f"{prefix}_integrated"] = native, integrated
        components[f"{prefix}_native"], components[f"{prefix}_integrated"] = native, integrated
        diagnostics[prefix] = model.diagnostics(full_view, split["test"])
        np.savez_compressed(run/f"{prefix}_diagnostics.npz", cells=split["test"], **diagnostics[prefix])
        files += [f"{prefix}{suffix}" for suffix in (".pt", ".json", "_complete.json", "_memory.json", "_diagnostics.npz")]
        del model, memory, source_view, full_view, validation_view
        gc.collect()
    np.savez_compressed(run/"components.npz", **components)
    files += ["components.npz"]
    bind_files(run, "point_complete.json", [run/name for name in files], config)
    # Point states and source/validation selections are now saved. Open only
    # designated support and test-scoring labels after this boundary.
    truth = np.asarray(torch.load(config["dataset_path"], weights_only=False, map_location="cpu")["y"], float)
    curves, adapters = support_curves(bases, dataset, split, truth)
    curves["split_seed"], curves["seed"], curves["target_huc4"] = int(huc4), seed, huc4
    old_curves = pd.read_parquet(prepared["previous"]/"support_curves.parquet")
    pd.concat([old_curves, curves], ignore_index=True).to_parquet(run/"support_curves.parquet", index=False)
    write_json(run/"support_adapters.json", adapters)
    previous = pd.read_parquet(prepared["previous"]/"predictions.parquet")
    template = previous[previous.model_name.eq("unmonitored_integrated")].copy()
    np.testing.assert_array_equal(template.cell, split["test"])
    np.testing.assert_array_equal(template.y_true, truth.ravel()[split["test"]])
    frames = [previous.copy()]
    for name, grid in bases.items():
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, grid.ravel()[split["test"]]
        prefix = name.rsplit("_", 1)[0]
        for key in ("prior_mass", "entropy"):
            frame[f"attention_{key}"] = diagnostics[prefix][key].mean(-1)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    files += ["point_complete.json", "support_adapters.json"]
    for name in ("predictions.parquet", "support_curves.parquet"):
        bind_product(run, name, config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json",
        "support_curves.parquet", "support_curves.meta.json")], config)
    progress("complete", {"new_neural_fits": 3, "reused_pair_references": 10, "new_forest_fits": 0,
                          "query_cells": len(split["test"])})
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--huc4", nargs="+", choices=HUC4_BLOCKS, default=list(HUC4_BLOCKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_current_source_attention_geographical_v1.py",
                 "scripts/verify_doc_current_source_attention_geographical_v1.py",
                 "scripts/run_doc_source_innovation_geographical_v1.py", "scripts/run_doc_geographical_confirmation_v1.py",
                 "scripts/run_doc_daily_hydro_residual_v1.py", "scripts/run_doc_source_retrieval_v1.py",
                 "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py",
                 "tests/test_current_source_attention.py", str(SOURCE_DECISION), str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve executed fitting code; changed code requires a new version")
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
