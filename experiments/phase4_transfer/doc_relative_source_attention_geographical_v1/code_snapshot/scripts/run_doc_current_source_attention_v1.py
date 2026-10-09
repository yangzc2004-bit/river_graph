"""Learn hydrology-conditioned current donors in the existing DOC residual."""
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
from run_doc_source_innovation_learning_v1 import (
    INPUT_FIELDS,
    PARENT,
    PROBE,
    SOURCE_INPUTS,
)
from run_doc_source_innovation_learning_v1 import ROOT as LEARNING
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.current_source_attention import CurrentSourceAttentionResidual
from river_graph.models.current_source_candidates import (
    attention_input_view,
    nested_source_attention_candidates,
    source_attention_candidates,
)
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary

ROOT = Path("experiments/phase4_transfer/doc_current_source_attention_v1")
MODES = {"attention_real": ("real", "learned"), "attention_historical": ("historical", "learned"),
         "attention_fixed": ("real", "fixed_prior")}
ARMS = tuple(f"{mode}_{kind}" for mode in MODES for kind in ("native", "integrated"))


def prepare(partition, seed):
    """Reuse immutable source roles, inputs and all ten double-held references."""
    name = f"split{partition}_seed{seed}"
    parent, learning, source, probe = (root/"runs"/name for root in (PARENT, LEARNING, SOURCE_INPUTS, PROBE))
    configs = {}
    for path in (parent, learning, source, probe):
        configs[str(path)] = json.loads((path/"config.json").read_text())
        verify_files(path, "complete.json", configs[str(path)])
    old = configs[str(parent)]
    for kind in ("dataset", "mask"):
        if sha256_file(old[f"{kind}_path"]) != old[f"{kind}_hash"]:
            raise ValueError("source dataset or roles changed")
    data = strip_auxiliary_water(torch.load(old["dataset_path"], weights_only=False, map_location="cpu"))
    with np.load(old["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    data["y"] = development_labels(data, split)
    t = data["y"].shape[1]
    ids, val_ids = np.unique(split["train"]//t), np.unique(split["val"]//t)
    if np.intersect1d(ids, np.unique(np.r_[split["val"], split["test"]]//t)).size:
        raise ValueError("receiving stations overlap the source library")
    with np.load(source/"joint_source_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_station_ids"], ids)
        source_inputs = {key: saved[key].copy() for key in INPUT_FIELDS}
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], split["val"])
        validation_inputs = {key: saved[key].copy() for key in INPUT_FIELDS}
        base = saved["context"].copy()
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    folds, references = station_folds(split["train"], t, seed), {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = learning/f"pair_reference_{a}_{b}"
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads(prefix.with_suffix(".json").read_text()),
                                  "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
    ecology = np.full((len(data["site_no"]), 9), np.nan)
    ecology[ids] = source_inputs["env"]
    hydro = source_inputs["extra"][..., 30:38]
    if not np.isfinite(hydro).all() or ((hydro < 0) | (hydro > 1)).any():
        raise ValueError("daily hydrology must retain its existing bounded semantics")
    source_candidates, roles = nested_source_attention_candidates(data["y"], data["site_no"], data["months"],
        ecology, split["train"], folds, references, hydro)
    library = SourceDOCInnovationLibrary.load(probe/"source_library.npz")
    np.testing.assert_array_equal(library.source_names_, np.asarray(data["site_no"], str)[ids])
    validation_candidates = source_attention_candidates(library, np.asarray(data["site_no"], str)[val_ids],
        validation_inputs["env"], data["months"], library.source_names_, hydro)
    features = {}
    with np.load(learning/"innovation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_ids"], ids)
        np.testing.assert_array_equal(saved["validation_ids"], val_ids)
        np.testing.assert_array_equal(saved["cells"], split["val"])
        for mode in ("real", "historical"):
            features[mode] = (saved[f"source_{mode}"].copy(), saved[f"validation_{mode}"].copy())
    source_mask, val_mask = np.zeros(data["y"].shape, bool), np.zeros(data["y"].shape, bool)
    source_mask.ravel()[split["train"]] = True
    _, query = support_query_cells(split, target_role="val", k=0, n_months=t)
    val_mask.ravel()[query] = True
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    for module in ("spatial", "temporal", "decay"):
        getattr(retained, module).load_state_dict(getattr(retained, f"_initial_{module}_state"))
    settings = {**retained._config(), "extra_dim": retained.extra_dim+3,
                "interaction_indices": (*retained.interaction_indices, retained.extra_dim),
                "epochs": 30, "patience": 5}
    return {"parent": parent, "learning": learning, "source": source, "probe": probe, "old": old,
            "data": data, "split": split, "source_ids": ids, "validation_ids": val_ids,
            "source_inputs": source_inputs, "validation_inputs": validation_inputs,
            "source_candidates": source_candidates, "validation_candidates": validation_candidates,
            "source_roles": roles, "features": features, "source_mask": source_mask[ids],
            "validation_mask": val_mask[val_ids], "source_base": np.maximum(0., np.expm1(oof[ids])),
            "validation_base": base, "oof": oof, "retained": retained, "settings": settings}


def model_views(prepared, mode):
    source_features, validation_features = prepared["features"][mode]
    source = {**prepared["source_inputs"], "extra": np.concatenate(
        [prepared["source_inputs"]["extra"], source_features], -1)}
    validation = {**prepared["validation_inputs"], "extra": np.concatenate(
        [prepared["validation_inputs"]["extra"], validation_features], -1)}
    return (attention_input_view(source, prepared["source_candidates"], mode),
            attention_input_view(validation, prepared["validation_candidates"], mode))


def run_one(root, partition, seed, runtime):
    run = root/"runs"/f"split{partition}_seed{seed}"
    if (run/"complete.json").exists():
        config = json.loads((run/"config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("completed attention execution changed")
        verify_files(run, "complete.json", config)
        return
    started = time.monotonic()
    prepared = prepare(partition, seed)
    old, data, split = prepared["old"], prepared["data"], prepared["split"]
    ids, val_ids, val = prepared["source_ids"], prepared["validation_ids"], split["val"]
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "q90_threshold_train", "split_seed", "seed")}, "experiment": "doc_current_source_attention_v1",
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*json.loads((prepared["learning"]/"config.json").read_text())["models"], *ARMS],
        "epochs": 30, "patience": 5, "lookback": 12, "attention_heads": 2, "attention_dimensions": 32,
        "new_forest_fits": 0, "reused_pair_references": 10, "selection_role": "source_validation",
        "evaluation_role": "source_validation_only", "receiving_water_quality": "none",
        "mechanism": "current GRU/ecology/hydro reallocates actual source seasonal innovations"}
    for name in ("parent", "learning", "source", "probe"):
        config[f"{name}_run"] = str(prepared[name])
        config[f"{name}_completion_hash"] = sha256_file(prepared[name]/"complete.json")
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if digest({k: v for k, v in saved.items() if k != "started_at"}) != digest(
                {k: v for k, v in config.items() if k != "started_at"}):
            raise ValueError("saved attention fitting version changed")
        config = saved
    else:
        write_json(run/"config.json", config)

    def progress(stage, row):
        record = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    files = ["config.json", "source_library_roles.json", "attention_candidates.npz"]
    write_json(run/"source_library_roles.json", prepared["source_roles"])
    np.savez_compressed(run/"attention_candidates.npz",
        **{f"source_{k}": v for k, v in prepared["source_candidates"].items()},
        **{f"validation_{k}": v for k, v in prepared["validation_candidates"].items()})
    previous = pd.read_parquet(prepared["learning"]/"predictions.parquet")
    frames = [previous.copy()]
    template = previous[previous.model_name.eq("unmonitored_residual")].copy()
    np.testing.assert_array_equal(template.cell, val)
    t = data["y"].shape[1]
    local_val = np.searchsorted(val_ids, val//t), val % t
    full_base = np.zeros(data["y"].shape)
    full_base[val_ids] = prepared["validation_base"]
    retained = prepared["retained"]
    for prefix, (source_mode, attention_mode) in MODES.items():
        source_view, val_view = model_views(prepared, source_mode)
        stage = f"{prefix}_complete.json"
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = CurrentSourceAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt",
                weights_only=False, map_location="cpu"))
        else:
            model = CurrentSourceAttentionResidual(retained.spatial, retained.temporal, retained.decay,
                attention_mode=attention_mode, **prepared["settings"])
            model.fit(source_view, prepared["source_base"], np.asarray(data["y"])[ids], prepared["source_mask"],
                val_view, prepared["validation_base"], np.asarray(data["y"])[val_ids], prepared["validation_mask"],
                tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                progress=lambda row, arm=prefix: progress(arm, row))
            torch.save(model.to_payload(), run/f"{prefix}.pt")
            write_json(run/f"{prefix}.json", model.to_dict())
            bind_files(run, stage, [run/f"{prefix}.pt", run/f"{prefix}.json"], config)
        prediction = model.predict(val_view, prepared["validation_base"])
        memory = EcologicalResidualTransfer().fit(np.asarray(data["regime"]), split["train"],
            np.maximum(0., np.expm1(prepared["oof"].ravel()[split["train"]])),
            np.asarray(data["y"]).ravel()[split["train"]], n_months=t, validation_cells=val,
            validation_y=np.asarray(data["y"]).ravel()[val], validation_context=full_base.ravel()[val],
            validation_temporal=prediction[local_val], selection_role="source_validation")
        full_native = full_base.copy()
        full_native[val_ids] = prediction
        integrated = memory.predict(full_base, full_native).ravel()[val]
        write_json(run/f"{prefix}_memory.json", memory.to_dict())
        diagnostics = model.diagnostics(val_view, np.ravel_multi_index(local_val, (len(val_ids), t)))
        np.savez_compressed(run/f"{prefix}_diagnostics.npz", cells=val, **diagnostics)
        for kind, values in (("native", prediction[local_val]), ("integrated", integrated)):
            frame = template.copy()
            frame["model_name"], frame["y_pred"] = f"{prefix}_{kind}", values
            for name in ("prior_mass", "entropy"):
                frame[f"attention_{name}"] = diagnostics[name].mean(-1)
            frames.append(frame)
        files += [f"{prefix}{suffix}" for suffix in (".pt", ".json", "_complete.json", "_memory.json", "_diagnostics.npz")]
        del model, memory, source_view, val_view
        gc.collect()
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"new_neural_fits": 3, "reused_pair_references": 10, "new_forest_fits": 0})


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
    for name in ("scripts/run_ladder.py", "scripts/run_doc_current_source_attention_v1.py",
                 "scripts/verify_doc_current_source_attention_v1.py", "tests/test_current_source_attention.py",
                 "scripts/run_doc_source_innovation_learning_v1.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_unified_doc_spatial.py", str(ROOT/"study_plan.md")):
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
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == "__main__":
    main()
