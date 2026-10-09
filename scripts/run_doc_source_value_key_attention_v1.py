"""Let permitted source DOC state inform the existing donor attention keys."""
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
from run_doc_current_availability_attention_v1 import ROOT as PRECEDING
from run_doc_current_availability_attention_v1 import (
    model_views as previous_model_views,
)
from run_doc_current_availability_attention_v1 import prepare as previous_prepare
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_value_key_attention import (
    SourceDOCKeyAttentionResidual,
    nested_source_state_key_candidates,
    source_key_rms,
    source_state_key_candidates,
    source_state_key_view,
)

ROOT = Path("experiments/phase4_transfer/doc_source_value_key_attention_v1")
MODES = {"key_current": ("current", "learned"), "key_seasonal": ("seasonal", "learned"),
         "key_zero": ("zero", "learned")}
ARMS = tuple(f"{mode}_{kind}" for mode in MODES for kind in ("native", "integrated"))


def prepare(partition, seed):
    prepared = previous_prepare(partition, seed)
    previous = PRECEDING/"runs"/f"split{partition}_seed{seed}"
    verify_files(previous, "complete.json", json.loads((previous/"config.json").read_text()))
    prepared["preceding_attention"] = previous
    data, split = prepared["data"], prepared["split"]
    ids, val_ids = prepared["source_ids"], prepared["validation_ids"]
    t = data["y"].shape[1]
    folds, references = station_folds(split["train"], t, seed), {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = prepared["learning"]/f"pair_reference_{a}_{b}"
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads(prefix.with_suffix(".json").read_text()),
                "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
    ecology = np.full((len(data["site_no"]), 9), np.nan)
    ecology[ids] = prepared["source_inputs"]["env"]
    hydro = prepared["source_inputs"]["extra"][..., 30:38]
    prepared["source_candidates"], prepared["source_roles"] = nested_source_state_key_candidates(
        data["y"], data["site_no"], data["months"], ecology, split["train"], folds, references, hydro)
    library = prepared["relative_library"]
    prepared["source_key_rms"] = source_key_rms(library)
    prepared["validation_candidates"] = source_state_key_candidates(library,
        np.asarray(data["site_no"], str)[val_ids], prepared["validation_inputs"]["env"],
        data["months"], library.source_names_, hydro, rms=prepared["source_key_rms"])
    return prepared


def model_views(prepared, mode):
    source, validation = previous_model_views(prepared, "real")
    return (source_state_key_view(source, prepared["source_candidates"], mode),
            source_state_key_view(validation, prepared["validation_candidates"], mode))


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
        "q90_threshold_train", "split_seed", "seed")}, "experiment": "doc_source_value_key_attention_v1",
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*json.loads((prepared["preceding_attention"]/"config.json").read_text())["models"], *ARMS],
        "epochs": 30, "patience": 5, "lookback": 12, "attention_heads": 2, "attention_dimensions": 32,
        "current_source_requires_previous_observation": False,
        "candidate_count": 20, "source_key_channels": 2, "source_key_rms": prepared["source_key_rms"],
        "source_key_normalization": "permitted OOF log1p residual RMS; zero not centered",
        "aggregate_inputs": "original matched earlier-season population unchanged",
        "new_forest_fits": 0, "reused_pair_references": 10, "selection_role": "source_validation",
        "evaluation_role": "source_validation_only", "receiving_water_quality": "none",
        "mechanism": "condition donor allocation on current/seasonal source DOC residual state; actual values unchanged",
        "source_value_units": "dimensionless OOF log1p residual including seasonal mean",
        "receiving_scale": "one plus frozen native receiving reference; first-order conversion",
        "source_reference_tertiles": prepared["source_reference_tertiles"]}
    for name in ("parent", "learning", "source", "probe", "native_attention", "preceding_attention"):
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

    files = ["config.json", "source_library_roles.json", "attention_candidates.npz", "source_key_normalization.json",
             "relative_source_library.npz", "relative_source_library_units.json"]
    prepared["relative_library"].save(run/"relative_source_library.npz")
    write_json(run/"source_key_normalization.json", {"full_source_rms": prepared["source_key_rms"],
        "nested_source_rms": [r["source_key_rms"] for r in prepared["source_roles"]],
        "normalization_role": "source training only; query fold excluded"})
    write_json(run/"relative_source_library_units.json", {
        "source_value_units": config["source_value_units"], "source_only": True,
        "seasonal_centering": "source OOF log1p residual split into per-donor/calendar-month mean and departure",
        "receiving_scale": config["receiving_scale"]})
    write_json(run/"source_library_roles.json", prepared["source_roles"])
    np.savez_compressed(run/"attention_candidates.npz",
        **{f"source_{k}": v for k, v in prepared["source_candidates"].items()},
        **{f"validation_{k}": v for k, v in prepared["validation_candidates"].items()})
    previous = pd.read_parquet(prepared["preceding_attention"]/"predictions.parquet")
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
            model = SourceDOCKeyAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt",
                weights_only=False, map_location="cpu"))
        else:
            model = SourceDOCKeyAttentionResidual(retained.spatial, retained.temporal, retained.decay,
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
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_value_key_attention_v1.py",
                 "scripts/verify_doc_source_value_key_attention_v1.py",
                 "scripts/run_doc_current_availability_attention_v1.py",
                 "tests/test_source_value_key_attention.py", "tests/test_current_source_attention.py",
                 "scripts/run_doc_source_innovation_learning_v1.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_current_source_attention_v1.py", "tests/test_relative_source_attention.py",
                 "scripts/run_doc_relative_source_attention_v1.py", "tests/test_source_level_attention.py",
                 "scripts/run_doc_source_level_attention_v1.py", "tests/test_source_current_availability.py",
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
