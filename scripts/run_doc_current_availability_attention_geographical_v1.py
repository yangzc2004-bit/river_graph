"""Replicate current-only donor availability across fixed DOC regions."""
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
from run_doc_current_availability_attention_v1 import ARMS, MODES
from run_doc_current_availability_attention_v1 import ROOT as SOURCE
from run_doc_current_source_attention_geographical_v1 import ROOT as NATIVE_ATTENTION
from run_doc_current_source_attention_geographical_v1 import prepare as native_prepare
from run_doc_geographical_confirmation_v1 import fit_memory, support_curves
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    HUC4_BLOCKS,
)
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
    nested_current_available_candidates,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_level_attention import source_level_input_view

PRECEDING = Path("experiments/phase4_transfer/doc_source_level_attention_geographical_v1")
ROOT = Path("experiments/phase4_transfer/doc_current_availability_attention_geographical_v1")
SOURCE_DECISION = SOURCE/"research_decision.md"


def subset_view(view, station_ids):
    return {key: value if key == "donor_hydro_bank" else value[station_ids] for key, value in view.items()}


def prepare(huc4, seed):
    """Reuse baseline inputs; open current donors without earlier-year requirements."""
    prepared = native_prepare(huc4, seed)
    native = NATIVE_ATTENTION/"runs"/f"huc4_{huc4}_seed{seed}"
    verify_files(native, "complete.json", json.loads((native/"config.json").read_text()))
    prepared["native_attention"] = native
    preceding = PRECEDING/"runs"/f"huc4_{huc4}_seed{seed}"
    verify_files(preceding, "complete.json", json.loads((preceding/"config.json").read_text()))
    prepared["preceding_attention"] = preceding
    data, split = prepared["data"], prepared["split"]
    ids, t = prepared["source_ids"], data["y"].shape[1]
    folds, references = station_folds(split["train"], t, seed), {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = prepared["previous"]/f"pair_reference_{a}_{b}"
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads(prefix.with_suffix(".json").read_text()),
                "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
    hydro = prepared["source_inputs"]["extra"][..., 30:38]
    prepared["source_candidates"], prepared["source_roles"] = nested_current_available_candidates(
        data["y"], data["site_no"], data["months"], prepared["full_inputs"]["env"],
        split["train"], folds, references, hydro)
    library_ids, residual = relative_source_residual_grid(data["y"], prepared["oof"], split["train"])
    np.testing.assert_array_equal(library_ids, ids)
    library = SourceDOCInnovationLibrary().fit(np.asarray(data["site_no"], str)[ids], data["months"],
        prepared["source_inputs"]["env"], residual)
    prepared["relative_library"] = library
    prepared["full_candidates"] = current_available_candidates(library, data["site_no"],
        prepared["full_inputs"]["env"], data["months"], library.source_names_, hydro)
    prepared["source_reference_tertiles"] = np.quantile(
        prepared["arrays"][1][prepared["arrays"][3]], [1/3, 2/3]).tolist()
    return prepared


def model_views(prepared, mode):
    source_features, full_features = prepared["features"]["historical" if mode == "historical" else "real"]
    source = {**prepared["source_inputs"],
        "attention_reference": np.nan_to_num(prepared["arrays"][1], nan=0.), "extra": np.concatenate([prepared["source_inputs"]["extra"], source_features], -1)}
    full = {**prepared["full_inputs"], "attention_reference": prepared["context"], "extra": np.concatenate([prepared["full_inputs"]["extra"], full_features], -1)}
    return (source_level_input_view(source, prepared["source_candidates"], mode),
            source_level_input_view(full, prepared["full_candidates"], mode))


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
    previous_config = json.loads((prepared["preceding_attention"]/"config.json").read_text())
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "target_huc4", "split_seed", "seed", "q90_threshold_train", "support_adapter", "query_policy")},
        "experiment": ROOT.name, "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime_snapshot_hash": runtime, "models": [*previous_config["models"], *ARMS],
        "source_decision": str(SOURCE_DECISION), "source_decision_hash": sha256_file(SOURCE_DECISION),
        "epochs": 30, "patience": 5, "attention_heads": 2, "attention_dimensions": 32,
        "new_forest_fits": 0, "reused_pair_references": 10, "selection_role": "source_validation",
        "lookback": 12, "receiving_water_quality": "none",
        "current_source_requires_previous_observation": False,
        "aggregate_inputs": "original matched earlier-season population unchanged",
        "mechanism": "expand individual current donor availability only",
        "evaluation_role": "retrospective ST357 geographical replication; not external validation",
        "source_value_units": "dimensionless OOF log1p residual including seasonal mean",
        "receiving_scale": "one plus frozen native receiving reference",
        "source_reference_tertiles": prepared["source_reference_tertiles"]}
    for name in ("parent", "previous", "native_attention", "preceding_attention"):
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

    files = ["config.json", "source_library_roles.json", "attention_candidates.npz", "readout_normalization.json",
             "relative_source_library.npz", "relative_source_library_units.json"]
    prepared["relative_library"].save(run/"relative_source_library.npz")
    write_json(run/"relative_source_library_units.json", {
        "source_value_units": config["source_value_units"], "source_only": True,
        "seasonal_centering": "source OOF log1p residual split into donor seasonal mean and departure",
        "receiving_scale": config["receiving_scale"]})
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
            model = AvailableSourceAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt", weights_only=False, map_location="cpu"))
        else:
            retained = prepared["retained"]
            model = AvailableSourceAttentionResidual(retained.spatial, retained.temporal, retained.decay,
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
    old_curves = pd.read_parquet(prepared["preceding_attention"]/"support_curves.parquet")
    pd.concat([old_curves, curves], ignore_index=True).to_parquet(run/"support_curves.parquet", index=False)
    write_json(run/"support_adapters.json", adapters)
    previous = pd.read_parquet(prepared["preceding_attention"]/"predictions.parquet")
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
    progress("complete", {"new_neural_fits": 2, "reused_pair_references": 10, "new_forest_fits": 0,
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
    for name in ("scripts/run_ladder.py", "scripts/run_doc_current_availability_attention_geographical_v1.py",
                 "scripts/verify_doc_current_availability_attention_geographical_v1.py",
                 "scripts/run_doc_source_innovation_geographical_v1.py",
                 "scripts/run_doc_current_source_attention_geographical_v1.py",
                 "scripts/run_doc_current_source_attention_v1.py",
                 "scripts/run_doc_current_availability_attention_v1.py", "tests/test_source_current_availability.py",
                 "scripts/run_doc_source_level_attention_v1.py",
                 "scripts/run_doc_source_level_attention_geographical_v1.py", "tests/test_source_level_attention.py", "scripts/run_doc_geographical_confirmation_v1.py",
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
