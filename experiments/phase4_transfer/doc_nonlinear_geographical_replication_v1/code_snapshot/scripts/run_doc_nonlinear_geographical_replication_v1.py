"""Refit the source-fixed nonlinear DOC head on saved whole-region tasks."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import (
    DAILY_ROOT,
    fit_memory,
    support_curves,
)
from run_doc_geographical_confirmation_v1 import (
    MODELS as PRIOR_MODELS,
)
from run_doc_geographical_confirmation_v1 import (
    ROOT as PARENT,
)
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    HUC4_BLOCKS,
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.nonlinear_native_residual import NonlinearNativeResidual
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_nonlinear_geographical_replication_v1")
SOURCE_DECISION = Path("experiments/phase4_transfer/doc_nonlinear_native_residual_v1/research_decision.md")
ARMS = ("nonlinear_native_residual", "nonlinear_native_integrated")
MODELS = (*PRIOR_MODELS, *ARMS)


def run_one(root, huc4, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"huc4_{huc4}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    for record in ("complete.json", "backbone_complete.json", "trees_complete.json", "current_complete.json"):
        verify_files(parent, record, old)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "target_huc4", "split_seed", "seed", "q90_threshold_train")}, "experiment": root.name,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "source_decision": str(SOURCE_DECISION), "source_decision_hash": sha256_file(SOURCE_DECISION),
        "models": list(MODELS), "epochs": 30, "patience": 5, "tail_weight": 2, "readout_width": 32,
        "main_change": "fixed nonlinear readout; matched initial backbone and source views",
        "selection_role": "source_validation", "evaluation_role": "retrospective whole-HUC4 replication",
        "historical_scope": "already evaluated ST357 tasks; not independent external validation",
        "support_adapter": old["support_adapter"], "query_policy": old["query_policy"]}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if any(saved[key] != config[key] for key in ("runtime_snapshot_hash", "parent_completion_hash", "source_decision_hash")):
            raise ValueError("saved nonlinear geographical execution changed")
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
            raise ValueError("geographical data or role assignment changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    truth = np.asarray(dataset["y"], dtype=np.float64).copy()
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    shape = truth.shape
    months = shape[1]
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], shape)
    expert = UnifiedDOCReconstructor.load(parent/"backbone", dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    with np.load(parent/"components.npz", allow_pickle=False) as saved:
        context = saved["environment"].copy()
    with np.load(parent/"oof.npz", allow_pickle=False) as saved:
        oof = saved["new"].copy()
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0., np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(dataset)["full"], n_months=months)
    _, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    # Identical old query inputs, including preprocessing and causal history.
    test_ids = np.unique(split["test"]//months)
    with np.load(parent/"test_inputs.npz", allow_pickle=False) as saved:
        for key, value in full_inputs.items():
            np.testing.assert_array_equal(value[test_ids], saved[key])
        np.testing.assert_array_equal(context[test_ids], saved["context"])
    if (run/"native_complete.json").exists():
        verify_files(run, "native_complete.json", config)
        model = NonlinearNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    else:
        retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
        for name in ("spatial", "temporal", "decay"):
            getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
        model = NonlinearNativeResidual(retained.spatial, retained.temporal, retained.decay,
            readout_width=config["readout_width"], **retained._config())
        model.fit(*arrays, tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                  progress=lambda row: progress("nonlinear_native", row))
        torch.save(model.to_payload(), run/"native.pt")
        write_json(run/"native.json", model.to_dict())
        bind_files(run, "native_complete.json", [run/"native.pt", run/"native.json"], config)
    delta = model.predict_delta(full_inputs)
    native = np.maximum(0., context+model.selected_scale_*delta)
    memory = fit_memory(dataset, split, oof, context, native, months)
    integrated = memory.predict(context, native)
    write_json(run/"memory.json", memory.to_dict())
    write_json(run/"readout_normalization.json", extra["normalization"])
    np.savez_compressed(run/"components.npz", environment=context, local_temporal_delta=delta,
                        native=native, integrated=integrated)
    np.savez_compressed(run/"test_inputs.npz", **{key: value[test_ids] for key, value in full_inputs.items()},
                        context=context[test_ids], cell=split["test"])
    bind_files(run, "point_complete.json", [run/name for name in ("native.pt", "native.json", "memory.json",
               "readout_normalization.json", "components.npz", "test_inputs.npz")], config)
    # Open scoring labels and designated support only after point/state saving.
    bases = dict(zip(ARMS, (native, integrated), strict=True))
    curves, adapters = support_curves(bases, dataset, split, truth)
    curves["split_seed"], curves["seed"], curves["target_huc4"] = int(huc4), seed, huc4
    prior_curves = pd.read_parquet(parent/"support_curves.parquet")
    pd.concat([prior_curves, curves], ignore_index=True).to_parquet(run/"support_curves.parquet", index=False)
    write_json(run/"support_adapters.json", adapters)
    prior = pd.read_parquet(parent/"predictions.parquet")
    frames = [prior.copy()]
    template = prior[prior.model_name.eq("unmonitored_integrated")].copy()
    cells = split["test"]
    np.testing.assert_array_equal(template.cell, cells)
    np.testing.assert_array_equal(template.y_true, truth.ravel()[cells])
    for name, grid in bases.items():
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, grid.ravel()[cells]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    files = ["config.json", "native.pt", "native.json", "native_complete.json", "memory.json",
        "readout_normalization.json", "components.npz", "test_inputs.npz", "point_complete.json", "support_adapters.json"]
    for name in ("predictions.parquet", "support_curves.parquet"):
        bind_product(run, name, config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json",
               "support_curves.parquet", "support_curves.meta.json")], config)
    progress("complete", {"models": len(MODELS), "query_cells": len(cells), "best_epoch": model.best_epoch_})
    del model, expert, dataset, features, full_inputs, arrays, memory
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--huc4", nargs="+", default=list(HUC4_BLOCKS), choices=HUC4_BLOCKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_nonlinear_geographical_replication_v1.py",
        "scripts/analyze_doc_nonlinear_geographical_replication_v1.py", "scripts/run_doc_geographical_confirmation_v1.py",
        "scripts/run_doc_source_retrieval_v1.py", "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py",
        str(ROOT/"study_plan.md"), str(SOURCE_DECISION)):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve saved nonlinear geographical execution after code changes")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    for huc4 in args.huc4:
        for seed in args.seeds:
            run_one(args.root, huc4, seed, digest(snapshot))


if __name__ == "__main__":
    main()
