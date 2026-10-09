"""Test time-aligned source innovations on the retained unmonitored DOC model."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_joint_source_states_v1 import ROOT as SOURCE_INPUTS
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    corrected_prediction,
    select_correction,
    source_residual_grid,
)

ROOT = Path("experiments/phase4_transfer/doc_source_innovation_transfer_v1")
ARMS = ("source_innovation_integrated", "source_history_integrated")


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    source = SOURCE_INPUTS/"runs"/parent.name
    verify_files(source, "complete.json", json.loads((source/"config.json").read_text()))
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
              "q90_threshold_train", "split_seed", "seed")}, "experiment": "doc_source_innovation_transfer_v1",
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "source_inputs_run": str(source), "source_inputs_hash": sha256_file(source/"joint_source_inputs.npz"),
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "models": [*old["models"], *ARMS], "candidate_count": 20, "control_seed": 42,
        "alpha_grid": [0., .25, .5, 1.], "zero_prior_weight": 1.,
        "main_change": "native source DOC same-month innovation; earlier-season matched control",
        "selection_role": "source_validation", "evaluation_role": "source_validation_only",
        "neural_or_forest_refit": False, "source_statistics": "fixed seasonal source-OOF means",
        "source_receiving_roles": "station-disjoint; target DOC/pH/conductance never in library"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        for key in ("runtime_snapshot_hash", "parent_completion_hash", "source_inputs_hash"):
            if saved[key] != config[key]:
                raise ValueError("saved innovation execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("source inputs or roles changed")
    data = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as masks:
        train, val = masks["train"].copy(), masks["val"].copy()
        receiving = np.unique(np.r_[val, masks["test"]]//data["y"].shape[1])
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        source_ids, residuals = source_residual_grid(data["y"], saved["pred_z"], train)
    if np.intersect1d(source_ids, receiving).size:
        raise ValueError("source library contains receiving stations")
    with np.load(source/"joint_source_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_station_ids"], source_ids)
        ecology = saved["env"].copy()
    months = np.asarray(data["months"], str)
    names = np.asarray(data["site_no"], str)
    library = SourceDOCInnovationLibrary(candidate_count=20, control_seed=42).fit(
        names[source_ids], months, ecology, residuals)
    library.save(run/"source_library.npz")
    write_json(run/"source_library.json", library.to_dict())
    receiver_ids = np.unique(val//len(months))
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], val)
        receiver_ecology = saved["env"].copy()
    components = library.predict_components(names[receiver_ids], receiver_ecology, months)
    index = np.searchsorted(receiver_ids, val//len(months)), val % len(months)
    previous = pd.read_parquet(parent/"predictions.parquet")
    template = previous[previous.model_name.eq("unmonitored_integrated")].copy()
    np.testing.assert_array_equal(template.cell, val)
    base, truth = template.y_pred.to_numpy(), template.y_true.to_numpy()
    np.testing.assert_array_equal(truth, np.asarray(data["y"]).ravel()[val])
    selections, frames = {}, [previous.copy()]
    for arm, key in zip(ARMS, ("real_innovation", "historical_innovation"), strict=True):
        innovation = components[key][index]
        selected = select_correction(base, innovation, truth)
        selections[arm] = selected
        frame = template.copy()
        frame["model_name"] = arm
        frame["y_pred"] = corrected_prediction(base, innovation, selected["alpha"])
        frame["source_innovation"] = innovation
        frame["source_alpha"] = selected["alpha"]
        for field in ("support_count", "all_current_support_count", "weight_mass"):
            frame[f"source_{field}"] = components[field][index]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    write_json(run/"selection.json", selections)
    write_json(run/"donors.json", components["donors"])
    write_json(run/"roles.json", {"source_station_ids": source_ids.tolist(),
        "source_stations": names[source_ids].tolist(), "receiving_station_ids": receiving.tolist(),
        "validation_station_ids": receiver_ids.tolist(), "labels_in_library": "source train only",
        "source_doc_cells": len(train), "receiving_chemistry_used": False})
    np.savez_compressed(run/"receiver_inputs.npz", station=names[receiver_ids], ecology=receiver_ecology,
        months=months, cells=val, receiver_ids=receiver_ids)
    files = ["config.json", "source_library.npz", "source_library.json", "selection.json", "donors.json",
             "roles.json", "receiver_inputs.npz"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    record = {"run": run.name, "stage": "complete", "models": len(config["models"]),
        "alpha_real": selections[ARMS[0]]["alpha"], "alpha_history": selections[ARMS[1]]["alpha"],
        "real_mae": selections[ARMS[0]]["validation_mae"], "history_mae": selections[ARMS[1]]["validation_mae"],
        "retained_mae": float(np.abs(base-truth).mean()),
        "support_fraction": float((components["support_count"][index] > 0).mean()),
        "elapsed_seconds": time.monotonic()-started}
    write_json(root/"progress.json", record)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_innovation_transfer_v1.py",
                 "scripts/analyze_doc_source_innovation_transfer_v1.py", "tests/test_source_doc_innovations.py",
                 "scripts/run_doc_joint_source_states_v1.py", "scripts/run_doc_unmonitored_residual_v1.py",
                 "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve saved execution; use a new version if fitting code changes")
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
