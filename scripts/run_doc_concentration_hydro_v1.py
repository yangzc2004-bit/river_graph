"""Fit small source-OOF concentration/hydro corrections on existing DOC experts."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_tail_residual_v1 import bind_product
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import development_labels
from river_graph.models.concentration_hydro_calibration import (
    ConcentrationHydroCalibration,
    hydro_calibration_inputs,
)

ROOT = Path("experiments/phase4_transfer/doc_concentration_hydro_v1")
MODES = ("log_affine", "concentration_hydro")


def run_one(root, partition, seed, runtime):
    started = time.monotonic()
    parent = PARENT/"runs"/f"split{partition}_seed{seed}"
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    run = root/"runs"/parent.name
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
              "q90_threshold_train", "split_seed", "seed")},
        "experiment": "doc_concentration_hydro_v1", "parent_run": str(parent),
        "parent_completion_hash": sha256_file(parent/"complete.json"), "runtime_snapshot_hash": runtime,
        "started_at": datetime.now(timezone.utc).isoformat(), "selection_role": "source_validation",
        "calibration_fit_role": "source_training_oof", "evaluation_role": "source_validation_only",
        "models": [*old["models"], *(f"{arm}_{mode}" for mode in MODES for arm in ("trees", "native", "integrated"))],
        "correction_modes": list(MODES), "ridge": .1, "smooth_eps": .05,
        "feature_clip_source_sd": 3, "concentration_knot_quantile": .75,
        "coefficient_bounds": [-3, 3], "normalization_role": "source_training_oof",
        "main_change": "source concentration/hydro bias correction, existing neural and memory states fixed",
        "target_information": "no receiving-site DOC/pH/conductivity; no local chemical histories"}
    if (run/"config.json").exists():
        saved = json.loads((run/"config.json").read_text())
        if saved["runtime_snapshot_hash"] != runtime or saved["parent_completion_hash"] != config["parent_completion_hash"]:
            raise ValueError("saved correction execution changed")
        config = saved
    else:
        write_json(run/"config.json", config)
    if (run/"complete.json").exists():
        verify_files(run, "complete.json", config)
        return
    if sha256_file(config["dataset_path"]) != config["dataset_hash"] or sha256_file(config["mask_path"]) != config["mask_hash"]:
        raise ValueError("source data or role assignments changed")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    dataset["y"] = development_labels(dataset, split)
    months = dataset["y"].shape[1]
    train, val = split["train"], split["val"]
    if np.intersect1d(train//months, val//months).size:
        raise ValueError("training and validation stations must be disjoint")
    oof_records = json.loads((parent/"oof_records.json").read_text())
    for fold in oof_records:
        if set(fold["hidden_stations"]) & set(fold["fitted_stations"]):
            raise ValueError("source held station entered its OOF tree")
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy().ravel()
    present = np.zeros(oof.size, bool)
    present[train] = True
    if not np.isfinite(oof[present]).all() or not np.isnan(oof[~present]).all():
        raise ValueError("OOF inputs must cover exactly source training cells")
    source_base = np.maximum(0., np.expm1(oof[train]))
    source_truth = np.asarray(dataset["y"]).ravel()[train]
    hydro = hydro_calibration_inputs(dataset).reshape(-1, 6)
    previous = pd.read_parquet(parent/"predictions.parquet")
    frames = [previous.copy()]
    template = previous[previous.model_name.eq("station_hidden_trees")].copy()
    np.testing.assert_array_equal(template.cell, val)
    np.testing.assert_array_equal(template.y_true, np.asarray(dataset["y"]).ravel()[val])
    base = template.y_pred.to_numpy(float)
    corrections = {"trees": np.zeros(len(val)),
        "native": previous[previous.model_name.eq("unmonitored_residual")].y_pred.to_numpy(float)-base,
        "integrated": previous[previous.model_name.eq("unmonitored_integrated")].y_pred.to_numpy(float)-base}
    for name in ("unmonitored_residual", "unmonitored_integrated"):
        np.testing.assert_array_equal(previous[previous.model_name.eq(name)].cell, val)
    files = ["config.json", "calibration_inputs.npz"]
    np.savez_compressed(run/"calibration_inputs.npz", source_prediction=source_base,
        source_truth=source_truth, source_hydro=hydro[train], source_cell=train,
        query_prediction=base, query_hydro=hydro[val], query_cell=val,
        native_residual=corrections["native"], integrated_residual=corrections["integrated"])
    for mode in MODES:
        model = ConcentrationHydroCalibration(mode, ridge=config["ridge"], smooth_eps=config["smooth_eps"])
        model.fit(source_base, source_truth, hydro[train], fit_role="source_training_oof")
        prediction = model.predict(base, hydro[val])
        write_json(run/f"{mode}.json", model.to_dict())
        files.append(f"{mode}.json")
        for arm, residual in corrections.items():
            frame = template.copy()
            frame["model_name"] = f"{arm}_{mode}"
            frame["y_pred"] = np.maximum(0., prediction+residual)
            frame["calibration_delta"] = prediction-base
            frame["retained_residual"] = residual
            frames.append(frame)
        record = {"run": run.name, "stage": mode, "source_rows": len(train), "validation_rows": len(val),
            "source_mae_before": model.source_mae_before_, "source_mae_after": model.source_mae_after_,
            "target_labels_read": False, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel["hydro_availability"] = np.asarray(dataset["x_mask"]).reshape(-1, 2).sum(1)[panel.cell]
    panel["reference_prediction"] = pd.Series(base, index=val).reindex(panel.cell).to_numpy()
    panel["calibration_delta"] = panel.calibration_delta.fillna(0.)
    panel["retained_residual"] = panel.retained_residual.fillna(panel.y_pred-panel.reference_prediction)
    panel.to_parquet(run/"predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    record = {"run": run.name, "stage": "complete", "models": len(config["models"]),
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
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_concentration_hydro_v1.py",
                 "scripts/analyze_doc_concentration_hydro_v1.py", "tests/test_concentration_hydro_calibration.py",
                 "scripts/run_doc_unmonitored_residual_v1.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_unified_doc_spatial.py", str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    target = args.root/"runtime_snapshot.json"
    if target.exists() and json.loads(target.read_text()) != snapshot:
        raise ValueError("saved calibration execution changed; retain it and use another version")
    if not target.exists():
        write_json(target, snapshot)
        for name in snapshot:
            destination = args.root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == "__main__":
    main()
