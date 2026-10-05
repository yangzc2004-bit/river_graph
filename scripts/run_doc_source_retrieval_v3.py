"""Develop robust source response profiles with the fixed residual-contrast operator."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from run_doc_source_retrieval_v2 import PRIOR, run_one
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.doc_source_data import nested_source_episodes
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_response_profiles import SourceResponseBank

ROOT = Path("experiments/phase4_transfer/doc_source_retrieval_v3")


def freeze(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_retrieval_v1.py",
                 "scripts/run_doc_source_retrieval_v2.py", "scripts/run_doc_source_retrieval_v3.py",
                 str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve the running version after code changes")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def prepare_bank(root, partition, seed, runtime):
    start = time.monotonic()
    previous = PRIOR / "runs" / f"split{partition}_seed{seed}"
    old = json.loads((previous / "config.json").read_text())
    verify_files(previous, "complete.json", old)
    run = root / "banks" / previous.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("saved bank execution differs")
        verify_files(run, "complete.json", config)
        return run
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "seed", "split_seed")},
        "parent_run": str(previous), "parent_completion_hash": sha256_file(previous / "complete.json"),
        "runtime_snapshot_hash": runtime, "donor_bank": "nested_OOF_smooth_native_MAE_season_hydro_profiles",
        "ridge": .1, "max_donors": 20, "outer_folds": 5, "inner_folds": 5,
        "residual_scale": "unchanged source-station-balanced native absolute forest error"}
    write_json(run / "config.json", config)
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    monthly = Path(old["parent_run"])
    source_config = json.loads((monthly / "config.json").read_text())
    forest = joblib.load(Path(source_config["source_run"]) / "context.joblib")
    folds = station_folds(split["train"], dataset["y"].shape[1], seed)

    def residuals(index, cells, prediction, truth):
        np.savez_compressed(run / f"nested_residuals_{index}.npz", cells=cells,
                            oof_prediction=prediction, truth=truth)

    def progress(record):
        record = {key: len(value) if key in ("excluded_stations", "donor_stations") else value
                  for key, value in record.items()}
        entry = {"run": run.name, **record, "elapsed_seconds": time.monotonic()-start}
        write_json(root / "progress.json", entry)
        print(json.dumps(entry), flush=True)

    episodes, records = nested_source_episodes(forest, dataset, split, folds,
        bank_factory=SourceResponseBank, residual_callback=residuals, progress=progress)
    np.savez_compressed(run / "nested_episodes.npz", **{f"{key}{i}": entry[key]
        for i, entry in enumerate(episodes) for key in ("query_cells", "context")})
    write_json(run / "nested_banks.json", [entry["bank"].to_dict() for entry in episodes])
    write_json(run / "nested_fold_records.json", records)
    with np.load(Path(source_config["oof_run"]) / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].ravel()[split["train"]]
    bank = SourceResponseBank().fit(np.asarray(dataset["regime"])[:, 4:13], dataset["x"], dataset["x_mask"],
        dataset["months"], split["train"], np.maximum(0, np.expm1(oof)),
        np.asarray(dataset["y"]).ravel()[split["train"]], station_names=dataset["site_no"])
    write_json(run / "source_bank.json", bank.to_dict())
    files = ["config.json", "nested_episodes.npz", "nested_banks.json", "nested_fold_records.json", "source_bank.json"]
    files += [f"nested_residuals_{i}.npz" for i in range(5)]
    bind_files(run, "complete.json", [run / name for name in files], config)
    del forest, episodes, dataset
    gc.collect()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    runtime = freeze(args.root)
    for partition in args.splits:
        for seed in args.seeds:
            bank = prepare_bank(args.root, partition, seed, runtime)
            run_one(args.root, partition, seed, runtime, bank_run=bank, experiment_name="doc_source_retrieval_v3")


if __name__ == "__main__":
    main()
