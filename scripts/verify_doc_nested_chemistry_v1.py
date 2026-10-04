"""Replay the separate chemical increment without fitting any model."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_doc_nested_chemistry_v1 import (
    NESTED,
    PARTITIONS,
    ROOT,
    SEEDS,
    build_products,
    load_parent,
    make_episodes,
)
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.nested_chemical_adapter import NestedChemicalAdapter


def verify_one(root, partition, seed, runtime):
    run = root / "runs" / f"split{partition}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    if (config["runtime_snapshot_hash"] != runtime or config["target_evaluation"]
            or config["split_seed"] != partition or config["seed"] != seed):
        raise ValueError("Nested increment identity/scope differs")
    verify_files(run, "complete.json", config)
    prior = Path(config["parent_run"])
    if sha256_file(prior / "complete.json") != config["parent_completion_hash"]:
        raise ValueError("Parent completion changed")
    for name, expected in config["parent_bindings"].items():
        if sha256_file(prior / name) != expected:
            raise ValueError(f"Parent component changed: {name}")
    _, split, full, labels, months, legacy, coordinates, active, _, states, references = load_parent(prior)
    saved = {cv: pd.read_parquet(run / filename) for cv, filename in (
        (False, "validation.parquet"), (True, "cv_predictions.parquet"))}
    count = 0
    for mode, name in NESTED.items():
        adapter = NestedChemicalAdapter.from_dict(json.loads((run / f"nested_{mode}.json").read_text()))
        episodes = make_episodes(full, references, labels, split, months, legacy, states, coordinates[mode], active)
        for cv in saved:
            replay = build_products(adapter, episodes, references, name, cross_validation=cv).sort_values(["k", "cell"])
            product = saved[cv][saved[cv].model_name.eq(name)].sort_values(["k", "cell"])
            for field in ("cell", "k", "station", "month", "y_pred", "base_pred", "adaptation_delta",
                          "chemical_delta", "chemical_support_count", "conditional_fold"):
                np.testing.assert_array_equal(replay[field].to_numpy(), product[field].to_numpy())
            count += len(product)
    print(f"{run.name}: exact nested/fold prediction replay", flush=True)
    return {"run": run.name, "status": "verified", "replayed_rows": count,
            "target_doc_read": False, "new_neural_fits": 0, "new_forest_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    results = []
    for partition in PARTITIONS:
        for seed in SEEDS:
            results.append(verify_one(args.root, partition, seed, runtime))
            gc.collect()
    write_json(args.root / "verification" / "replay.json", {
        "verifier_hash": sha256_file(__file__), "runtime_snapshot_hash": runtime,
        "status": "verified", "results": results})


if __name__ == "__main__":
    main()
