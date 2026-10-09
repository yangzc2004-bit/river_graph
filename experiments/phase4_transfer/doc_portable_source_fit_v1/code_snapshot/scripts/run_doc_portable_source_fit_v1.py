"""Refit fixed DOC procedures for deployment using only ST357 source data."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from portable_doc_reconstructor_v1 import PortableDOCReconstructor
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, DATASET
from run_unified_doc_spatial import digest, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file

ROOT = Path("experiments/phase4_transfer/doc_portable_source_fit_v1")
SOURCE_ROLES = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split142.npz")


def deployment_split(observed, reference_split):
    """Retain split142 validation stations; use all other ST stations as source.

    Membership reads only observation identities. Historical internal test
    stations become legitimate source stations for a genuinely independent
    external deployment. This split supplies no new internal test score.
    """
    mask = np.asarray(observed, bool)
    if mask.ndim != 2 or not mask.shape[1]:
        raise ValueError("source observation mask must be a station/month grid")
    validation = np.asarray(reference_split["val"])
    if (validation.ndim != 1 or validation.dtype.kind not in "iu" or not len(validation)
            or (validation < 0).any() or (validation >= mask.size).any()
            or not mask.ravel()[validation].all()):
        raise ValueError("reference validation identities must be observed source cells")
    held = np.unique(validation//mask.shape[1])
    cells = np.flatnonzero(mask)
    receiving = np.isin(cells//mask.shape[1], held)
    return {"train": cells[~receiving], "val": cells[receiving],
            "test": np.empty(0, dtype=np.int64), "context": np.empty(0, dtype=np.int64)}


def prepare_source_roles(root):
    dataset = torch.load(DATASET, weights_only=False, map_location="cpu")
    with np.load(SOURCE_ROLES, allow_pickle=False) as saved:
        reference = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    split = deployment_split(dataset["y_mask"], reference)
    path = root/"source_roles.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            for role in split:
                np.testing.assert_array_equal(previous[role], split[role])
    else:
        np.savez_compressed(path, **split)
    months = dataset["y"].shape[1]
    write_json(root/"source_roles.json", {"dataset_hash": sha256_file(DATASET),
        "reference_role_file": str(SOURCE_ROLES), "reference_role_hash": sha256_file(SOURCE_ROLES),
        "source_mask_hash": sha256_file(path),
        "train_stations": len(np.unique(split["train"]//months)),
        "validation_stations": len(np.unique(split["val"]//months)),
        "train_cells": len(split["train"]), "validation_cells": len(split["val"]),
        "external_labels_used": False,
        "scope": "source-only deployment refit; historical internal test roles are absorbed into source"})
    load_daily_pack(DAILY_ROOT, sha256_file(DATASET), tuple(dataset["y"].shape))
    return path, DAILY_ROOT/"daily_features.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    mask_path, daily_path = prepare_source_roles(args.root)
    if args.prepare_only:
        print((args.root/"source_roles.json").read_text(), flush=True)
        return
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_portable_source_fit_v1.py",
                 "scripts/fit_portable_doc_sources_v1.py", "scripts/portable_doc_reconstructor_v1.py",
                 "scripts/run_doc_geographical_confirmation_v1.py", "scripts/run_doc_unmonitored_trees_v1.py",
                 "scripts/run_doc_source_retrieval_v1.py", "scripts/run_doc_daily_hydro_residual_v1.py",
                 "scripts/run_doc_tail_residual_v1.py", "scripts/run_unified_doc_spatial.py",
                 str(ROOT/"study_plan.md"), str(ROOT/"deployment_protocol.json")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("preserve the saved deployment execution after code changes")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for seed in args.seeds:
        predictor = PortableDOCReconstructor.fit(dataset_path=DATASET,
            mask_path=mask_path, daily_path=daily_path, workdir=args.root/"runs"/f"seed{seed}",
            seed=seed, runtime_snapshot_hash=digest(snapshot), progress_path=args.root/"progress.json")
        export = args.root/"exports"/f"seed{seed}"
        if not export.exists():
            predictor.save(export)
        else:
            loaded = PortableDOCReconstructor.load(export)
            if loaded.metadata["source_completion_sha256"] != predictor.metadata["source_completion_sha256"]:
                raise ValueError("cached deployment export has a different fitted source")
        del predictor
        gc.collect()


if __name__ == "__main__":
    main()
