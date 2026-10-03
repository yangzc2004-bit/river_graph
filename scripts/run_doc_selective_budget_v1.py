"""Complete matched DOC loss training with a 120-epoch maximum budget."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from run_doc_selective_residual_v1 import PRIOR, run_one
from run_unified_doc_spatial import digest, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file

ROOT = Path("experiments/phase4_transfer/doc_selective_residual_v2")


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_selective_residual_v1.py", "scripts/run_doc_selective_budget_v1.py",
                 "experiments/phase4_transfer/doc_selective_residual_v1/study_plan.md",
                 str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution changed; preserve this version and use another root")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime, epochs=120, patience=5)
            gc.collect()


if __name__ == "__main__":
    main()
