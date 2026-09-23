"""Generate all benchmark masks for one dataset.

Usage: python scripts/generate_masks.py \
    --dataset data/processed/mississippi_graph_graphfix_st357.pt \
    --out-dir experiments/masks_stcore_v1

E3 connectivity is taken from the dataset's own ``edge_index``/``site_no`` so
the held-out split is defined on the evaluation graph, never on some other
cohort's edge list. A ``masks_provenance.json`` with input and output hashes is
written next to the masks.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.masks import (
    make_e1,
    make_e2_partial,
    make_e2_strict,
    make_e3,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_v02.pt")
    ap.add_argument("--out-dir", default="experiments/masks")
    args = ap.parse_args()

    d = torch.load(args.dataset, weights_only=False)
    y_mask = d["y_mask"].numpy()
    months = pd.DatetimeIndex(d["months"])
    sites = [str(s) for s in d["site_no"]]
    ei = d["edge_index"]
    edges = pd.DataFrame({
        "source": [sites[int(i)] for i in ei[0]],
        "target": [sites[int(i)] for i in ei[1]],
    })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks = {**make_e1(y_mask)}
    masks["e2a_strict"] = make_e2_strict(y_mask, months)
    masks["e2b_partial"] = make_e2_partial(y_mask, months)
    for s in (42, 43, 44):
        masks[f"e3_spatial_seed{s}"] = make_e3(y_mask, edges, sites, seed=s)

    written = {}
    for name, split in masks.items():
        np.savez(out / f"{name}.npz", **split)
        written[f"{name}.npz"] = sha256_file(out / f"{name}.npz")
        tot = len(split["train"]) + len(split["val"]) + len(split["test"])
        print(f"{name}: train={len(split['train'])} val={len(split['val'])} "
              f"test={len(split['test'])} (total {tot})")

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_masks.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "dataset": args.dataset,
        "dataset_sha256": sha256_file(Path(args.dataset)),
        "edges_source": "dataset.edge_index + dataset.site_no",
        "edge_count": int(ei.shape[1]),
        "mask_hashes": written,
    }
    (out / "masks_provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"wrote {out / 'masks_provenance.json'}")


if __name__ == "__main__":
    main()
