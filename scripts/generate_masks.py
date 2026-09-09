"""Generate all benchmark masks (experiments/masks/*.npz).

Usage: python scripts/generate_masks.py
"""

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

OUT = Path("experiments/masks")


def main() -> None:
    d = torch.load("data/processed/mississippi_graph_v02.pt", weights_only=False)
    y_mask = d["y_mask"].numpy()
    months = pd.DatetimeIndex(d["months"])
    sites = d["site_no"]
    edges = pd.read_csv("data/processed/graph_edges.csv", dtype=str)

    OUT.mkdir(parents=True, exist_ok=True)
    masks = {**make_e1(y_mask)}
    masks["e2a_strict"] = make_e2_strict(y_mask, months)
    masks["e2b_partial"] = make_e2_partial(y_mask, months)
    for s in (42, 43, 44):
        masks[f"e3_spatial_seed{s}"] = make_e3(y_mask, edges, sites, seed=s)

    for name, split in masks.items():
        np.savez(OUT / f"{name}.npz", **split)
        tot = len(split["train"]) + len(split["val"]) + len(split["test"])
        print(f"{name}: train={len(split['train'])} val={len(split['val'])} "
              f"test={len(split['test'])} (total {tot})")


if __name__ == "__main__":
    main()
