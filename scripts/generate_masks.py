"""Generate all benchmark masks (experiments/masks/*.npz).

Usage: python scripts/generate_masks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.masks import make_e1, make_e2, make_e3

OUT = Path("experiments/masks")


def main() -> None:
    d = torch.load("data/processed/mississippi_graph_v02.pt", weights_only=False)
    y_mask = d["y_mask"].numpy()
    months = pd.DatetimeIndex(d["months"])
    sites = d["site_no"]
    edges = pd.read_csv("data/processed/graph_edges.csv", dtype=str)

    OUT.mkdir(parents=True, exist_ok=True)
    masks = {**make_e1(y_mask)}
    masks["e2_temporal"] = make_e2(y_mask, months)
    masks["e3_spatial"] = make_e3(y_mask, edges, sites)

    for name, split in masks.items():
        np.savez(OUT / f"{name}.npz", **split)
        tot = len(split["train"]) + len(split["val"]) + len(split["test"])
        print(f"{name}: train={len(split['train'])} val={len(split['val'])} "
              f"test={len(split['test'])} (total {tot})")


if __name__ == "__main__":
    main()
