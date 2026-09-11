"""Reproducibility check: same config, N repeats, measure spread.

Runs H2X on e2b_partial `repeats` times, twice: once with default threading,
once single-threaded (to isolate float-reduction nondeterminism).

Usage: python scripts/check_reproducibility.py
"""

import numpy as np
import torch

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics
from river_graph.models.gcn import GCNDocModel

REPEATS = 3


def run_once(ds, split):
    pred = GCNDocModel(variant="river", architecture="transport_enc",
                       env_encoder=True).fit_predict(ds, split)
    y = ds["y"].numpy()
    return metrics(y.ravel()[split["test"]], pred.ravel()[split["test"]])


def main() -> None:
    ds = load_dataset("data/processed/mississippi_graph_v04.pt")
    split = load_mask("e2b_partial")
    for threads in (None, 1):
        if threads:
            torch.set_num_threads(threads)
        res = []
        for i in range(REPEATS):
            m = run_once(ds, split)
            res.append(m["mae"])
            print(f"threads={threads or 'default'} run {i + 1}: MAE={m['mae']:.4f} "
                  f"R2={m['r2']:.4f}", flush=True)
        arr = np.array(res)
        print(f"threads={threads or 'default'}: MAE mean={arr.mean():.4f} "
              f"std={arr.std():.4f} spread={arr.max() - arr.min():.4f}", flush=True)


if __name__ == "__main__":
    main()
