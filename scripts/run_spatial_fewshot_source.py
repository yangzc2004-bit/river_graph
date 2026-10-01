"""Few-shot DOC adaptation for spatially held-out stations.

The strict E3 benchmark hides every target-station label.  This companion
experiment opens K deterministic target-month support labels and evaluates the
remaining months.  Support labels enter prediction context only; the forest is
still fitted on source stations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import (
    load_split,
    predict_query_models,
    station_descriptors,
)

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
KS = (0, 1, 3, 5)


def fewshot_split(split: dict, k: int) -> tuple[dict, np.ndarray]:
    out = {name: np.asarray(cells, dtype=np.int64).copy() for name, cells in split.items()}
    target = np.asarray(out["test"], dtype=np.int64)
    support: list[int] = []
    for station in np.unique(target // T):
        cells = np.sort(target[target // T == station])
        if k:
            positions = np.linspace(0, len(cells) - 1, min(k, len(cells)), dtype=int)
            support.extend(cells[np.unique(positions)].tolist())
    support_array = np.asarray(sorted(set(support)), dtype=np.int64)
    out["test"] = np.setdiff1d(target, support_array, assume_unique=False)
    out["context"] = np.unique(np.concatenate([out.get("context", np.array([], dtype=np.int64)), support_array]))
    return out, support_array


def run(args: argparse.Namespace) -> None:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    base = load_split(MASK)
    desc = station_descriptors(data)
    y = data["y"].numpy().reshape(-1)
    rows: list[dict] = []
    for seed in args.seeds:
        for k in KS:
            split, support = fewshot_split(base, k)
            query = np.asarray(split["test"], dtype=np.int64)
            fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
            eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
            pred = predict_query_models(
                data, split, np.asarray(split["train"], dtype=np.int64), query,
                fit_x, eval_x, desc, 40, seed, args.n_estimators, 4, 4,
            )
            rows.append({"seed": seed, "k": k, "support_cells": len(support),
                         "query_cells": len(query), **metrics(y[query], pred)})
    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "metrics.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
