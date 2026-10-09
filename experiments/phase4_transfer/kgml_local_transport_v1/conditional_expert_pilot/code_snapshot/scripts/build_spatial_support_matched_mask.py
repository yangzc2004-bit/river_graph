"""Build a station holdout whose graph-support mix resembles frozen E3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
E3_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")


def support_stats(dataset, cells, source_stations, edges):
    y_mask = dataset["y_mask"].numpy()
    n, t = y_mask.shape
    visible = y_mask.copy()
    visible[list(set(range(n)) - set(source_stations)), :] = False
    upstream = [[] for _ in range(n)]
    for u, v in edges:
        upstream[v].append(u)
    support = []
    for cell in cells:
        station, month = divmod(int(cell), t)
        support.append(any(visible[u, month] for u in upstream[station]))
    support = np.asarray(support, dtype=bool)
    stations = np.unique(np.asarray(cells) // t)
    station_support = []
    for station in stations:
        station_cells = np.asarray(cells)[np.asarray(cells) // t == station]
        station_support.append(bool(support[np.isin(np.asarray(cells), station_cells)].any()))
    return {
        "cells": len(cells),
        "support_cells": int(support.sum()),
        "support_fraction": float(support.mean()) if len(support) else 0.0,
        "stations": len(stations),
        "support_stations": int(sum(station_support)),
        "support_station_fraction": float(np.mean(station_support)) if station_support else 0.0,
    }


def build(output: Path, *, seed: int = 20260930, validation_stations: int = 20) -> dict:
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    y_mask = dataset["y_mask"].numpy()
    n, t = y_mask.shape
    with np.load(E3_MASK, allow_pickle=False) as frozen:
        e3_test = np.asarray(frozen["test"], dtype=np.int64)
    test_stations = set(e3_test // t)
    source = np.asarray(sorted(set(range(n)) - test_stations), dtype=int)
    edges = dataset["edge_index"].numpy().T

    test_source = set(source)
    test_stats = support_stats(dataset, e3_test, test_source, edges)
    target_cell_fraction = test_stats["support_fraction"]
    target_station_fraction = test_stats["support_station_fraction"]
    target_supported_stations = round(validation_stations * target_station_fraction)
    rng = np.random.default_rng(seed)
    best = None
    observed_cells = np.arange(n * t)[y_mask.ravel()]
    for _ in range(50000):
        held = set(rng.choice(source, size=validation_stations, replace=False).tolist())
        train_stations = set(source) - held
        val_cells = observed_cells[np.isin(observed_cells // t, list(held))]
        stats = support_stats(dataset, val_cells, train_stations, edges)
        cross_edges = sum(u in train_stations and v in held for u, v in edges)
        if cross_edges < 3:
            continue
        score = (
            abs(stats["support_fraction"] - target_cell_fraction)
            + 0.6 * abs(stats["support_station_fraction"] - target_station_fraction)
            + 0.03 * abs(stats["support_stations"] - target_supported_stations)
        )
        if best is None or score < best[0]:
            best = (score, held, stats, cross_edges)
    if best is None:
        raise RuntimeError("could not find a support-matched validation block")
    _, held, val_stats, cross_edges = best
    train_stations = set(source) - held
    train = observed_cells[np.isin(observed_cells // t, list(train_stations))]
    val = observed_cells[np.isin(observed_cells // t, list(held))]
    test = e3_test[y_mask.ravel()[e3_test]]
    if np.intersect1d(train, val).size or np.intersect1d(train, test).size or np.intersect1d(val, test).size:
        raise ValueError("spatial roles overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, train=train, val=val, test=test, context=np.array([], dtype=np.int64))
    report = {
        "selection_seed": seed,
        "selection_rule": "random station holdout matched to frozen E3 topological upstream-support mix",
        "validation_stations": [int(x) for x in sorted(held)],
        "train_station_count": len(train_stations),
        "validation_station_count": len(held),
        "test_station_count": len(test_stations),
        "train_cells": len(train), "validation_cells": len(val), "test_cells": len(test),
        "train_to_validation_upstream_edges": int(cross_edges),
        "frozen_e3_support": test_stats,
        "validation_support": val_stats,
        "source_e3_mask": str(E3_MASK),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    print(json.dumps(build(args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
