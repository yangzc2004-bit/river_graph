"""Build a spatial-validation version of the frozen E3 station holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
E3_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")


def build(output: Path, validation_huc6: str = "101900") -> dict:
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    nodes = pd.read_csv(NODES, dtype=str)
    huc6 = nodes["huc_cd"].str.zfill(8).str[:6]
    if validation_huc6 not in set(huc6):
        raise ValueError(f"HUC6 not found: {validation_huc6}")
    observed = dataset["y_mask"].numpy().ravel()
    with np.load(E3_MASK, allow_pickle=False) as frozen:
        e3_test = np.asarray(frozen["test"], dtype=np.int64)
    t = dataset["y"].shape[1]
    e3_test_stations = set(e3_test // t)
    validation_stations = set(np.flatnonzero(huc6.eq(validation_huc6))) - e3_test_stations
    if len(validation_stations) < 5:
        raise ValueError("spatial validation block has too few source stations")
    all_stations = np.arange(dataset["y"].shape[0])
    train_stations = set(all_stations) - validation_stations - e3_test_stations
    train = np.flatnonzero(observed & np.isin(np.arange(len(observed)) // t, list(train_stations)))
    val = np.flatnonzero(observed & np.isin(np.arange(len(observed)) // t, list(validation_stations)))
    test = e3_test[observed[e3_test]]
    if np.intersect1d(train, val).size or np.intersect1d(train, test).size or np.intersect1d(val, test).size:
        raise ValueError("spatial validation roles overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, train=train, val=val, test=test, context=np.array([], dtype=np.int64))
    report = {
        "validation_huc6": validation_huc6,
        "validation_stations": [int(x) for x in sorted(validation_stations)],
        "train_stations": len(train_stations),
        "validation_station_count": len(validation_stations),
        "test_station_count": len(e3_test_stations),
        "train_cells": len(train),
        "validation_cells": len(val),
        "test_cells": len(test),
        "source_e3_mask": str(E3_MASK),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/"
                                     "spatial_validation_e3/masks/e3_spatial_validation.npz"))
    parser.add_argument("--validation-huc6", default="101900")
    args = parser.parse_args()
    print(build(args.output, args.validation_huc6))


if __name__ == "__main__":
    main()
