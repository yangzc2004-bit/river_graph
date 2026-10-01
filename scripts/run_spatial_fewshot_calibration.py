"""Nested target-station support calibration for source regional experts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_fewshot_source import fewshot_split
from run_spatial_source_selection import (
    load_split,
    predict_query_models,
    station_descriptors,
)

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
KS = (1, 3, 5)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def support_shift(data: dict, split: dict, support: np.ndarray, query: np.ndarray,
                  base_pred: np.ndarray, alpha: float) -> np.ndarray:
    z = np.log1p(data["y"].numpy().reshape(-1))
    out = np.log1p(np.maximum(base_pred, 0.0)).copy()
    support_stations = support // T
    query_stations = query // T
    for station in np.unique(query_stations):
        s = support[support_stations == station]
        q = query_stations == station
        if len(s) == 0:
            continue
        out[q] += alpha * (float(z[s].mean()) - float(out[q].mean()))
    return np.expm1(out)


def run(args: argparse.Namespace) -> None:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _rows, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    desc = station_descriptors(data)
    y = data["y"].numpy().reshape(-1)
    validation: list[dict] = []
    selected: dict[int, float] = {}
    internal_target = {
        "train": internal["train"],
        "val": np.array([], dtype=np.int64),
        "test": internal["val"],
        "context": np.array([], dtype=np.int64),
    }
    for seed in args.seeds:
        for k in KS:
            split, support = fewshot_split(internal_target, k)
            query = np.asarray(split["test"], dtype=np.int64)
            fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
            eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
            base = predict_query_models(data, split, np.asarray(split["train"], dtype=np.int64), query,
                                        fit_x, eval_x, desc, 40, seed, args.n_estimators, 4, 4)
            for alpha in ALPHAS:
                pred = support_shift(data, split, support, query, base, alpha)
                validation.append({"seed": seed, "k": k, "alpha": alpha,
                                   **metrics(y[query], pred)})
    val = pd.DataFrame(validation)
    for k in KS:
        selected[k] = float(val[val.k == k].groupby("alpha").mae.mean().sort_values().index[0])
    test_rows: list[dict] = []
    for seed in args.seeds:
        for k in KS:
            split, support = fewshot_split(outer, k)
            query = np.asarray(split["test"], dtype=np.int64)
            fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
            eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
            base = predict_query_models(data, split, np.asarray(split["train"], dtype=np.int64), query,
                                        fit_x, eval_x, desc, 40, seed, args.n_estimators, 4, 4)
            pred = support_shift(data, split, support, query, base, selected[k])
            test_rows.append({"seed": seed, "k": k, "alpha": selected[k],
                              **metrics(y[query], pred)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.out_dir / "validation.csv", index=False)
    pd.DataFrame(test_rows).to_csv(args.out_dir / "test.csv", index=False)
    print(val.groupby(["k", "alpha"]).mae.mean().unstack())
    print("selected", selected)
    print(pd.DataFrame(test_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
