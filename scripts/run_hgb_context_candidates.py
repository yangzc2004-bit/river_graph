"""Validation-selected HistGradientBoosting DOC context baseline.

This is an analysis-only candidate search.  It keeps the existing RF-context
feature visibility protocol and evaluates test only for the validation winner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

MASKS = {
    "e1_r20_seed42": Path("experiments/masks_stcore_v1/e1_r20_seed42.npz"),
    "e2a_strict": Path("experiments/masks_stcore_v1/e2a_strict.npz"),
    "e2b_partial": Path("experiments/masks_stcore_v1/e2b_partial.npz"),
    "e3_spatial_seed42": Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz"),
}


def candidates(seed: int, max_iter: int) -> dict[str, HistGradientBoostingRegressor]:
    common = {
        "max_iter": max_iter,
        "early_stopping": False,
        "random_state": seed,
        "max_bins": 255,
    }
    return {
        "hgb_lr05_leaf31": HistGradientBoostingRegressor(
            **common, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.0
        ),
        "hgb_lr05_leaf63": HistGradientBoostingRegressor(
            **common, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.0
        ),
        "hgb_lr10_leaf31": HistGradientBoostingRegressor(
            **common, learning_rate=0.10, max_leaf_nodes=31, l2_regularization=0.0
        ),
        "hgb_lr05_leaf31_l2": HistGradientBoostingRegressor(
            **common, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0
        ),
    }


def run_one(mask_name: str, seed: int, max_iter: int) -> pd.DataFrame:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    with np.load(MASKS[mask_name], allow_pickle=False) as saved:
        split = {k: np.asarray(saved[k], dtype=np.int64) for k in saved.files if k in {"train", "val", "test", "context"}}
    target = target_values(data, "log1p").reshape(-1)
    y = data["y"].numpy().reshape(-1)
    train = split["train"]
    val = split.get("val", np.array([], dtype=np.int64))
    test = split["test"]
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    test_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
    rows: list[dict] = []
    for name, model in candidates(seed, max_iter).items():
        model.fit(fit_x[train], target[train])
        for role, cells, x in (("val", val, fit_x), ("test", test, test_x)):
            pred = np.expm1(model.predict(x[cells]))
            rows.append({"mask": mask_name, "seed": seed, "model": name, "role": role, **metrics(y[cells], pred)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--masks", nargs="+", choices=tuple(MASKS), default=list(MASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-iter", type=int, default=250)
    args = parser.parse_args()
    frames = [run_one(m, s, args.max_iter) for s in args.seeds for m in args.masks]
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    val = table[table.role.eq("val")].sort_values(["seed", "mask", "mae"])
    selected = val.groupby(["seed", "mask"], as_index=False).first()[["seed", "mask", "model", "mae"]]
    selected = selected.rename(columns={"model": "selected_model", "mae": "selected_val_mae"})
    test = table[table.role.eq("test")].merge(selected[["seed", "mask", "selected_model"]], left_on=["seed", "mask", "model"], right_on=["seed", "mask", "selected_model"])
    test = test.rename(columns={"mae": "selected_test_mae"})
    selected.to_csv(args.out.with_name("hgb_validation_selection.csv"), index=False)
    test.to_csv(args.out.with_name("hgb_selected_test_metrics.csv"), index=False)
    (args.out.with_name("hgb_spec.json")).write_text(json.dumps({"max_iter": args.max_iter, "seeds": args.seeds, "masks": args.masks}, indent=2) + "\n")
    print(selected.to_string(index=False))
    print(test[["seed", "mask", "selected_model", "selected_test_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
