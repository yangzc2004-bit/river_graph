"""Headwater-conditioned ExtraTrees expert for DOC spatial transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.temporal_h2x import _as_tensor
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

DATASET = Path(DATASETS["doc"])
MASKS = {
    "e3_supportmatched": Path(
        "experiments/phase4_transfer/kgml_local_transport_v1/"
        "spatial_validation_e3_supportmatched/masks/"
        "e3_spatial_validation_supportmatched.npz"
    ),
    "e3_test": Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz"),
}


def load(mask_name: str) -> tuple[dict, dict, Path]:
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    path = MASKS[mask_name]
    with np.load(path, allow_pickle=False) as saved:
        split = {k: np.asarray(saved[k], dtype=np.int64)
                 for k in ("train", "val", "test", "context") if k in saved.files}
    return data, split, path


def headwater_stations(data: dict) -> np.ndarray:
    n = int(data["y"].shape[0])
    edge = np.asarray(data["edge_index"], dtype=np.int64)
    indegree = np.bincount(edge[1], minlength=n)
    return indegree == 0


def fit_route(data: dict, split: dict, eval_role: str, *, seed: int,
              leaf: int, n_estimators: int) -> dict:
    _n, t = data["y"].shape
    source = np.asarray(split["train"], dtype=np.int64)
    eval_cells = np.asarray(split[eval_role], dtype=np.int64)
    y = target_values(data, "log1p").reshape(-1)
    x_fit = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    x_eval = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
    is_head = headwater_stations(data)
    source_head = source[is_head[source // t]]
    eval_head = is_head[eval_cells // t]
    global_model = ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=leaf, max_features=1.0,
        n_jobs=4, random_state=seed,
    ).fit(x_fit[source], y[source])
    head_model = ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=leaf, max_features=1.0,
        n_jobs=4, random_state=seed + 101,
    ).fit(x_fit[source_head], y[source_head])
    global_pred = np.expm1(global_model.predict(x_eval[eval_cells]))
    head_pred = np.expm1(head_model.predict(x_eval[eval_cells]))
    final = np.where(eval_head, head_pred, global_pred)
    truth = _as_tensor(data["y"]).numpy().reshape(-1)[eval_cells]
    rows = {
        "seed": seed, "leaf": leaf, "global_mae": metrics(truth, global_pred)["mae"],
        "route_mae": metrics(truth, final)["mae"],
        "head_global_mae": metrics(truth[eval_head], global_pred[eval_head])["mae"] if eval_head.any() else np.nan,
        "head_route_mae": metrics(truth[eval_head], final[eval_head])["mae"] if eval_head.any() else np.nan,
        "nonhead_global_mae": metrics(truth[~eval_head], global_pred[~eval_head])["mae"] if (~eval_head).any() else np.nan,
        "nonhead_route_mae": metrics(truth[~eval_head], final[~eval_head])["mae"] if (~eval_head).any() else np.nan,
        "n_eval": len(eval_cells), "n_head_eval": int(eval_head.sum()),
        "n_source_head": len(source_head),
    }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--leaves", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    data, split, _ = load("e3_supportmatched")
    for seed in args.seeds:
        for leaf in args.leaves:
            rows.append({"mask": "e3_supportmatched", **fit_route(
                data, split, "val", seed=seed, leaf=leaf, n_estimators=args.n_estimators)})
    validation = pd.DataFrame(rows)
    selected = validation.groupby("leaf", as_index=False)["route_mae"].mean().sort_values("route_mae").iloc[0]
    data, split, _ = load("e3_test")
    test_rows = []
    for seed in args.seeds:
        test_rows.append({"mask": "e3_test", **fit_route(
            data, split, "test", seed=seed, leaf=int(selected["leaf"]), n_estimators=args.n_estimators)})
    test = pd.DataFrame(test_rows)
    validation.to_csv(args.out_dir / "validation_candidates.csv", index=False)
    test.to_csv(args.out_dir / "selected_test_metrics.csv", index=False)
    config = {"dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET),
              "selection": "minimum mean route MAE on support-matched validation",
              "seeds": args.seeds, "leaves": args.leaves, "n_estimators": args.n_estimators,
              "selected_leaf": int(selected["leaf"])}
    (args.out_dir / "spec.json").write_text(json.dumps(config, indent=2) + "\n")
    print("selected", selected.to_dict())
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
