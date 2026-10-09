"""Validation-selected multi-hop context ExtraTrees for DOC E3 transfer."""

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
    role_visible,
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
HOPS = (1, 2, 3)
LEAVES = (2, 4, 8)


def load(mask_name: str) -> tuple[dict, dict, Path]:
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    path = MASKS[mask_name]
    with np.load(path, allow_pickle=False) as saved:
        split = {k: np.asarray(saved[k], dtype=np.int64)
                 for k in ("train", "val", "test", "context") if k in saved.files}
    return data, split, path


def multihop_features(data: dict, split: dict, roles: tuple[str, ...], max_hop: int) -> np.ndarray:
    n, t = data["y"].shape
    values = target_values(data, "log1p").reshape(n, t)
    visible = role_visible(split, roles, (n, t))
    edge = np.asarray(data["edge_index"], dtype=np.int64)
    adjacency = np.zeros((n, n), dtype=float)
    for source, target in edge.T:
        adjacency[target, source] = 1.0
    power = adjacency.copy()
    observed = np.where(visible, values, 0.0)
    blocks = [build_rf_features(data, split, roles, target_transform="log1p", include_network=True)]
    for _hop in range(1, max_hop + 1):
        mask = power > 0
        count_up = mask.astype(float) @ visible.astype(float)
        sum_up = mask.astype(float) @ observed
        count_down = mask.astype(float).T @ visible.astype(float)
        sum_down = mask.astype(float).T @ observed
        blocks.append(np.stack([
            sum_up / np.maximum(count_up, 1),
            count_up / np.maximum(mask.sum(1)[:, None], 1),
            sum_down / np.maximum(count_down, 1),
            count_down / np.maximum(mask.sum(0)[:, None], 1),
        ], axis=-1).reshape(n * t, 4).astype(np.float32))
        power = (power @ adjacency) > 0
    return np.concatenate(blocks, axis=-1)


def fit_predict(data: dict, split: dict, eval_role: str, *, seed: int,
                max_hop: int, leaf: int, n_estimators: int) -> dict:
    y_z = target_values(data, "log1p").reshape(-1)
    source = np.asarray(split["train"], dtype=np.int64)
    cells = np.asarray(split[eval_role], dtype=np.int64)
    x_fit = multihop_features(data, split, FIT_ROLES, max_hop)
    x_eval = multihop_features(data, split, TEST_ROLES, max_hop)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=leaf, max_features=1.0,
        n_jobs=4, random_state=seed,
    ).fit(x_fit[source], y_z[source])
    prediction = np.expm1(model.predict(x_eval[cells]))
    truth = _as_tensor(data["y"]).numpy().reshape(-1)[cells]
    return {"seed": seed, "max_hop": max_hop, "leaf": leaf,
            **metrics(truth, prediction), "n": len(cells)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data, split, _ = load("e3_supportmatched")
    validation_rows = []
    for seed in args.seeds:
        for max_hop in HOPS:
            for leaf in LEAVES:
                validation_rows.append({"mask": "e3_supportmatched", **fit_predict(
                    data, split, "val", seed=seed, max_hop=max_hop,
                    leaf=leaf, n_estimators=args.n_estimators)})
    validation = pd.DataFrame(validation_rows)
    selected = (validation.groupby(["max_hop", "leaf"], as_index=False)["mae"].mean()
                .sort_values("mae").iloc[0])
    data, split, _ = load("e3_test")
    test_rows = []
    for seed in args.seeds:
        test_rows.append({"mask": "e3_test", **fit_predict(
            data, split, "test", seed=seed, max_hop=int(selected["max_hop"]),
            leaf=int(selected["leaf"]), n_estimators=args.n_estimators)})
    test = pd.DataFrame(test_rows)
    validation.to_csv(args.out_dir / "validation_candidates.csv", index=False)
    test.to_csv(args.out_dir / "selected_test_metrics.csv", index=False)
    config = {"dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET),
              "selection": "minimum mean support-matched validation MAE",
              "hops": HOPS, "leaves": LEAVES, "seeds": args.seeds,
              "n_estimators": args.n_estimators,
              "selected_hop": int(selected["max_hop"]), "selected_leaf": int(selected["leaf"])}
    (args.out_dir / "spec.json").write_text(json.dumps(config, indent=2) + "\n")
    print("selected", selected.to_dict())
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
