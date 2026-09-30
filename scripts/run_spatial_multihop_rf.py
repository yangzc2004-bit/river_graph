"""Test explicit multi-hop upstream context for spatial extrapolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.models.kgml_local_transport import (
    build_rf_features,
    role_visible,
    target_values,
)

DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
MASKS = {
    "supportmatched": Path(
        "experiments/phase4_transfer/kgml_local_transport_v1/"
        "spatial_validation_e3_supportmatched/masks/"
        "e3_spatial_validation_supportmatched.npz"
    ),
}


def multihop_features(dataset, split, roles, transform: str, max_hop: int) -> np.ndarray:
    base = build_rf_features(dataset, split, roles, target_transform=transform, include_network=True)
    n, t = dataset["y"].shape
    values = target_values(dataset, transform)
    visible = role_visible(split, roles, (n, t))
    edge = dataset["edge_index"].numpy()
    adjacency = np.zeros((n, n), dtype=float)
    for source, target in edge.T:
        adjacency[target, source] = 1.0
    power = adjacency.copy()
    blocks = [base]
    observed = np.where(visible, values, 0.0)
    for hop in range(2, max_hop + 1):
        power = (power @ adjacency > 0).astype(float)
        count = power @ visible.astype(float)
        total = power @ observed
        blocks.append(np.stack([
            total / np.maximum(count, 1),
            count / np.maximum(power.sum(1)[:, None], 1),
        ], -1).reshape(n * t, 2).astype(np.float32))
    return np.concatenate(blocks, -1)


def load_task(mask_name: str):
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    with np.load(MASKS[mask_name], allow_pickle=False) as saved:
        split = {role: saved[role] for role in ("train", "val", "test", "context")}
    return dataset, split


def run_one(dataset, split, seed: int, max_hop: int, n_estimators: int):
    transform = "log1p"
    y = target_values(dataset, transform).ravel()
    train = split["train"]
    x_train = multihop_features(dataset, split, ("train",), transform, max_hop)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=4)
    model.fit(x_train[train], y[train])
    rows = []
    for role, roles in (("val", ("train",)), ("test", ("train", "val", "context"))):
        cells = split[role]
        x = multihop_features(dataset, split, roles, transform, max_hop)
        pred_z = model.predict(x[cells])
        pred = np.expm1(pred_z)
        truth = dataset["y"].numpy().ravel()[cells]
        row = {"seed": seed, "max_hop": max_hop, "role": role, "n": len(cells), **metrics(truth, pred)}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mask", choices=tuple(MASKS), default="supportmatched")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-hops", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset, split = load_task(args.mask)
    rows = []
    for hop in args.max_hops:
        if hop < 1 or hop > 3:
            raise ValueError("max hops must be 1, 2 or 3")
        for seed in args.seeds:
            rows.extend(run_one(dataset, split, seed, hop, 200))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "spec.json").write_text(json.dumps({
        "mask": args.mask, "max_hops": args.max_hops, "seeds": args.seeds,
        "features": "RF-context plus exact directed upstream distance-2/3 summaries",
        "target_transform": "log1p", "test_selection": "validation only",
    }, indent=2) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
