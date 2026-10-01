"""Probe exact-hop network context features for spatial DOC extrapolation.

This is a small, read-only model probe.  It extends the current RF-context
feature matrix with four label-free summaries at each directed hop:
upstream mean, upstream visible fraction, downstream mean, and downstream
visible fraction.  The matrix is built separately for the fit and test
visibility roles, so held-out station labels are never read.  It is intended
to compare the feature idea before promoting it into the production model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    role_visible,
    target_values,
)


def _load(mask_path: Path) -> tuple[dict, dict]:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    with np.load(mask_path, allow_pickle=False) as saved:
        split = {key: np.asarray(saved[key], dtype=np.int64)
                 for key in saved.files}
    return data, split


def _hop_matrices(data: dict, max_hop: int) -> list[np.ndarray]:
    n = int(data["y"].shape[0])
    edge_index = np.asarray(data["edge_index"], dtype=np.int64)
    adjacency = np.zeros((n, n), dtype=np.float32)
    # Dataset edges point source/upstream -> target/downstream.
    adjacency[edge_index[1], edge_index[0]] = 1.0
    power = adjacency.copy()
    result = []
    for _ in range(max_hop):
        result.append((power > 0).astype(np.float32))
        power = power @ adjacency
        power[power > 0] = 1.0
    return result


def _hop_features(data: dict, split: dict, roles: tuple[str, ...],
                  transformed_y: np.ndarray, hop_matrices: list[np.ndarray]) -> np.ndarray:
    n, t = data["y"].shape
    visible = role_visible(split, roles, (n, t))
    values = transformed_y.reshape(n, t)
    visible_values = np.where(visible, values, 0.0)
    visible_float = visible.astype(np.float32)
    blocks: list[np.ndarray] = []
    for matrix in hop_matrices:
        upstream_count = matrix @ visible_float
        upstream_sum = matrix @ visible_values
        downstream_count = matrix.T @ visible_float
        downstream_sum = matrix.T @ visible_values
        blocks.extend([
            upstream_sum / np.maximum(upstream_count, 1.0),
            upstream_count / max(n - 1, 1),
            downstream_sum / np.maximum(downstream_count, 1.0),
            downstream_count / max(n - 1, 1),
        ])
    features = np.concatenate(blocks, axis=-1).reshape(n * t, -1)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite multi-hop features")
    return features.astype(np.float32)


def run(mask_path: Path, seed: int, max_hop: int, leaves: tuple[int, ...],
        n_estimators: int, n_jobs: int) -> pd.DataFrame:
    data, split = _load(mask_path)
    transformed = target_values(data, "log1p")
    train = np.asarray(split["train"], dtype=np.int64)
    val = np.asarray(split.get("val", []), dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    base_fit = build_rf_features(data, split, FIT_ROLES,
                                 target_transform="log1p", include_network=True)
    base_test = build_rf_features(data, split, TEST_ROLES,
                                  target_transform="log1p", include_network=True)
    hops = _hop_matrices(data, max_hop)
    extra_fit = _hop_features(data, split, FIT_ROLES, transformed, hops)
    extra_test = _hop_features(data, split, TEST_ROLES, transformed, hops)
    fit_x = np.concatenate([base_fit, extra_fit], axis=1)
    test_x = np.concatenate([base_test, extra_test], axis=1)
    target = transformed.reshape(-1)
    y_raw = np.asarray(data["y"].numpy()).reshape(-1)
    rows: list[dict] = []
    for leaf in leaves:
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
            max_features=1.0, min_samples_leaf=leaf,
        )
        model.fit(fit_x[train], target[train])
        test_pred = np.expm1(model.predict(test_x[test]))
        row = {"mask": mask_path.stem, "seed": seed,
               "max_hop": max_hop, "min_samples_leaf": leaf,
               "n_features": fit_x.shape[1], **metrics(y_raw[test], test_pred)}
        if len(val):
            val_pred = np.expm1(model.predict(fit_x[val]))
            row["val_mae"] = metrics(y_raw[val], val_pred)["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path,
                        default=Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hop", type=int, default=3)
    parser.add_argument("--leaves", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.mask, args.seed, args.max_hop, tuple(args.leaves),
                 args.n_estimators, args.n_jobs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
