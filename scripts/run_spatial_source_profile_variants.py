"""Compare source-station descriptors for spatial regionalization."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import load_split, predict_query_models
from sklearn.preprocessing import StandardScaler

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
KS = (20, 40, 80, 160)


def monthly_descriptors(data: dict) -> np.ndarray:
    x = data["x"].numpy().astype(float)
    xm = data["x_mask"].numpy().astype(float)
    n, t, c = x.shape
    month = np.arange(t) % 12
    blocks: list[np.ndarray] = []
    for j in range(c):
        for m in range(12):
            take = month == m
            den = np.maximum(xm[:, take, j].sum(1), 1.0)
            blocks.extend([(x[:, take, j] * xm[:, take, j]).sum(1) / den,
                           xm[:, take, j].mean(1)])
    blocks.extend([data["static"].numpy()[:, j] for j in range(data["static"].shape[1])])
    blocks.extend([data["regime"].numpy()[:, j] for j in range(data["regime"].shape[1])])
    edge = data["edge_index"].numpy()
    attr = data["edge_attr"].numpy().astype(float)
    blocks.extend([np.bincount(edge[0], minlength=n), np.bincount(edge[1], minlength=n)])
    for j in range(attr.shape[1]):
        blocks.append(np.bincount(edge[1], weights=attr[:, j], minlength=n))
    out = np.stack(blocks, axis=1)
    return StandardScaler().fit_transform(np.nan_to_num(out)).astype(np.float32)


def run(args: argparse.Namespace) -> None:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    desc = monthly_descriptors(data)
    internal, _rows, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    y = data["y"].numpy().reshape(-1)
    rows: list[dict] = []
    for seed in args.seeds:
        for split, role in ((internal, "val"), (outer, "test")):
            cells = np.asarray(split[role], dtype=np.int64)
            fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p",
                                      include_network=True)
            eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p",
                                       include_network=True)
            for k in KS:
                pred = predict_query_models(
                    data, split, np.asarray(split["train"], dtype=np.int64), cells,
                    fit_x, eval_x, desc, k, seed, args.n_estimators, 4, 4,
                )
                rows.append({"seed": seed, "role": role, "k": k,
                             **metrics(y[cells], pred)})
    out = pd.DataFrame(rows)
    val = out[out.role == "val"]
    selected = int(val.groupby("k").mae.mean().sort_values().index[0])
    test = out[(out.role == "test") & (out.k == selected)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "summary.txt").write_text(
        f"selected_k={selected}\n" + val.groupby("k").mae.mean().to_string()
        + "\n" + test.to_string(index=False) + "\n"
    )
    print(val.groupby("k").mae.mean())
    print("selected", selected)
    print(test.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
