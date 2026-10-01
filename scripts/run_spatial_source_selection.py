"""Source-station similarity transfer for DOC spatial extrapolation.

This is a validation-first regionalization baseline inspired by learned
auxiliary-basin selection: each query station receives an ExtraTrees context
model fitted on the k source stations most similar in label-free hydro,
regime, static and graph descriptors.  k is chosen only on a nested
station-held-out split; the frozen E3 stations are scored once afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

T = 654
OUTER_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {k: np.asarray(saved[k], dtype=np.int64) for k in saved.files
                if k in {"train", "val", "test", "context"}}


def station_descriptors(data: dict) -> np.ndarray:
    """Label-free station descriptor used for source selection."""
    x = data["x"].numpy().astype(np.float64)
    xm = data["x_mask"].numpy().astype(np.float64)
    n, _t, c = x.shape
    blocks: list[np.ndarray] = []
    for j in range(c):
        a, v = x[:, :, j], xm[:, :, j]
        den = np.maximum(v.sum(1), 1.0)
        mean = (a * v).sum(1) / den
        sd = np.sqrt(((a - mean[:, None]) ** 2 * v).sum(1) / den)
        qs = []
        for q in (0.10, 0.50, 0.90):
            qs.append(np.asarray([
                np.quantile(a[i, v[i] > 0], q) if np.any(v[i] > 0) else 0.0
                for i in range(n)
            ]))
        blocks.extend([mean, sd, *qs, v.mean(1)])
    blocks.extend([data["static"].numpy()[:, j] for j in range(data["static"].shape[1])])
    blocks.extend([data["regime"].numpy()[:, j] for j in range(data["regime"].shape[1])])
    edge = data["edge_index"].numpy()
    attr = data["edge_attr"].numpy().astype(np.float64)
    blocks.extend([
        np.bincount(edge[0], minlength=n),
        np.bincount(edge[1], minlength=n),
    ])
    for j in range(attr.shape[1]):
        blocks.append(np.bincount(edge[1], weights=attr[:, j], minlength=n))
    out = np.stack(blocks, axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(out).astype(np.float32)


def nearest_sources(desc: np.ndarray, source: np.ndarray, query: int, k: int) -> np.ndarray:
    d = np.square(desc[source] - desc[query]).sum(axis=1)
    return source[np.argsort(d)[: min(int(k), len(source))]]


def predict_query_models(data: dict, split: dict, fit_cells: np.ndarray,
                         query_cells: np.ndarray, fit_x: np.ndarray,
                         query_x: np.ndarray, desc: np.ndarray,
                         k: int, seed: int, n_estimators: int,
                         leaf: int, jobs: int) -> np.ndarray:
    _n, t = data["y"].shape
    source = np.unique(fit_cells // t)
    target = np.unique(query_cells // t)
    by_station = {int(s): fit_cells[fit_cells // t == s] for s in source}
    yz = target_values(data, "log1p").reshape(-1)
    pred = np.empty(len(query_cells), dtype=np.float64)
    positions = {int(c): i for i, c in enumerate(query_cells)}
    for station in target:
        cells = np.flatnonzero(query_cells // t == station)
        if not len(cells):
            continue
        neighbors = nearest_sources(desc, source, int(station), k)
        train = np.concatenate([by_station[int(s)] for s in neighbors])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=seed,
            n_jobs=jobs,
        )
        model.fit(fit_x[train], yz[train])
        p = np.expm1(model.predict(query_x[query_cells[cells]]))
        for local, cell in zip(p, query_cells[cells], strict=True):
            pred[positions[int(cell)]] = local
    return pred


def run_nested(data: dict, outer: dict, desc: np.ndarray, seeds: list[int], ks: list[int],
               n_estimators: int, leaf: int, jobs: int) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    internal, val_rows, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    val_rows = np.asarray(val_rows, dtype=int)
    rows = []
    for seed in seeds:
        fit_x = build_rf_features(data, internal, FIT_ROLES, target_transform="log1p", include_network=True)
        val_x = fit_x
        y = data["y"].numpy().reshape(-1)
        for k in ks:
            pred = predict_query_models(data, internal, internal["train"], internal["val"], fit_x, val_x,
                                        desc, k, seed, n_estimators, leaf, jobs)
            rows.append({"seed": seed, "k": k, "role": "internal_val", **metrics(y[internal["val"]], pred)})
    val = pd.DataFrame(rows)
    selected_k = int(val.groupby("k").mae.mean().sort_values().index[0])
    out_rows = []
    for seed in seeds:
        fit_x = build_rf_features(data, outer, FIT_ROLES, target_transform="log1p", include_network=True)
        test_x = build_rf_features(data, outer, TEST_ROLES, target_transform="log1p", include_network=True)
        y = data["y"].numpy().reshape(-1)
        pred = predict_query_models(data, outer, outer["train"], outer["test"], fit_x, test_x,
                                    desc, selected_k, seed, n_estimators, leaf, jobs)
        out_rows.append({"seed": seed, "k": selected_k, "role": "outer_e3_test", **metrics(y[outer["test"]], pred)})
    return val, selected_k, pd.DataFrame(out_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--ks", nargs="+", type=int, default=[20, 40, 80, 160])
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--leaf", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(OUTER_MASK)
    desc = station_descriptors(data)
    val, selected, test = run_nested(data, outer, desc, args.seeds, args.ks,
                                      args.n_estimators, args.leaf, args.jobs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.out_dir / "internal_validation.csv", index=False)
    test.to_csv(args.out_dir / "outer_test.csv", index=False)
    spec = {"seeds": args.seeds, "ks": args.ks, "selected_k": selected,
            "n_estimators": args.n_estimators, "leaf": args.leaf,
            "descriptor": "hydro climatology + static + regime + graph degree/edge sums",
            "selection": "internal station holdout only; outer E3 scored once"}
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(val.groupby("k").mae.agg(["mean", "std"]).to_string())
    print("selected_k", selected)
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
