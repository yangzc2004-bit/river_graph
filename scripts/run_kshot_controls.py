#!/usr/bin/env python3
"""Task 2 controls: is K-shot gain just local calibration?

Same nested support/query tasks as run_kshot.py. Methods:
- gnn_true          GNN with true support values/positions
- gnn_shuf_values   same support sites, values permuted within the set
- gnn_shuf_sites    same values, moved to other non-query sites
- mean_bias         K=0 GNN + local mean residual on support
- nearest           nearest support by river hops
- idw               inverse-hop-distance weighted support mean
- local_mean        mean of support values (no GNN)

GNN methods batch one forward pass per (method, K) over all tasks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, metrics
from river_graph.experiments.kshot import (
    DEFAULT_REGIONS,
    K_LIST,
    MIN_QUERY,
    label_tasks,
    make_region_split,
    make_support_query_tasks,
    region_station_index,
    validate_split,
    validate_tasks,
)
from river_graph.models.gcn import GCNDocModel


def build_model(seed: int, max_epochs: int, patience: int) -> GCNDocModel:
    return GCNDocModel(
        architecture="transport",
        variant="river",
        edge_set="river",
        edge_direction="both",
        seed=seed,
        lr=1e-3,
        max_epochs=max_epochs,
        patience=patience,
    )


def hop_lookup(dataset: dict):
    g = nx.Graph()
    n = dataset["y"].shape[0]
    g.add_nodes_from(range(n))
    g.add_edges_from(dataset["edge_index"].T.tolist())
    length = dict(nx.all_pairs_shortest_path_length(g))

    def hops(a: int, b: int) -> int | None:
        return length.get(a, {}).get(b)

    return hops


def cells_frame(tasks, y, preds_by_key, method, seed) -> pd.DataFrame:
    """preds_by_key: {(k, task_key): y_pred full grid or dict flat->pred}."""
    _n, t = y.shape
    rows = []
    for task in tasks:
        task_key = (task.month_index, task.query)
        for k, support in sorted(task.support_by_k.items()):
            pred = preds_by_key.get((k, task_key))
            if pred is None:
                continue
            for flat in task.query:
                row, col = flat // t, flat % t
                if isinstance(pred, dict):
                    yp = pred.get(flat)
                    if yp is None:
                        continue
                    yp = float(yp)
                else:
                    yp = float(pred[row, col])
                rows.append(
                    {
                        "region": task.region,
                        "seed": seed,
                        "month": task.month,
                        "month_index": task.month_index,
                        "k": k,
                        "method": method,
                        "flat": flat,
                        "row": row,
                        "col": col,
                        "y_true": float(y[row, col]),
                        "y_pred": yp,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    ap.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--k-list", nargs="+", type=int, default=list(K_LIST))
    ap.add_argument("--min-query", type=int, default=MIN_QUERY)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out-dir", default="experiments/kshot_controls_st357")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)

    dataset = load_dataset(args.dataset)
    y_mask = np.asarray(dataset["y_mask"])
    y = np.asarray(dataset["y"])
    sites = [str(s) for s in dataset["site_no"]]
    months = [str(m) for m in dataset["months"]]
    nodes = pd.read_csv(args.nodes, dtype={"site_no": str})
    k_list = tuple(args.k_list)
    hops = hop_lookup(dataset)
    t = y.shape[1]
    y_flat = y.ravel()

    (out / "protocol.json").write_text(
        json.dumps(
            {
                "parent": "experiments/kshot_st357",
                "controls": [
                    "gnn_true",
                    "gnn_shuf_values",
                    "gnn_shuf_sites",
                    "mean_bias",
                    "nearest",
                    "idw",
                    "local_mean",
                ],
                "regions": args.regions,
                "seeds": args.seeds,
                "k_list": list(k_list),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for region in args.regions:
        rows = region_station_index(sites, nodes, region)
        tasks = label_tasks(
            make_support_query_tasks(
                y_mask, rows, months, k_list=k_list, min_query=args.min_query, seed=42
            ),
            region,
        )
        split = make_region_split(y_mask, rows, seed=42)
        problems = validate_split(split, y_mask) + validate_tasks(
            tasks, y_mask, k_list=k_list, min_query=args.min_query
        )
        if problems:
            raise SystemExit(f"protocol validation failed for {region}: {problems}")
        print(f"[{region}] tasks={len(tasks)}", flush=True)

        for seed in args.seeds:
            tag = f"{region}_s{seed}"
            t0 = time.perf_counter()
            model = build_model(seed, args.max_epochs, args.patience)
            model.fit(dataset, split)
            pred0 = model.predict()
            fit_s = time.perf_counter() - t0
            print(f"[{tag}] fit {fit_s:.1f}s", flush=True)
            rng = np.random.default_rng(seed + 7)

            # ---- GNN methods: one forward per (method, K) ----
            for method in ("gnn_true", "gnn_shuf_values", "gnn_shuf_sites"):
                by_k: dict[int, list] = {}
                for task in tasks:
                    for k, support in sorted(task.support_by_k.items()):
                        by_k.setdefault(k, []).append((task, support))
                preds = {}
                for k, items in sorted(by_k.items()):
                    if k == 0:
                        grid = pred0
                        for task, support in items:
                            preds[(0, (task.month_index, task.query))] = grid
                        continue
                    if method == "gnn_true":
                        support_all = [c for _, s in items for c in s]
                        grid = model.predict(
                            extra_visible=np.asarray(support_all, dtype=np.int64)
                        )
                        for task, support in items:
                            preds[(k, (task.month_index, task.query))] = grid
                    elif method == "gnn_shuf_values":
                        # one forward per K: permute values across all support cells
                        support_all = [c for _, s in items for c in s]
                        vals = y_flat[support_all].copy()
                        rng.shuffle(vals)
                        grid = model.predict(
                            extra_visible=np.asarray(support_all, dtype=np.int64),
                            extra_values=vals,
                        )
                        for task, support in items:
                            preds[(k, (task.month_index, task.query))] = grid
                    else:  # gnn_shuf_sites
                        # one forward per K: each support value moves to a
                        # random non-query site in the same month
                        dest, vals = [], []
                        for task, support in items:
                            month = task.month_index
                            forb = set(support) | set(task.query)
                            pool = [
                                i * t + month
                                for i in range(y.shape[0])
                                if y_mask[i, month] and (i * t + month) not in forb
                            ]
                            if len(pool) < len(support):
                                for s in support:
                                    dest.append(s)
                                    vals.append(y_flat[s])
                                continue
                            pick = rng.choice(pool, size=len(support), replace=False)
                            dest.extend(int(p) for p in pick)
                            vals.extend(float(y_flat[s]) for s in support)
                        grid = model.predict(
                            extra_visible=np.asarray(dest, dtype=np.int64),
                            extra_values=np.asarray(vals, dtype=np.float32),
                        )
                        for task, support in items:
                            preds[(k, (task.month_index, task.query))] = grid
                frame = cells_frame(tasks, y, preds, method, seed)
                frame.to_parquet(out / "cells" / f"{tag}__{method}.parquet", index=False)
                print(f"[{tag}] {method} done", flush=True)

            # mean_bias: one K=0 grid + per-task offset
            mb_preds = {}
            for task in tasks:
                for k, support in sorted(task.support_by_k.items()):
                    key = (k, (task.month_index, task.query))
                    if k == 0:
                        mb_preds[key] = pred0
                    else:
                        s = list(support)
                        bias = float(np.mean(y_flat[s] - pred0.ravel()[s]))
                        mb_preds[key] = pred0 + bias
            cells_frame(tasks, y, mb_preds, "mean_bias", seed).to_parquet(
                out / "cells" / f"{tag}__mean_bias.parquet", index=False
            )

            # nearest / idw / local_mean: no GNN
            for method in ("nearest", "idw", "local_mean"):
                preds = {}
                for task in tasks:
                    for k, support in sorted(task.support_by_k.items()):
                        if k == 0:
                            preds[(k, (task.month_index, task.query))] = pred0
                            continue
                        local = {}
                        for q in task.query:
                            qrow, _qcol = q // t, q % t
                            if method == "local_mean":
                                local[q] = float(np.mean(y_flat[list(support)]))
                            elif method == "nearest":
                                best, best_h = None, 10**9
                                for s in support:
                                    h = hops(s // t, qrow)
                                    if h is not None and h < best_h:
                                        best, best_h = s, h
                                local[q] = float(y_flat[best]) if best is not None else np.nan
                            else:  # idw
                                wsum, ysum = 0.0, 0.0
                                for s in support:
                                    h = hops(s // t, qrow)
                                    if h is None:
                                        continue
                                    w = 1.0 / (h + 1.0)
                                    wsum += w
                                    ysum += w * y_flat[s]
                                local[q] = ysum / wsum if wsum > 0 else np.nan
                        preds[(k, (task.month_index, task.query))] = local
                cells_frame(tasks, y, preds, method, seed).to_parquet(
                    out / "cells" / f"{tag}__{method}.parquet", index=False
                )

            # summary metrics
            all_cells = pd.concat(
                [pd.read_parquet(p) for p in sorted((out / "cells").glob(f"{tag}__*.parquet"))],
                ignore_index=True,
            )
            summ = []
            for (method, k), sub in all_cells.groupby(["method", "k"]):
                ok = np.isfinite(sub.y_true) & np.isfinite(sub.y_pred)
                m = metrics(sub.y_true.to_numpy()[ok], sub.y_pred.to_numpy()[ok])
                summ.append(
                    {"region": region, "seed": seed, "method": method, "k": k, **m}
                )
            pd.DataFrame(summ).to_csv(out / f"{tag}.csv", index=False)
            print(f"[{tag}] controls written in {time.perf_counter() - t0:.1f}s", flush=True)
            for row in pd.DataFrame(summ).sort_values(["k", "method"]).itertuples(index=False):
                if int(row.k) in (0, 3, 5):
                    print(
                        f"  K={int(row.k):2d} {row.method:14s} MAE={row.mae:.3f} R2={row.r2:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
