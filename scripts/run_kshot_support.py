#!/usr/bin/env python3
"""K-shot with a lightweight support encoder (Task 3 case B).

Pipeline per (region, seed):
1. Fit H2 with the region's DOC labels hidden.
2. pred0 = H2 predictions with no local support.
3. Fit SupportEncoder on episodic support/query tasks drawn from non-region rows.
4. Evaluate on the region's nested same-month tasks against local_mean / mean_bias / GNN.
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
from river_graph.models.support_encoder import (
    analytic_blend,
    encode_predict,
    fit_support_encoder,
    residual_idw,
    sample_episodes,
)


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
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out-dir", default="experiments/kshot_support_st357")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)

    dataset = load_dataset(args.dataset)
    y_mask = np.asarray(dataset["y_mask"])
    y = np.asarray(dataset["y"])
    y_flat = y.ravel()
    sites = [str(s) for s in dataset["site_no"]]
    months = [str(m) for m in dataset["months"]]
    nodes = pd.read_csv(args.nodes, dtype={"site_no": str})
    k_list = tuple(args.k_list)
    hops = hop_lookup(dataset)
    t = y.shape[1]

    (out / "protocol.json").write_text(
        json.dumps(
            {
                "methods": [
                    "gnn_true",
                    "mean_bias",
                    "local_mean",
                    "residual_idw",
                    "analytic_blend",
                    "support_encoder",
                ],
                "regions": args.regions,
                "seeds": args.seeds,
                "k_list": list(k_list),
                "episodes": args.episodes,
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
        region_set = set(int(i) for i in rows)
        train_rows = sorted({int(f) // t for f in split["train"]})
        print(f"[{region}] tasks={len(tasks)} train_rows={len(train_rows)}", flush=True)

        for seed in args.seeds:
            tag = f"{region}_s{seed}"
            t0 = time.perf_counter()
            model = build_model(seed, args.max_epochs, args.patience)
            model.fit(dataset, split)
            pred0 = model.predict()
            print(f"[{tag}] H2 fit {time.perf_counter() - t0:.1f}s", flush=True)

            batch = sample_episodes(
                pred0, y, y_mask, train_rows, hops, t,
                k_list=(1, 3, 5), n_episodes=args.episodes, seed=seed,
            )
            enc = fit_support_encoder(batch, seed=seed)
            print(
                f"[{tag}] support encoder n={batch.feats.shape[0]} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

            rows_out = []
            # Batch GNN true-support predictions: one forward per K.
            gnn_grids: dict[int, np.ndarray] = {0: pred0}
            by_k: dict[int, list] = {}
            for task in tasks:
                for k, support in sorted(task.support_by_k.items()):
                    by_k.setdefault(int(k), []).append(list(support))
            for k, supports in by_k.items():
                if k == 0:
                    continue
                support_all = [c for s in supports for c in s]
                gnn_grids[k] = model.predict(
                    extra_visible=np.asarray(support_all, dtype=np.int64)
                )

            for task in tasks:
                for k, support in sorted(task.support_by_k.items()):
                    k = int(k)
                    support = list(support)
                    query = list(task.query)
                    if k == 0:
                        for name in (
                            "gnn_true",
                            "mean_bias",
                            "local_mean",
                            "residual_idw",
                            "analytic_blend",
                            "support_encoder",
                        ):
                            for q in query:
                                qrow, qcol = q // t, q % t
                                rows_out.append(
                                    {
                                        "region": region,
                                        "seed": seed,
                                        "month": task.month,
                                        "month_index": task.month_index,
                                        "k": k,
                                        "method": name,
                                        "flat": q,
                                        "y_true": float(y_flat[q]),
                                        "y_pred": float(pred0[qrow, qcol]),
                                    }
                                )
                        continue
                    bias = float(np.mean(y_flat[support] - pred0.ravel()[support]))
                    mean_bias = pred0 + bias
                    lm = float(np.mean(y_flat[support]))
                    local_mean = np.full_like(pred0, lm)
                    gnn_grid = gnn_grids[k]
                    rid = residual_idw(pred0, y, support, query, hops, t)
                    ab = analytic_blend(pred0, y, support, query, hops, t)
                    se = encode_predict(enc, pred0, y, support, query, hops, t)
                    for q in query:
                        qrow, qcol = q // t, q % t
                        yt = float(y_flat[q])
                        base = {
                            "region": region,
                            "seed": seed,
                            "month": task.month,
                            "month_index": task.month_index,
                            "k": k,
                            "flat": q,
                            "y_true": yt,
                        }
                        for name, val in (
                            ("gnn_true", float(gnn_grid[qrow, qcol])),
                            ("mean_bias", float(mean_bias[qrow, qcol])),
                            ("local_mean", float(local_mean[qrow, qcol])),
                            ("residual_idw", float(rid[q])),
                            ("analytic_blend", float(ab[q])),
                            ("support_encoder", float(se[q])),
                        ):
                            rows_out.append({**base, "method": name, "y_pred": val})

            cells = pd.DataFrame(rows_out)
            cells.to_parquet(out / "cells" / f"{tag}.parquet", index=False)
            summ = []
            for (method, k), sub in cells.groupby(["method", "k"]):
                m = metrics(sub.y_true.to_numpy(), sub.y_pred.to_numpy())
                summ.append({"region": region, "seed": seed, "method": method, "k": k, **m})
            pd.DataFrame(summ).to_csv(out / f"{tag}.csv", index=False)
            print(f"[{tag}] done {time.perf_counter() - t0:.1f}s", flush=True)
            for row in pd.DataFrame(summ).sort_values(["k", "method"]).itertuples(index=False):
                if int(row.k) in (0, 3, 5):
                    print(
                        f"  K={int(row.k):2d} {row.method:16s} MAE={row.mae:.3f} R2={row.r2:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
