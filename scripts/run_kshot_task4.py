#!/usr/bin/env python3
"""Task 4: multi-watershed K-shot curves with adaptive support blending.

Default adapter is analytic_blend (closed-form gate). Also reports
gate_blend (learned g), local_mean, mean_bias, and gnn_true.
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
    sample_episodes,
)

DEFAULT_TASK4_REGIONS = (
    "10300101",
    "7050002",
    "5090101",
    "10180001",
    "10130201",
    "11010001",
)


def hop_lookup(dataset: dict):
    g = nx.Graph()
    n = dataset["y"].shape[0]
    g.add_nodes_from(range(n))
    g.add_edges_from(dataset["edge_index"].T.tolist())
    length = dict(nx.all_pairs_shortest_path_length(g))
    return lambda a, b: length.get(a, {}).get(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    ap.add_argument("--regions", nargs="+", default=list(DEFAULT_TASK4_REGIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--k-list", nargs="+", type=int, default=list(K_LIST))
    ap.add_argument("--min-query", type=int, default=MIN_QUERY)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out-dir", default="experiments/kshot_task4")
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
                "methods": ["analytic_blend", "gate_blend", "local_mean", "mean_bias", "gnn_true"],
                "default_adapter": "analytic_blend",
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

    all_summ = []
    for region in args.regions:
        rows = region_station_index(sites, nodes, region)
        if len(rows) == 0:
            print(f"[{region}] no stations, skip", flush=True)
            continue
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
        if not tasks:
            print(f"[{region}] no feasible tasks, skip", flush=True)
            continue
        train_rows = sorted({int(f) // t for f in split["train"]})
        print(f"[{region}] stations={len(rows)} tasks={len(tasks)}", flush=True)

        for seed in args.seeds:
            tag = f"{region}_s{seed}"
            t0 = time.perf_counter()
            model = GCNDocModel(
                architecture="transport",
                variant="river",
                edge_set="river",
                edge_direction="both",
                seed=seed,
                lr=1e-3,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            model.fit(dataset, split)
            pred0 = model.predict()
            batch = sample_episodes(
                pred0, y, y_mask, train_rows, hops, t,
                k_list=(1, 3, 5), n_episodes=args.episodes, seed=seed,
            )
            enc = fit_support_encoder(batch, seed=seed)
            print(f"[{tag}] fit+enc {time.perf_counter() - t0:.1f}s", flush=True)

            gnn_grids = {0: pred0}
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

            rows_out = []
            for task in tasks:
                for k, support in sorted(task.support_by_k.items()):
                    k = int(k)
                    support = list(support)
                    query = list(task.query)
                    if k == 0:
                        for name in (
                            "analytic_blend", "gate_blend", "local_mean", "mean_bias", "gnn_true"
                        ):
                            for q in query:
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
                                        "y_pred": float(pred0[q // t, q % t]),
                                    }
                                )
                        continue
                    bias = float(np.mean(y_flat[support] - pred0.ravel()[support]))
                    mean_bias = pred0 + bias
                    lm = float(np.mean(y_flat[support]))
                    local_mean = np.full_like(pred0, lm)
                    gnn_grid = gnn_grids[k]
                    ab = analytic_blend(pred0, y, support, query, hops, t)
                    gb = encode_predict(enc, pred0, y, support, query, hops, t)
                    for q in query:
                        qrow, qcol = q // t, q % t
                        base = {
                            "region": region,
                            "seed": seed,
                            "month": task.month,
                            "month_index": task.month_index,
                            "k": k,
                            "flat": q,
                            "y_true": float(y_flat[q]),
                        }
                        for name, val in (
                            ("analytic_blend", ab[q]),
                            ("gate_blend", gb[q]),
                            ("local_mean", float(local_mean[qrow, qcol])),
                            ("mean_bias", float(mean_bias[qrow, qcol])),
                            ("gnn_true", float(gnn_grid[qrow, qcol])),
                        ):
                            rows_out.append({**base, "method": name, "y_pred": float(val)})

            cells = pd.DataFrame(rows_out)
            cells.to_parquet(out / "cells" / f"{tag}.parquet", index=False)
            summ = []
            for (method, k), sub in cells.groupby(["method", "k"]):
                m = metrics(sub.y_true.to_numpy(), sub.y_pred.to_numpy())
                summ.append({"region": region, "seed": seed, "method": method, "k": k, **m})
            sdf = pd.DataFrame(summ)
            sdf.to_csv(out / f"{tag}.csv", index=False)
            all_summ.append(sdf)
            print(f"[{tag}] done {time.perf_counter() - t0:.1f}s", flush=True)
            for row in sdf.sort_values(["k", "method"]).itertuples(index=False):
                if int(row.k) in (0, 1, 3, 5) and row.method in (
                    "analytic_blend", "gate_blend", "local_mean"
                ):
                    print(
                        f"  K={int(row.k):2d} {row.method:15s} MAE={row.mae:.3f}",
                        flush=True,
                    )

    if all_summ:
        summary = pd.concat(all_summ, ignore_index=True)
        summary.to_csv(out / "task4_metrics.csv", index=False)
        print(f"wrote {out}/task4_metrics.csv")


if __name__ == "__main__":
    main()
