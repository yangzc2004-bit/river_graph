#!/usr/bin/env python3
"""Run K-shot pseudo-new-watershed experiments (Task 1).

One training per (region, seed) with the region's DOC labels hidden; each K
only changes inference-time support visibility. Query cells are fixed per
task month across K.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import load_dataset, metrics
from river_graph.experiments.kshot import (
    DEFAULT_REGIONS,
    K_LIST,
    MIN_QUERY,
    label_tasks,
    make_region_split,
    make_support_query_tasks,
    region_station_index,
    tasks_to_frame,
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


def river_distances(dataset: dict, src: int, dst: int) -> int | None:
    """Undirected hop distance between two node indices."""
    g = nx.Graph()
    n = dataset["y"].shape[0]
    g.add_nodes_from(range(n))
    g.add_edges_from(dataset["edge_index"].T.tolist())
    try:
        return int(nx.shortest_path_length(g, src, dst))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def evaluate_k(
    model: GCNDocModel,
    dataset: dict,
    tasks,
    k_list: tuple[int, ...],
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every (task, K). Returns (metrics_long, cells_long)."""
    metric_rows = []
    cell_rows = []
    # Group support cells by K across tasks so each K is one forward pass.
    for k in k_list:
        support_all = []
        query_cells = []
        for task in tasks:
            if k not in task.support_by_k:
                continue
            support_all.extend(task.support_by_k[k])
            for flat in task.query:
                query_cells.append((task, flat))
        if not query_cells:
            continue
        pred = model.predict(extra_visible=np.asarray(support_all, dtype=np.int64))
        pred_flat = pred.ravel()
        y_flat = y.ravel()
        for task, flat in query_cells:
            row, col = flat // y.shape[1], flat % y.shape[1]
            cell_rows.append(
                {
                    "region": task.region,
                    "month": task.month,
                    "month_index": task.month_index,
                    "k": k,
                    "flat": flat,
                    "row": row,
                    "col": col,
                    "y_true": float(y_flat[flat]),
                    "y_pred": float(pred_flat[flat]),
                }
            )
        yt = np.array([y_flat[f] for _, f in query_cells], dtype=float)
        yp = np.array([pred_flat[f] for _, f in query_cells], dtype=float)
        m = metrics(yt, yp)
        m["k"] = k
        m["n_tasks"] = len({id(task) for task, _ in query_cells})
        metric_rows.append(m)
    return pd.DataFrame(metric_rows), pd.DataFrame(cell_rows)


def attach_support_query_distances(
    cells: pd.DataFrame,
    dataset: dict,
    tasks,
    sites: list[str],
) -> pd.DataFrame:
    """Add min river hop from each query cell to that task's support at same K."""
    hop_cache: dict[tuple[int, int], int | None] = {}

    def hop(a: int, b: int) -> int | None:
        key = (min(a, b), max(a, b))
        if key not in hop_cache:
            hop_cache[key] = river_distances(dataset, a, b)
        return hop_cache[key]

    by_task = {(t.region, t.month, t.month_index): t for t in tasks}
    dists = []
    for row in cells.itertuples(index=False):
        task = by_task[(row.region, row.month, row.month_index)]
        support = task.support_by_k[int(row.k)]
        q_row = int(row.row)
        if not support:
            dists.append(np.nan)
            continue
        hops = []
        for flat in support:
            s_row = flat // dataset["y"].shape[1]
            h = hop(s_row, q_row)
            if h is not None:
                hops.append(h)
        dists.append(min(hops) if hops else np.nan)
    out = cells.copy()
    out["min_support_hops"] = dists
    return out


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
    ap.add_argument("--out-dir", default="experiments/kshot_st357")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny epochs + one seed for protocol smoke test")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.max_epochs = 2
        args.patience = 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)

    dataset = load_dataset(args.dataset)
    y_mask = np.asarray(dataset["y_mask"])
    y = np.asarray(dataset["y"])
    sites = [str(s) for s in dataset["site_no"]]
    months = [str(m) for m in dataset["months"]]
    nodes = pd.read_csv(args.nodes, dtype={"site_no": str})
    k_list = tuple(args.k_list)

    protocol = {
        "dataset": args.dataset,
        "regions": args.regions,
        "k_list": list(k_list),
        "min_query": args.min_query,
        "seeds": args.seeds,
        "architecture": "transport",
        "edge_direction": "both",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "smoke": bool(args.smoke),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_metrics = []
    for region in args.regions:
        rows = region_station_index(sites, nodes, region)
        if len(rows) == 0:
            raise SystemExit(f"region {region} has no stations in this dataset")
        tasks = label_tasks(
            make_support_query_tasks(
                y_mask, rows, months, k_list=k_list, min_query=args.min_query, seed=42
            ),
            region_name=region,
        )
        task_df = tasks_to_frame(tasks)
        task_df.to_csv(out / "tasks" / f"{region}.csv", index=False)

        split = make_region_split(y_mask, rows, seed=42)
        problems = validate_split(split, y_mask) + validate_tasks(
            tasks, y_mask, k_list=k_list, min_query=args.min_query
        )
        if problems:
            raise SystemExit(f"protocol validation failed for {region}: {problems}")

        print(
            f"[{region}] stations={len(rows)} held_out_cells={len(split['test'])} "
            f"train={len(split['train'])} val={len(split['val'])} tasks={len(tasks)}"
        )
        if not tasks:
            print(f"[{region}] no feasible same-month K tasks, skip")
            continue

        for seed in args.seeds:
            tag = f"{region}_s{seed}"
            t0 = time.perf_counter()
            model = build_model(seed, args.max_epochs, args.patience)
            model.fit(dataset, split)
            train_s = time.perf_counter() - t0
            m_df, c_df = evaluate_k(model, dataset, tasks, k_list, y)
            c_df = attach_support_query_distances(c_df, dataset, tasks, sites)
            c_df.to_parquet(out / "predictions" / f"{tag}.parquet", index=False)
            m_df.insert(0, "region", region)
            m_df.insert(1, "seed", seed)
            m_df.insert(2, "train_seconds", round(train_s, 2))
            m_df.to_csv(out / "results" / f"{tag}.csv", index=False)
            all_metrics.append(m_df)
            print(f"[{tag}] trained {train_s:.1f}s")
            for row in m_df.itertuples(index=False):
                print(
                    f"  K={int(row.k):2d}  n={int(row.n):4d}  "
                    f"MAE={row.mae:.3f}  R2={row.r2:.3f}  logR2={row.log_r2:.3f}"
                )

    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        summary.to_csv(out / "kshot_metrics.csv", index=False)
        print(f"wrote {out}/kshot_metrics.csv")


if __name__ == "__main__":
    main()
