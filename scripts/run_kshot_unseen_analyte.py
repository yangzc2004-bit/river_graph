#!/usr/bin/env python3
"""Task 6: K-shot on a held-out analyte (model never trained on this target).

Base predictor is month-of-year climatology from non-region stations (K=0).
Support methods then read a few local observations of the *unseen* analyte.

Also supports --mode heldin: train H2 on the analyte itself (region still
hidden) as a ceiling. Compare the two to see how much of the K-curve survives
without ever training on the target.
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
from river_graph.models.support_encoder import analytic_blend


def hop_lookup(dataset: dict):
    g = nx.Graph()
    n = dataset["y"].shape[0]
    g.add_nodes_from(range(n))
    g.add_edges_from(dataset["edge_index"].T.tolist())
    length = dict(nx.all_pairs_shortest_path_length(g))
    return lambda a, b: length.get(a, {}).get(b)


def climatology_base(y, y_mask, train_rows: list[int], months: list[str]) -> np.ndarray:
    """(N, T) month-of-year mean from train rows, fallback global train mean."""
    y = np.asarray(y, dtype=np.float64)
    y_mask = np.asarray(y_mask)
    n, t = y.shape
    mo = np.array([int(m[5:7]) for m in months])
    base = np.zeros((n, t), dtype=np.float64)
    gmean = float(y[y_mask].mean()) if y_mask.any() else 0.0
    train_mask = y_mask[train_rows]
    train_y = y[train_rows]
    for m in range(1, 13):
        cols = np.flatnonzero(mo == m)
        if cols.size == 0:
            continue
        cell_mask = train_mask[:, cols]
        mu = float(train_y[:, cols][cell_mask].mean()) if cell_mask.any() else gmean
        base[:, cols] = mu
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_ph_st357.pt")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    ap.add_argument("--regions", nargs="+", default=["11010001", "10180001", "10130201"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--k-list", nargs="+", type=int, default=list(K_LIST))
    ap.add_argument("--min-query", type=int, default=MIN_QUERY)
    ap.add_argument("--mode", choices=["heldout", "heldin"], default="heldout")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out-dir", default="experiments/kshot_unseen_ph")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)

    dataset = load_dataset(args.dataset)
    y_mask = np.asarray(dataset["y_mask"])
    y = np.asarray(dataset["y"], dtype=np.float64)
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
                "mode": args.mode,
                "dataset": args.dataset,
                "methods": ["climatology", "local_mean", "mean_bias", "analytic_blend"]
                + (["h2"] if args.mode == "heldin" else []),
                "regions": args.regions,
                "seeds": args.seeds,
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
            print(f"[{region}] no tasks, skip", flush=True)
            continue
        train_rows = sorted({int(f) // t for f in split["train"]})
        clim = climatology_base(y, y_mask, train_rows, months)
        print(f"[{region}] tasks={len(tasks)} mode={args.mode}", flush=True)

        for seed in args.seeds:
            tag = f"{region}_s{seed}"
            t0 = time.perf_counter()
            pred0 = clim
            model = None
            if args.mode == "heldin":
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
            print(f"[{tag}] base ready {time.perf_counter() - t0:.1f}s", flush=True)

            # batch H2 by K when heldin
            h2_by_k = {}
            if model is not None:
                by_k: dict[int, list[int]] = {}
                for task in tasks:
                    for k, support in task.support_by_k.items():
                        by_k.setdefault(int(k), []).extend(support)
                h2_by_k[0] = pred0
                for k, support_all in by_k.items():
                    if k == 0:
                        continue
                    h2_by_k[k] = model.predict(
                        extra_visible=np.asarray(support_all, dtype=np.int64)
                    )

            rows_out = []
            for task in tasks:
                for k, support in sorted(task.support_by_k.items()):
                    k = int(k)
                    support = list(support)
                    query = list(task.query)
                    if k == 0:
                        method_pred = {
                            "climatology": lambda q: clim,
                            "local_mean": lambda q: clim,
                            "mean_bias": lambda q: clim,
                            "analytic_blend": lambda q: clim,
                        }
                    else:
                        bias = float(np.mean(y_flat[support] - pred0.ravel()[support]))
                        mean_bias = pred0 + bias
                        lm = float(np.mean(y_flat[support]))
                        local_mean = np.full_like(pred0, lm)
                        ab = analytic_blend(pred0, y, support, query, hops, t)

                        def pick(name, q):
                            if name == "climatology":
                                return clim
                            if name == "local_mean":
                                return local_mean
                            if name == "mean_bias":
                                return mean_bias
                            return None

                        method_pred = {
                            "climatology": pick,
                            "local_mean": pick,
                            "mean_bias": pick,
                            "analytic_blend": None,
                        }
                    for q in query:
                        qrow, qcol = q // t, q % t
                        base_row = {
                            "region": region,
                            "seed": seed,
                            "month": task.month,
                            "month_index": task.month_index,
                            "k": k,
                            "flat": q,
                            "y_true": float(y_flat[q]),
                        }
                        if k == 0:
                            for name in ("climatology", "local_mean", "mean_bias", "analytic_blend"):
                                rows_out.append(
                                    {
                                        **base_row,
                                        "method": name,
                                        "y_pred": float(clim[qrow, qcol]),
                                    }
                                )
                        else:
                            for name in ("climatology", "local_mean", "mean_bias"):
                                rows_out.append(
                                    {
                                        **base_row,
                                        "method": name,
                                        "y_pred": float(pick(name, q)[qrow, qcol]),
                                    }
                                )
                            rows_out.append(
                                {**base_row, "method": "analytic_blend", "y_pred": float(ab[q])}
                            )
                        if model is not None:
                            rows_out.append(
                                {
                                    **base_row,
                                    "method": "h2",
                                    "y_pred": float(h2_by_k[k][qrow, qcol]),
                                }
                            )

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
                if int(row.k) in (0, 3, 5):
                    print(f"  K={int(row.k):2d} {row.method:14s} MAE={row.mae:.3f}", flush=True)

    if all_summ:
        pd.concat(all_summ, ignore_index=True).to_csv(out / "unseen_metrics.csv", index=False)
        print(f"wrote {out}/unseen_metrics.csv")


if __name__ == "__main__":
    main()
