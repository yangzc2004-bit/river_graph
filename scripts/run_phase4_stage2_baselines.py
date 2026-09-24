"""Run the frozen ST357 Stage-2 support baselines without model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from river_graph.experiments.transfer import (
    ANALYTES,
    DATASETS,
    K_VALUES,
    array,
    availability_tasks,
    file_hash,
    load_bundle,
)
from river_graph.models.support_encoder import analytic_blend

HUC6 = ("101302", "101900", "102701", "103001", "051002")
TASK_SEEDS = (42, 43, 44)


def _climatology(y, mask, source_rows, months):
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n, t = y.shape
    mo = np.array([int(str(m)[5:7]) for m in months])
    source_rows = np.asarray(source_rows, dtype=int)
    values = y[source_rows][mask[source_rows]]
    fallback = float(values.mean()) if len(values) else 0.0
    pred = np.full((n, t), fallback, dtype=np.float64)
    for month in range(1, 13):
        cols = np.flatnonzero(mo == month)
        if not len(cols):
            continue
        chunk = y[np.ix_(source_rows, cols)]
        chunk_mask = mask[np.ix_(source_rows, cols)]
        pred[:, cols] = float(chunk[chunk_mask].mean()) if chunk_mask.any() else fallback
    return pred


def _hops(ds):
    graph = nx.Graph()
    n = len(ds["site_no"])
    graph.add_nodes_from(range(n))
    graph.add_edges_from(array(ds["edge_index"]).T.tolist())
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    return lambda a, b: lengths.get(int(a), {}).get(int(b))


def _hash_spec(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/stage2_baselines_v1")
    ap.add_argument("--spec", default="experiments/phase4_transfer/stage2_baseline_spec_v1.md")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    datasets, summaries, nodes = load_bundle(
        DATASETS,
        "data/processed/graph_nodes_graphfix_st357.csv",
        "data/processed/graph_edges_graphfix_st357.csv",
    )
    nodes = nodes.reset_index()
    huc6 = nodes["huc6"].astype(str).to_numpy()
    all_cells, summary = [], []
    for analyte in ANALYTES:
        ds = datasets[analyte]
        y, mask = array(ds["y"]).astype(float), array(ds["y_mask"]).astype(bool)
        months = [str(m)[:7] for m in ds["months"]]
        hops = _hops(ds)
        for basin in HUC6:
            target_rows = np.flatnonzero(huc6 == basin)
            source_rows = np.flatnonzero(huc6 != basin)
            if len(target_rows) == 0:
                continue
            base = _climatology(y, mask, source_rows, months)
            for task_seed in TASK_SEEDS:
                tasks = availability_tasks(mask, target_rows, months, task_seed)
                if not tasks:
                    continue
                for task_index, task in enumerate(tasks):
                    query = list(task["query_cells"])
                    for k in K_VALUES:
                        support = list(task["support_cells_by_k"][str(k)])
                        if support:
                            support_y = y.ravel()[support]
                            bias = float(np.mean(support_y - base.ravel()[support]))
                            local = float(np.mean(support_y))
                            bias_pred = base + bias
                            local_pred = np.full_like(base, local)
                            blend = analytic_blend(base, y, support, query, hops, y.shape[1])
                        else:
                            bias_pred, local_pred = base, base
                            blend = {q: float(base.ravel()[q]) for q in query}
                        for q in query:
                            pred = {
                                "climatology": float(base.ravel()[q]),
                                "local_mean": float(local_pred.ravel()[q]),
                                "mean_bias": float(bias_pred.ravel()[q]),
                                "analytic_blend": float(blend[q]),
                            }
                            row = {
                                "analyte": analyte, "basin": basin, "task_seed": task_seed,
                                "task_index": task_index, "month": task["month"],
                                "month_index": task["month_index"], "k": k,
                                "flat": q, "station": str(ds["site_no"][q // y.shape[1]]),
                                "y_true": float(y.ravel()[q]), **{f"pred_{m}": v for m, v in pred.items()},
                            }
                            all_cells.append(row)
                cells = pd.DataFrame([r for r in all_cells if r["analyte"] == analyte and r["basin"] == basin and r["task_seed"] == task_seed])
                cells.to_parquet(out / "cells" / f"{analyte}_{basin}_seed{task_seed}.parquet", index=False)
                for k, sub in cells.groupby("k"):
                    for method in ("climatology", "local_mean", "mean_bias", "analytic_blend"):
                        err = sub[f"pred_{method}"] - sub.y_true
                        log_err = np.log1p(np.maximum(sub[f"pred_{method}"], 0)) - np.log1p(np.maximum(sub.y_true, 0))
                        q90 = sub.y_true >= sub.y_true.quantile(0.9)
                        summary.append({
                            "analyte": analyte, "basin": basin, "task_seed": task_seed,
                            "k": int(k), "method": method, "n": len(sub),
                            "mae": float(np.mean(np.abs(err))),
                            "log1p_mae": float(np.mean(np.abs(log_err))),
                            "q90_mae": float(np.mean(np.abs(err[q90]))) if q90.any() else None,
                        })
    pd.DataFrame(all_cells).to_parquet(out / "all_cells.parquet", index=False)
    pd.DataFrame(summary).to_csv(out / "metrics.csv", index=False)
    manifest = {
        "version": "phase4_stage2_baselines_v1",
        "spec": str(args.spec), "spec_sha256": _hash_spec(Path(args.spec)),
        "datasets": {a: {"path": DATASETS[a], "sha256": summaries[a]["sha256"]} for a in ANALYTES},
        "nodes_sha256": file_hash("data/processed/graph_nodes_graphfix_st357.csv"),
        "task_seeds": list(TASK_SEEDS), "k_values": list(K_VALUES), "huc6": list(HUC6),
        "script_sha256": file_hash(__file__), "training_started": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(all_cells), "summary_rows": len(summary), "out": str(out)}))


if __name__ == "__main__":
    main()
