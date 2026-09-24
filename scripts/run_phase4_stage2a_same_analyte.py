"""Stage-2A same-analyte support diagnostic on frozen HUC6 components.

This is intentionally a diagnostic arm. It measures whether the fixed
support/query construction contains local information before any cross-analyte
model is considered. It must not be reported as held-out-analyte transfer.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
    object_hash,
)
from river_graph.models.support_encoder import analytic_blend

REGIONS = "experiments/kshot_protocol_v2/regions.json"
SEEDS = (42, 43, 44)


def climatology(y, mask, source_rows, months):
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    rows = np.asarray(source_rows, dtype=int)
    month_of_year = np.array([int(str(m)[5:7]) for m in months])
    source_values = y[rows][mask[rows]]
    fallback = float(source_values.mean()) if len(source_values) else 0.0
    pred = np.full_like(y, fallback, dtype=float)
    for month in range(1, 13):
        cols = np.flatnonzero(month_of_year == month)
        chunk = y[np.ix_(rows, cols)]
        chunk_mask = mask[np.ix_(rows, cols)]
        pred[:, cols] = float(chunk[chunk_mask].mean()) if chunk_mask.any() else fallback
    return pred


def hops(ds):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(ds["site_no"])))
    graph.add_edges_from(array(ds["edge_index"]).T.tolist())
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    return lambda a, b: lengths.get(int(a), {}).get(int(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/stage2a_same_analyte_v2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = Path("experiments/phase4_transfer/stage2_baseline_spec_v1.md")
    region_path = Path(REGIONS)
    config = {
        "role": "same_analyte_basin_adaptation_diagnostic",
        "datasets": DATASETS, "regions": REGIONS, "task_seeds": list(SEEDS),
        "k_values": list(K_VALUES), "methods": ["climatology", "local_mean", "mean_bias", "analytic_blend"],
        "q90_threshold": "source_rows_only",
    }
    config_hash = object_hash(config)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force:
        old = json.loads(manifest_path.read_text())
        if old.get("config_hash") != config_hash:
            raise SystemExit("existing output has a different config; use --force in a new review")
    datasets, summaries, nodes = load_bundle(
        DATASETS, "data/processed/graph_nodes_graphfix_st357.csv",
        "data/processed/graph_edges_graphfix_st357.csv",
    )
    regions = json.loads(region_path.read_text())["primary"]
    site_count = len(datasets["doc"]["site_no"])
    rows_out, task_manifest, visibility_roles = [], [], []
    for analyte in ANALYTES:
        ds = datasets[analyte]
        y, mask = array(ds["y"]).astype(float), array(ds["y_mask"]).astype(bool)
        months = [str(m)[:7] for m in ds["months"]]
        hop = hops(ds)
        for region in regions:
            basin = "051002" if region["code"] == "510020" else region["code"]
            target_rows = np.asarray(region["task_component_rows"], dtype=int)
            if len(target_rows) == 0 or target_rows.max() >= site_count:
                raise ValueError(f"invalid component rows for {basin}")
            if not np.all(nodes.iloc[target_rows]["huc6"].to_numpy() == basin):
                raise ValueError(f"component rows do not match canonical HUC6 {basin}")
            hide_rows = np.asarray(region["hide_rows"], dtype=int)
            if not np.all(nodes.iloc[hide_rows]["huc6"].to_numpy() == basin):
                raise ValueError(f"hide rows do not match canonical HUC6 {basin}")
            source_rows = np.setdiff1d(np.arange(site_count), hide_rows)
            visibility_roles.append({
                "analyte": analyte, "basin": basin,
                "hide_rows_sha256": object_hash(hide_rows.tolist()),
                "target_component_rows_sha256": object_hash(target_rows.tolist()),
                "source_rows_sha256": object_hash(source_rows.tolist()),
                "n_hide_rows": len(hide_rows), "n_target_component_rows": len(target_rows),
                "n_source_rows": len(source_rows),
            })
            base = climatology(y, mask, source_rows, months)
            source_values = y[source_rows][mask[source_rows]]
            q90 = float(np.quantile(source_values, 0.9)) if len(source_values) else None
            for seed in SEEDS:
                tasks = availability_tasks(mask, target_rows, months, seed)
                for task_index, task in enumerate(tasks):
                    task_manifest.append({
                        "analyte": analyte, "basin": basin, "seed": seed,
                        "task_index": task_index, "month_index": task["month_index"],
                        "query_cells": task["query_cells"], "support_cells_by_k": task["support_cells_by_k"],
                    })
                    for k in K_VALUES:
                        support = list(task["support_cells_by_k"][str(k)])
                        query = list(task["query_cells"])
                        if support:
                            values = y.ravel()[support]
                            bias = float(np.mean(values - base.ravel()[support]))
                            local = float(np.mean(values))
                            bias_pred = base + bias
                            local_pred = np.full_like(base, local)
                            blend = analytic_blend(base, y, support, query, hop, y.shape[1])
                        else:
                            bias_pred, local_pred = base, base
                            blend = {q: float(base.ravel()[q]) for q in query}
                        for q in query:
                            row = {
                                "analyte": analyte, "basin": basin, "task_seed": seed,
                                "task_index": task_index, "month": task["month"],
                                "month_index": task["month_index"], "k": k,
                                "flat": q, "station": str(ds["site_no"][q // y.shape[1]]),
                                "y_true": float(y.ravel()[q]),
                                "pred_climatology": float(base.ravel()[q]),
                                "pred_local_mean": float(local_pred.ravel()[q]),
                                "pred_mean_bias": float(bias_pred.ravel()[q]),
                                "pred_analytic_blend": float(blend[q]),
                                "source_q90_threshold": q90,
                                "visibility_role": "same_analyte_support_diagnostic",
                                "config_hash": config_hash,
                                "hide_rows_sha256": object_hash(hide_rows.tolist()),
                                "target_component_rows_sha256": object_hash(target_rows.tolist()),
                            }
                            rows_out.append(row)
    cells = pd.DataFrame(rows_out)
    cells_path = out / "cells.parquet"
    cells.to_parquet(cells_path, index=False)
    task_hash = object_hash(task_manifest)
    runtime = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
    manifest = {
        "version": "phase4_stage2a_same_analyte_v2",
        "role": config["role"], "config_hash": config_hash,
        "spec_sha256": file_hash(spec), "region_spec_sha256": file_hash(region_path),
        "task_manifest_sha256": task_hash, "datasets": {a: summaries[a]["sha256"] for a in ANALYTES},
        "runner_sha256": file_hash(Path(__file__)),
        "visibility_roles": visibility_roles,
        "nodes_sha256": file_hash("data/processed/graph_nodes_graphfix_st357.csv"),
        "edges_sha256": file_hash("data/processed/graph_edges_graphfix_st357.csv"),
        "runtime": runtime, "runtime_hash": object_hash(runtime),
        "cells_sha256": file_hash(cells_path), "n_rows": len(cells),
        "training_started": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "task_manifest.json").write_text(json.dumps(task_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(cells), "tasks": len(task_manifest), "out": str(out)}))


if __name__ == "__main__":
    main()
