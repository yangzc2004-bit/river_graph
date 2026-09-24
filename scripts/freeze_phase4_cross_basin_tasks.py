"""Freeze cross-basin multi-analyte roles and K-shot tasks without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from river_graph.experiments.transfer import (
    ANALYTES,
    DATASETS,
    K_VALUES,
    array,
    array_hash,
    availability_tasks,
    cross_basin_view,
    file_hash,
    load_bundle,
    make_cross_basin_split,
    object_hash,
)

REGIONS = Path("experiments/kshot_protocol_v2/regions.json")
SEEDS = (42, 43, 44)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/cross_basin_tasks_v1")
    ap.add_argument("--spec", default="experiments/phase4_transfer/spec_v2_cross_basin.md")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets, summaries, nodes = load_bundle(
        DATASETS,
        "data/processed/graph_nodes_graphfix_st357.csv",
        "data/processed/graph_edges_graphfix_st357.csv",
    )
    labels = np.stack([array(datasets[a]["y"]) for a in ANALYTES])
    observed = np.stack([array(datasets[a]["y_mask"]).astype(bool) for a in ANALYTES])
    months = [str(m)[:7] for m in datasets["doc"]["months"]]
    regions = json.loads(REGIONS.read_text())["primary"]
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists; use a new version or --force")
    task_rows, role_rows = [], []
    for region in regions:
        basin = "051002" if region["code"] == "510020" else region["code"]
        hide_rows = np.asarray(region["hide_rows"], dtype=int)
        component_rows = np.asarray(region["task_component_rows"], dtype=int)
        if not np.all(nodes.iloc[hide_rows]["huc6"].to_numpy() == basin):
            raise ValueError(f"hide rows do not match {basin}")
        if not np.all(nodes.iloc[component_rows]["huc6"].to_numpy() == basin):
            raise ValueError(f"component rows do not match {basin}")
        for seed in SEEDS:
            split = make_cross_basin_split(observed, hide_rows, seed)
            role_rows.append({
                "basin": basin, "seed": seed,
                "hide_rows": hide_rows.tolist(), "component_rows": component_rows.tolist(),
                "observed_hash": array_hash(split["observed"]),
                "train_hash": array_hash(split["train"]), "val_hash": array_hash(split["val"]),
                "test_hash": array_hash(split["test"]),
            })
            for analyte_index, analyte in enumerate(ANALYTES):
                tasks = availability_tasks(observed[analyte_index], component_rows, months, seed)
                for task_index, task in enumerate(tasks):
                    for k in K_VALUES:
                        view = cross_basin_view(
                            labels, observed, split["train"], split["val"],
                            target=analyte_index, target_rows=hide_rows,
                            support=task["support_cells_by_k"][str(k)],
                            query=task["query_cells"],
                        )
                        task_rows.append({
                            "analyte": analyte, "target": analyte_index, "basin": basin,
                            "seed": seed, "task_index": task_index,
                            "month": task["month"], "month_index": task["month_index"], "k": k,
                            "support_cells": task["support_cells_by_k"][str(k)],
                            "query_cells": task["query_cells"],
                            "support_count": len(task["support_cells_by_k"][str(k)]),
                            "fit_mask_hash": array_hash(view["fit_mask"]),
                            "selection_mask_hash": array_hash(view["selection_mask"]),
                            "target_source_mask_hash": array_hash(view["target_analyte_source_mask"]),
                            "visibility_role": "cross_basin_target_analyte_source_visible",
                        })
    task_manifest = {
        "version": "phase4_cross_basin_tasks_v1",
        "spec": str(args.spec), "spec_sha256": file_hash(args.spec),
        "regions_sha256": file_hash(REGIONS), "datasets": {a: summaries[a]["sha256"] for a in ANALYTES},
        "nodes_sha256": file_hash("data/processed/graph_nodes_graphfix_st357.csv"),
        "edges_sha256": file_hash("data/processed/graph_edges_graphfix_st357.csv"),
        "k_values": list(K_VALUES), "seeds": list(SEEDS),
        "roles": role_rows, "tasks": task_rows,
        "roles_hash": object_hash(role_rows), "tasks_hash": object_hash(task_rows),
        "training_started": False,
    }
    (out / "manifest.json").write_text(json.dumps(task_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"roles": len(role_rows), "tasks": len(task_rows), "out": str(out)}))


if __name__ == "__main__":
    raise SystemExit(main())
