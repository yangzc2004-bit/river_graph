"""Run frozen no-training baselines for the cross-basin transfer route.

The task/query construction is read from ``cross_basin_tasks_v1``.  This
script only fits deterministic source aggregations; it never uses target query
labels to form a prediction, select a method, or choose a threshold.
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
    array,
    array_hash,
    file_hash,
    load_bundle,
    make_cross_basin_split,
    object_hash,
)
from river_graph.experiments.transfer_baselines import (
    METHODS,
    all_baseline_predictions,
    ecology_month_climatology,
    source_month_climatology,
)

NODES = "data/processed/graph_nodes_graphfix_st357.csv"
EDGES = "data/processed/graph_edges_graphfix_st357.csv"
TASKS = "experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json"
SPEC = "experiments/phase4_transfer/spec_v2_cross_basin.md"
BASELINE_SPEC = "experiments/phase4_transfer/stage2b_cross_basin_spec_v1.md"


def _hops(ds):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(ds["site_no"])))
    graph.add_edges_from(array(ds["edge_index"]).T.tolist())
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    return lambda a, b: lengths.get(int(a), {}).get(int(b))


def _source_selection(
    base: np.ndarray,
    eco_base: np.ndarray,
    val_mask: np.ndarray,
    y: np.ndarray,
    analyte: str,
    basin: str,
    seed: int,
) -> list[dict]:
    """Score source-only candidates on source validation labels."""
    indices = np.flatnonzero(val_mask.ravel())
    if not len(indices):
        raise ValueError(f"{analyte}/{basin}/seed{seed}: empty source validation")
    candidates = {
        "climatology": base.ravel(),
        "eco_month_climatology": eco_base.ravel(),
    }
    rows = []
    for order, method in enumerate(("climatology", "eco_month_climatology")):
        err = np.abs(candidates[method][indices] - y.ravel()[indices])
        rows.append(
            {
                "analyte": analyte,
                "basin": basin,
                "seed": seed,
                "method": method,
                "source_val_n": len(indices),
                "source_val_mae": float(err.mean()),
                "tie_order": order,
            }
        )
    return rows


def _verify_role(role: dict, split: dict[str, np.ndarray], target: int) -> None:
    for name in ("observed", "train", "val", "test"):
        if array_hash(split[name]) != role[f"{name}_hash"]:
            raise ValueError(f"role {role['basin']}/seed{role['seed']} {name} hash mismatch")
    if split["train"][:, role["hide_rows"], :].any() or split["val"][:, role["hide_rows"], :].any():
        raise ValueError("target HUC6 labels entered source roles")
    if split["train"][target, role["hide_rows"], :].any():
        raise ValueError("target analyte target-basin labels entered fit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--baseline-spec", default=BASELINE_SPEC)
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/stage2b_cross_basin_v1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    task_path, spec_path, baseline_spec_path = (
        Path(args.tasks), Path(args.spec), Path(args.baseline_spec)
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_manifest = out / "manifest.json"
    if output_manifest.exists() and not args.force:
        raise SystemExit(f"{output_manifest} exists; use --force or a new version")

    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    if task_manifest.get("training_started"):
        raise ValueError("frozen task manifest is marked training_started")
    if file_hash(spec_path) != task_manifest["spec_sha256"]:
        raise ValueError("task manifest/spec hash mismatch")
    if object_hash(task_manifest["tasks"]) != task_manifest["tasks_hash"]:
        raise ValueError("task manifest task hash mismatch")
    if object_hash(task_manifest["roles"]) != task_manifest["roles_hash"]:
        raise ValueError("task manifest role hash mismatch")

    datasets, summaries, _nodes = load_bundle(DATASETS, NODES, EDGES)
    if {a: summaries[a]["sha256"] for a in ANALYTES} != task_manifest["datasets"]:
        raise ValueError("dataset hash mismatch against frozen task manifest")
    if file_hash(NODES) != task_manifest["nodes_sha256"] or file_hash(EDGES) != task_manifest["edges_sha256"]:
        raise ValueError("node/edge hash mismatch against frozen task manifest")
    labels = np.stack([array(datasets[a]["y"]).astype(float) for a in ANALYTES])
    observed = np.stack([array(datasets[a]["y_mask"]).astype(bool) for a in ANALYTES])
    months = [str(m)[:7] for m in datasets["doc"]["months"]]
    hops = _hops(datasets["doc"])
    role_map = {(str(r["basin"]), int(r["seed"])): r for r in task_manifest["roles"]}
    if len(role_map) != len(task_manifest["roles"]):
        raise ValueError("duplicate frozen role")

    task_groups: dict[tuple[str, int, str], list[dict]] = {}
    for task in task_manifest["tasks"]:
        key = (str(task["basin"]), int(task["seed"]), str(task["analyte"]))
        task_groups.setdefault(key, []).append(task)

    prediction_rows, selection_rows, role_rows = [], [], []
    for (basin, seed, analyte), tasks in sorted(task_groups.items()):
        role = role_map[(basin, seed)]
        target = ANALYTES.index(analyte)
        split = make_cross_basin_split(observed, role["hide_rows"], seed)
        _verify_role(role, split, target)
        ds = datasets[analyte]
        y = labels[target]
        fit_mask = split["train"][target]
        source_q90 = float(np.quantile(y[fit_mask], 0.9))
        base = source_month_climatology(y, fit_mask, months)
        eco_base = ecology_month_climatology(y, fit_mask, months, array(ds["regime"]))
        selection_rows.extend(
            _source_selection(base, eco_base, split["val"][target], y, analyte, basin, seed)
        )
        role_rows.append(
            {
                "analyte": analyte,
                "basin": basin,
                "seed": seed,
                "fit_cells": int(split["train"][target].sum()),
                "selection_cells": int(split["val"][target].sum()),
                "target_support_cells": int(sum(int(t["support_count"]) for t in tasks) // len(task_groups[(basin, seed, analyte)])),
                "fit_mask_hash": array_hash(split["train"]),
                "selection_mask_hash": array_hash(split["val"]),
                "visibility_role": "cross_basin_target_analyte_source_visible",
            }
        )
        for task in sorted(tasks, key=lambda row: (int(row["task_index"]), int(row["k"]))):
            support = np.asarray(task["support_cells"], dtype=int)
            query = np.asarray(task["query_cells"], dtype=int)
            if len(support) != int(task["support_count"]):
                raise ValueError("support count differs from frozen task")
            if np.intersect1d(support, query).size:
                raise ValueError("frozen support/query overlap")
            if len(query) == 0 or not np.isin(query // y.shape[1], role["component_rows"]).all():
                raise ValueError("query outside frozen target component")
            if not np.isin(support // y.shape[1], role["hide_rows"]).all():
                raise ValueError("support outside frozen target basin")
            if not observed[target].ravel()[query].all() or not observed[target].ravel()[support].all():
                raise ValueError("support/query contains unobserved cell")
            support_grid = np.zeros_like(y)
            support_grid.ravel()[support] = y.ravel()[support]
            preds = all_baseline_predictions(base, eco_base, support_grid, support, query, hops)
            for q in query:
                # y_true is copied only for final scoring after prediction is
                # fully determined. It is never passed to a baseline method.
                for method in METHODS:
                    prediction_rows.append(
                        {
                            "analyte": analyte,
                            "basin": basin,
                            "task_seed": seed,
                            "task_index": int(task["task_index"]),
                            "month": str(task["month"]),
                            "month_index": int(task["month_index"]),
                            "k": int(task["k"]),
                            "flat": int(q),
                            "station": str(ds["site_no"][int(q) // y.shape[1]]),
                            "model_name": method,
                            "y_pred": float(preds[method][int(q)]),
                            "y_true": float(y.ravel()[q]),
                            "source_q90_threshold": source_q90,
                            "support_count": int(task["support_count"]),
                            "support_hash": object_hash(support.tolist()),
                            "query_hash": object_hash(query.tolist()),
                            "visibility_role": task["visibility_role"],
                            "query_labels_used_for_prediction": False,
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    roles = pd.DataFrame(role_rows)
    predictions_path = out / "predictions.parquet"
    selection_path = out / "source_selection.csv"
    roles_path = out / "roles.csv"
    predictions.to_parquet(predictions_path, index=False)
    selections.to_csv(selection_path, index=False)
    roles.to_csv(roles_path, index=False)
    # Select one source-only baseline per analyte/HUC6 by averaging the three
    # independent source validation splits. No target query row is involved.
    selected = (
        selections.groupby(["analyte", "basin", "method"], as_index=False)["source_val_mae"].mean()
        .sort_values(["analyte", "basin", "source_val_mae", "method"])
        .drop_duplicates(["analyte", "basin"], keep="first")
        .rename(columns={"method": "selected_method", "source_val_mae": "selected_source_val_mae"})
    )
    selected.to_csv(out / "selected_baselines.csv", index=False)
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "networkx": nx.__version__,
    }
    config = {
        "version": "phase4_stage2b_cross_basin_v1",
        "methods": list(METHODS),
        "task_manifest": str(task_path),
        "spec": str(spec_path),
        "baseline_spec": str(baseline_spec_path),
        "query_labels_used_for_prediction": False,
        "source_selection": "source validation labels only; no target query labels",
        "k_values": list(task_manifest["k_values"]),
    }
    manifest = {
        **config,
        "config_hash": object_hash(config),
        "task_manifest_sha256": file_hash(task_path),
        "task_manifest_tasks_hash": task_manifest["tasks_hash"],
        "task_manifest_roles_hash": task_manifest["roles_hash"],
        "spec_sha256": file_hash(spec_path),
        "baseline_spec_sha256": file_hash(baseline_spec_path),
        "datasets": {a: {"path": DATASETS[a], "sha256": summaries[a]["sha256"]} for a in ANALYTES},
        "nodes_sha256": file_hash(NODES),
        "edges_sha256": file_hash(EDGES),
        "runner_sha256": file_hash(Path(__file__)),
        "baseline_module_sha256": file_hash(Path("src/river_graph/experiments/transfer_baselines.py")),
        "runtime": runtime,
        "runtime_hash": object_hash(runtime),
        "predictions_sha256": file_hash(predictions_path),
        "source_selection_sha256": file_hash(selection_path),
        "selected_baselines_sha256": file_hash(out / "selected_baselines.csv"),
        "roles_sha256": file_hash(roles_path),
        "n_prediction_rows": len(predictions),
        "n_tasks": int(predictions[["analyte", "basin", "task_seed", "task_index", "k"]].drop_duplicates().shape[0]),
        "training_started": False,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(predictions), "tasks": manifest["n_tasks"], "out": str(out)}))


if __name__ == "__main__":
    main()
