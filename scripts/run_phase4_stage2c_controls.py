"""Plan and (after explicit review) execute Stage-2C control arms.

The default action is plan-only.  Execution requires an existing plan and an
explicit ``--ack-plan-sha256`` value so a control matrix cannot be trained
before its identity and visibility contract has been reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from river_graph.experiments.transfer_stage2c import (
    ARM_CONFIGS,
    ARMS,
    ECORF_TRAINING,
    GNN_TRAINING,
    ecological_time_features,
    validate_arm_configs,
)

NODES = "data/processed/graph_nodes_graphfix_st357.csv"
EDGES = "data/processed/graph_edges_graphfix_st357.csv"
TASKS = "experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json"
SPEC = "experiments/phase4_transfer/stage2c_controls_v1_spec.md"
K_VALUES = (0, 1, 3, 5)


def _load_context(task_path: Path, spec_path: Path) -> dict:
    task_manifest = json.loads(task_path.read_text(encoding="utf-8"))
    if task_manifest.get("training_started"):
        raise ValueError("cross-basin task manifest is marked training_started")
    if object_hash(task_manifest["roles"]) != task_manifest["roles_hash"]:
        raise ValueError("cross-basin role hash mismatch")
    if object_hash(task_manifest["tasks"]) != task_manifest["tasks_hash"]:
        raise ValueError("cross-basin task hash mismatch")
    if file_hash(Path(task_manifest["spec"])) != task_manifest["spec_sha256"]:
        raise ValueError("cross-basin route spec hash mismatch")
    datasets, summaries, _nodes = load_bundle(DATASETS, NODES, EDGES)
    for analyte in ANALYTES:
        if summaries[analyte]["sha256"] != task_manifest["datasets"][analyte]:
            raise ValueError(f"dataset hash mismatch: {analyte}")
    if file_hash(NODES) != task_manifest["nodes_sha256"]:
        raise ValueError("node metadata hash mismatch")
    if file_hash(EDGES) != task_manifest["edges_sha256"]:
        raise ValueError("edge metadata hash mismatch")
    if tuple(task_manifest["k_values"]) != K_VALUES:
        raise ValueError("Stage-2C requires K={0,1,3,5}")
    if file_hash(spec_path) == "":  # pragma: no cover - defensive path check
        raise ValueError("missing Stage-2C spec")
    labels = np.stack([array(datasets[a]["y"]).astype(float) for a in ANALYTES])
    observed = np.stack([array(datasets[a]["y_mask"]).astype(bool) for a in ANALYTES])
    roles = {}
    splits = {}
    for role in task_manifest["roles"]:
        key = (str(role["basin"]), int(role["seed"]))
        if key in roles:
            raise ValueError(f"duplicate role: {key}")
        split = make_cross_basin_split(observed, role["hide_rows"], int(role["seed"]))
        for name in ("observed", "train", "val", "test"):
            if array_hash(split[name]) != role[f"{name}_hash"]:
                raise ValueError(f"role hash mismatch {key} {name}")
        if split["train"][:, role["hide_rows"], :].any() or split["val"][:, role["hide_rows"], :].any():
            raise ValueError(f"target basin label visible in source role {key}")
        roles[key] = role
        splits[key] = split
    tasks_by_unit = {}
    for task in task_manifest["tasks"]:
        key = (str(task["basin"]), int(task["seed"]), str(task["analyte"]))
        role_key = key[:2]
        if role_key not in roles:
            raise ValueError(f"task refers to unknown role: {key}")
        target = ANALYTES.index(key[2])
        split = splits[role_key]
        if task["fit_mask_hash"] != array_hash(split["train"]):
            raise ValueError(f"task fit mask mismatch: {key}")
        if task["selection_mask_hash"] != array_hash(split["val"]):
            raise ValueError(f"task selection mask mismatch: {key}")
        if task["target_source_mask_hash"] != array_hash(split["train"][target]):
            raise ValueError(f"task target source mask mismatch: {key}")
        if int(task["k"]) not in K_VALUES:
            raise ValueError(f"unsupported K: {task['k']}")
        support, query = np.asarray(task["support_cells"], dtype=int), np.asarray(task["query_cells"], dtype=int)
        if len(support) != int(task["support_count"]):
            raise ValueError(f"support count mismatch: {key}")
        if np.intersect1d(support, query).size:
            raise ValueError(f"support/query overlap: {key}")
        _n, t = labels.shape[1:]
        rows = np.asarray(roles[role_key]["hide_rows"], dtype=int)
        component = np.asarray(roles[role_key]["component_rows"], dtype=int)
        if len(support) and not np.isin(support // t, rows).all():
            raise ValueError(f"support outside target basin: {key}")
        if len(query) == 0 or not np.isin(query // t, component).all():
            raise ValueError(f"query outside target component: {key}")
        if not observed[target].ravel()[support].all() or not observed[target].ravel()[query].all():
            raise ValueError(f"unobserved support/query: {key}")
        tasks_by_unit.setdefault(key, []).append(task)
    for key, tasks in tasks_by_unit.items():
        if {int(t["k"]) for t in tasks} != set(K_VALUES):
            raise ValueError(f"incomplete K ladder: {key}")
    return {
        "task_manifest": task_manifest,
        "datasets": datasets,
        "summaries": summaries,
        "labels": labels,
        "observed": observed,
        "roles": roles,
        "splits": splits,
        "tasks_by_unit": tasks_by_unit,
    }


def build_plan(context: dict, task_path: Path, spec_path: Path, runtime_hash: str) -> dict:
    """Create the exact arm/unit matrix without importing torch or training."""
    validate_arm_configs()
    task_manifest = context["task_manifest"]
    units = []
    for unit_key in sorted(context["tasks_by_unit"]):
        basin, seed, analyte = unit_key
        target = ANALYTES.index(analyte)
        role = context["roles"][(basin, seed)]
        for arm in ARMS:
            cfg = {
                "version": "phase4_stage2c_controls_v1",
                "arm": arm,
                "arm_config": ARM_CONFIGS[arm],
                "analyte": analyte,
                "target_index": target,
                "basin": basin,
                "seed": seed,
                "k_values": list(K_VALUES),
                "task_manifest": str(task_path),
                "task_manifest_sha256": file_hash(task_path),
                "tasks_hash": task_manifest["tasks_hash"],
                "roles_hash": task_manifest["roles_hash"],
                "route_spec": task_manifest["spec"],
                "route_spec_sha256": task_manifest["spec_sha256"],
                "stage2c_spec": str(spec_path),
                "stage2c_spec_sha256": file_hash(spec_path),
                "dataset_path": DATASETS[analyte],
                "dataset_sha256": context["summaries"][analyte]["sha256"],
                "nodes_sha256": file_hash(NODES),
                "edges_sha256": file_hash(EDGES),
                "fit_mask_hash": role["train_hash"],
                "selection_mask_hash": role["val_hash"],
                "test_mask_hash": role["test_hash"],
                "hide_rows": role["hide_rows"],
                "component_rows": role["component_rows"],
                "runtime_snapshot_hash": runtime_hash,
                "training_budget": ECORF_TRAINING if arm == "ecorf" else GNN_TRAINING,
                "query_labels_used_for_prediction": False,
            }
            units.append({
                "unit": f"{arm}__{analyte}__{basin}__seed{seed}",
                "arm": arm,
                "analyte": analyte,
                "basin": basin,
                "seed": seed,
                "config": cfg,
                "config_hash": object_hash(cfg),
                "task_count": len(context["tasks_by_unit"][unit_key]),
                "task_counts_by_k": {
                    str(k): sum(int(t["k"]) == k for t in context["tasks_by_unit"][unit_key])
                    for k in K_VALUES
                },
            })
    plan = {
        "version": "phase4_stage2c_controls_v1",
        "status": "plan_only_pending_identity_review",
        "arms": list(ARMS),
        "units": units,
        "n_units": len(units),
        "n_gnn_units": sum(u["arm"] != "ecorf" for u in units),
        "n_ecorf_units": sum(u["arm"] == "ecorf" for u in units),
        "expected_compute": {
            "gnn_fits": sum(u["arm"] != "ecorf" for u in units),
            "ecorf_fits": sum(u["arm"] == "ecorf" for u in units),
            "gnn_full_grid_inference_calls": sum(
                u["task_count"] for u in units if u["arm"] != "ecorf"
            ),
            "ecorf_full_grid_inference_calls": sum(
                u["task_count"] // len(K_VALUES) for u in units if u["arm"] == "ecorf"
            ),
            "note": "one GNN prediction per frozen task/K; EcoRF full-grid fit once and repeated over K",
        },
        "k_values": list(K_VALUES),
        "task_manifest": str(task_path),
        "task_manifest_sha256": file_hash(task_path),
        "tasks_hash": task_manifest["tasks_hash"],
        "roles_hash": task_manifest["roles_hash"],
        "stage2c_spec": str(spec_path),
        "stage2c_spec_sha256": file_hash(spec_path),
        "training_started": False,
        "query_labels_used_for_prediction": False,
        "runtime_snapshot_hash": runtime_hash,
    }
    return plan


def _ecorf_predict(dataset: dict, split: dict, seed: int) -> np.ndarray:
    """Fit label-free-feature EcoRF on source train cells and predict full grid."""
    from sklearn.ensemble import RandomForestRegressor

    features = ecological_time_features(dataset)
    y = np.log1p(array(dataset["y"]).astype(float)).ravel()
    train = np.asarray(split["train"], dtype=int)
    model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(features[train], y[train])
    return np.expm1(model.predict(features).reshape(array(dataset["y"]).shape))


def _execute(context: dict, plan: dict, out: Path) -> None:
    """Execute reviewed units; caller must have supplied the plan hash."""

    from river_graph.models.gcn import GCNDocModel

    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    selection_rows, completed = [], []
    for unit in plan["units"]:
        arm, analyte, basin, seed = unit["arm"], unit["analyte"], unit["basin"], int(unit["seed"])
        role_key = (basin, seed)
        split_full = context["splits"][role_key]
        target = ANALYTES.index(analyte)
        dataset = dict(context["datasets"][analyte])
        split = {
            "train": np.flatnonzero(split_full["train"][target].ravel()),
            "val": np.flatnonzero(split_full["val"][target].ravel()),
            "test": np.flatnonzero(split_full["test"][target].ravel()),
        }
        if arm == "ecorf":
            pred0 = _ecorf_predict(dataset, split, seed)
        else:
            cfg = ARM_CONFIGS[arm]
            model = GCNDocModel(
                architecture=cfg["architecture"],
                edge_set=cfg["edge_set"],
                edge_direction=cfg["edge_direction"],
                env_groups=cfg["env_groups"],
                env_encoder=cfg["env_encoder"],
                variant="river",
                seed=seed,
                **GNN_TRAINING,
            )
            model.fit(dataset, split)
            source_train = split["train"]
            pred0 = model.predict(only_visible=source_train)
        # Source validation score is a selection diagnostic only. It does not
        # open target query labels or alter the arm/model.
        y = array(dataset["y"]).astype(float)
        val = split["val"]
        selection_rows.append(
            {
                "unit": unit["unit"],
                "arm": arm,
                "analyte": analyte,
                "basin": basin,
                "seed": seed,
                "source_val_n": len(val),
                "source_val_mae": float(np.mean(np.abs(pred0.ravel()[val] - y.ravel()[val]))),
                "config_hash": unit["config_hash"],
            }
        )
        rows_out = []
        for task in sorted(context["tasks_by_unit"][(basin, seed, analyte)], key=lambda row: (row["task_index"], row["k"])):
            k = int(task["k"])
            support = np.asarray(task["support_cells"], dtype=int)
            query = np.asarray(task["query_cells"], dtype=int)
            if arm == "ecorf" or k == 0:
                pred = pred0
            else:
                support_y = y.ravel()[support]
                pred = model.predict(
                    only_visible=source_train,
                    extra_visible=support,
                    extra_values=support_y,
                )
            for q in query:
                rows_out.append(
                    {
                        "model_name": arm,
                        "analyte": analyte,
                        "basin": basin,
                        "task_seed": seed,
                        "task_index": int(task["task_index"]),
                        "month": str(task["month"]),
                        "month_index": int(task["month_index"]),
                        "k": k,
                        "flat": int(q),
                        "station": str(dataset["site_no"][int(q) // y.shape[1]]),
                        "y_pred": float(pred.ravel()[q]),
                        "uncertainty": np.nan,
                        "uncertainty_status": "not_applicable_no_interval",
                        "support_count": int(task["support_count"]),
                        "visibility_role": task["visibility_role"],
                        "query_labels_used_for_prediction": False,
                        "config_hash": unit["config_hash"],
                    }
                )
        path = pred_dir / f"{unit['unit']}.parquet"
        pd.DataFrame(rows_out).to_parquet(path, index=False)
        completed.append({**unit, "prediction_sha256": file_hash(path), "n_rows": len(rows_out)})
    pd.DataFrame(selection_rows).to_csv(out / "source_selection.csv", index=False)
    (out / "completed_units.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/stage2c_controls_v1")
    ap.add_argument("--execute", action="store_true", help="run reviewed units; plan-only is the default")
    ap.add_argument("--ack-plan-sha256", default=None, help="required hash of an existing reviewed execution_plan.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    validate_arm_configs()
    task_path, spec_path, out = Path(args.tasks), Path(args.spec), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "execution_plan.json"
    if plan_path.exists() and not args.force and not args.execute:
        raise SystemExit(f"{plan_path} exists; use --force to regenerate")
    # runtime snapshot is computed before training and is bound into every unit
    # identity. Import lazily so plan-only remains independent of torch.
    from river_graph.experiments.provenance import runtime_code_snapshot_sha256

    context = _load_context(task_path, spec_path)
    runtime_hash = runtime_code_snapshot_sha256()
    if args.execute:
        if not plan_path.is_file():
            raise SystemExit("execution requires an existing reviewed execution_plan.json")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        want = file_hash(plan_path)
        if args.ack_plan_sha256 != want:
            raise SystemExit(f"plan review acknowledgment mismatch; expected {want}")
        if plan.get("runtime_snapshot_hash") != runtime_hash:
            raise SystemExit("runtime snapshot changed since plan freeze; regenerate and review plan")
        if plan.get("training_started"):
            raise SystemExit("plan already marked training_started")
        _execute(context, plan, out)
        plan["training_started"] = True
        plan["status"] = "executed"
        plan["completed_units_sha256"] = file_hash(out / "completed_units.json")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        manifest_path = out / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "status": "executed",
                    "training_started": True,
                    "completed_units_sha256": plan["completed_units_sha256"],
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "executed", "units": plan["n_units"], "out": str(out)}))
        return
    plan = build_plan(context, task_path, spec_path, runtime_hash)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    plan_sha = file_hash(plan_path)
    manifest = {
        "version": "phase4_stage2c_controls_v1",
        "status": "plan_only_pending_identity_review",
        "execution_plan_sha256": plan_sha,
        "task_manifest_sha256": file_hash(task_path),
        "tasks_hash": context["task_manifest"]["tasks_hash"],
        "roles_hash": context["task_manifest"]["roles_hash"],
        "stage2c_spec_sha256": file_hash(spec_path),
        "datasets": {a: context["summaries"][a]["sha256"] for a in ANALYTES},
        "nodes_sha256": file_hash(NODES),
        "edges_sha256": file_hash(EDGES),
        "runtime_snapshot_hash": runtime_hash,
        "n_units": plan["n_units"],
        "n_gnn_units": plan["n_gnn_units"],
        "n_ecorf_units": plan["n_ecorf_units"],
        "training_started": False,
        "query_labels_used_for_prediction": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "STATUS.md").write_text(
        "# Stage 2C controls status\n\n"
        "Plan-only freeze complete. No model has been trained and no prediction "
        "product has been generated. Review `execution_plan.json` before using "
        "the following explicit command:\n\n"
        f"```bash\n.venv/bin/python scripts/run_phase4_stage2c_controls.py --execute "
        f"--ack-plan-sha256 {plan_sha}\n```\n\n"
        f"The reviewed plan contains {plan['n_units']} units ({plan['n_gnn_units']} "
        f"GNN and {plan['n_ecorf_units']} EcoRF fits).\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "plan_only_pending_identity_review", "units": plan["n_units"], "out": str(out)}))


if __name__ == "__main__":
    main()
