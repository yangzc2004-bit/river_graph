"""Freeze and execute the bounded Stage-2 support-integrity audit.

The default action is plan-only.  Execution requires the reviewed plan hash.
This runner fits only the already frozen H2X arm and writes one label-free
product per analyte × target basin × seed.  Query labels are consumed only by
the separate evaluator after the product audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.support_integrity import (
    SHUFFLE_MODES,
    site_shuffle_destination,
    stable_seed,
    validate_shuffle_record,
    value_shuffle_permutation,
)
from river_graph.experiments.transfer import (
    ANALYTES,
    DATASETS,
    array,
    file_hash,
    object_hash,
)
from river_graph.experiments.transfer_stage2c import ARM_CONFIGS, GNN_TRAINING

TASKS = Path("experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json")
STAGE2C_SPEC = Path("experiments/phase4_transfer/stage2c_controls_v1_1_spec.md")
SPEC = Path("experiments/phase4_transfer/stage2_support_integrity_v1_spec.md")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
EDGES = Path("data/processed/graph_edges_graphfix_st357.csv")
OUT = Path("experiments/phase4_transfer/stage2_support_integrity_v1")


def _load_context(task_path: Path):
    # Reuse the already audited Stage-2C loader.  It verifies the frozen task
    # roles, datasets, masks, and exact task inventory before this audit adds
    # its own shuffle manifest.
    try:
        from scripts.run_phase4_stage2c_controls import _load_context as load
    except ModuleNotFoundError:  # direct execution from scripts/
        from run_phase4_stage2c_controls import _load_context as load

    return load(task_path, STAGE2C_SPEC)


def _task_key(task: dict) -> tuple:
    return (
        str(task["analyte"]),
        str(task["basin"]),
        int(task["seed"]),
        int(task["task_index"]),
        int(task["k"]),
    )


def _build_shuffle_manifest(context: dict) -> dict:
    """Construct all task-level controls using masks but never target values."""

    records = []
    datasets = context["datasets"]
    for unit_key in sorted(context["tasks_by_unit"]):
        basin, seed, analyte = unit_key
        dataset = datasets[analyte]
        observed_flat = array(dataset["y_mask"]).astype(bool).ravel()
        # y_mask is (station, month); use its second dimension explicitly.
        n_months = int(array(dataset["y_mask"]).shape[1])
        target_rows = np.asarray(context["roles"][(basin, seed)]["hide_rows"], dtype=np.int64)
        for task in sorted(context["tasks_by_unit"][unit_key], key=lambda row: (int(row["task_index"]), int(row["k"]))):
            support = np.asarray(task["support_cells"], dtype=np.int64)
            query = np.asarray(task["query_cells"], dtype=np.int64)
            base = {
                "analyte": analyte,
                "basin": basin,
                "seed": int(seed),
                "task_index": int(task["task_index"]),
                "month": str(task["month"]),
                "month_index": int(task["month_index"]),
                "k": int(task["k"]),
                "support_cells": support.tolist(),
                "query_cells": query.tolist(),
                "support_count": int(task["support_count"]),
            }
            if int(task["k"]) <= 1:
                value_record = {
                    **base,
                    "mode": "value_shuffle",
                    "status": "not_identifiable_k0" if int(task["k"]) == 0 else "not_identifiable_k1",
                    "value_permutation": list(range(int(task["k"]))),
                    "candidate_count": None,
                    "destination_cells": [],
                    "shuffle_seed": stable_seed(analyte, basin, seed, task["task_index"], task["k"], "value"),
                }
            else:
                permutation = value_shuffle_permutation(
                    support,
                    seed=stable_seed(analyte, basin, seed, task["task_index"], task["k"], "value"),
                )
                value_record = {
                    **base,
                    "mode": "value_shuffle",
                    "status": "identifiable",
                    "value_permutation": permutation.tolist(),
                    "candidate_count": None,
                    "destination_cells": [],
                    "shuffle_seed": stable_seed(analyte, basin, seed, task["task_index"], task["k"], "value"),
                }
            records.append(value_record)
            site = site_shuffle_destination(
                support,
                query,
                target_rows,
                int(task["month_index"]),
                n_months,
                observed_flat,
                seed=stable_seed(analyte, basin, seed, task["task_index"], task["k"], "site"),
            )
            records.append(
                {
                    **base,
                    "mode": "site_shuffle",
                    **site,
                    "shuffle_seed": stable_seed(analyte, basin, seed, task["task_index"], task["k"], "site"),
                }
            )
    for record in records:
        validate_shuffle_record(record, n_months=int(array(datasets[record["analyte"]]["y_mask"]).shape[1]))
    return {
        "version": "phase4_stage2_support_integrity_v1",
        "status": "frozen_before_execution",
        "task_manifest_sha256": file_hash(TASKS),
        "datasets": {a: context["summaries"][a]["sha256"] for a in ANALYTES},
        "nodes_sha256": file_hash(NODES),
        "edges_sha256": file_hash(EDGES),
        "spec_sha256": file_hash(SPEC),
        "n_records": len(records),
        "records": records,
        "query_labels_used": False,
    }


def _build_plan(context: dict, task_path: Path, runtime_hash: str) -> tuple[dict, dict]:
    task_manifest = context["task_manifest"]
    units = []
    for unit_key in sorted(context["tasks_by_unit"]):
        basin, seed, analyte = unit_key
        role = context["roles"][(basin, seed)]
        cfg = {
            "version": "phase4_stage2_support_integrity_v1",
            "arm": "h2x_full",
            "arm_config": ARM_CONFIGS["h2x_full"],
            "analyte": analyte,
            "target_index": ANALYTES.index(analyte),
            "basin": basin,
            "seed": int(seed),
            "k_values": [0, 1, 3, 5],
            "task_manifest": str(task_path),
            "task_manifest_sha256": file_hash(task_path),
            "tasks_hash": task_manifest["tasks_hash"],
            "roles_hash": task_manifest["roles_hash"],
            "stage2c_spec": str(STAGE2C_SPEC),
            "stage2c_spec_sha256": file_hash(STAGE2C_SPEC),
            "support_integrity_spec": str(SPEC),
            "support_integrity_spec_sha256": file_hash(SPEC),
            "runner_sha256": file_hash(Path(__file__)),
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
            "training_budget": GNN_TRAINING,
            "query_labels_used_for_prediction": False,
        }
        units.append(
            {
                "unit": f"h2x_full__{analyte}__{basin}__seed{seed}",
                "arm": "h2x_full",
                "analyte": analyte,
                "basin": basin,
                "seed": int(seed),
                "config": cfg,
                "config_hash": object_hash(cfg),
                "task_count": len(context["tasks_by_unit"][unit_key]),
            }
        )
    shuffle = _build_shuffle_manifest(context)
    plan = {
        "version": "phase4_stage2_support_integrity_v1",
        "status": "plan_only_pending_identity_review",
        "units": units,
        "n_units": len(units),
        "task_manifest": str(task_path),
        "task_manifest_sha256": file_hash(task_path),
        "tasks_hash": task_manifest["tasks_hash"],
        "roles_hash": task_manifest["roles_hash"],
        "support_integrity_spec": str(SPEC),
        "support_integrity_spec_sha256": file_hash(SPEC),
        "runner_sha256": file_hash(Path(__file__)),
        "runtime_snapshot_hash": runtime_hash,
        "training_budget": GNN_TRAINING,
        "query_labels_used_for_prediction": False,
        "shuffle_manifest_sha256": object_hash(shuffle),
        "training_started": False,
    }
    return plan, shuffle


def _record_index(shuffle: dict) -> dict[tuple, dict]:
    return {
        (str(r["analyte"]), str(r["basin"]), int(r["seed"]), int(r["task_index"]), int(r["k"]), str(r["mode"])): r
        for r in shuffle["records"]
    }


def _write_progress(plan: dict, out: Path) -> None:
    completed = []
    for unit in plan["units"]:
        path = out / "predictions" / f"{unit['unit']}.parquet"
        sidecar = out / "sidecars" / f"{unit['unit']}.json"
        if path.is_file() and sidecar.is_file():
            completed.append({**unit, "prediction_sha256": file_hash(path), "sidecar_sha256": file_hash(sidecar)})
    completed.sort(key=lambda row: row["unit"])
    (out / "completed_units.json").write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")


def _execute(context: dict, plan: dict, shuffle: dict, out: Path, units: list[dict]) -> None:
    from river_graph.models.gcn import GCNDocModel

    records = _record_index(shuffle)
    pred_dir, side_dir = out / "predictions", out / "sidecars"
    pred_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    for unit in units:
        pred_path = pred_dir / f"{unit['unit']}.parquet"
        side_path = side_dir / f"{unit['unit']}.json"
        if pred_path.is_file() and side_path.is_file():
            old = json.loads(side_path.read_text(encoding="utf-8"))
            if old.get("config_hash") == unit["config_hash"]:
                continue
        key = (unit["basin"], int(unit["seed"]))
        split_full = context["splits"][key]
        target = ANALYTES.index(unit["analyte"])
        dataset = dict(context["datasets"][unit["analyte"]])
        y = array(dataset["y"]).astype(float)
        source_train = np.flatnonzero(split_full["train"][target].ravel())
        source_val = np.flatnonzero(split_full["val"][target].ravel())
        model = GCNDocModel(
            architecture="transport_enc",
            edge_set="river",
            edge_direction="both",
            env_groups=None,
            env_encoder=True,
            variant="river",
            seed=int(unit["seed"]),
            **GNN_TRAINING,
        )
        model.fit(dataset, {"train": source_train, "val": source_val, "test": np.asarray([], dtype=int)})
        pred0 = model.predict(only_visible=source_train)
        rows_out = []
        task_rows = context["tasks_by_unit"][(unit["basin"], int(unit["seed"]), unit["analyte"])]
        for task in sorted(task_rows, key=lambda row: (int(row["task_index"]), int(row["k"]))):
            task_index, k = int(task["task_index"]), int(task["k"])
            support = np.asarray(task["support_cells"], dtype=np.int64)
            query = np.asarray(task["query_cells"], dtype=np.int64)
            mode_preds = {"true": pred0 if k == 0 else model.predict(only_visible=source_train, extra_visible=support, extra_values=y.ravel()[support])}
            value_record = records[(unit["analyte"], unit["basin"], int(unit["seed"]), task_index, k, "value_shuffle")]
            if value_record["status"] == "identifiable":
                permutation = np.asarray(value_record["value_permutation"], dtype=np.int64)
                mode_preds["value_shuffle"] = model.predict(only_visible=source_train, extra_visible=support, extra_values=y.ravel()[support][permutation])
            site_record = records[(unit["analyte"], unit["basin"], int(unit["seed"]), task_index, k, "site_shuffle")]
            if site_record["status"] == "identifiable":
                destinations = np.asarray(site_record["destination_cells"], dtype=np.int64)
                mode_preds["site_shuffle"] = model.predict(only_visible=source_train, extra_visible=destinations, extra_values=y.ravel()[support])
            for mode in SHUFFLE_MODES:
                record = value_record if mode == "value_shuffle" else site_record if mode == "site_shuffle" else {"status": "identifiable" if k >= 0 else "not_applicable_k0", "candidate_count": None}
                pred_grid = mode_preds.get(mode)
                for q in query.tolist():
                    rows_out.append(
                        {
                            "model_name": "h2x_full",
                            "support_mode": mode,
                            "analyte": unit["analyte"],
                            "basin": unit["basin"],
                            "task_seed": int(unit["seed"]),
                            "task_index": task_index,
                            "month": str(task["month"]),
                            "month_index": int(task["month_index"]),
                            "k": k,
                            "flat": int(q),
                            "station": str(dataset["site_no"][int(q) // y.shape[1]]),
                            "y_pred": float(pred_grid.ravel()[q]) if pred_grid is not None else np.nan,
                            "support_count": int(task["support_count"]),
                            "visibility_role": task["visibility_role"],
                            "query_labels_used_for_prediction": False,
                            "shuffle_status": str(record.get("status", "identifiable")),
                            "candidate_count": record.get("candidate_count"),
                            "config_hash": unit["config_hash"],
                        }
                    )
        pd.DataFrame(rows_out).to_parquet(pred_path, index=False)
        sidecar = {
            "version": "phase4_stage2_support_integrity_v1",
            "unit": unit["unit"],
            "config_hash": unit["config_hash"],
            "task_manifest_sha256": plan["task_manifest_sha256"],
            "shuffle_manifest_sha256": plan["shuffle_manifest_sha256"],
            "support_integrity_spec_sha256": plan["support_integrity_spec_sha256"],
            "runtime_snapshot_hash": plan["runtime_snapshot_hash"],
            "query_labels_used_for_prediction": False,
            "retrospective": False,
        }
        side_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        _write_progress(plan, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--ack-plan-sha256")
    ap.add_argument("--only-unit")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    task_path, out = Path(args.tasks), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "execution_plan.json"
    from river_graph.experiments.provenance import runtime_code_snapshot_sha256

    context = _load_context(task_path)
    runtime_hash = runtime_code_snapshot_sha256()
    if not args.execute:
        if plan_path.exists() and not args.force:
            raise SystemExit(f"{plan_path} exists; use --force to regenerate")
        plan, shuffle = _build_plan(context, task_path, runtime_hash)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        (out / "shuffle_manifest.json").write_text(json.dumps(shuffle, indent=2) + "\n", encoding="utf-8")
        plan_sha = file_hash(plan_path)
        (out / "manifest.json").write_text(json.dumps({**plan, "execution_plan_sha256": plan_sha}, indent=2) + "\n", encoding="utf-8")
        (out / "STATUS.md").write_text(
            "# Stage-2 support-integrity audit\n\n"
            "Plan-only freeze complete. Review `execution_plan.json` and "
            "`shuffle_manifest.json` before execution. Site-shuffle tasks "
            "without enough alternative same-month target stations are marked "
            "`not_identifiable_by_design` and are never silently reused.\n\n"
            f"Plan SHA256: `{plan_sha}`\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": plan["status"], "units": plan["n_units"], "plan_sha256": plan_sha}))
        return
    if not plan_path.is_file():
        raise SystemExit("execution requires a frozen execution_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.ack_plan_sha256 != file_hash(plan_path):
        raise SystemExit("plan review acknowledgment mismatch")
    if plan["runtime_snapshot_hash"] != runtime_hash:
        raise SystemExit("runtime snapshot changed since plan freeze")
    shuffle = json.loads((out / "shuffle_manifest.json").read_text(encoding="utf-8"))
    if object_hash(shuffle) != plan["shuffle_manifest_sha256"]:
        raise SystemExit("shuffle manifest hash mismatch")
    if args.finalize:
        _write_progress(plan, out)
        completed = json.loads((out / "completed_units.json").read_text(encoding="utf-8"))
        if len(completed) != int(plan["n_units"]):
            raise SystemExit(f"cannot finalize: {len(completed)}/{plan['n_units']} units complete")
        plan.update({"status": "executed", "training_started": True, "completed_units_sha256": file_hash(out / "completed_units.json")})
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "executed", "units": plan["n_units"]}))
        return
    units = plan["units"]
    if args.only_unit:
        units = [u for u in units if u["unit"] == args.only_unit]
        if len(units) != 1:
            raise SystemExit(f"unknown unit: {args.only_unit}")
    _execute(context, plan, shuffle, out, units)
    _write_progress(plan, out)
    completed = json.loads((out / "completed_units.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": "in_progress", "completed": len(completed), "units": plan["n_units"]}))


if __name__ == "__main__":
    main()
