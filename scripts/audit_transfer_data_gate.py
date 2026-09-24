#!/usr/bin/env python3
"""Run the Stage-1 transfer data gate; no model fitting is performed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from river_graph.experiments.transfer import (
    ANALYTES,
    DATASETS,
    external_checks,
    file_hash,
    load_bundle,
)
from river_graph.experiments.transfer_data import (
    availability_audit,
    label_content_bindings,
    qc_replay,
    unified_labels,
)


def read_candidates(path: str | None) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = payload if isinstance(payload, list) else payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise TypeError("candidate manifest must be a list or contain candidates")
    return candidates


def assess_external(candidate: dict, limits: dict) -> dict:
    paths = candidate.get("analytes", {})
    missing = [a for a in ANALYTES if a not in paths]
    result = {"name": candidate.get("name", "unnamed"), "huc8": candidate.get("huc8"), "eligible": False,
              "errors": [f"missing analyte path: {a}" for a in missing]}
    if missing:
        return result
    try:
        _datasets, summaries, _nodes = load_bundle(
            paths, candidate["nodes"], candidate["edges"]
        )
        checks = external_checks(summaries, limits=limits)
        errors = [f"{a}.{name}" for a, values in checks.items() for name, ok in values.items() if not ok]
        result.update({"eligible": not errors, "errors": errors, "summaries": summaries,
                       "nodes_sha256": file_hash(candidate["nodes"]),
                       "edges_sha256": file_hash(candidate["edges"])})
    except Exception as exc:  # noqa: BLE001 - candidate failures are reportable gate outcomes
        result["errors"].append(str(exc))
    return result


def markdown(report: dict) -> str:
    lines = [
        "# Phase 4 transfer data gate",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Stage-1 status: **{report['stage1_status']}**",
        "",
        "The audit uses only dataset availability, metadata, graph identity, and frozen QC facts.",
        "It does not load predictions or test labels for selection.",
        "",
        "## ST357 local bundle",
        "",
        "| analyte | stations | months | observed station-months | largest active component | sha256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for a, s in report["local"]["summaries"].items():
        lines.append(f"| {a} | {s['stations']} | {s['months']} | {s['observed_station_months']} | "
                     f"{s['active_largest_component_fraction']:.3f} | `{s['sha256'][:12]}…` |")
    lines.extend(["", f"Local errors: `{len(report['local']['errors'])}`"])
    if report["local"].get("qc_replay"):
        qc = report["local"]["qc_replay"]
        lines.extend([
            "", "## QC replay and task availability", "",
            f"Raw-cache replay: **{qc['passed']}** ({qc['raw_files']} files; retrospective only).",
            "",
            "| analyte | task ID | HUC6 | K=5 task-months | post-2020 task-months | query cells |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in report["local"].get("task_availability", []):
            lines.append(
                f"| {row['analyte']} | {row['task_id']} | {row['canonical_huc6']} | "
                f"{row['task_months']} | {row['post2020_task_months']} | {row['query_cells']} |"
            )
        lines.extend([
            "", "K=5 inventories are availability diagnostics, not a new confirmatory split.",
            "Existing task IDs are preserved; `510020` maps to canonical HUC6 `051002`.",
        ])
    if report["candidates"]:
        lines.extend(["", "## External candidates", "", "| candidate | eligible | errors |", "|---|---:|---:|"])
        for c in report["candidates"]:
            lines.append(f"| {c['name']} | {c['eligible']} | {len(c['errors'])} |")
    else:
        lines.extend(["", "## External candidates", "", "No candidate manifest supplied; external validation is pending."])
    lines.extend(["", "No model training is authorized by this report.", ""])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-manifest")
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/data_gate")
    ap.add_argument("--allow-external-pending", action="store_true")
    args = ap.parse_args()
    limits = {
        "minimum_stations": 50,
        "minimum_months": 36,
        "minimum_station_months_per_analyte": 10000,
        "minimum_largest_component_fraction": 0.8,
    }
    try:
        _datasets, summaries, nodes = load_bundle(
            DATASETS, "data/processed/graph_nodes_graphfix_st357.csv",
            "data/processed/graph_edges_graphfix_st357.csv"
        )
        qc, raw_manifest = qc_replay(Path("data/raw/wqp_results"), _datasets)
        mask_table, task_table, task_files, aliases = availability_audit(
            _datasets, nodes, Path("experiments/kshot_protocol_v2/regions.json")
        )
        local = {"eligible": True, "errors": [], "summaries": summaries,
                 "nodes": {"rows": len(nodes), "huc6": sorted(nodes.huc6.unique().tolist()),
                           "sha256": file_hash("data/processed/graph_nodes_graphfix_st357.csv")},
                 "edges": {"sha256": file_hash("data/processed/graph_edges_graphfix_st357.csv")},
                 "qc_replay": qc, "mask_availability": mask_table.to_dict("records"),
                 "task_availability": task_table.to_dict("records"),
                 "task_aliases": aliases, "label_bindings": label_content_bindings(_datasets),
                 "task_files": task_files, "raw_file_count": len(raw_manifest),
                 "raw_manifest_sha256": qc["raw_manifest_sha256"]}
        raw_manifest_records = raw_manifest.to_dict("records")
        if not qc["passed"]:
            local["eligible"] = False
            local["errors"].append("retrospective raw-cache QC replay failed")
    except Exception as exc:  # noqa: BLE001 - report failed gate, preserve details
        local = {"eligible": False, "errors": [str(exc)], "summaries": {}}
        raw_manifest_records = []
    candidates = [assess_external(c, limits) for c in read_candidates(args.candidate_manifest)]
    candidates = sorted(candidates, key=lambda c: (c.get("huc8") or "", c["name"]))
    selected = next((c["name"] for c in candidates if c["eligible"]), None)
    status = "local_data_failed" if not local["eligible"] else (
        "pass" if selected else ("external_no_eligible_candidate" if candidates else "external_pending")
    )
    report = {
        "version": "phase4_transfer_data_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage1_status": status,
        "thresholds": limits,
        "local": local,
        "candidates": candidates,
        "selected_external": selected,
        "training_authorized": False,
        "selection_rule": "availability-only before model results are inspected; canonical HUC8 then name tie-break",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if local.get("eligible"):
        pd.DataFrame(raw_manifest_records).to_csv(out / "st357_raw_file_manifest.csv", index=False)
        pd.DataFrame(local["mask_availability"]).to_csv(out / "st357_mask_availability.csv", index=False)
        pd.DataFrame(local["task_availability"]).to_csv(out / "st357_kshot_availability.csv", index=False)
        (out / "st357_qc_replay.json").write_text(json.dumps(local["qc_replay"], indent=2) + "\n", encoding="utf-8")
        (out / "st357_task_aliases.json").write_text(json.dumps(local["task_aliases"], indent=2) + "\n", encoding="utf-8")
        (out / "st357_label_bindings.json").write_text(json.dumps(local["label_bindings"], indent=2) + "\n", encoding="utf-8")
        # Keep the observed-only long table inspectable; it is not a training view.
        unified_labels(_datasets, summaries, nodes).to_parquet(out / "st357_observed_labels.parquet", index=False)
    (out / "transfer_data_gate.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out / "transfer_data_gate.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"stage1_status": status, "selected_external": selected}))
    if status == "pass" or (status == "external_pending" and args.allow_external_pending):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
