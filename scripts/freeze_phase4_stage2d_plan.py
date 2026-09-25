"""Freeze the conditional-fusion feasibility matrix without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from river_graph.experiments.conditional_fusion import (
    FEATURE_COLUMNS,
    fixed_weight_grid,
)
from river_graph.experiments.transfer import ANALYTES, DATASETS, file_hash, object_hash

TASKS = Path("experiments/phase4_transfer/cross_basin_tasks_v1/manifest.json")
SPEC = Path("experiments/phase4_transfer/stage2d_conditional_fusion_v1_spec.md")
OUT = Path("experiments/phase4_transfer/stage2d_conditional_fusion_v1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "execution_plan.json"
    if plan_path.exists() and not args.force:
        raise SystemExit(f"{plan_path} exists; use --force to regenerate")
    manifest = json.loads(TASKS.read_text(encoding="utf-8"))
    basins = sorted({str(t["basin"]) for t in manifest["tasks"]})
    seeds = sorted({int(t["seed"]) for t in manifest["tasks"]})
    units = []
    for analyte in ANALYTES:
        for basin in basins:
            for seed in seeds:
                units.append({"analyte": analyte, "basin": basin, "seed": seed})
    plan = {
        "version": "phase4_stage2d_conditional_fusion_v1",
        "status": "plan_only_pending_protocol_review",
        "query_labels_used": False,
        "task_manifest_sha256": file_hash(TASKS),
        "spec_sha256": file_hash(SPEC),
        "task_manifest": str(TASKS),
        "spec": str(SPEC),
        "datasets": {a: {"path": DATASETS[a], "sha256": file_hash(DATASETS[a])} for a in ANALYTES},
        "analytes": list(ANALYTES),
        "basins": basins,
        "seeds": seeds,
        "k_values": [0, 1, 3, 5],
        "candidate_ladder": [
            "fixed_weight_grid",
            "disagreement_gate",
            "support_quality_gate",
            "small_logistic_or_isotonic_gate",
        ],
        "fixed_weight_grid": fixed_weight_grid().tolist(),
        "feature_columns": list(FEATURE_COLUMNS),
        "units": units,
        "n_units": len(units),
        "training_started": False,
    }
    plan["plan_sha256"] = object_hash(plan)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (out / "STATUS.md").write_text(
        "# Stage-2D conditional-fusion feasibility\n\n"
        "Plan-only freeze. No target query labels have been opened and no new "
        "model has been trained. The next executable step is a source-only "
        "pseudo-target gate calibration under the candidate ladder.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": plan["status"], "units": len(units), "plan": str(plan_path)}))


if __name__ == "__main__":
    main()
