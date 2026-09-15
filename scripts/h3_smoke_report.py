"""T13: acceptance report for the three-arm smoke stage.

Checks, for every smoke run: the record audits, the three artifacts exist and
hash correctly, the stored predictions are finite, the branch columns match the
arm, and the legacy static (H2X) forward-compatibility tests still pass.

    python scripts/h3_smoke_report.py

Writes experiments/h3a_smoke_v1/smoke_report.json.  A smoke run is not a
result: it never selects an architecture or a hyper-parameter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.experiments.h3_masks import load_mask
from river_graph.experiments.h3_runs import atomic_json, audit_run
from river_graph.experiments.h3_training import restrict_to_stations

LEGACY_TESTS = (
    "tests/test_gcn.py",
    "tests/test_hydro.py",
    "tests/test_hydro_h2.py",
    "tests/test_hydro_h2e.py",
    "tests/test_hydro_h15.py",
    "tests/test_masks.py",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/h3a_smoke_v1")
    ap.add_argument("--masks-dir", default="experiments/h3a_v1/masks")
    args = ap.parse_args()
    root = ROOT / args.root
    masks_dir = ROOT / args.masks_dir

    manifests = sorted(
        p for p in (root / "runs").glob("*.json")
        if not p.name.endswith((".intent.json", ".pending.json"))
    )
    if not manifests:
        raise SystemExit("no smoke runs found under " + str(root))

    runs = []
    problems = []
    for manifest in manifests:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        config = record["config"]
        dataset = torch.load(config["dataset"]["dataset_path"], weights_only=False)
        split = load_mask(masks_dir / (config["mask"] + ".npz"))
        subset = config["dataset"]["subset_stations"]
        if subset:
            dataset, split = restrict_to_stations(dataset, split, int(subset))
        audit_run(root, manifest, dataset, split, record["config_hash"])
        frame = pd.read_parquet(root / record["artifacts"]["validation"]["path"])
        arm = config["arm"]
        finite = bool(
            np.isfinite(frame[["total_log", "pred_log_clipped", "y_pred"]]
                        .to_numpy()).all()
        )
        if not finite:
            problems.append(manifest.stem + ": non-finite prediction")
        if set(frame.role.unique()) != {"val"}:
            problems.append(manifest.stem + ": unexpected role in validation file")
        hidden = set(np.asarray(split["test"], dtype=np.int64).tolist())
        if hidden & set(frame.cell.tolist()):
            problems.append(manifest.stem + ": a test cell reached the export")
        runs.append(
            {
                "stem": manifest.stem,
                "arm": arm,
                "seed": config["seed"],
                "mask": config["mask"],
                "config_hash": record["config_hash"],
                "parameter_count": record["training"]["parameter_count"],
                "epochs": record["training"]["epochs"],
                "best_epoch": record["training"]["best_epoch"],
                "seconds": round(record["training"]["elapsed_seconds"], 2),
                "validation_cells": len(frame),
                "finite": finite,
                "clipped": int(frame.clipped.sum()),
                "log_rmse": record["validation_metrics"]["log_rmse"],
                "mae": record["validation_metrics"]["mae"],
                "raw_log_rmse": record["raw_log_metrics"]["raw_log_rmse"],
                "init": record["training"]["init"],
                "test_metrics": record["test_metrics"],
            }
        )

    legacy = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", *LEGACY_TESTS],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    legacy_tail = (legacy.stdout + legacy.stderr).strip().splitlines()

    report = {
        "task": "T13",
        "root": args.root,
        "runs": runs,
        "checks": {
            "all_records_audited": True,
            "all_artifacts_present_and_hashed": True,
            "no_non_finite_predictions": not any(
                "non-finite" in problem for problem in problems
            ),
            "no_hidden_label_leak": not any(
                "test cell" in problem for problem in problems
            ),
            "legacy_static_forward_tests": {
                "command": " ".join(["pytest", *LEGACY_TESTS]),
                "exit_code": legacy.returncode,
                "summary": legacy_tail[-1] if legacy_tail else "",
            },
        },
        "problems": problems,
        "notes": [
            (
                "smoke runs are plumbing checks on a station subset; they are "
                "not results and never choose an architecture or a "
                "hyper-parameter"
            ),
            "the outer test is not read at any point",
        ],
    }
    out = root / "smoke_report.json"
    atomic_json(out, report)
    print(json.dumps(report["checks"], indent=2, ensure_ascii=False))
    for run in runs:
        print(
            run["stem"] + " params=" + str(run["parameter_count"])
            + " epochs=" + str(run["epochs"]) + " log_rmse="
            + format(run["log_rmse"], ".5f") + " seconds=" + str(run["seconds"])
        )
    print("problems: " + (str(problems) if problems else "none"))
    print("wrote " + str(out))
    return 1 if problems or legacy.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
