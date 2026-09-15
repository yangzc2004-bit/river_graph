"""T12/T19: the separate frozen evaluation entry point (outer test export).

This is the only script allowed to read the outer test.  It is locked until
the T17 expansion decision records a promotion AND the exact run set that
decision was taken on is still present, complete and identical:

  * the recorded artifact manifest (per-run config hash and checkpoint hash)
    must match a manifest recomputed from the run records on disk;
  * the whole 8-scenario x 5-seed x 3-arm grid must be present;
  * every record must pass the strict identity check.

A pilot promotion alone is not enough, and a partially filled expansion grid
is not enough.  A river-network contribution claim additionally requires the
T18 no-message control, so it must be requested explicitly.

    python scripts/run_h3_frozen_eval.py --dry-run
    python scripts/run_h3_frozen_eval.py
    python scripts/run_h3_frozen_eval.py --claim network

The outer test here is a development benchmark that has been inspected
repeatedly during earlier work; it is not a fresh confirmation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import load_mask
from river_graph.experiments.h3_runs import (
    artifact_entries,
    artifact_manifest_hash,
    atomic_json,
    audit_run,
    expected_config_for,
    require_decision,
    task_grid,
)
from river_graph.experiments.h3_training import (
    H3Trainer,
    load_protocol,
    restrict_to_stations,
)
from river_graph.experiments.provenance import sha256_file

NO_MESSAGE_ARM = "h3a_no_message"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--protocol", default="configs/h3a_v1.json")
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--seed", action="append", type=int, default=None)
    ap.add_argument("--mask", action="append", default=None)
    ap.add_argument("--claim", choices=("architecture", "network"),
                    default="architecture",
                    help="a network claim additionally requires the T18 arm")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def collect(root: Path) -> dict:
    records = {}
    for path in sorted((root / "runs").glob("*.json")):
        if path.name.endswith((".intent.json", ".pending.json")):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        record["_manifest"] = str(path)
        key = (record["config"]["arm"], int(record["config"]["seed"]),
               record["config"]["mask"])
        if key in records:
            raise SystemExit("duplicate run record for " + str(key))
        records[key] = record
    return records


def unlock(root: Path, protocol: dict, claim: str) -> dict:
    """Every condition that must hold before the outer test may be read."""
    decision = require_decision(root, "expand", "export the outer test")
    if not decision.get("complete"):
        raise SystemExit(
            "refusing to export the outer test: the expansion decision is not "
            "marked complete"
        )
    records = collect(root)
    grid = [(arm, int(seed), mask)
            for arm, seed, mask in task_grid("expand", protocol)]
    missing = sorted(set(grid) - set(records))
    if missing:
        raise SystemExit(
            "refusing to export the outer test: " + str(len(missing))
            + " expansion configurations are missing, e.g. " + str(missing[:5])
        )
    allowed = set(grid) | {(NO_MESSAGE_ARM, seed, mask) for _, seed, mask in grid}
    extra = sorted(set(records) - allowed)
    if extra:
        raise SystemExit(
            "refusing to export the outer test: unexpected runs present "
            + str(extra[:5])
        )
    entries = artifact_entries(
        [(records[key], Path(records[key]["_manifest"])) for key in grid]
    )
    digest = artifact_manifest_hash(entries)
    if digest != decision.get("artifact_manifest_sha256"):
        raise SystemExit(
            "refusing to export the outer test: the run set on disk no longer "
            "matches the expansion decision (manifest "
            + str(decision.get("artifact_manifest_sha256"))[:12] + " vs "
            + digest[:12] + ")"
        )
    if claim == "network":
        have = [key for key in records if key[0] == NO_MESSAGE_ARM]
        if len(have) != len(grid):
            raise SystemExit(
                "refusing a network-contribution claim: the T18 no-message "
                "control is incomplete (" + str(len(have)) + "/"
                + str(len(grid)) + ")"
            )
    return {
        "decision": decision,
        "records": records,
        "grid": grid,
        "manifest_sha256": digest,
    }


def export_one(record: dict, root: Path, protocol: dict, masks_dir: Path) -> dict:
    config = record["config"]
    dataset = torch.load(config["dataset"]["dataset_path"], weights_only=False)
    split = load_mask(masks_dir / (config["mask"] + ".npz"))
    if config["dataset"]["subset_stations"]:
        dataset, split = restrict_to_stations(
            dataset, split, int(config["dataset"]["subset_stations"])
        )
    expected, expected_hash = expected_config_for(record, protocol, masks_dir)
    audit_run(root, Path(record["_manifest"]), dataset, split,
              expected_hash=expected_hash, expected_config=expected,
              strict_identity=True)

    trainer = H3Trainer(config["arm"], int(config["seed"]), protocol)
    trainer.prepare(dataset, split)
    trainer.build_model()
    checkpoint = torch.load(
        root / record["artifacts"]["checkpoint"]["path"], weights_only=False
    )
    trainer.model.load_state_dict(checkpoint["state_dict"])
    grid = trainer.predict_grid(role="full")

    truth = dataset["y"].numpy()
    cells = np.asarray(split["test"], dtype=np.int64)
    ntime = truth.shape[1]
    frame = pd.DataFrame(
        {
            "station": [dataset["site_no"][int(c) // ntime] for c in cells],
            "month": [str(dataset["months"][int(c) % ntime]) for c in cells],
            "y_true": truth.ravel()[cells],
            "y_pred": grid["y_pred"].ravel()[cells],
            "total_log": grid["total_log"].ravel()[cells],
            "pred_log_clipped": grid["pred_log_clipped"].ravel()[cells],
            "clipped": grid["clipped"].ravel()[cells],
        }
    )
    for key in ("arm", "seed", "mask", "scenario"):
        frame[key] = config[key]
    return {"frame": frame, "cells": cells, "grid": grid, "config": config}


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(ROOT / args.protocol)
    root = ROOT / (args.root or protocol["result_root"])
    masks_dir = (
        Path(args.masks_dir) if args.masks_dir else ROOT / protocol["masks"]["dir"]
    )
    unlocked = unlock(root, protocol, args.claim)
    records = unlocked["records"]
    selected = [
        key for key in unlocked["grid"]
        if (not args.arm or key[0] in args.arm)
        and (not args.seed or key[1] in args.seed)
        and (not args.mask or key[2] in args.mask)
    ]
    print(
        "frozen evaluation unlocked: claim=" + args.claim
        + " expansion decision outcome=" + unlocked["decision"]["outcome"]
        + " manifest=" + unlocked["manifest_sha256"][:12]
        + " runs=" + str(len(selected)),
        flush=True,
    )
    if args.dry_run:
        for arm, seed, mask in selected:
            print("would export " + arm + " s" + str(seed) + " " + mask, flush=True)
        return 0

    out_dir = Path(args.out) if args.out else root / "frozen_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, seed, mask in selected:
        record = records[(arm, seed, mask)]
        item = export_one(record, root, protocol, masks_dir)
        frame = item["frame"]
        path = out_dir / (Path(record["_manifest"]).stem + "__test.parquet")
        temp = path.with_name(path.name + ".tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(path)
        cells = item["cells"]
        row = {
            "arm": arm,
            "seed": seed,
            "mask": mask,
            "scenario": record["config"]["scenario"],
            "cells": len(cells),
            "clipped_cells": int(item["grid"]["clipped"].ravel()[cells].sum()),
            "clipped_fraction": float(item["grid"]["clipped"].ravel()[cells].mean()),
            **metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy()),
            "predictions": path.relative_to(ROOT).as_posix(),
            "predictions_sha256": sha256_file(path),
        }
        rows.append(row)
        print("exported " + path.name, flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "test_per_run.csv", index=False)
    summary = (
        table.groupby(["arm", "scenario"])[
            ["mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2", "pbias"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "test_summary.csv", index=False)
    atomic_json(
        out_dir / "frozen_eval_manifest.json",
        {
            "root": protocol["result_root"],
            "runs": len(rows),
            "claim": args.claim,
            "protocol_version": protocol["protocol_version"],
            "protocol_revision": protocol.get("protocol_revision"),
            "expansion_decision": unlocked["decision"]["outcome"],
            "artifact_manifest_sha256": unlocked["manifest_sha256"],
            "no_message_arm_included": bool(
                any(row["arm"] == NO_MESSAGE_ARM for row in rows)
            ),
            "caveat": (
                "the outer test has been inspected repeatedly during earlier "
                "development work; treat it as a development benchmark, not a "
                "fresh confirmation set"
            ),
        },
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
