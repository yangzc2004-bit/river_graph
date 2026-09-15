"""T12/T19: the separate frozen evaluation entry point (outer test export).

This is the only script allowed to read the outer test.  It is locked until
experiments/h3a_v1/pilot_decision.json records a promotion, and it never
trains: it reloads the frozen best weights of already verified runs, rebuilds
the predictions under the fully-open visibility arrangement, and reports the
outer test per arm / mask / seed.

    python scripts/run_h3_frozen_eval.py --stage expand --dry-run
    python scripts/run_h3_frozen_eval.py --stage expand

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
    ROOT as H3_ROOT,  # noqa: F401
)
from river_graph.experiments.h3_runs import (
    atomic_json,
    audit_run,
)
from river_graph.experiments.h3_training import (
    H3Trainer,
    load_protocol,
    restrict_to_stations,
)
from river_graph.experiments.provenance import sha256_file

DEFAULT_ROOT = "experiments/h3a_v1"
DECISION = "pilot_decision.json"
REQUIRED_OUTCOME = "promote"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--protocol", default="configs/h3a_v1.json")
    ap.add_argument("--masks-dir", default="experiments/h3a_v1/masks")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--seed", action="append", type=int, default=None)
    ap.add_argument("--mask", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def decision_gate(root: Path) -> dict:
    path = root / DECISION
    if not path.is_file():
        raise SystemExit(
            "frozen evaluation is locked: " + str(path)
            + " does not exist. The outer test may only be exported after the "
            "T16 pilot decision promotes H3A."
        )
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("outcome") != REQUIRED_OUTCOME:
        raise SystemExit(
            "frozen evaluation is locked: the pilot decision outcome is "
            + repr(decision.get("outcome")) + ", not " + repr(REQUIRED_OUTCOME)
        )
    return decision


def select_records(root: Path, args) -> list[tuple[dict, Path]]:
    records = []
    for path in sorted((root / "runs").glob("*.json")):
        if path.name.endswith((".intent.json", ".pending.json")):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        config = record["config"]
        if args.arm and config["arm"] not in args.arm:
            continue
        if args.seed and config["seed"] not in args.seed:
            continue
        if args.mask and config["mask"] not in args.mask:
            continue
        records.append((record, path))
    return records


def rebuild(record: dict, protocol: dict, masks_dir: Path) -> tuple[dict, dict]:
    config = record["config"]
    dataset = torch.load(config["dataset"]["dataset_path"], weights_only=False)
    split = load_mask(masks_dir / (config["mask"] + ".npz"))
    if config["dataset"]["subset_stations"]:
        dataset, split = restrict_to_stations(
            dataset, split, int(config["dataset"]["subset_stations"])
        )
    return dataset, split


def main(argv=None) -> int:
    args = parse_args(argv)
    root = ROOT / args.root
    masks_dir = ROOT / args.masks_dir
    protocol = load_protocol(ROOT / args.protocol)
    decision = decision_gate(root)
    records = select_records(root, args)
    if not records:
        print("no verified runs matched the frozen evaluation filter")
        return 1
    print(
        "frozen evaluation unlocked by decision outcome=" + decision["outcome"]
        + "; " + str(len(records)) + " runs",
        flush=True,
    )
    if args.dry_run:
        for record, path in records:
            print("would export " + path.stem, flush=True)
        return 0

    out_dir = Path(args.out) if args.out else root / "frozen_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record, manifest in records:
        dataset, split = rebuild(record, protocol, masks_dir)
        audit_run(root, manifest, dataset, split, record["config_hash"])
        config = record["config"]
        trainer = H3Trainer(config["arm"], int(config["seed"]), protocol)
        trainer.prepare(dataset, split)
        trainer.build_model()
        checkpoint = torch.load(
            root / record["artifacts"]["checkpoint"]["path"], weights_only=False
        )
        trainer.model.load_state_dict(checkpoint["state_dict"])
        grid = trainer.predict_grid(role="full")

        truth = dataset["y"].numpy()
        role = np.full(truth.size, "", dtype=object)
        cells = np.asarray(split["test"], dtype=np.int64)
        role[cells] = "test"
        role = role.reshape(truth.shape)
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
        path = out_dir / (manifest.stem + "__test.parquet")
        temp = path.with_name(path.name + ".tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(path)
        row = {
            "arm": config["arm"],
            "seed": config["seed"],
            "mask": config["mask"],
            "scenario": config["scenario"],
            "cells": len(cells),
            "clipped_fraction": float(grid["clipped"].ravel()[cells].mean()),
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
            "root": args.root,
            "runs": len(rows),
            "masks_dir": args.masks_dir,
            "protocol_version": protocol["protocol_version"],
            "decision": decision,
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
