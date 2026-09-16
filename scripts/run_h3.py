"""T12: task list and run entry point for the H3-A round.

    python scripts/run_h3.py --dry-run --stage pilot
    python scripts/run_h3.py --stage smoke --workers 3
    python scripts/run_h3.py --stage pilot --mask e2b_partial --seed 0 --workers 3
    python scripts/run_h3.py --stage pilot --workers 6
    python scripts/run_h3.py --verify-only --stage pilot

Stages
------
smoke  3 short runs on a 150-station subset, at most 3 epochs each
       (not a result and never a basis for choosing an architecture)
pilot  ENV/H2X/H3A x the three development masks x seeds 0/1/2 = 27 runs
expand ENV/H2X/H3A x 8 key masks x seeds 0-4 = 120 runs, of which the 27
       pilot runs are reused by identity (T17, conditional on T16)

The outer test is never read here.  Test export has its own frozen entry
point (scripts/run_h3_frozen_eval.py) which stays locked until the pilot
decision promotes H3A.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.experiments.h3_runs import (
    SOURCE_FILES,
    require_decision,
    task_grid,
    train_and_store,
)
from river_graph.experiments.h3_training import load_protocol
from river_graph.experiments.provenance import sha256_file

DEFAULT_DATASET = "data/processed/mississippi_graph_v07.pt"
DEFAULT_PROTOCOL = "configs/h3a_v1.json"
STAGE_ROOTS = {
    "smoke": "experiments/h3a_smoke_v1r3",
    "pilot": "experiments/h3a_v1r3",
    "expand": "experiments/h3a_v1r3",
    # the T18 control shares the root: it is the same experiment family and its
    # records are distinguished by the arm, not by a separate directory
    "no_message": "experiments/h3a_v1r3",
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=("smoke", "pilot", "expand", "no_message"),
        default="pilot",
    )
    ap.add_argument("--root", default=None)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--masks-dir", default="experiments/h3a_v1r3/masks")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--arm", action="append", default=None,
                    help="restrict to an arm (repeatable)")
    ap.add_argument("--seed", action="append", type=int, default=None,
                    help="restrict to a training seed (repeatable)")
    ap.add_argument("--mask", action="append", default=None,
                    help="restrict to a mask name (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the exact pending and reusable configurations only")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-verify stored runs; never trains")
    ap.add_argument("--verify", action="store_true",
                    help="run the independent recomputation after training")
    return ap.parse_args(argv)


def check_inputs(args, protocol) -> None:
    dataset = ROOT / args.dataset
    if not dataset.is_file():
        raise SystemExit("dataset not found: " + str(dataset))
    masks_dir = ROOT / args.masks_dir
    if not masks_dir.is_dir():
        raise SystemExit(
            "masks not found: " + str(masks_dir)
            + " (run scripts/build_h3_masks.py first)"
        )
    for name in protocol["masks"]["key_scenarios"]:
        if not (masks_dir / (name + ".npz")).is_file():
            raise SystemExit("missing mask file: " + name)
    if protocol["protocol_version"] != json.loads(
        (ROOT / args.protocol).read_text(encoding="utf-8")
    )["protocol_version"]:
        raise SystemExit("protocol version mismatch")


def protocol_fingerprint(protocol) -> dict:
    return {
        "protocol_version": protocol["protocol_version"],
        "source_sha256": {
            path: sha256_file(ROOT / path)[:16] for path in SOURCE_FILES
        },
    }


def job(payload):
    arm, seed, stage, protocol, mask, masks_dir, dataset, root, cpu_threads = payload
    report = SimpleNamespace(trained=0, skipped=0, pending=0)
    record = train_and_store(
        arm,
        seed,
        stage,
        protocol,
        mask,
        Path(masks_dir),
        dataset,
        Path(root),
        cpu_threads=cpu_threads,
        report=report,
    )
    return arm, seed, mask, record.get("status"), report


def run_verify(args, root) -> int:
    from verify_h3 import main as verify_main

    argv = [
        "--root",
        str(Path(args.root or STAGE_ROOTS[args.stage]).as_posix()),
        "--protocol",
        args.protocol,
        "--stage",
        args.stage,
        "--masks-dir",
        args.masks_dir,
    ]
    if args.arm:
        if len(args.arm) != 1:
            raise SystemExit("--verify-only accepts at most one --arm")
        argv += ["--arm", args.arm[0]]
    if args.seed:
        if len(args.seed) != 1:
            raise SystemExit("--verify-only accepts at most one --seed")
        argv += ["--seed", str(args.seed[0])]
    if args.mask:
        if len(args.mask) != 1:
            raise SystemExit("--verify-only accepts at most one --mask")
        argv += ["--mask", args.mask[0]]
    original = sys.argv
    try:
        sys.argv = ["verify_h3.py", *argv]
        return verify_main()
    finally:
        sys.argv = original


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    protocol = load_protocol(ROOT / args.protocol)
    check_inputs(args, protocol)
    root = Path(args.root or STAGE_ROOTS[args.stage])
    if not root.is_absolute():
        root = ROOT / root
    masks_dir = ROOT / args.masks_dir

    if args.verify_only:
        return run_verify(args, root)

    grid = task_grid(args.stage, protocol, args.arm, args.seed, args.mask)
    # Stage gates: expansion needs the pilot decision, and the T18 no-message
    # control needs the expansion decision.  A dry run is a read-only listing,
    # so it reports the gate instead of being blocked by it.
    gates = []
    if args.stage == "expand":
        gates.append(("pilot", "run the T17 expansion grid"))
    if args.stage == "no_message":
        gates.append(("expand", "run the T18 no-message control"))
    for kind, action in gates:
        try:
            require_decision(root, kind, action)
        except SystemExit as exc:
            if args.dry_run:
                print("GATE: " + str(exc), flush=True)
                return 1
            raise
    fingerprint = protocol_fingerprint(protocol)
    print(
        "stage=" + args.stage + " tasks=" + str(len(grid)) + " root="
        + str(root) + " workers=" + str(args.workers),
        flush=True,
    )
    print("protocol=" + json.dumps(fingerprint, ensure_ascii=False), flush=True)

    if args.dry_run:
        report = SimpleNamespace(trained=0, skipped=0, pending=0)
        for arm, seed, mask in grid:
            train_and_store(
                arm, seed, args.stage, protocol, mask, masks_dir, args.dataset,
                root, cpu_threads=args.cpu_threads, dry_run=True, report=report,
            )
        print(
            "dry-run: pending=" + str(report.pending) + " reusable="
            + str(report.skipped) + " (no training performed)",
            flush=True,
        )
        return 0

    reports = SimpleNamespace(trained=0, skipped=0, pending=0)
    if args.workers == 1:
        for arm, seed, mask in grid:
            result = job((arm, seed, args.stage, protocol, mask, str(masks_dir),
                          args.dataset, str(root), args.cpu_threads))
            reports.trained += result[4].trained
            reports.skipped += result[4].skipped
    else:
        payloads = [
            (arm, seed, args.stage, protocol, mask, str(masks_dir),
             args.dataset, str(root), args.cpu_threads)
            for arm, seed, mask in grid
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(job, payload): payload[:3] for payload in payloads}
            try:
                for done in as_completed(futures):
                    arm, seed, mask, status, report = done.result()
                    reports.trained += report.trained
                    reports.skipped += report.skipped
                    print(
                        "FINISHED " + str(arm) + " seed=" + str(seed) + " "
                        + str(mask) + " status=" + str(status),
                        flush=True,
                    )
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                print("interrupted; completed runs stay on disk and are reused",
                      flush=True)
                return 130
    print(
        "trained=" + str(reports.trained) + " reused=" + str(reports.skipped),
        flush=True,
    )
    if args.verify:
        return run_verify(args, root)
    print("verify with: python scripts/verify_h3.py --stage " + args.stage, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
