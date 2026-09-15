"""T01: record the implementation baseline before any H3 code exists.

Writes experiments/h3a_v1/preflight.json with:
  * the exact Git state (HEAD, tracked-file modifications, untracked paths) so
    user work in progress can never be confused with an H3 regression;
  * content hashes of every H2X-related source file and of the frozen inputs
    (dataset, masks, edge table), so "same name, different bytes" is detectable;
  * the baseline pytest result (which tests already fail before the change).

Re-running this script is cheap and must reproduce the same hashes as long as
neither the sources nor the data changed.

Usage: python scripts/h3_preflight.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = (
    # H2X / graph model stack
    "src/river_graph/models/gcn.py",
    "src/river_graph/models/hydro.py",
    "src/river_graph/dataset.py",
    # frozen benchmark infrastructure
    "src/river_graph/experiments/masks.py",
    "src/river_graph/experiments/evaluate.py",
    "src/river_graph/experiments/predictions.py",
    "src/river_graph/experiments/provenance.py",
    "src/river_graph/experiments/gate_runs.py",
    "src/river_graph/experiments/gate_audit.py",
    # entry points
    "scripts/run_gnn.py",
    "scripts/run_ladder.py",
    "scripts/generate_masks.py",
    "scripts/run_dynamic_gate.py",
    # tests that guard the frozen behaviour
    "tests/test_gcn.py",
    "tests/test_hydro.py",
    "tests/test_hydro_h2.py",
    "tests/test_hydro_h2e.py",
    "tests/test_hydro_h15.py",
    "tests/test_masks.py",
)

DATA_FILES = (
    "data/processed/mississippi_graph_v02.pt",
    "data/processed/mississippi_graph_v03.pt",
    "data/processed/mississippi_graph_v04.pt",
    "data/processed/mississippi_graph_v02_smoke50.pt",
    "data/processed/graph_edges.csv",
    "data/processed/graph_nodes.csv",
    "data/processed/edge_features.csv",
)

MASKS_DIR = "experiments/masks"

# The tests that pin the frozen static (H2X) forward path and the recent
# gate/trajectory work. Recorded separately so a regression in exactly these
# guards is visible at a glance.
BASELINE_TEST_TARGETS = (
    "tests/test_gcn.py",
    "tests/test_hydro.py",
    "tests/test_hydro_h2.py",
    "tests/test_hydro_h2e.py",
    "tests/test_hydro_h15.py",
    "tests/test_masks.py",
    "tests/test_predictions.py",
    "tests/test_dynamic_gate.py",
    "tests/test_gate_audit.py",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def file_record(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        return {"path": relative, "exists": False}
    stat = path.stat()
    return {
        "path": relative,
        "exists": True,
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_state() -> dict:
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    tracked_modified, untracked, staged = [], [], []
    for line in status.splitlines():
        code, _, path = line[:2], line[2], line[3:]
        path = path.strip().strip('"')
        if code == "??":
            untracked.append(path)
        else:
            (staged if code[0] not in " ?" else tracked_modified).append(path)
    return {
        "head": run_git("rev-parse", "HEAD").strip(),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "remotes": [
            line.split()[-1]
            for line in run_git("remote", "-v").splitlines()
            if line.endswith("(fetch)")
        ],
        "staged": sorted(staged),
        "tracked_modified": sorted(tracked_modified),
        "untracked": sorted(untracked),
        "clean": not status.strip(),
    }


def dependency_versions() -> dict:
    versions = {}
    for package in (
        "torch",
        "torch-geometric",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "networkx",
        "pytest",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_pytest(targets: list[str], timeout: int) -> dict:
    command = [sys.executable, "-m", "pytest", "-q", "--no-header",
               "-p", "no:cacheprovider", *targets]
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        check=False,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()
    return {
        "command": " ".join(command[1:]),
        "exit_code": result.returncode,
        "seconds": round(time.perf_counter() - started, 2),
        "summary": tail[-1] if tail else "",
        "output_tail": tail[-40:],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="experiments/h3a_v1/preflight.json")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the pytest baseline (hashes and Git state only)")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    os.chdir(ROOT)
    payload = {
        "task": "T01",
        "generated_at_epoch": time.time(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
        "git": git_state(),
        "dependencies": dependency_versions(),
        "source_sha256": {p: file_record(p) for p in SOURCE_FILES},
        "data_sha256": {p: file_record(p) for p in DATA_FILES},
        "mask_sha256": {
            p.name: file_record(str(p.relative_to(ROOT)).replace("\\", "/"))
            for p in sorted((ROOT / MASKS_DIR).glob("*.npz"))
        },
        "notes": [
            (
                "Git tracked_modified/untracked list the pre-existing user work "
                "in progress; H3 work is additive and must not revert or "
                "overwrite it."
            ),
            (
                "The pytest baseline records the pre-change state so a later "
                "failure can be attributed to the H3 change instead of a "
                "pre-existing problem."
            ),
        ],
    }
    if not args.skip_tests:
        payload["baseline_tests"] = run_pytest(list(BASELINE_TEST_TARGETS),
                                               args.timeout)
        payload["baseline_tests_full"] = run_pytest([], args.timeout)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temp, out)

    tests = payload.get("baseline_tests_full", {})
    print(f"wrote {out}")
    print(f"git clean={payload['git']['clean']} "
          f"tracked_modified={len(payload['git']['tracked_modified'])} "
          f"untracked={len(payload['git']['untracked'])}")
    if tests:
        print(f"baseline pytest: exit={tests['exit_code']} {tests['summary']}")


if __name__ == "__main__":
    main()
