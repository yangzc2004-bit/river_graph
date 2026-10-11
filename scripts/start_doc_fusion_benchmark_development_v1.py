"""Prepare an isolated run_ladder workspace without changing historical hashes.

Start only after tests pass. The worker writes a new versioned protocol,
predictions, and provenance. Existing training code and results stay intact.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from river_graph.experiments.provenance import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "experiments/phase4_transfer/doc_fusion_benchmark_development_v1"
WORKSPACE = REPOSITORY / "cache/doc_fusion_benchmark_development_v1/runtime"
DISPATCHER = '''"""Versioned training entry point for the matched-input DOC development benchmark.

The benchmark freezes its protocol before fitting and refuses changed prediction
provenance on resumption. The original repository run_ladder is unchanged.
"""
from run_doc_fusion_benchmark_development_v1 import main

if __name__ == "__main__":
    main()
'''


def prepare(repository=REPOSITORY, workspace=WORKSPACE):
    marker = workspace / "workspace.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        for name, expected in record["source_hashes"].items():
            if sha256_file(repository / name) != expected or sha256_file(workspace / name) != expected:
                raise ValueError(f"source changed after workspace preparation: {name}")
        if (workspace / "scripts/run_ladder.py").read_text() != DISPATCHER:
            raise ValueError("isolated training dispatcher changed")
        return workspace
    if workspace.exists():
        raise ValueError("unfinished workspace exists; inspect it before preparing a new version")
    workspace.mkdir(parents=True)
    hashes = {}
    for folder in ("src", "scripts", "tests"):
        shutil.copytree(repository / folder, workspace / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for folder in ("src", "scripts", "tests"):
        for path in (repository / folder).rglob("*.py"):
            if "__pycache__" not in path.parts and path.relative_to(repository).as_posix() != "scripts/run_ladder.py":
                hashes[path.relative_to(repository).as_posix()] = sha256_file(path)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(repository / name, workspace / name)
        hashes[name] = sha256_file(repository / name)
    for name in ("data", "experiments", ".venv"):
        (workspace / name).symlink_to(repository / name, target_is_directory=True)
    (workspace / "scripts/run_ladder.py").write_text(DISPATCHER)
    record = {"repository": str(repository), "source_hashes": hashes,
        "original_dispatcher_sha256": sha256_file(repository / "scripts/run_ladder.py"),
        "isolated_dispatcher_sha256": sha256_file(workspace / "scripts/run_ladder.py")}
    marker.write_text(json.dumps(record, indent=2) + "\n")
    return workspace


def command(workspace, output, repository, threads, *, freeze_only=False):
    args = ["uv", "run", "--no-sync", "python", "scripts/run_ladder.py",
            "--repository", str(repository), "--root", str(output), "--threads", str(threads)]
    if freeze_only:
        args.append("--freeze-only")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(workspace / "src")
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    environment["OPENBLAS_NUM_THREADS"] = str(threads)
    return args, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("positive thread count required")
    workspace = prepare()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        print(json.dumps({"stage": "workspace_prepared", "workspace": str(workspace)}))
        return
    # Freeze and verify dependencies synchronously before detaching a worker.
    freeze_args, environment = command(workspace, OUTPUT, REPOSITORY, args.threads, freeze_only=True)
    subprocess.run(freeze_args, cwd=workspace, env=environment, check=True)
    pid_path = OUTPUT / "training.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            print(json.dumps({"stage": "already_running", "pid": pid, "root": str(OUTPUT)}))
            return
    run_args, environment = command(workspace, OUTPUT, REPOSITORY, args.threads)
    with (OUTPUT / "training.log").open("a") as log:
        process = subprocess.Popen(run_args, cwd=workspace, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(process.pid) + "\n")
    (OUTPUT / "launch.json").write_text(json.dumps({"pid": process.pid, "command": run_args,
        "cwd": str(workspace), "log": str(OUTPUT / "training.log"),
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "original_dispatcher_sha256": sha256_file(REPOSITORY / "scripts/run_ladder.py")}, indent=2) + "\n")
    print(json.dumps({"stage": "started", "pid": process.pid, "root": str(OUTPUT)}))


if __name__ == "__main__":
    main()
