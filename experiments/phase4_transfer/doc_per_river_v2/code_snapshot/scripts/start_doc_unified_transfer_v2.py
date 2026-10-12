"""Prepare an immutable runtime and launch the unified model via run_ladder."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from river_graph.experiments.provenance import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY/"experiments/phase4_transfer/doc_unified_transfer_v2"
WORKSPACE = REPOSITORY/"cache/doc_unified_transfer_v2/runtime"
DISPATCHER = '"""Frozen unified DOC entry."""\nfrom run_doc_unified_transfer_v2 import main\n\nif __name__ == "__main__":\n    main()\n'


def prepare():
    marker = WORKSPACE/"workspace.json"
    if marker.exists():
        for name, expected in json.loads(marker.read_text())["source_hashes"].items():
            if sha256_file(REPOSITORY/name) != expected or sha256_file(WORKSPACE/name) != expected:
                raise ValueError(f"execution source changed: {name}")
        if (WORKSPACE/"scripts/run_ladder.py").read_text() != DISPATCHER:
            raise ValueError("dispatcher changed")
        return
    if WORKSPACE.exists():
        raise ValueError("unfinished runtime exists; inspect before resuming")
    WORKSPACE.mkdir(parents=True)
    hashes = {}
    for folder in ("src", "scripts", "tests"):
        shutil.copytree(REPOSITORY/folder, WORKSPACE/folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in (REPOSITORY/folder).rglob("*.py"):
            name = path.relative_to(REPOSITORY).as_posix()
            if "__pycache__" not in path.parts and name != "scripts/run_ladder.py":
                hashes[name] = sha256_file(path)
    shutil.copytree(REPOSITORY/"docs", WORKSPACE/"docs", ignore=shutil.ignore_patterns("*.png", "*.pdf"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(REPOSITORY/name, WORKSPACE/name)
        hashes[name] = sha256_file(REPOSITORY/name)
    for name in ("data", "experiments", ".venv"):
        (WORKSPACE/name).symlink_to(REPOSITORY/name, target_is_directory=True)
    (WORKSPACE/"scripts/run_ladder.py").write_text(DISPATCHER)
    marker.write_text(json.dumps({"repository": str(REPOSITORY), "source_hashes": hashes}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    prepare()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(WORKSPACE/"src")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "2"
    command = ["uv", "run", "--no-sync", "python", "scripts/run_ladder.py", "--repository", str(REPOSITORY),
               "--root", str(OUTPUT), "--workers", str(args.workers)]
    subprocess.run([*command, "--freeze-only"], cwd=WORKSPACE, env=environment, check=True)
    if args.freeze_only:
        return
    pid_path = OUTPUT/"training.pid"
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            print(json.dumps({"stage": "already_running"}))
            return
    with (OUTPUT/"training.log").open("a") as log:
        worker = subprocess.Popen(command, cwd=WORKSPACE, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    pid_path.write_text(str(worker.pid)+"\n")
    (OUTPUT/"launch.json").write_text(json.dumps({"pid": worker.pid, "command": command, "cwd": str(WORKSPACE),
        "protocol_sha256": sha256_file(OUTPUT/"protocol.json")}, indent=2)+"\n")
    print(json.dumps({"stage": "started", "pid": worker.pid, "root": str(OUTPUT)}))


if __name__ == "__main__":
    main()
