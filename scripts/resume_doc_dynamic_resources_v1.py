"""Resource-only execution amendment; retain frozen models, masks and selection."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from run_doc_dynamic_reconstruction_v1 import (
    CODE,
    DATASET,
    ROOT,
    analyze,
    choose,
    config,
    optional_dependencies,
    score_task,
)
from run_doc_fusion_benchmark_development_v1 import utc_now, verify
from run_doc_river_architecture_comparison_v1 import digest, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY / "cache/doc_dynamic_reconstruction_v1/runtime"
AMENDMENT = ROOT / "resource_amendment.json"


def select_pending(pending, neural_arms, active_neural, max_neural):
    return next((i for i, task in enumerate(pending)
                 if task["arm"] not in neural_arms or active_neural < max_neural), None)


def freeze(workers, max_neural):
    assert not (ROOT / "all_selections.json").exists(), "No amendment after target scoring starts"
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for name in CODE:
        if name == "scripts/run_ladder.py":
            continue
        assert sha256_file(name) == protocol["runtime_snapshot"][name], name
    commands = subprocess.check_output(["ps", "-axo", "pid,ppid,command"], text=True)
    adopted = []
    parents = []
    for line in commands.splitlines()[1:]:
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid, _parent, command = fields
        if "scripts/run_ladder.py" not in command or str(ROOT.resolve()) not in command:
            continue
        if "--task " in command:
            task_id = command.split("--task ", 1)[1].split()[0]
            task = next(t for t in protocol["tasks"] if t["id"] == task_id)
            adopted.append({"pid": int(pid), "task": task})
        elif "uv run" not in command:
            parents.append(int(pid))
    assert len(parents) == 1 and len(adopted) <= protocol["workers"], (parents, adopted)
    ledger = {"frozen_at": utc_now(), "protocol_sha256": sha256_file(ROOT / "protocol.json"),
              "protocol_hash": digest(protocol), "old_workers": protocol["workers"],
              "workers": workers, "max_simultaneous_neural": max_neural,
              "per_task_threads": 1, "adopted_workers": adopted, "old_orchestrators": parents,
              "completed_fits_at_amendment": len(list((ROOT / "fits").glob("*/selection_complete.json"))),
              "script_sha256": sha256_file(__file__),
              "test_sha256": sha256_file("tests/test_doc_dynamic_resources_v1.py"),
              "reason": "Increase independent fit concurrency on ten logical CPUs; retain four-neural memory cap. Original freeze, predictions and training/selection code remain immutable.",
              "scientific_settings_changed": False, "target_scoring_started": False}
    assert not AMENDMENT.exists(), "Amendment already frozen"
    write_json(AMENDMENT, ledger)
    (ROOT / "code_snapshot/scripts/resume_doc_dynamic_resources_v1.py").write_bytes(Path(__file__).read_bytes())
    for pid in parents:
        # Workers have their own Python processes and log descriptors; only the
        # dispatch loop is replaced, and already-running fits continue.
        os.kill(pid, signal.SIGTERM)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-neural", type=int, default=4)
    args = parser.parse_args()
    if args.freeze_only:
        print(json.dumps(freeze(args.workers, args.max_neural)))
        return
    amendment = json.loads(AMENDMENT.read_text())
    assert sha256_file(__file__) == amendment["script_sha256"]
    assert sha256_file(ROOT / "protocol.json") == amendment["protocol_sha256"]
    protocol = json.loads((ROOT / "protocol.json").read_text())
    adopted = {item["task"]["id"]: item for item in amendment["adopted_workers"]}
    pending = []
    for task in protocol["tasks"]:
        directory = ROOT / "fits" / task["id"]
        if (directory / "selection_complete.json").exists():
            verify(directory, "selection_complete.json", config(protocol, task))
        elif task["id"] not in adopted:
            pending.append(task)
    running = [{**item, "process": None, "log": None} for item in adopted.values()
               if not (ROOT / "fits" / item["task"]["id"] / "selection_complete.json").exists()]
    while pending or running:
        kept = []
        for item in running:
            task = item["task"]
            directory = ROOT / "fits" / task["id"]
            if item["process"] is None:
                if (directory / "selection_complete.json").exists():
                    verify(directory, "selection_complete.json", config(protocol, task))
                    continue
                status = subprocess.check_output(["ps", "-p", str(item["pid"]), "-o", "stat="], text=True).strip()
                if status.startswith("Z"):
                    raise RuntimeError(f"adopted worker failed: {task['id']}")
                kept.append(item)
            elif item["process"].poll() is None:
                kept.append(item)
            else:
                item["log"].close()
                assert item["process"].returncode == 0, task["id"]
                verify(directory, "selection_complete.json", config(protocol, task))
        running = kept
        while pending and len(running) < amendment["workers"]:
            count = sum(i["task"]["arm"] in protocol["neural_arms"] for i in running)
            index = select_pending(pending, protocol["neural_arms"], count, amendment["max_simultaneous_neural"])
            if index is None:
                break
            task = pending.pop(index)
            log = (ROOT / "worker_logs" / f"{task['id']}.log").open("a")
            process = subprocess.Popen([sys.executable, "scripts/run_ladder.py", "--root", str(ROOT.resolve()), "--task", task["id"]], cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
            running.append({"task": task, "process": process, "pid": process.pid, "log": log})
        time.sleep(.5)
    optional_dependencies()
    torch.set_num_threads(1)
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    with np.load(protocol["daily"], allow_pickle=False) as archive:
        daily = archive["full"].copy()
    choose(ROOT, protocol)
    write_json(ROOT / "all_selections.json", {"protocol_hash": digest(protocol),
               "fit_receipts": {t["id"]: sha256_file(ROOT / "fits" / t["id"] / "selection_complete.json") for t in protocol["tasks"]},
               "selected_recipe_sha256": sha256_file(ROOT / "selected_recipe.json"),
               "resource_amendment_sha256": sha256_file(AMENDMENT), "test_used": False, "frozen_at": utc_now()})
    with threadpool_limits(limits=1):
        for task in protocol["tasks"]:
            score_task(ROOT, task, protocol, data, daily)
    analyze(ROOT, protocol)
    write_json(ROOT / "batch_complete.json", {"protocol_hash": digest(protocol), "tasks": len(protocol["tasks"]),
               "resource_amendment_sha256": sha256_file(AMENDMENT), "completed_at": utc_now()})
    print(json.dumps({"stage": "complete", "tasks": len(protocol["tasks"])}), flush=True)


if __name__ == "__main__":
    main()
