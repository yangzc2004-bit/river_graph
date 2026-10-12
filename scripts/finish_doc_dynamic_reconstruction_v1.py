"""Finish the authorized one-off experiment after its existing trainer completes."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from river_graph.experiments.provenance import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "experiments/phase4_transfer/doc_dynamic_reconstruction_v1"


def main():
    status = ROOT / "postprocessing_status.json"
    while not (ROOT / "batch_complete.json").exists():
        pid = int((ROOT / "training.pid").read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError("Trainer exited before completing its batch; inspect resource_training.log") from error
        time.sleep(30)
    for step, script in (("independent_verification", "verify_doc_dynamic_reconstruction_v1.py"),
                         ("report", "report_doc_dynamic_reconstruction_v1.py")):
        status.write_text(json.dumps({"stage": step, "script": script}, indent=2) + "\n")
        subprocess.run(["uv", "run", "--no-sync", "python", f"scripts/{script}"], cwd=REPOSITORY, check=True)
    result = {"stage": "complete", "independent_audit_passed": True,
              "summary_sha256": sha256_file(ROOT / "analysis/summary.csv"),
              "report_sha256": sha256_file(REPOSITORY / "docs/doc_dynamic_reconstruction_results_20261012.md"),
              "script_sha256": sha256_file(__file__)}
    status.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
