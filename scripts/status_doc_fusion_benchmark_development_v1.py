"""Inspect benchmark process and source-side progress without opening test scores."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("experiments/phase4_transfer/doc_fusion_benchmark_development_v1")


def main():
    pid_file = ROOT / "training.pid"
    pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
    running = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            pass
    progress = ROOT / "progress.json"
    rows = []
    for region in ("1013", "1019", "0708", "1030", "1101"):
        run = ROOT / "regions" / f"huc4_{region}"
        rows.append({"region": region, "completed_candidate_fits": len(list((run / "fits").glob("*/*/seed*/fit_complete.json"))),
            "planned_candidate_fits": 21, "selection_frozen": (run / "selection_complete.json").exists(),
            "target_scored": (run / "test_complete.json").exists()})
    log = ROOT / "training.log"
    recent = log.read_text().splitlines()[-25:] if log.exists() else []
    errors = [line for line in recent if "Traceback" in line or "Error:" in line]
    print(json.dumps({"root": str(ROOT), "pid": pid, "running": running,
        "completed": (ROOT / "batch_complete.json").exists(),
        "completed_candidate_fits": sum(row["completed_candidate_fits"] for row in rows),
        "planned_candidate_fits": 105, "regions": rows,
        "progress": json.loads(progress.read_text()) if progress.exists() else {},
        "recent_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
