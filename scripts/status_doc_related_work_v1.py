"""Read official-backbone training status without opening target scores."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("experiments/phase4_transfer/doc_related_work_v1")


def main():
    pid_path = ROOT / "training.pid"
    pid = int(pid_path.read_text()) if pid_path.exists() else None
    running = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            pass
    regions = []
    for region in ("1013", "1019", "0708", "1030", "1101"):
        path = ROOT / "regions" / f"huc4_{region}"
        regions.append({"region": region,
            "completed_candidate_fits": len(list((path / "fits").glob("*/*/seed*/fit_complete.json"))),
            "planned_candidate_fits": 12, "selection_frozen": (path / "selection_complete.json").exists(),
            "target_scored": (path / "test_complete.json").exists()})
    log = ROOT / "training.log"
    recent = log.read_text().splitlines()[-30:] if log.exists() else []
    progress = ROOT / "progress.json"
    print(json.dumps({"root": str(ROOT), "pid": pid, "running": running,
        "completed": (ROOT / "batch_complete.json").exists(), "planned_candidate_fits": 60,
        "completed_candidate_fits": sum(row["completed_candidate_fits"] for row in regions),
        "regions": regions, "progress": json.loads(progress.read_text()) if progress.exists() else {},
        "recent_errors": [line for line in recent if "Traceback" in line or "Error:" in line]}, indent=2))


if __name__ == "__main__":
    main()
