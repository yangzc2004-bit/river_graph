"""Read-only progress inspection without opening target DOC predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("experiments/phase4_transfer/doc_fusion_component_comparison_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    regions = []
    for run in sorted((ROOT / "regions").glob("huc4_*")):
        path = run / "progress.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        log = ROOT / f"worker_{run.name.removeprefix('huc4_')}.log"
        stage_records = []
        error = None
        if log.exists():
            for line in log.read_text().splitlines():
                if line.startswith("{"):
                    item = json.loads(line)
                    if "stage_complete" in item:
                        stage_records.append(item)
                elif "Error" in line or "Traceback" in line:
                    error = line
        regions.append({"region": run.name,
            "completed_fits": len(list((run / "fits").glob("*/seed*/fit_complete.json"))),
            "finished": (run / "complete.json").exists(),
            "current_specification": record.get("specification"), "seed": record.get("seed"),
            "epoch": record.get("epoch"), "stages": stage_records, "error": error})
    if args.compact:
        print(f"Fits: {sum(row['completed_fits'] for row in regions)}/165")
        for row in regions:
            stages = {stage["stage_complete"]: stage["selected"] for stage in row["stages"]}
            print(f"{row['region']}: {row['completed_fits']}/33, seed {row['seed']}, "
                  f"epoch {row['epoch']}, {row['current_specification']}, selected {stages}, "
                  f"complete {row['finished']}, error {row['error']}")
    else:
        print(json.dumps({"completed_fits": sum(row["completed_fits"] for row in regions),
                          "planned_fits": 165, "regions": regions}, indent=2))


if __name__ == "__main__":
    main()
