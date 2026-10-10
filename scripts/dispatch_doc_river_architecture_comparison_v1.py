"""Run disjoint late-region packages in an isolated worker, then adopt atomically.

No model or protocol is changed. The main run_ladder process verifies and skips
these completed packages when it reaches them. A started canonical package is
never overwritten; two processes never write the same progress file.
"""
from __future__ import annotations

import json
import subprocess

from run_doc_river_architecture_comparison_v1 import (
    ROOT,
    digest,
    freeze,
    verify_run,
    write_json,
)


def main():
    import torch

    protocol = json.loads((ROOT / "protocol.json").read_text())
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    worker = ROOT / "late_region_worker"
    if freeze(worker, protocol["settings"]) != protocol:
        raise ValueError("worker must use exactly the frozen canonical protocol")
    adopted = []
    for region in ("1101", "1030"):
        for seed in protocol["seeds"]:
            name = f"huc4_{region}_seed{seed}"
            destination = ROOT / "runs" / name
            if (destination / "complete.json").exists():
                verify_run(destination, json.loads((destination / "config.json").read_text()))
                continue
            # Keep the repository's mandatory training entry point, including
            # immutable prediction/provenance checks, for every worker package.
            subprocess.run(["uv", "run", "--no-sync", "python", "scripts/run_ladder.py",
                            "--experiment", "doc-river-architecture-comparison-v1",
                            "--root", str(worker), "--region", region, "--seed", str(seed)], check=True)
            source = worker / "runs" / name
            config = json.loads((source / "config.json").read_text())
            verify_run(source, config)
            if destination.exists():
                print(f"{name}: canonical package already started; preserving both", flush=True)
                continue
            try:
                source.rename(destination)
            except FileExistsError:
                print(f"{name}: concurrent canonical start; preserving both", flush=True)
                continue
            adopted.append({"run": name, "config_hash": digest(config)})
            write_json(worker / "adopted_packages.json", adopted)
            print(f"{name}: adopted verified complete package", flush=True)


if __name__ == "__main__":
    main()
