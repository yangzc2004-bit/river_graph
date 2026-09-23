"""Aggregate the current-code frozen direction/topology baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MASKS = [
    "e2a_strict",
    "e2b_partial",
    "e3_spatial_seed42",
    "e3_spatial_seed43",
    "e3_spatial_seed44",
]
SEEDS = [42, 43, 44]
DIRECTIONS = ["both", "up", "down", "empty"]


def read_condition(results_dir: Path, prefix: str, direction: str, seed: int) -> dict:
    path = results_dir / f"FRZ_{prefix}_{direction}_river_s{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {mask: data[mask] for mask in MASKS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/analysis_frozen_baselines")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prefix, results in (
        ("st", Path("experiments/results_frozen_st357")),
        ("370", Path("experiments/results_frozen_370")),
    ):
        for seed in SEEDS:
            records = {
                direction: read_condition(results, prefix, direction, seed)
                for direction in DIRECTIONS
            }
            for mask in MASKS:
                row = {"dataset": prefix, "seed": seed, "mask": mask}
                for direction in DIRECTIONS:
                    for metric in ("r2", "log_r2", "mae", "rmse"):
                        row[f"{direction}_{metric}"] = records[direction][mask][metric]
                for direction in ("down", "up", "empty"):
                    for metric in ("r2", "log_r2", "mae", "rmse"):
                        row[f"delta_{direction}_{metric}"] = (
                            row[f"{direction}_{metric}"] - row[f"both_{metric}"]
                        )
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "direction_topology_frozen.csv", index=False)
    summary = result.groupby(["dataset", "mask"], as_index=False)[
        [
            "delta_down_r2",
            "delta_up_r2",
            "delta_empty_r2",
            "delta_down_log_r2",
            "delta_up_log_r2",
            "delta_empty_log_r2",
            "delta_down_mae",
            "delta_up_mae",
            "delta_empty_mae",
        ]
    ].agg(["mean", "std"])
    summary.to_csv(out / "direction_topology_frozen_summary.csv", index=False)
    print(result.to_string(index=False))
    print("\nmean by dataset and mask")
    print(summary.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
