"""Summarize the matched non-ST context intervention."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics

CONDITIONS = {
    "st357": (
        Path("experiments/predictions_graphfix_st357"),
        "ST_H2_both_s{seed}_river_s{seed}",
        "ST_H2_downstream_s{seed}_river_s{seed}",
    ),
    "context_only": (
        Path("experiments/predictions_direction_intervention_context370"),
        "INT_ctx_both_river_s{seed}",
        "INT_ctx_down_river_s{seed}",
    ),
    "full_labels": (
        Path("experiments/predictions_direction_intervention_full370"),
        "INT_full_both_river_s{seed}",
        "INT_full_down_river_s{seed}",
    ),
}


def load_pair(
    pred_dir: Path, both_name: str, down_name: str, mask: str
) -> pd.DataFrame:
    both = pd.read_parquet(pred_dir / f"{both_name}__{mask}.parquet")
    down = pd.read_parquet(pred_dir / f"{down_name}__{mask}.parquet")
    both = both[both.split == "test"][["station", "month", "y_true", "y_pred"]]
    down = down[down.split == "test"][["station", "month", "y_true", "y_pred"]]
    both = both.rename(columns={"y_true": "y_true_both", "y_pred": "y_pred_both"})
    down = down.rename(columns={"y_true": "y_true_down", "y_pred": "y_pred_down"})
    out = both.merge(down, on=["station", "month"], how="inner", validate="one_to_one")
    if out.empty or not np.allclose(out.y_true_both, out.y_true_down, equal_nan=True):
        raise ValueError(f"invalid paired predictions for {mask}")
    out["y_true"] = out.y_true_both
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/analysis_direction_intervention.csv")
    args = ap.parse_args()
    rows = []
    for condition, (pred_dir, both_template, down_template) in CONDITIONS.items():
        for seed in (42, 43, 44):
            for mask in ("e2a_strict", "e2b_partial"):
                pair = load_pair(
                    pred_dir,
                    both_template.format(seed=seed),
                    down_template.format(seed=seed),
                    mask,
                )
                both = metrics(pair.y_true.to_numpy(), pair.y_pred_both.to_numpy())
                down = metrics(pair.y_true.to_numpy(), pair.y_pred_down.to_numpy())
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "mask": mask,
                        "n_cells": len(pair),
                        "n_stations": pair.station.nunique(),
                        "both_r2": both["r2"],
                        "downstream_r2": down["r2"],
                        "delta_r2": down["r2"] - both["r2"],
                        "both_log_r2": both["log_r2"],
                        "downstream_log_r2": down["log_r2"],
                        "delta_log_r2": down["log_r2"] - both["log_r2"],
                        "both_mae": both["mae"],
                        "downstream_mae": down["mae"],
                        "delta_mae": down["mae"] - both["mae"],
                    }
                )
    result = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    mean = result.groupby("condition", as_index=False)[
        ["delta_r2", "delta_log_r2", "delta_mae"]
    ].mean()
    mean.to_csv(out.with_name(out.stem + "_mean.csv"), index=False)
    print(result.to_string(index=False))
    print("\nmean by condition")
    print(mean.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
