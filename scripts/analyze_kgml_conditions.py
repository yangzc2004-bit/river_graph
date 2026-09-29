"""Describe where the source-isolated river-message branch helps.

This is a post-hoc diagnostic of the frozen K2 products.  It does not choose
an endpoint or a model; it reports paired error differences by support and
hydrologic context so the next model change can target an observed regime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file


def station_bootstrap(
    frame: pd.DataFrame, value: str, *, seed: int = 42, reps: int = 3000
) -> tuple[float, float, float]:
    """Return mean gain and a station-clustered percentile interval."""
    grouped = frame.groupby("station")[value].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    observed = float(sums.sum() / counts.sum())
    if len(sums) < 2:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_arm(root: Path, arm: str, mask: str) -> list[pd.DataFrame]:
    frames = []
    for meta_path in sorted((root / "runs").glob(f"{arm}__doc__{mask}*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["config"]["arm"] != arm:
            continue
        prediction_path = meta_path.parent / "test_predictions.parquet"
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        frame = pd.read_parquet(prediction_path)
        required = {
            "cell", "station", "month", "y_true", "final_pred", "age_group",
            "upstream_support_group", "flow", "graph_delta",
        }
        if not required <= set(frame.columns):
            raise ValueError(f"missing diagnostic columns in {prediction_path}")
        if not frame.visibility_role.eq("test").all() or frame.visible_input.any():
            raise ValueError(f"test visibility mismatch: {prediction_path}")
        frame["seed"] = meta["config"]["seed"]
        frame["absolute_error"] = np.abs(frame.y_true - frame.final_pred)
        frames.append(frame)
    if len(frames) != 3:
        raise ValueError(f"expected three {arm} products for {mask}, found {len(frames)}")
    return frames


def paired_cells(root: Path, mask: str) -> pd.DataFrame:
    message = pd.concat(load_arm(root, "residual_msgdelta", mask), ignore_index=True)
    null = pd.concat(load_arm(root, "residual_msgnull", mask), ignore_index=True)
    key = ["cell", "station", "month", "y_true"]
    stable = ["age_group", "upstream_support_group", "flow"]
    message = message[key + stable + ["absolute_error", "graph_delta"]]
    null = null[key + ["absolute_error"]]
    grouped_message = message.groupby(key + stable, as_index=False).mean(numeric_only=True)
    grouped_null = null.groupby(key, as_index=False).mean(numeric_only=True)
    paired = grouped_message.merge(grouped_null, on=key, suffixes=("_message", "_null"), validate="one_to_one")
    paired["gain_mae"] = paired.absolute_error_null - paired.absolute_error_message
    paired["flow_bin"] = pd.cut(
        paired.flow,
        bins=[-np.inf, 0, 1, 10, 100, 1_000, 10_000, 100_000, np.inf],
        labels=["zero", "0-1", "1-10", "10-100", "100-1k", "1k-10k", "10k-100k", ">100k"],
        right=False,
    ).astype(str)
    return paired


def summarize(paired: pd.DataFrame, mask: str) -> pd.DataFrame:
    rows = []
    for variable in ("age_group", "upstream_support_group", "flow_bin"):
        for level, group in paired.groupby(variable, dropna=False, observed=False):
            gain, low, high = station_bootstrap(group, "gain_mae", seed=42)
            rows.append(
                {
                    "mask": mask,
                    "variable": variable,
                    "level": str(level),
                    "n_cells": len(group),
                    "n_stations": group.station.nunique(),
                    "mean_gain_mae": gain,
                    "ci_low": low,
                    "ci_high": high,
                    "message_mae": float(group.absolute_error_message.mean()),
                    "null_mae": float(group.absolute_error_null.mean()),
                    "mean_abs_message_delta": float(group.graph_delta.abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k2_source_isolation_v3"),
    )
    args = parser.parse_args()
    all_pairs, summaries = [], []
    for mask in ("e2a_strict", "e3_spatial_seed42"):
        paired = paired_cells(args.root, mask)
        paired["mask"] = mask
        all_pairs.append(paired)
        summaries.append(summarize(paired, mask))
    cells = pd.concat(all_pairs, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    cells.to_csv(args.root / "condition_cells.csv", index=False)
    summary.to_csv(args.root / "condition_summary.csv", index=False)
    report = [
        "# K2 message gain by observation and hydrologic context",
        "",
        "Positive gain means the message-only model has lower absolute error than the zero-message null.",
        "Bins are descriptive diagnostics on the frozen test products; they were not used to select a model.",
        "",
        summary.to_string(index=False),
        "",
        "## Reading",
        "",
        "The temporal mask has no visible-upstream support in the test query, so its upstream-support contrast is not identifiable.",
        "The spatial mask provides the relevant upstream-support contrast. Intervals are station-clustered and should be read as diagnostic evidence, not a new primary endpoint.",
        "",
    ]
    (args.root / "condition_verdict.md").write_text("\n".join(report))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
