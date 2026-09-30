"""Analyze the RF-context spatial residual pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import sha256_file

ARMS = ("rf_context", "residual_context_nomsg", "residual_context_msgdelta",
        "residual_context_both")
CANDIDATES = ("residual_context_msgdelta", "residual_context_both")


def station_bootstrap(delta: pd.Series, stations: pd.Series, *, seed: int = 42,
                      reps: int = 5000) -> tuple[float, float, float]:
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    # Use the same sampled station indices for numerator and denominator.
    indices = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    return float(sums.sum() / counts.sum()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def load_runs(root: Path) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    rows = []
    frames = {}
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if cfg["arm"] not in ARMS:
            continue
        pred = meta_path.parent / "test_predictions.parquet"
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        frame = pd.read_parquet(pred)
        required = {"cell", "station", "y_true", "final_pred", "graph_delta", "upstream_support_group"}
        if not required <= set(frame.columns):
            raise ValueError(f"missing columns in {pred}")
        if frame.cell.duplicated().any() or not np.isfinite(frame[["y_true", "final_pred"]]).all().all():
            raise ValueError(f"invalid query product: {pred}")
        if not frame.visibility_role.eq("test").all() or frame.visible_input.any():
            raise ValueError(f"test visibility mismatch: {pred}")
        if cfg["arm"] == "residual_context_nomsg":
            np.testing.assert_array_equal(frame.graph_delta.to_numpy(), np.zeros(len(frame)))
            np.testing.assert_allclose(frame.final_pred, frame.context_pred, rtol=0, atol=0)
        rows.append({"arm": cfg["arm"], "seed": cfg["seed"], "mask": cfg["mask"],
                     "prediction_path": str(pred), **metrics(frame.y_true.to_numpy(), frame.final_pred.to_numpy())})
        frames[(cfg["arm"], cfg["seed"])] = frame
    result = pd.DataFrame(rows)
    if set(result.arm) != set(ARMS) or len(result) != 12:
        raise ValueError("spatial pilot matrix is incomplete")
    if result.groupby("arm").seed.nunique().min() != 3:
        raise ValueError("spatial pilot seed matrix is incomplete")
    return result, frames


def seedmean_cells(frames: dict[tuple[str, int], pd.DataFrame]) -> dict[str, pd.DataFrame]:
    pooled = {}
    for arm in ARMS:
        parts = []
        for seed in (42, 43, 44):
            frame = frames[(arm, seed)].copy()
            frame["abs_error"] = np.abs(frame.y_true - frame.final_pred)
            parts.append(frame[["cell", "station", "y_true", "abs_error", "upstream_support_group"]])
        merged = pd.concat(parts, ignore_index=True)
        pooled[arm] = merged.groupby(
            ["cell", "station", "y_true", "upstream_support_group"], as_index=False
        ).abs_error.mean()
    return pooled


def analyze(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs, frames = load_runs(root)
    cells = seedmean_cells(frames)
    base = cells["rf_context"]
    null = cells["residual_context_nomsg"]
    summary_rows = []
    bootstrap_rows = []
    for arm in ARMS:
        errors = cells[arm].abs_error
        summary_rows.append({"arm": arm, "n_cells": len(errors), "n_stations": cells[arm].station.nunique(),
                             "mae": float(errors.mean())})
    for arm in CANDIDATES:
        candidate = cells[arm]
        for reference_name, reference in (("rf_context", base), ("context_nomsg", null)):
            merged = reference.merge(candidate, on=["cell", "station", "y_true", "upstream_support_group"],
                                     suffixes=("_reference", "_candidate"), validate="one_to_one")
            gain = merged.abs_error_reference - merged.abs_error_candidate
            mean, lo, hi = station_bootstrap(gain, merged.station, seed=44 if reference_name == "rf_context" else 45)
            bootstrap_rows.append({"candidate": arm, "reference": reference_name,
                                   "mae_candidate": float(merged.abs_error_candidate.mean()),
                                   "mae_reference": float(merged.abs_error_reference.mean()),
                                   "gain_mae": mean, "gain_pct": 100 * mean / merged.abs_error_reference.mean(),
                                   "ci_low": lo, "ci_high": hi})
    support_rows = []
    for group, group_frame in base.groupby("upstream_support_group"):
        for arm in ARMS:
            candidate = cells[arm]
            subset = candidate[candidate.upstream_support_group == group]
            support_rows.append({"support_group": group, "arm": arm, "n_cells": len(subset),
                                 "n_stations": subset.station.nunique(), "mae": float(subset.abs_error.mean())})
    summary = pd.DataFrame(summary_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    support = pd.DataFrame(support_rows)
    summary.to_csv(root / "spatial_summary.csv", index=False)
    bootstrap.to_csv(root / "spatial_bootstrap.csv", index=False)
    support.to_csv(root / "spatial_support_summary.csv", index=False)
    runs.to_csv(root / "spatial_runs.csv", index=False)
    report = ["# Spatial context residual pilot verdict", "",
              ("The RF-context arm is the explicit current-month network baseline. "
               "The context_nomsg arm is an exact zero-message null: its prediction equals RF-context. "
               "Positive gain means a graph candidate has lower absolute error."), "",
              "## Mean test MAE", "", summary.to_string(index=False), "",
              "## Paired station bootstrap", "", bootstrap.to_string(index=False), "",
              "## Upstream support strata", "", support.to_string(index=False), "",
              "## Interpretation", "",
              ("The upstream residual branch is the primary spatial candidate. "
               "Its gain must be read together with the support strata: stations without a visible upstream source "
               "cannot receive a target-message correction. The both-direction branch is a spatial interpolation "
               "diagnostic because held-out stations can have visible downstream neighbors; it is not a one-way "
               "transport claim."), ""]
    (root / "spatial_context_verdict.md").write_text("\n".join(report))
    return summary, bootstrap, support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/spatial_context_e3_pilot"))
    args = parser.parse_args()
    summary, bootstrap, support = analyze(args.root)
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
