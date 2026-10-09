"""Analyze the local-residual versus message-residual cross-analyte pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file


def station_bootstrap(frame: pd.DataFrame, *, seed: int = 42, reps: int = 5000):
    grouped = frame.groupby("station").gain.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    return point, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def pooled(root: Path, analyte: str, mask: str, arm: str) -> pd.DataFrame:
    pieces = []
    pattern = f"{arm}__{analyte}__{mask}*/meta.json"
    for meta_path in sorted((root / "runs").glob(pattern)):
        meta = json.loads(meta_path.read_text())
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        frame = pd.read_parquet(meta_path.parent / "test_predictions.parquet")
        frame = frame[["cell", "station", "month", "y_true", "final_pred"]].copy()
        frame["error"] = np.abs(frame.y_true - frame.final_pred)
        pieces.append(frame)
    if len(pieces) != 3:
        raise ValueError(f"expected three seeds for {analyte} {mask} {arm}, found {len(pieces)}")
    frame = pd.concat(pieces, ignore_index=True)
    return frame.groupby(["cell", "station", "month", "y_true"], as_index=False).error.mean()


def analyze(root: Path) -> pd.DataFrame:
    rows = []
    for analyte in ("ph", "spec_conductance"):
        for mask in ("e2a_strict", "e3_spatial_seed42"):
            local = pooled(root, analyte, mask, "residual_nomsg").rename(columns={"error": "local_error"})
            message = pooled(root, analyte, mask, "residual_msgdelta").rename(columns={"error": "message_error"})
            merged = local.merge(message, on=["cell", "station", "month", "y_true"], validate="one_to_one")
            frame = merged[["station"]].copy()
            frame["gain"] = merged.local_error - merged.message_error
            point, low, high = station_bootstrap(frame)
            rows.append({"analyte": analyte, "mask": mask, "n_cells": len(merged),
                         "n_stations": merged.station.nunique(),
                         "local_mae": float(merged.local_error.mean()),
                         "message_mae": float(merged.message_error.mean()),
                         "gain_mae": point, "ci_low": low, "ci_high": high,
                         "gain_pct": 100 * point / merged.local_error.mean()})
    result = pd.DataFrame(rows)
    if len(result) != 4:
        raise ValueError("cross-analyte matrix is incomplete")
    result.to_csv(root / "multianalyte_summary.csv", index=False)
    report = ["# K6 cross-analyte KGML verdict", "",
              "Positive gain means the upstream message residual improves on the local temporal residual.", "",
              result.to_string(index=False), "",
              "The comparison is paired by station-month and averaged over three training seeds.", ""]
    (root / "multianalyte_verdict.md").write_text("\n".join(report))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k6_multianalyte"))
    args = parser.parse_args()
    print(analyze(args.root).to_string(index=False))


if __name__ == "__main__":
    main()
