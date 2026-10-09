"""Audit the gated dual-channel KGML pilot against prior message branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file


def bootstrap(frame: pd.DataFrame, *, seed: int = 42, reps: int = 5000):
    grouped = frame.groupby("station").gain.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    return point, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def pooled_error(root: Path, arm: str, mask: str) -> pd.DataFrame:
    pieces = []
    for meta_path in sorted((root / "runs").glob(f"{arm}__doc__{mask}*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["config"]["arm"] != arm:
            continue
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        frame = pd.read_parquet(meta_path.parent / "test_predictions.parquet")
        frame = frame[["cell", "station", "month", "y_true", "final_pred"]].copy()
        frame["error"] = np.abs(frame.y_true - frame.final_pred)
        pieces.append(frame)
    if len(pieces) != 3:
        raise ValueError(f"expected 3 products for {root} {arm} {mask}, found {len(pieces)}")
    frame = pd.concat(pieces, ignore_index=True)
    return frame.groupby(["cell", "station", "month", "y_true"], as_index=False).error.mean()


def audit(root: Path, references: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for mask in ("e2a_strict", "e3_spatial_seed42"):
        parts = {
            "dual": pooled_error(root, "residual_msgdual", mask),
            "null": pooled_error(root, "residual_msgnull", mask),
            "all": pooled_error(references["all"], "residual_msgdelta", mask),
            "target_only": pooled_error(references["target_only"], "residual_msgdelta", mask),
            "hydro_ecology": pooled_error(references["hydro_ecology"], "residual_msgdelta", mask),
        }
        merged = parts["dual"].rename(columns={"error": "dual_error"})
        for name in ("null", "all", "target_only", "hydro_ecology"):
            merged = merged.merge(parts[name].rename(columns={"error": f"{name}_error"}),
                                  on=["cell", "station", "month", "y_true"], validate="one_to_one")
        for reference in ("null", "all", "target_only", "hydro_ecology"):
            group = merged[["station"]].copy()
            group["gain"] = merged[f"{reference}_error"] - merged.dual_error
            point, low, high = bootstrap(group)
            rows.append({"mask": mask, "comparison": f"dual_vs_{reference}",
                         "n_cells": len(merged), "n_stations": merged.station.nunique(),
                         "dual_mae": float(merged.dual_error.mean()),
                         "reference_mae": float(merged[f"{reference}_error"].mean()),
                         "gain_mae": point, "ci_low": low, "ci_high": high,
                         "gain_pct": 100 * point / merged[f"{reference}_error"].mean()})
    result = pd.DataFrame(rows)
    if len(result) != 8:
        raise ValueError("dual gate comparison is incomplete")
    result.to_csv(root / "dual_gate_summary.csv", index=False)
    report = ["# K5 dual-channel gate verdict", "",
              "Positive gain means the gated dual-channel model has lower error.", "",
              result.to_string(index=False), "",
              "The dual branch is interpreted as a predictive mixture of input channels; gate weights are not causal transport coefficients.", ""]
    (root / "dual_gate_verdict.md").write_text("\n".join(report))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k5_dual_gate"))
    parser.add_argument("--all-root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k2_source_isolation_v3"))
    parser.add_argument("--target-root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k4_channel_isolation/target_only"))
    parser.add_argument("--hydro-root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k4_channel_isolation/hydro_ecology"))
    args = parser.parse_args()
    print(audit(args.root, {"all": args.all_root, "target_only": args.target_root,
                            "hydro_ecology": args.hydro_root}).to_string(index=False))


if __name__ == "__main__":
    main()
