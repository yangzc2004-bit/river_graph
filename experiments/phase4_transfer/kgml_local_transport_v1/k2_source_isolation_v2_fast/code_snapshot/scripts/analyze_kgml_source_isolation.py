"""Audit the KGML message-only source-isolation pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ARMS = ("residual_msgdelta", "residual_msgnull")


def station_bootstrap(delta: pd.Series, stations: pd.Series, *, seed: int = 42,
                      reps: int = 5000) -> tuple[float, float, float]:
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.mean()
    values = grouped.to_numpy()
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(reps, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if cfg["arm"] not in ARMS:
            continue
        prediction_path = meta_path.parent / "test_predictions.parquet"
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        frame = pd.read_parquet(prediction_path)
        required = {"cell", "station", "month", "y_true", "final_pred", "graph_delta"}
        if not required <= set(frame.columns):
            raise ValueError(f"missing columns in {prediction_path}")
        if frame.cell.duplicated().any() or not np.isfinite(frame[["y_true", "final_pred", "graph_delta"]]).all().all():
            raise ValueError(f"invalid prediction product: {prediction_path}")
        if not frame.visibility_role.eq("test").all() or frame.visible_input.any():
            raise ValueError(f"test visibility mismatch: {prediction_path}")
        rows.append({"arm": cfg["arm"], "mask": cfg["mask"], "seed": cfg["seed"],
                     "prediction_path": str(prediction_path),
                     "graph_delta_mean": float(frame.graph_delta.mean()),
                     "graph_delta_sd": float(frame.graph_delta.std(ddof=0)),
                     "mae": float(np.abs(frame.y_true - frame.final_pred).mean())})
    result = pd.DataFrame(rows)
    if len(result) != 12 or set(result.arm) != set(ARMS):
        raise ValueError("source-isolation matrix is incomplete")
    if result.groupby(["arm", "mask"]).seed.nunique().min() != 3:
        raise ValueError("source-isolation seeds are incomplete")
    return result


def paired_predictions(runs: pd.DataFrame, mask: str) -> pd.DataFrame:
    pooled = []
    for arm in ARMS:
        parts = []
        for _, row in runs[(runs["mask"] == mask) & (runs["arm"] == arm)].iterrows():
            frame = pd.read_parquet(row.prediction_path)
            parts.append(frame[["cell", "station", "month", "y_true", "final_pred", "graph_delta"]])
        frame = pd.concat(parts, ignore_index=True)
        frame = frame.groupby(["cell", "station", "month", "y_true"], as_index=False).mean(numeric_only=True)
        pooled.append(frame.rename(columns={"final_pred": f"{arm}_pred", "graph_delta": f"{arm}_delta"}))
    paired = pooled[0]
    for frame in pooled[1:]:
        paired = paired.merge(frame, on=["cell", "station", "month", "y_true"], validate="one_to_one")
    return paired


def analyze(root: Path) -> pd.DataFrame:
    runs = load_runs(root)
    rows = []
    cells = []
    for mask in sorted(runs["mask"].unique()):
        paired = paired_predictions(runs, mask)
        cells.append(paired.assign(mask=mask))
        delta = (np.abs(paired.y_true - paired["residual_msgnull_pred"])
                 - np.abs(paired.y_true - paired["residual_msgdelta_pred"]))
        mean, low, high = station_bootstrap(delta, paired.station)
        rows.append({"mask": mask, "mae_message": float(np.abs(paired.y_true - paired.residual_msgdelta_pred).mean()),
                     "mae_null": float(np.abs(paired.y_true - paired.residual_msgnull_pred).mean()),
                     "gain_mae": mean, "gain_pct": 100 * mean / np.abs(paired.y_true - paired.residual_msgnull_pred).mean(),
                     "ci_low": low, "ci_high": high,
                     "message_delta_abs_mean": float(np.abs(paired.residual_msgdelta_delta).mean()),
                     "null_delta_abs_max": float(np.abs(paired.residual_msgnull_delta).max())})
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "source_isolation_summary.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(root / "source_isolation_cells.csv", index=False)
    runs.to_csv(root / "source_isolation_runs.csv", index=False)
    report = ["# KGML source-isolation verdict", "", "## Paired station bootstrap", "",
              summary.to_string(index=False), "", "## Interpretation", "",
              ("The message-only arm is compared with an identical empty-message null. "
               "A positive interval supports a directed-message contribution; an interval "
               "covering zero means that K1's residual improvement is not identified as river transport. "
               "The null graph delta must be exactly zero by construction."), ""]
    (root / "source_isolation_verdict.md").write_text("\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k2_source_isolation"))
    args = parser.parse_args()
    print(analyze(args.root).to_string(index=False))


if __name__ == "__main__":
    main()
