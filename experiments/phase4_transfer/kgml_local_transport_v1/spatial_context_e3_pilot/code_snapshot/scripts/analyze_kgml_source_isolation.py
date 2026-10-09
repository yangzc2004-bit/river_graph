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
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    index = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[index].sum(1) / counts[index].sum(1)
    return float(sums.sum() / counts.sum()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if cfg["arm"] not in ARMS:
            continue
        if cfg.get("residual_definition") != "zero_preserving_message_gru_v1":
            raise ValueError("development architecture is not a source-isolation result")
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
        if cfg["arm"] == "residual_msgnull":
            np.testing.assert_array_equal(frame.graph_delta.to_numpy(), np.zeros(len(frame)))
            np.testing.assert_array_equal(frame.final_pred, frame.local_pred)
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
            frame["absolute_error"] = np.abs(frame.y_true - frame.final_pred)
            parts.append(frame[["cell", "station", "month", "y_true", "final_pred", "graph_delta", "absolute_error", "seed"]])
        frame = pd.concat(parts, ignore_index=True)
        if not frame.groupby("cell").seed.nunique().eq(3).all():
            raise ValueError("query cells differ between seeds")
        frame = frame.drop(columns="seed")
        frame = frame.groupby(["cell", "station", "month", "y_true"], as_index=False).mean(numeric_only=True)
        pooled.append(frame.rename(columns={"final_pred": f"{arm}_pred", "graph_delta": f"{arm}_delta",
                                           "absolute_error": f"{arm}_error"}))
    paired = pooled[0]
    for frame in pooled[1:]:
        if set(paired.cell) != set(frame.cell):
            raise ValueError("query cells differ between arms")
        paired = paired.merge(frame, on=["cell", "station", "month", "y_true"], validate="one_to_one")
    return paired


def analyze(root: Path) -> pd.DataFrame:
    runs = load_runs(root)
    rows = []
    cells = []
    for mask in sorted(runs["mask"].unique()):
        paired = paired_predictions(runs, mask)
        cells.append(paired.assign(mask=mask))
        delta = paired.residual_msgnull_error - paired.residual_msgdelta_error
        mean, low, high = station_bootstrap(delta, paired.station)
        rows.append({"mask": mask, "mae_message": float(paired.residual_msgdelta_error.mean()),
                     "mae_null": float(paired.residual_msgnull_error.mean()),
                     "gain_mae": mean, "gain_pct": 100 * mean / paired.residual_msgnull_error.mean(),
                     "ci_low": low, "ci_high": high,
                     "message_delta_abs_mean": float(np.abs(paired.residual_msgdelta_delta).mean()),
                     "null_delta_abs_max": float(np.abs(paired.residual_msgnull_delta).max())})
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "source_isolation_summary.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(root / "source_isolation_cells.csv", index=False)
    runs.to_csv(root / "source_isolation_runs.csv", index=False)
    report = ["# KGML source-isolation verdict", "",
              ("Estimator: cell-weighted MAE averaged over training seeds; paired station bootstrap. "
               "This is not the error of an averaged prediction ensemble."), "", "## Paired station bootstrap", "",
              summary.to_string(index=False), "", "## Interpretation", "",
              ("The message-only arm is compared with an identical empty-message null. "
               "The null is structurally RF-local, not an independently learned local residual. "
               "A positive interval establishes an improvement over that base in this design; "
               "it does not by itself distinguish correct upstream correspondence from topology-dependent "
               "bias correction. K1's learned no-message arm remains the stronger comparison. "
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
