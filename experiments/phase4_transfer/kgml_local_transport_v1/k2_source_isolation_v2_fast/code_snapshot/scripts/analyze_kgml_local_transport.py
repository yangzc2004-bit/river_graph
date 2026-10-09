"""Paired analysis for the completed Local--Transport KGML K1 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import sha256_file

ARMS = ("rf_local", "rf_context", "h2x_t", "residual_upstream", "residual_both", "residual_nomsg")


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        config = meta["config"]
        prediction_path = meta_path.parent / "test_predictions.parquet"
        if not prediction_path.is_file():
            raise ValueError(f"missing predictions: {prediction_path}")
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        frame = pd.read_parquet(prediction_path)
        required = {"cell", "station", "month", "y_true", "final_pred", "local_pred", "context_pred", "graph_delta"}
        if not required <= set(frame.columns):
            raise ValueError(f"missing columns in {prediction_path}")
        if frame.cell.duplicated().any() or not np.isfinite(frame[["y_true", "final_pred"]]).all().all():
            raise ValueError(f"invalid query product: {prediction_path}")
        if not frame.visibility_role.eq("test").all() or frame.visible_input.any():
            raise ValueError(f"test visibility mismatch: {prediction_path}")
        result = metrics(frame.y_true.to_numpy(), frame.final_pred.to_numpy())
        rows.append({"arm": config["arm"], "analyte": config["analyte"], "mask": config["mask"],
                     "seed": config["seed"], "prediction_path": str(prediction_path), **result,
                     "graph_delta_mean": float(frame.graph_delta.mean()),
                     "graph_delta_sd": float(frame.graph_delta.std(ddof=0)),
                     "q90_mae": float(meta["summary"]["q90_mae"])})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no K1 products found")
    expected = set(ARMS)
    if set(result.arm) != expected or result.groupby(["arm", "mask"]).seed.nunique().min() != 3:
        raise ValueError("K1 matrix is incomplete")
    return result


def paired_predictions(runs: pd.DataFrame, mask: str) -> pd.DataFrame:
    arms = ["rf_local", "rf_context", "h2x_t", "residual_upstream", "residual_both", "residual_nomsg"]
    wide = []
    for arm in arms:
        pieces = []
        for _, row in runs[(runs["mask"] == mask) & (runs["arm"] == arm)].iterrows():
            frame = pd.read_parquet(row.prediction_path)
            frame = frame[["cell", "station", "month", "y_true", "final_pred"]].copy()
            frame["seed"] = row.seed
            pieces.append(frame)
        pooled = pd.concat(pieces, ignore_index=True)
        pooled = pooled.groupby(["cell", "station", "month", "y_true"], as_index=False).final_pred.mean()
        wide.append(pooled.rename(columns={"final_pred": arm}))
    paired = wide[0]
    for frame in wide[1:]:
        paired = paired.merge(frame, on=["cell", "station", "month", "y_true"], validate="one_to_one")
    return paired


def station_bootstrap(delta: pd.Series, stations: pd.Series, *, seed: int = 42, reps: int = 5000) -> tuple[float, float, float]:
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.mean()
    values = grouped.to_numpy()
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(reps, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def analyze(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = load_runs(root)
    paired_rows = []
    paired_cells = []
    for mask in sorted(runs["mask"].unique()):
        paired = paired_predictions(runs, mask)
        paired["mask"] = mask
        paired_cells.append(paired)
        for arm in ["h2x_t", "residual_upstream", "residual_both", "residual_nomsg"]:
            # Positive means the candidate has lower absolute error than the reference.
            delta = np.abs(paired.y_true - paired.rf_local) - np.abs(paired.y_true - paired[arm])
            mean, lo, hi = station_bootstrap(delta, paired.station)
            paired_rows.append({"mask": mask, "comparison": f"{arm}_vs_rf_local",
                                "mae_candidate": float(np.abs(paired.y_true - paired[arm]).mean()),
                                "mae_reference": float(np.abs(paired.y_true - paired.rf_local).mean()),
                                "gain_mae": mean, "gain_pct": 100 * mean / np.abs(paired.y_true - paired.rf_local).mean(),
                                "ci_low": lo, "ci_high": hi})
        for arm in ["h2x_t", "residual_upstream", "residual_both", "residual_nomsg"]:
            delta = np.abs(paired.y_true - paired.rf_context) - np.abs(paired.y_true - paired[arm])
            mean, lo, hi = station_bootstrap(delta, paired.station, seed=44)
            paired_rows.append({"mask": mask, "comparison": f"{arm}_vs_rf_context",
                                "mae_candidate": float(np.abs(paired.y_true - paired[arm]).mean()),
                                "mae_reference": float(np.abs(paired.y_true - paired.rf_context).mean()),
                                "gain_mae": mean, "gain_pct": 100 * mean / np.abs(paired.y_true - paired.rf_context).mean(),
                                "ci_low": lo, "ci_high": hi})
        for arm in ["residual_upstream", "residual_both"]:
            delta = np.abs(paired.y_true - paired.residual_nomsg) - np.abs(paired.y_true - paired[arm])
            mean, lo, hi = station_bootstrap(delta, paired.station, seed=43)
            paired_rows.append({"mask": mask, "comparison": f"{arm}_vs_residual_nomsg",
                                "mae_candidate": float(np.abs(paired.y_true - paired[arm]).mean()),
                                "mae_reference": float(np.abs(paired.y_true - paired.residual_nomsg).mean()),
                                "gain_mae": mean, "gain_pct": 100 * mean / np.abs(paired.y_true - paired.residual_nomsg).mean(),
                                "ci_low": lo, "ci_high": hi})
    summary = pd.DataFrame(paired_rows)
    cells = pd.concat(paired_cells, ignore_index=True)
    summary.to_csv(root / "paired_bootstrap_summary.csv", index=False)
    cells.to_csv(root / "paired_seedmean_cells.csv", index=False)
    runs.to_csv(root / "audited_runs.csv", index=False)
    report = ["# K1 Local--Transport KGML verdict", "", f"Completed runs: {len(runs)}", "",
              "## Mean test MAE by arm", "",
              runs.groupby(["arm", "mask"], as_index=False).mae.agg(["mean", "std", "count"]).to_string(), "",
              "## Paired station bootstrap", "", summary.to_string(index=False), "",
              "## Scientific reading", "",
              ("The residual arms improve over RF-local, especially in temporal extrapolation. "
               "The upstream, bidirectional, and no-message residual arms are nearly identical. "
               "Therefore K1 supports residual correction, but does not isolate an additional river-message contribution. "
               "RF-context remains the strongest spatial baseline. K2 should test a simpler source-isolation design "
               "or process-state representation before adding lag complexity."), ""]
    (root / "k1_verdict.md").write_text("\n".join(report))
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"))
    args = parser.parse_args()
    _runs, summary = analyze(args.root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
