"""Validation-selected additive combination of K1 and K2 residual products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.temporal_h2x import inverse_target, transform_target

K1_ROOT = Path("experiments/phase4_transfer/kgml_local_transport_v1/k1")
K2_ROOT = Path("experiments/phase4_transfer/kgml_local_transport_v1/k2_source_isolation_v3")
MASKS = ("e2a_strict", "e3_spatial_seed42")
SEEDS = (42, 43, 44)
ALPHAS = np.linspace(-1.0, 4.0, 101)


def read_pair(mask: str, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k1 = K1_ROOT / "runs" / f"residual_nomsg__doc__{mask}__seed{seed}"
    k2 = K2_ROOT / "runs" / f"residual_msgdelta__doc__{mask}__seed{seed}"
    m1, m2 = json.loads((k1 / "meta.json").read_text()), json.loads((k2 / "meta.json").read_text())
    if m1["config"]["seed"] != seed or m2["config"]["seed"] != seed:
        raise ValueError("seed identity mismatch")
    if m1["config"]["mask"] != mask or m2["config"]["mask"] != mask:
        raise ValueError("mask identity mismatch")
    if m1["config"]["dataset_sha256"] != m2["config"]["dataset_sha256"]:
        raise ValueError("dataset identity mismatch")
    for meta, root in ((m1, k1), (m2, k2)):
        for name, expected in meta["artifacts"].items():
            path = root / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
    return m1, pd.read_parquet(k1 / "val_predictions.parquet"), pd.read_parquet(k1 / "test_predictions.parquet"), \
        pd.read_parquet(k2 / "val_predictions.parquet"), pd.read_parquet(k2 / "test_predictions.parquet")


def combine_frame(base: pd.DataFrame, message: pd.DataFrame, rf, alpha: float) -> np.ndarray:
    if not np.array_equal(base.cell, message.cell):
        raise ValueError("K1 and K2 query cells differ")
    transform = rf.target_transform
    local = np.asarray(base.local_pred.to_numpy(), dtype=np.float32).copy()
    local_z = (transform_target(torch.as_tensor(local), transform).numpy() - rf.target_mu) / rf.target_sd
    combined_z = local_z + base.graph_delta_std.to_numpy() + alpha * message.graph_delta_std.to_numpy()
    return inverse_target(torch.as_tensor(combined_z * rf.target_sd + rf.target_mu), transform).numpy()


def station_bootstrap(delta: pd.Series, stations: pd.Series, *, seed: int = 42,
                      reps: int = 5000) -> tuple[float, float, float]:
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    index = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[index].sum(1) / counts[index].sum(1)
    return float(sums.sum() / counts.sum()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def analyze(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, paired_rows = [], []
    root.mkdir(parents=True, exist_ok=True)
    for mask in MASKS:
        for seed in SEEDS:
            meta, k1_val, k1_test, k2_val, k2_test = read_pair(mask, seed)
            rf = joblib.load(meta["config"]["rf_bundle_path"])
            val_losses = []
            for alpha in ALPHAS:
                pred = combine_frame(k1_val, k2_val, rf, float(alpha))
                val_losses.append(float(np.abs(k1_val.y_true.to_numpy() - pred).mean()))
            best_idx = int(np.argmin(val_losses))
            alpha = float(ALPHAS[best_idx])
            combined = combine_frame(k1_test, k2_test, rf, alpha)
            local_residual = k1_test.final_pred.to_numpy()
            message_only = combine_frame(k1_test, k2_test, rf, 1.0)
            rf_context = k1_test.context_pred.to_numpy()
            row = {"mask": mask, "seed": seed, "alpha_val": alpha,
                   "val_mae_at_alpha": val_losses[best_idx],
                   "local_residual_mae": float(np.abs(k1_test.y_true - local_residual).mean()),
                   "message_only_mae": float(np.abs(k1_test.y_true - message_only).mean()),
                   "additive_mae": float(np.abs(k1_test.y_true - combined).mean()),
                   "rf_context_mae": float(np.abs(k1_test.y_true - rf_context).mean()),
                   "additive_q90_mae": float(np.abs(k1_test.loc[k1_test.y_true >= meta["summary"]["q90_threshold_train"], "y_true"] - combined[k1_test.y_true.to_numpy() >= meta["summary"]["q90_threshold_train"]]).mean())}
            rows.append(row)
            paired_rows.append({"mask": mask, "seed": seed, "station": k1_test.station.to_numpy(),
                                "y_true": k1_test.y_true.to_numpy(), "local_residual": local_residual,
                                "message_only": message_only, "additive": combined, "rf_context": rf_context})
    runs = pd.DataFrame(rows)
    runs.to_csv(root / "additive_seed_metrics.csv", index=False)
    summary = runs.groupby("mask", as_index=False).agg({"alpha_val": "mean", "val_mae_at_alpha": "mean",
        "local_residual_mae": "mean", "message_only_mae": "mean", "additive_mae": "mean",
        "rf_context_mae": "mean", "additive_q90_mae": "mean"})
    summary.to_csv(root / "additive_summary.csv", index=False)
    bootstrap_rows = []
    for mask in MASKS:
        parts = [row for row in paired_rows if row["mask"] == mask]
        frame = pd.DataFrame({"station": np.concatenate([r["station"] for r in parts]),
                              "local_error": np.concatenate([np.abs(r["y_true"] - r["local_residual"]) for r in parts]),
                              "additive_error": np.concatenate([np.abs(r["y_true"] - r["additive"]) for r in parts]),
                              "context_error": np.concatenate([np.abs(r["y_true"] - r["rf_context"]) for r in parts])})
        for candidate, reference in (("additive_error", "local_error"), ("additive_error", "context_error")):
            mean, low, high = station_bootstrap(frame[reference] - frame[candidate], frame.station)
            bootstrap_rows.append({"mask": mask, "comparison": f"{candidate}_vs_{reference}",
                                   "gain_mae": mean, "ci_low": low, "ci_high": high,
                                   "gain_pct": 100 * mean / frame[reference].mean()})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(root / "additive_bootstrap.csv", index=False)
    report = ["# K3 additive complementarity diagnostic", "",
              "Alpha is selected from the validation query only. Test cells are used once for the rows below.", "",
              "## Seed-level summary", "", summary.to_string(index=False), "",
              "## Paired station bootstrap", "", bootstrap.to_string(index=False), "",
              "## Interpretation", "",
              ("The additive arm tests whether the K1 learned local residual and the K2 upstream message correction "
               "carry complementary signal. A positive additive gain over local residual supports a joint model; "
               "a validation-selected alpha near zero means the message branch is redundant after local correction. "
               "RF-context remains a fixed reference for explicit spatial information."), ""]
    (root / "additive_verdict.md").write_text("\n".join(report))
    return runs, bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k3_additive"))
    args = parser.parse_args()
    runs, bootstrap = analyze(args.root)
    print(runs.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
