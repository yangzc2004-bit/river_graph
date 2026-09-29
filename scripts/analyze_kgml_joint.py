"""Audit the joint local--message residual pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

K1_ROOT = Path("experiments/phase4_transfer/kgml_local_transport_v1/k1")
ARMS = ("residual_additive",)


def bootstrap(delta: pd.Series, stations: pd.Series, *, seed=42, reps=5000):
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    index = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[index].sum(1) / counts[index].sum(1)
    return float(sums.sum() / counts.sum()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def audit(root: Path):
    rows, paired_rows = [], []
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if cfg["arm"] not in ARMS:
            continue
        for name, expected in meta["artifacts"].items():
            path = meta_path.parent / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
        k3 = pd.read_parquet(meta_path.parent / "test_predictions.parquet")
        k1_path = K1_ROOT / "runs" / f"residual_nomsg__doc__{cfg['mask']}__seed{cfg['seed']}" / "test_predictions.parquet"
        k1 = pd.read_parquet(k1_path)
        if not np.array_equal(k3.cell, k1.cell):
            raise ValueError("query cells differ between K1 and K3")
        rows.append({"mask": cfg["mask"], "seed": cfg["seed"],
                     "joint_mae": float(np.abs(k3.y_true - k3.final_pred).mean()),
                     "local_residual_mae": float(np.abs(k1.y_true - k1.final_pred).mean()),
                     "rf_context_mae": float(np.abs(k1.y_true - k1.context_pred).mean()),
                     "joint_graph_delta_sd": float(k3.graph_delta.std(ddof=0)),
                     "joint_message_delta_sd": float(k3.message_delta.std(ddof=0))})
        paired_rows.append({"mask": cfg["mask"], "station": k3.station, "joint_error": np.abs(k3.y_true - k3.final_pred),
                            "local_error": np.abs(k1.y_true - k1.final_pred),
                            "context_error": np.abs(k1.y_true - k1.context_pred)})
    result = pd.DataFrame(rows)
    if len(result) != 6:
        raise ValueError(f"expected six joint products, found {len(result)}")
    result.to_csv(root / "joint_seed_metrics.csv", index=False)
    summary = result.groupby("mask", as_index=False).mean(numeric_only=True)
    summary.to_csv(root / "joint_summary.csv", index=False)
    boot = []
    # ``DataFrame.mask`` is a method, so use the explicit column lookup here.
    for mask in sorted(result["mask"].unique()):
        parts = [x for x in paired_rows if x["mask"] == mask]
        frame = pd.DataFrame({key: np.concatenate([p[key] for p in parts]) for key in
                              ("station", "joint_error", "local_error", "context_error")})
        for candidate, ref in (("joint_error", "local_error"), ("joint_error", "context_error")):
            mean, low, high = bootstrap(frame[ref] - frame[candidate], frame.station)
            boot.append({"mask": mask, "comparison": f"{candidate}_vs_{ref}", "gain_mae": mean,
                         "ci_low": low, "ci_high": high, "gain_pct": 100 * mean / frame[ref].mean()})
    bootstrap_table = pd.DataFrame(boot)
    bootstrap_table.to_csv(root / "joint_bootstrap.csv", index=False)
    report = ["# K3 joint local--message verdict", "", summary.to_string(index=False), "",
              "## Paired station bootstrap", "", bootstrap_table.to_string(index=False), "", 
              ("The joint model is useful when its gain over K1 learned local residual is positive and stable in the temporal holdout. "
               "RF-context is retained as the explicit-context reference."), ""]
    (root / "joint_verdict.md").write_text("\n".join(report))
    return result, bootstrap_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k3_joint"))
    args = parser.parse_args()
    result, boot = audit(args.root)
    print(result.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
