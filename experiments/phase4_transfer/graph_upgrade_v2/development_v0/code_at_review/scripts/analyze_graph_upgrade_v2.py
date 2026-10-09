"""Summarize graph-upgrade mechanism runs against existing matched baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/phase4_transfer/graph_upgrade_v2")
    ap.add_argument("--mechanism", default="m1")
    ap.add_argument(
        "--baseline-dir",
        default="experiments/phase4_transfer/temporal_h2x_v1/t3_formal",
    )
    ap.add_argument(
        "--rf-dir",
        default="experiments/phase4_transfer/model_upgrade_v1/u3_temporal_rf",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    upgrade = _read(root / args.mechanism / "metrics.csv")
    if upgrade.empty:
        raise SystemExit(f"no completed {args.mechanism} metrics at {root}")
    baseline = _read(Path(args.baseline_dir) / "metrics.csv")
    rf = _read(Path(args.rf_dir) / "metrics.csv")
    rows = []
    for (analyte, mask), group in upgrade.groupby(["analyte", "mask"]):
        row = {
            "mechanism": args.mechanism,
            "analyte": analyte,
            "mask": mask,
            "upgrade_mae_mean": group["mae"].mean(),
            "upgrade_mae_sd": group["mae"].std(ddof=1),
            "n_upgrade": len(group),
        }
        if not baseline.empty:
            b = baseline[(baseline["analyte"] == analyte) & (baseline["mask"] == mask)]
            if not b.empty:
                row["h2x_t_mae_mean"] = b.mae.mean()
                row["delta_vs_h2x_t_pct"] = 100 * (b.mae.mean() - row["upgrade_mae_mean"]) / b.mae.mean()
        if not rf.empty:
            r = rf[(rf["analyte"] == analyte) & (rf["mask"] == mask)]
            if not r.empty:
                row["temporal_rf_mae_mean"] = r.mae.mean()
                row["gap_vs_rf_pct"] = 100 * (row["upgrade_mae_mean"] - r.mae.mean()) / r.mae.mean()
        tail_values = []
        for run in group["run"]:
            full_path = root / args.mechanism / "runs" / run / "full_grid.parquet"
            if not full_path.is_file():
                continue
            full = pd.read_parquet(full_path)
            train = full[full["split"] == "train"]
            test = full[full["split"] == "test"].copy()
            if train.empty or test.empty:
                continue
            threshold = train["y_true"].quantile(.9)
            tail = test[test["y_true"] >= threshold]
            if not tail.empty:
                tail_values.append(float((tail["y_true"] - tail["y_pred"]).abs().mean()))
        if tail_values:
            row["q90_mae_mean"] = sum(tail_values) / len(tail_values)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["analyte", "mask"])
    out = Path(args.out) if args.out else root / args.mechanism / "analysis_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
