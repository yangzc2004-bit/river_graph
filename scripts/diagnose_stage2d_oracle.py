"""Exploratory oracle ceiling for conditional transfer fusion.

This script intentionally opens target labels.  It is a post-hoc design
diagnostic, never a primary endpoint or a model-selection result.  It asks
whether a convex blend of already frozen H2X K=5 and ecology-time baseline
predictions has enough headroom to repair the observed basin-specific harm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.transfer import ANALYTES, DATASETS, array, file_hash

BASELINE = Path("experiments/phase4_transfer/stage2b_cross_basin_v1/predictions.parquet")
H2X = Path("experiments/phase4_transfer/stage2_support_integrity_v1/label_free_predictions_audited.parquet")
OUT = Path("experiments/phase4_transfer/stage2d_conditional_fusion_v1/oracle_v0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASELINE)
    base = base[(base["model_name"] == "analytic_blend") & (base["k"] == 5)].copy()
    h2x = pd.read_parquet(H2X)
    h2x = h2x[(h2x["support_mode"] == "true") & (h2x["k"] == 5)].copy()
    keys = ["analyte", "basin", "task_seed", "task_index", "month", "month_index", "k", "flat"]
    base = base[keys + ["y_pred"]].rename(columns={"y_pred": "baseline_pred"})
    h2x = h2x[keys + ["y_pred"]].rename(columns={"y_pred": "h2x_pred"})
    joined = h2x.merge(base, on=keys, validate="one_to_one")
    labels = {a: array(torch.load(DATASETS[a], map_location="cpu", weights_only=False)["y"]).astype(float) for a in ANALYTES}
    joined["y_true"] = [float(labels[r.analyte].ravel()[int(r.flat)]) for r in joined.itertuples(index=False)]
    rows = []
    weights = np.linspace(0, 1, 21)
    for (analyte, basin), group in joined.groupby(["analyte", "basin"], sort=True):
        scores = []
        for weight in weights:
            pred = weight * group["h2x_pred"].to_numpy() + (1 - weight) * group["baseline_pred"].to_numpy()
            scores.append(float(np.mean(np.abs(pred - group["y_true"].to_numpy()))))
        best = int(np.argmin(scores))
        rows.append({
            "analyte": analyte,
            "basin": basin,
            "h2x_mae": scores[-1],
            "baseline_mae": scores[0],
            "oracle_weight_h2x": float(weights[best]),
            "oracle_mae": scores[best],
            "oracle_relative_reduction_vs_baseline_pct": float((scores[0] - scores[best]) / scores[0] * 100),
            "h2x_relative_reduction_vs_baseline_pct": float((scores[0] - scores[-1]) / scores[0] * 100),
            "n_query": len(group),
        })
    result = pd.DataFrame(rows)
    result.to_csv(out / "oracle_cell_summary.csv", index=False)
    fixed = []
    for analyte, group in joined.groupby("analyte", sort=True):
        scores = []
        for weight in weights:
            pred = weight * group["h2x_pred"].to_numpy() + (1 - weight) * group["baseline_pred"].to_numpy()
            scores.append(float(np.mean(np.abs(pred - group["y_true"].to_numpy()))))
        best = int(np.argmin(scores))
        fixed.append({"analyte": analyte, "oracle_fixed_weight_h2x": float(weights[best]), "oracle_fixed_mae": scores[best], "baseline_mae": scores[0], "h2x_mae": scores[-1]})
    pd.DataFrame(fixed).to_csv(out / "oracle_analyte_summary.csv", index=False)
    verdict = {
        "version": "stage2d_conditional_fusion_oracle_v0",
        "status": "exploratory_post_hoc_ceiling_diagnostic",
        "query_labels_used": True,
        "primary_claims_supported": False,
        "interpretation": "This result estimates whether a convex fusion has headroom; it does not authorize a target-calibrated gate or change any endpoint.",
        "baseline_sha256": file_hash(BASELINE),
        "h2x_sha256": file_hash(H2X),
        "n_rows": len(joined),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
