"""Summarize a fixed DOC family router from the existing five-seed fusion table.

This is a post-hoc performance summary: it does not fit a new model or read
terminal labels.  The route is fixed before aggregation: RF-context for E1/E3
and the local residual expert for the two temporal holdouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROUTE = {
    "e1_r20_seed42": "rf_context",
    "e2a_strict": "local_residual",
    "e2b_partial": "local_residual",
    "e3_spatial_seed42": "rf_context",
}


def analyze(input_path: Path, out_dir: Path) -> None:
    table = pd.read_csv(input_path)
    required = {"mask", "n", "rf_mae", "local_mae", "fusion_mae"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if set(table["mask"]) != set(ROUTE):
        raise ValueError("input families do not match the frozen route")
    table = table.copy()
    table["selected_model"] = table["mask"].map(ROUTE)
    table["router_mae"] = table.apply(
        lambda row: row["rf_mae"]
        if row["selected_model"] == "rf_context"
        else row["local_mae"],
        axis=1,
    )
    table["router_gain_vs_rf_pct"] = 100 * (table["rf_mae"] - table["router_mae"]) / table["rf_mae"]
    table["router_gain_vs_local_pct"] = 100 * (table["local_mae"] - table["router_mae"]) / table["local_mae"]

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "family_results.csv", index=False)
    metrics = {}
    total = table["n"].sum()
    for name in ("rf_mae", "local_mae", "fusion_mae", "router_mae"):
        metrics[name] = float((table["n"] * table[name]).sum() / total)
    summary = pd.DataFrame([{
        "n_families": len(table),
        "n_cells": int(total),
        "rf_context_mae": metrics["rf_mae"],
        "local_residual_mae": metrics["local_mae"],
        "learned_conditional_fusion_mae": metrics["fusion_mae"],
        "fixed_router_mae": metrics["router_mae"],
        "fixed_router_gain_vs_rf_pct": 100 * (metrics["rf_mae"] - metrics["router_mae"]) / metrics["rf_mae"],
        "fixed_router_gain_vs_local_pct": 100 * (metrics["local_mae"] - metrics["router_mae"]) / metrics["local_mae"],
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "verdict.md").write_text(
        "# Fixed DOC family router\n\n"
        "The route is fixed before aggregation: RF-context for E1 and E3, "
        "and the local residual expert for E2a and E2b. This summarizes the "
        "existing five-seed products and does not fit a new model.\n\n"
        + summary.to_string(index=False)
        + "\n\n"
        "The router is a performance candidate for a unified system; its "
        "family-specific rule should be confirmed on a new held-out batch "
        "before being presented as a general routing law.\n"
    )
    print(table[["mask", "selected_model", "n", "router_mae", "router_gain_vs_rf_pct"]].to_string(index=False))
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.out_dir)


if __name__ == "__main__":
    main()
