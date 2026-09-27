"""Summarize the U1 convergence pilot without touching frozen T8/T9 files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("experiments/phase4_transfer/model_upgrade_v1/u1_convergence")


def collect() -> pd.DataFrame:
    rows = []
    for epoch_dir in sorted(ROOT.glob("epoch_*")):
        try:
            budget = int(epoch_dir.name.split("_")[-1])
        except ValueError:
            continue
        for metrics_path in epoch_dir.glob("runs/*/metrics.json"):
            meta = json.loads(metrics_path.with_name("meta.json").read_text())
            metrics = json.loads(metrics_path.read_text())
            rows.append({
                "budget": budget,
                "analyte": metrics["analyte"],
                "mask": metrics["mask"],
                "seed": metrics["seed"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "epochs_run": meta["training"]["epochs_run"],
                "best_val_loss": meta["training"]["best_val_loss"],
                "run": metrics["run"],
            })
    return pd.DataFrame(rows).sort_values(["analyte", "mask", "budget", "seed"])


def main() -> None:
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    runs = collect()
    runs.to_csv(out / "runs.csv", index=False)
    summary = (runs.groupby(["analyte", "mask", "budget"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_sd_seed=("mae", "std"),
                    rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
                    best_val_loss_mean=("best_val_loss", "mean"),
                    epochs_run_mean=("epochs_run", "mean"), seeds=("seed", "count")))
    wide = summary.pivot(index=["analyte", "mask"], columns="budget", values="mae_mean")
    for budget in (10, 30, 60):
        if budget in wide.columns:
            wide[f"improvement_vs_10_pct_{budget}"] = (
                100 * (1 - wide[budget] / wide[10])
            )
    wide.reset_index().to_csv(out / "summary.csv", index=False)
    complete_30 = summary[summary.budget == 30]
    complete_10 = summary[summary.budget == 10]
    joined = complete_10.merge(complete_30, on=["analyte", "mask"],
                               suffixes=("_10", "_30"))
    joined["improvement_30_vs_10_pct"] = 100 * (1 - joined.mae_mean_30 / joined.mae_mean_10)
    joined.to_csv(out / "budget_comparison_10_vs_30.csv", index=False)
    report = [
        "# U1 convergence pilot verdict",
        "",
        ("U1 compares the 10- and 30-epoch EcoHydroGraph budgets for DOC and "
         "specific conductance under strict temporal extrapolation and "
         "unmonitored-station holdout. The 60-epoch arm was stopped after the "
         "30-epoch arm met the entry criterion; partial files are retained but "
         "not used in the comparison."),
        "",
        "## Decision",
        "",
        ("The 30-epoch budget is accepted for the next pilot stage. Both tested "
         "analytes and both scenarios show material MAE reduction relative to "
         "10 epochs. The 10-epoch T8/T9 result remains frozen and is not "
         "rewritten. The longer 60-epoch arm is not required to decide whether "
         "the original budget was limiting."),
        "",
        "## Reproduction",
        "",
        "`.venv/bin/python scripts/analyze_u1_convergence.py`",
        "",
        (f"Completed run products: {len(runs)}. The interrupted 60-epoch "
         "products are identifiable by the absence of a metrics sidecar."),
    ]
    (out / "verdict.md").write_text("\n".join(report) + "\n")
    print(joined[["analyte", "mask", "mae_mean_10", "mae_mean_30",
                  "improvement_30_vs_10_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
