"""Aggregate benchmark results into the summary table (Milestone 3.4).

Reads experiments/results/baselines.csv + gnn.csv, averages E1 over seeds,
and writes experiments/results/summary.md.

Usage: python scripts/summarize_results.py
"""

from pathlib import Path

import pandas as pd

RESULTS = Path("experiments/results")


def load() -> pd.DataFrame:
    frames = []
    for f in ["baselines.csv", "gnn.csv"]:
        p = RESULTS / f
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df["exp"] = df["mask"].str.replace(r"_seed\d+$", "", regex=True)
    return df


def main() -> None:
    df = load()
    # mean +- std over seeds within each experiment
    agg = (
        df.groupby(["model", "exp"])
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
             rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"))
        .reset_index()
    )
    agg["mae"] = agg.apply(
        lambda r: f"{r.mae_mean:.2f}±{r.mae_std:.2f}" if pd.notna(r.mae_std) else f"{r.mae_mean:.2f}",
        axis=1,
    )
    table = agg.pivot(index="model", columns="exp", values="mae")
    order = ["e1_r20", "e1_r40", "e1_r60", "e2a_strict", "e2b_partial", "e3_spatial"]
    table = table.reindex(columns=[c for c in order if c in table.columns])

    lines = ["# Benchmark summary", "",
             "MAE in mg/L (lower is better); E1 entries are mean±std over seeds 42/43/44.",
             ""]
    lines.append(table.to_markdown())
    lines.append("")
    lines.append("RMSE (mg/L), same layout:")
    lines.append("")
    rtab = agg.pivot(index="model", columns="exp", values="rmse_mean").round(2)
    rtab = rtab.reindex(columns=[c for c in order if c in rtab.columns])
    lines.append(rtab.to_markdown())
    lines.append("")
    lines.append("R2, same layout:")
    lines.append("")
    qtab = agg.pivot(index="model", columns="exp", values="r2_mean").round(2)
    qtab = qtab.reindex(columns=[c for c in order if c in qtab.columns])
    lines.append(qtab.to_markdown())

    out = RESULTS / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
