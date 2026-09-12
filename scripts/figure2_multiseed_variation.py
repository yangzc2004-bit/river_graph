"""Figure 2: training-seed variation of the multi-seed runs.

Unlike the label spread in an earlier revision of ``figure1_evolution.py``,
which took ``std()`` across the MASKS of a scenario, the spread drawn here is
the standard deviation across the 5 TRAINING SEEDS with the mask held fixed
(see ``scripts/analyze_multiseed.py``). Those two quantities answer different
questions and must not be substituted for each other.

What the error bars mean, exactly
---------------------------------
Each point is the equal-weight mean over one scenario's masks; the bar is the
spread of that scenario mean across the 5 training seeds. It describes how much
the result moves when only the random seed changes. It is NOT a per-cell
prediction uncertainty, and E2a/E2b are single masks, so their spread carries
no mask-to-mask information.

G0 is absent on purpose: it is a frozen single run with no seed spread, so it
has nothing to plot here and is shown only in Figure 1.

Usage:
    python scripts/figure2_multiseed_variation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path("experiments/analysis")
OUT = Path("experiments/figures")

MODELS = ["H1", "H2", "H2X"]
SCENARIOS = ["E1", "E2a", "E2b", "E3"]
COLORS = {"H1": "#1f4e79", "H2": "#c55a11", "H2X": "#2e8b57"}
METRICS = [("mae", "MAE (mg/L, lower is better)"),
           ("r2", "R² (higher is better)")]


class Figure2Error(RuntimeError):
    """Raised when the summary table cannot support the figure."""


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise Figure2Error(
            f"{path} not found; run scripts/analyze_multiseed.py first")
    df = pd.read_csv(path)
    missing = [f"{m}_{s}" for m in ("mae", "r2") for s in ("mean", "sd_seeds")
               if f"{m}_{s}" not in df.columns]
    if missing:
        raise Figure2Error(f"{path} lacks expected columns: {missing}")
    for scenario in SCENARIOS:
        for model in MODELS:
            sub = df[(df["scenario"] == scenario) & (df["model"] == model)]
            if len(sub) != 1:
                raise Figure2Error(
                    f"expected exactly 1 row for {model}/{scenario}, "
                    f"found {len(sub)}")
    return df


def draw(df: pd.DataFrame, out_base: Path, metric: str,
         ylabel: str) -> tuple[object, list[Path]]:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.26)
    xs = np.arange(len(SCENARIOS))
    offset = {"H1": -0.22, "H2": 0.0, "H2X": 0.22}

    for model in MODELS:
        means, sds = [], []
        for scenario in SCENARIOS:
            row = df[(df["model"] == model) & (df["scenario"] == scenario)].iloc[0]
            means.append(float(row[f"{metric}_mean"]))
            sds.append(float(row[f"{metric}_sd_seeds"]))
        means = np.array(means)
        sds = np.array(sds)
        x = xs + offset[model]
        ax.errorbar(x, means, yerr=sds, fmt="o-", color=COLORS[model],
                    lw=2, ms=7, capsize=4, label=model)
        for xi, mean, sd in zip(x, means, sds):
            ax.annotate(f"{mean:.3f}", (xi, mean + sd), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8.5,
                        color=COLORS[model])

    ax.set_xticks(xs, [f"{s}\n({'1 mask' if s in ('E2a', 'E2b') else '3 masks'})"
                       for s in SCENARIOS])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("scenario (error bars: standard deviation over 5 training seeds)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="model", framealpha=0.9)
    ax.set_title(
        "Figure 2 | Training-seed variation of the multi-seed runs\n"
        "spread = stdev across 5 TRAINING SEEDS with the mask fixed "
        "(not mask-to-mask, not per-cell)",
        fontsize=10)
    fig.text(0.5, 0.055,
             "G0 is absent: it is a frozen single run without a seed spread. "
             "E2a/E2b are single masks, so their bars carry no mask information.",
             ha="center", va="center", fontsize=9, color="#333333")

    paths = []
    for suffix, kwargs in (("png", {"dpi": 200}), ("svg", {}), ("pdf", {})):
        path = out_base.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        paths.append(path)
    return fig, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--analysis", default=str(ANALYSIS))
    ap.add_argument("--date", default="20260912")
    ap.add_argument("--basename", default="figure2_multiseed_variation")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.analysis) / f"multiseed_summary_{args.date}.csv"
    df = load_summary(summary_path)

    source = df[df["model"].isin(MODELS) & df["scenario"].isin(SCENARIOS)][
        ["model", "scenario", "mae_mean", "mae_sd_seeds",
         "r2_mean", "r2_sd_seeds", "rmse_mean", "rmse_sd_seeds"]].copy()
    source["n_masks"] = source["scenario"].map(
        lambda s: 1 if s in ("E2a", "E2b") else 3)
    source_csv = out_dir / f"{args.basename}_source_data.csv"
    source.to_csv(source_csv, index=False)
    print(f"saved {source_csv}")

    for metric, ylabel in METRICS:
        base = out_dir / f"{args.basename}_{metric}"
        fig, paths = draw(df, base, metric, ylabel)
        fig.canvas.draw()
        plt.close(fig)
        for path in paths:
            print(f"saved {path}")

    print()
    print("spread used (training-seed stdev, NOT mask stdev):")
    print(source[["model", "scenario", "mae_mean", "mae_sd_seeds",
                  "r2_mean", "r2_sd_seeds"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
