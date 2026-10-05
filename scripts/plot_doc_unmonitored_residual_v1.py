"""Plot the matched source-validation comparison for unmonitored DOC stations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from plot_doc_chemistry_confirmation_v1 import set_style

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_residual_v1")
MODELS = ("current_model", "matched_daily_trees", "station_hidden_trees",
          "unmonitored_residual", "unmonitored_integrated")
LABELS = ("Current model", "Original matched trees", "Station-hidden trees",
          "Station-hidden + GRU", "+ ecological memory")
COLORS = ("#7B838A", "#94ADA5", "#36796C", "#265D82", "#C87932")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    folder = args.root / "analysis"
    summary = pd.read_csv(folder / "summary.csv").set_index("model_name")
    partitions = pd.read_csv(folder / "partition_metrics.csv")
    effects = pd.read_csv(folder / "paired_effects.csv")
    if set(summary.index) != set(MODELS):
        raise ValueError("incomplete model comparison")
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.1))
    fig.subplots_adjust(left=.085, right=.97, bottom=.3, top=.77, wspace=.65)
    for ax, metric, title in zip(axes[:2], ("mae", "q90_mae"),
                               ("All DOC observations", "High-DOC observations"), strict=True):
        means = partitions.groupby("model_name")[metric].mean()
        np.testing.assert_allclose(means.loc[list(MODELS)], summary.loc[list(MODELS), metric])
        for i, model in enumerate(MODELS):
            color = COLORS[i]
            ax.plot(i, summary.loc[model, metric], "s", color=color, ms=5, zorder=4)
            values = partitions[partitions.model_name.eq(model)][metric].to_numpy()
            ax.scatter(i+np.linspace(-.15, .15, len(values)), values, s=12,
                       facecolors="white", edgecolors=color, linewidths=.7)
        ax.set(xticks=range(5), xticklabels=range(1, 6), xlabel="Model",
               ylabel="MAE (mg L$^{-1}$)", title=title)
        ax.grid(axis="y", lw=.5, color="#E8EBED")
    rows = []
    for i, reference in enumerate(("current_model", "station_hidden_trees", "matched_daily_trees")):
        selected = effects[effects.candidate.eq("unmonitored_integrated") & effects.reference.eq(reference)
                           & effects.region.eq("overall")]
        if len(selected) != 1:
            raise ValueError("missing or duplicate integrated comparison")
        row = selected.iloc[0]
        rows.append(row)
        axes[2].hlines(i, row.gain_ci_low_pct, row.gain_ci_high_pct, lw=1.5, color=COLORS[4])
        axes[2].plot(row.relative_gain_pct, i, "s", color=COLORS[4], ms=5)
    extent = np.r_[[r.gain_ci_low_pct for r in rows], [r.gain_ci_high_pct for r in rows], 0]
    pad = max(float(np.ptp(extent)), .1) * .1
    axes[2].set(xlim=(extent.min()-pad, extent.max()+pad), ylim=(2.5, -.5), yticks=range(3),
                yticklabels=("Current model", "Hidden trees", "Original trees"),
                xlabel="MAE reduction vs reference (%)", title="New integrated model")
    axes[2].axvline(0, color="#596168", lw=.7)
    axes[2].grid(axis="x", lw=.5, color="#E8EBED")
    for letter, ax in zip("abc", axes, strict=True):
        ax.text(-.22, 1.09, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.suptitle("DOC reconstruction without target-station water-quality history",
                 y=.96, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], marker="s", ls="none", color=color, label=f"{i+1}. {label}")
               for i, (label, color) in enumerate(zip(LABELS, COLORS, strict=True))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .11),
               ncol=3, frameon=False, fontsize=7.5)
    fig.text(.5, .07, "Source-validation development · 3 station partitions × 3 seeds · no geographical test scores",
             ha="center", fontsize=7, color="#596168")
    fig.text(.5, .035, "Squares: partition means; circles: individual partitions; intervals: paired station bootstrap (5,000 draws).",
             ha="center", fontsize=7, color="#596168")
    output = args.root / "figures"
    output.mkdir(exist_ok=True)
    products = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"doc_unmonitored_residual.{extension}"
        fig.savefig(path, dpi=300)
        products.append(path)
    plt.close(fig)
    inputs = [folder / f"{name}.csv" for name in ("summary", "partition_metrics", "paired_effects")]
    (output / "sources.json").write_text(json.dumps({"inputs": {str(p): sha256_file(p) for p in inputs},
        "outputs": {str(p): sha256_file(p) for p in products}}, indent=2)+"\n")
    print(output / "doc_unmonitored_residual.png")


if __name__ == "__main__":
    main()
