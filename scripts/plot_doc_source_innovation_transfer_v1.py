"""Source-backed comparison of current versus earlier-season DOC innovations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_source_innovation_transfer_v1 import ARMS, ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    summary = pd.read_csv(root/"analysis/summary.csv").set_index("model_name")
    effects = pd.read_csv(root/"analysis/paired_effects.csv")
    support = pd.read_csv(root/"analysis/source_support.csv").groupby("split_seed").mean(numeric_only=True)
    names = ["station_hidden_trees", "unmonitored_integrated", ARMS[1], ARMS[0]]
    labels = {"station_hidden_trees": "Strong trees", "unmonitored_integrated": "Retained complete",
              ARMS[1]: "Earlier-season source", ARMS[0]: "Same-month source"}
    colors = {"station_hidden_trees": "#a48264", "unmonitored_integrated": "#78878f",
              ARMS[1]: "#7e9cae", ARMS[0]: "#317e73"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    fig.subplots_adjust(left=.20, right=.97, top=.89, bottom=.17, wspace=.85, hspace=.65)
    for ax, metric, title in ((axes[0, 0], "mae", "DOC at water-quality-free receivers"),
                               (axes[1, 0], "q90_mae", "High-DOC error")):
        largest = float(summary.loc[names, metric].max())
        for i, name in enumerate(names):
            value = summary.loc[name, metric]
            ax.barh(i, value, color=colors[name], height=.6)
            ax.text(value+.015*largest, i, f"{value:.3f}", va="center", fontsize=8)
        ax.set(yticks=np.arange(len(names)), yticklabels=[labels[n] for n in names],
               xlim=(0, largest*1.2), ylim=(len(names)-.5, -.5), title=title, xlabel="MAE (mg L$^{-1}$)")
        ax.grid(axis="x", color="#eeeeee", lw=.6)
    contrasts = [(ARMS[0], "unmonitored_integrated", "Same month vs retained"),
                 (ARMS[1], "unmonitored_integrated", "Earlier season vs retained"),
                 (ARMS[0], ARMS[1], "Same month vs earlier season")]
    for i, (candidate, reference, label) in enumerate(contrasts):
        row = effects[(effects.candidate.eq(candidate)) & effects.reference.eq(reference) & effects.region.eq("overall")].iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=colors[candidate], lw=1.7)
        axes[0, 1].plot(row.relative_gain_pct, i, "s", color=colors[candidate], ms=5)
    axes[0, 1].axvline(0, color="#777777", lw=.7)
    axes[0, 1].set(yticks=np.arange(3), yticklabels=[row[2] for row in contrasts], ylim=(2.5, -.5),
                   xlabel="MAE reduction (%)", title="5,000 paired station draws")
    positions = np.arange(len(support))
    axes[1, 1].bar(positions-.16, support.current_coverage, .32, color="#bbccc4", label="Any current source")
    axes[1, 1].bar(positions+.16, support.matched_coverage, .32, color=colors[ARMS[0]], label="Matched current + past")
    axes[1, 1].set(xticks=positions, xticklabels=support.index, ylim=(0, 1.05),
                   xlabel="Source-development partition", ylabel="Fraction of query cells", title="Available source support")
    axes[1, 1].grid(axis="y", color="#eeeeee", lw=.6)
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.18, 1.10, letter, transform=ax.transAxes, fontweight="bold")
        ax.set_axisbelow(True)
    fig.suptitle("Time-aligned source DOC information in the retained model", fontsize=13, y=.97)
    fig.text(.55, .067, "Source DOC only · Receiving DOC/pH/conductance remain hidden · Current/past arms share donor availability",
             ha="center", fontsize=8, color="#697b83")
    fig.text(.55, .042, "3 partitions × 3 seeds · Source-validation selection includes zero correction · Development results",
             ha="center", fontsize=8, color="#697b83")
    output = root/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/f"source_innovation_comparison.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__),
        "selection_scope": "source_validation_development", "csvs": {name: sha256_file(root/"analysis"/name)
        for name in ("summary.csv", "paired_effects.csv", "source_support.csv")}})


if __name__ == "__main__":
    main()
