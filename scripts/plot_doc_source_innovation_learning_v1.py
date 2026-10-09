"""Plot matched source-innovation learning and explicit-feature controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_source_innovation_learning_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    summary = pd.read_csv(root/"analysis/summary.csv").set_index("model_name")
    effects, partitions = (pd.read_csv(root/"analysis"/name) for name in ("paired_effects.csv", "partition_metrics.csv"))
    real, historical, availability = [f"innovation_{mode}_integrated" for mode in ("real", "historical", "availability")]
    names = ["station_hidden_trees", "innovation_real_trees", "unmonitored_integrated", availability, historical, real]
    labels = {"station_hidden_trees": "Strong trees", "innovation_real_trees": "Trees + same-month source",
        "unmonitored_integrated": "Retained complete", availability: "GRU + source availability",
        historical: "GRU + earlier-season source", real: "GRU + same-month source"}
    colors = {"station_hidden_trees": "#ab896e", "innovation_real_trees": "#8b6954", "unmonitored_integrated": "#7c898e",
              availability: "#b6c9c1", historical: "#799caf", real: "#307f74"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.1))
    fig.subplots_adjust(left=.23, right=.97, top=.91, bottom=.19, wspace=.95, hspace=.65)
    for ax, metric, title in ((axes[0, 0], "mae", "Unmonitored-site DOC reconstruction"),
                              (axes[1, 0], "q90_mae", "High-DOC reconstruction")):
        largest = float(summary.loc[names, metric].max())
        for i, name in enumerate(names):
            value = summary.loc[name, metric]
            ax.barh(i, value, color=colors[name], height=.6)
            ax.text(value+.015*largest, i, f"{value:.3f}", va="center", fontsize=8)
        ax.set(yticks=np.arange(len(names)), yticklabels=[labels[n] for n in names], ylim=(len(names)-.5, -.5),
               xlabel="MAE (mg L$^{-1}$)", title=title, xlim=(0, largest*1.2))
        ax.grid(axis="x", color="#eeeeee", lw=.6)
    contrasts = [("unmonitored_integrated", "Retained complete"), (availability, "Availability only"),
                 (historical, "Earlier-season source"), ("innovation_real_trees", "Trees with same information")]
    for i, (reference, label) in enumerate(contrasts):
        row = effects[effects.candidate.eq(real) & effects.reference.eq(reference) & effects.region.eq("overall")].iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=colors[real], lw=1.7)
        axes[0, 1].plot(row.relative_gain_pct, i, "s", color=colors[real], ms=5)
    axes[0, 1].axvline(0, color="#777777", lw=.7)
    axes[0, 1].set(yticks=np.arange(len(contrasts)), yticklabels=[f"Same-month GRU\nvs {label}" for _, label in contrasts],
                   ylim=(len(contrasts)-.5, -.5), xlabel="MAE reduction (%)", title="95% paired station intervals")
    partition_ids = sorted(partitions.split_seed.unique())
    series = ["unmonitored_integrated", availability, historical, real]
    for offset, name in zip((-.18, -.06, .06, .18), series, strict=True):
        frame = partitions[partitions.model_name.eq(name)].set_index("split_seed")
        axes[1, 1].scatter(np.arange(len(partition_ids))+offset, frame.loc[partition_ids, "mae"],
            color=colors[name], s=32, label=labels[name])
    axes[1, 1].set(xticks=np.arange(len(partition_ids)), xticklabels=partition_ids,
                   xlabel="Source-development partition", ylabel="MAE (mg L$^{-1}$)", title="Matched partition averages")
    axes[1, 1].grid(axis="y", color="#eeeeee", lw=.6)
    handles, legend_labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=7.8, ncol=2,
               loc="center", bbox_to_anchor=(.65, .095), columnspacing=1.2)
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.18, 1.1, letter, transform=ax.transAxes, fontweight="bold")
        ax.set_axisbelow(True)
    fig.suptitle("Learning time-aligned source DOC information in the existing GRU", fontsize=13, y=.97)
    fig.text(.56, .045, "3 partitions × 3 seeds · Same reference/backbone · Matched support and causal historical controls",
             ha="center", fontsize=8, color="#687b83")
    fig.text(.56, .022, "Receiving water quality absent · 5,000 station draws · Source-development selection; geographical/external results unchanged",
             ha="center", fontsize=8, color="#687b83")
    output = root/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/f"source_innovation_learning_comparison.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__),
        "selection_scope": "source_validation_development", "csvs": {name: sha256_file(root/"analysis"/name)
        for name in ("summary.csv", "paired_effects.csv", "partition_metrics.csv")}})


if __name__ == "__main__":
    main()
