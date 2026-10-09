"""Publication-style comparison of learned donor choice and matched controls."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_current_source_attention_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    root = parser.parse_args().root
    summary = pd.read_csv(root/"analysis/summary.csv").set_index("model_name")
    effects = pd.read_csv(root/"analysis/paired_effects.csv")
    partitions = pd.read_csv(root/"analysis/partition_metrics.csv")
    real, fixed, historical = (f"attention_{mode}_integrated" for mode in ("real", "fixed", "historical"))
    names = ["innovation_real_trees", "unmonitored_integrated", "innovation_real_integrated", historical, fixed, real]
    labels = {"innovation_real_trees": "Trees + current source", "unmonitored_integrated": "Retained complete",
        "innovation_real_integrated": "Current-source readout", historical: "Earlier-season attention",
        fixed: "Fixed-prior attention", real: "Hydrology-conditioned attention"}
    colors = {"innovation_real_trees": "#a18770", "unmonitored_integrated": "#7d8b90",
        "innovation_real_integrated": "#80aba0", historical: "#7896af", fixed: "#acbdba", real: "#286f65"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3))
    fig.subplots_adjust(left=.24, right=.97, top=.90, bottom=.19, wspace=1.12, hspace=.68)
    for ax, metric, title in ((axes[0, 0], "mae", "All-observation K0 reconstruction"),
                              (axes[1, 0], "q90_mae", "High-DOC reconstruction")):
        largest = float(summary.loc[names, metric].max())
        for i, name in enumerate(names):
            value = summary.loc[name, metric]
            ax.barh(i, value, color=colors[name], height=.60)
            ax.text(value+.02*largest, i, f"{value:.3f}", fontsize=8, va="center")
        ax.set(yticks=np.arange(len(names)), yticklabels=[labels[n] for n in names],
               ylim=(len(names)-.5, -.5), xlim=(0, 1.23*largest), title=title, xlabel="MAE (mg L$^{-1}$)")
        ax.grid(axis="x", color="#eeeeee", lw=.6)
    references = ["unmonitored_integrated", "innovation_real_integrated", fixed, historical]
    for i, reference in enumerate(references):
        selected = effects[effects.candidate.eq(real) & effects.reference.eq(reference) & effects.region.eq("overall")]
        if len(selected) != 1:
            raise ValueError("attention interval comparison is ambiguous")
        row = selected.iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=colors[real], lw=1.7)
        axes[0, 1].plot(row.relative_gain_pct, i, "s", color=colors[real], ms=5)
    axes[0, 1].axvline(0, color="#777777", lw=.7)
    axes[0, 1].set(yticks=np.arange(len(references)), yticklabels=[f"Attention vs\n{labels[n]}" for n in references],
        ylim=(len(references)-.5, -.5), title="Paired station intervals (95%)", xlabel="MAE reduction (%)")
    series = ["innovation_real_integrated", historical, fixed, real]
    ids = sorted(partitions.split_seed.unique())
    for offset, name in zip((-.18, -.06, .06, .18), series, strict=True):
        frame = partitions[partitions.model_name.eq(name)].set_index("split_seed")
        axes[1, 1].scatter(np.arange(len(ids))+offset, frame.loc[ids, "mae"], s=30,
                           color=colors[name], label=labels[name])
    axes[1, 1].set(xticks=np.arange(len(ids)), xticklabels=ids, title="Matched source partitions",
                   xlabel="Source-development partition", ylabel="MAE (mg L$^{-1}$)")
    axes[1, 1].grid(axis="y", color="#eeeeee", lw=.6)
    handles, legend_labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center", bbox_to_anchor=(.65, .10), ncol=2,
               frameon=False, fontsize=7.8)
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.18, 1.12, letter, transform=ax.transAxes, fontweight="bold")
        ax.set_axisbelow(True)
    fig.suptitle("Selecting current source DOC information with the existing GRU", fontsize=13, y=.97)
    fig.text(.57, .044, "3 source partitions × 3 seeds · Same reference/backbone · Two matched donor-selection controls",
             ha="center", fontsize=8, color="#627b80")
    fig.text(.57, .022, "Receiving water quality absent · 5,000 station draws · Geographical and external products unchanged",
             ha="center", fontsize=8, color="#627b80")
    output = root/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/f"current_source_attention_comparison.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__),
        "selection_scope": "source_validation_development", "csvs": {name: sha256_file(root/"analysis"/name)
        for name in ("summary.csv", "paired_effects.csv", "partition_metrics.csv")}})


if __name__ == "__main__":
    main()
