"""Plot matched DOC concentration/objective comparisons from saved source results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_native_reference_objective_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = pd.read_csv(args.root/"analysis/summary.csv").set_index("model_name")
    effects = pd.read_csv(args.root/"analysis/paired_effects.csv")
    parts = pd.read_csv(args.root/"analysis/partition_metrics.csv")
    labels = {"unmonitored_integrated": "Retained complete model",
        "station_hidden_trees": "Retained log ExtraTrees",
        "native_target_trees": "Native ExtraTrees",
        "log_l1_boosting": "Log L1 boosting", "native_l1_boosting": "Native L1 boosting"}
    colors = dict(zip(labels, ("#77858d", "#347d72", "#c57a3e", "#6595b7", "#a47aa0"), strict=True))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.))
    fig.subplots_adjust(left=.2, right=.97, top=.9, bottom=.22, wspace=.8, hspace=.65)
    names = list(labels)
    for ax, metric, title in ((axes[0, 0], "mae", "All-observation K0"),
                              (axes[1, 0], "q90_mae", "High-DOC reconstruction")):
        for i, name in enumerate(names):
            value = summary.loc[name, metric]
            ax.barh(i, value, color=colors[name])
            ax.text(value+.015*summary.loc[names, metric].max(), i, f"{value:.3f}", va="center", fontsize=8)
        ax.set(yticks=range(len(names)), yticklabels=[labels[n] for n in names],
            xlim=(0, summary.loc[names, metric].max()*1.2), xlabel="MAE (mg L$^{-1}$)", title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#eeeeee", lw=.6)
    pairs = [(arm, "unmonitored_integrated") for arm in names[2:]]
    pairs += [("native_target_trees", "station_hidden_trees"), ("native_l1_boosting", "log_l1_boosting")]
    descriptions = []
    for i, (candidate, reference) in enumerate(pairs):
        row = effects[effects.candidate.eq(candidate) & effects.reference.eq(reference) & effects.region.eq("overall")].iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=colors[candidate], lw=1.5)
        axes[0, 1].plot(row.relative_gain_pct, i, "s", color=colors[candidate], ms=5)
        descriptions.append(f"{labels[candidate]}\nvs {labels[reference]}")
    axes[0, 1].axvline(0, color="#777777", lw=.8)
    axes[0, 1].set(yticks=range(len(pairs)), yticklabels=descriptions,
        ylim=(len(pairs)-.5, -.5), xlabel="Paired MAE reduction (%)", title="95% paired station intervals")
    ids = sorted(parts.split_seed.unique())
    for offset, name in zip(np.linspace(-.18, .18, len(names)), names, strict=True):
        group = parts[parts.model_name.eq(name)].set_index("split_seed")
        axes[1, 1].scatter(np.arange(len(ids))+offset, group.loc[ids, "mae"], color=colors[name], s=26, label=labels[name])
    axes[1, 1].set(xticks=range(len(ids)), xticklabels=ids, xlabel="Source-development partition",
        ylabel="MAE (mg L$^{-1}$)", title="Matched partition averages")
    axes[1, 1].legend(frameon=False, fontsize=7, loc="upper center",
        bbox_to_anchor=(.5, -.3), ncol=2)
    axes[1, 1].grid(axis="y", color="#eeeeee", lw=.6)
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.18, 1.09, letter, transform=ax.transAxes, fontweight="bold")
        ax.set_axisbelow(True)
    fig.suptitle("Aligning the environmental reference with DOC concentration error", fontsize=12, y=.97)
    fig.text(.54, .04, "3 source partitions × 3 seeds · Identical47 inputs and query cells · 5,000 paired station draws",
        ha="center", fontsize=8, color="#6b7b83")
    fig.text(.54, .02, "Source-development comparison; geographical and independent-basin queries remain untouched",
        ha="center", fontsize=8, color="#6b7b83")
    output = args.root/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output/f"source_development_comparison.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__),
        "source_role": "source_validation_development", "csvs": {name: sha256_file(args.root/"analysis"/name)
            for name in ("summary.csv", "paired_effects.csv", "partition_metrics.csv")}})


if __name__ == "__main__":
    main()
