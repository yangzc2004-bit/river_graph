"""Plot the source-only timing diagnostic with its observation support."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_doc_source_synchrony_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file


def main():
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    parts = pd.read_csv(ROOT/"analysis/partition_diagnostics.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9., 4.4))
    fig.subplots_adjust(left=.1, right=.97, top=.72, bottom=.27, wspace=.42)
    colors = {"nearest5": "#347d72", "farthest5": "#77858d"}
    labels = {"nearest5": "Nearest ecological sources", "farthest5": "Distant ecological sources"}
    for name in ("nearest5", "farthest5"):
        group = summary[summary.group.eq(name)].sort_values("lag")
        x = np.arange(len(group))
        axes[0].plot(x, group.real_corr, "o-", color=colors[name], label=labels[name])
        axes[0].plot(x, group.shuffle_corr, "s--", color=colors[name], alpha=.65,
                     label=f"{labels[name]}: year shuffle")
        available = parts[parts.group.eq(name)].groupby("lag").apply(
            lambda g: (g.eligible_stations/g.source_stations).mean(), include_groups=False)
        axes[1].plot(x, available.loc[group.lag], "o-", color=colors[name], label=labels[name])
    axes[0].axhline(0, lw=.8, color="#999999")
    axes[0].set(ylabel="Mean source residual correlation", title="Timing information beyond seasonality")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=7.5, ncol=2,
               loc="upper center", bbox_to_anchor=(.53, .91))
    axes[1].set(ylabel="Supported recipients (fraction)", ylim=(0, 1),
                title="Observation support")
    for letter, ax in zip("ab", axes, strict=True):
        ax.set(xticks=range(3), xticklabels=["0 (same month)", "1", "3"],
               xlabel="Source donor lag (months)")
        ax.grid(axis="y", color="#eeeeee", lw=.6)
        ax.set_axisbelow(True)
        ax.text(-.13, 1.08, letter, transform=ax.transAxes, fontweight="bold")
    fig.suptitle("Source DOC departures contain a same-month ecological-neighbour signal", y=.97, fontsize=11)
    fig.text(.53, .11, "Source training stations only · station-blocked OOF residuals · ≥12 common observed months",
             ha="center", color="#6b7b83", fontsize=8)
    fig.text(.53, .065, "Shuffle preserves each donor's calendar season and availability; pairs/partitions share observations",
             ha="center", color="#6b7b83", fontsize=8)
    fig.text(.53, .02, "Descriptive associations; receiving-site prediction improvement has not yet been evaluated",
             ha="center", color="#6b7b83", fontsize=8)
    output = ROOT/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/f"source_synchrony.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__),
        "inputs": {p.name: sha256_file(p) for p in [ROOT/"analysis/summary.csv", ROOT/"analysis/partition_diagnostics.csv"]}})


if __name__ == "__main__":
    main()
