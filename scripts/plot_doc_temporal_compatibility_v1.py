"""Plot the matched temporal refits using their saved numerical comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_temporal_compatibility_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

COLORS = {"current_fusion": "#737E87", "station_hidden_trees": "#2F7F71",
          "upgraded_fusion": "#CE7B2E"}
LABELS = {"current_fusion": "Current full temporal model", "station_hidden_trees": "Station-hidden trees",
          "upgraded_fusion": "Upgraded temporal procedure"}
FAMILIES = {"e2a_strict": "Unobserved periods", "e2b_partial": "Observation-assisted periods"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = pd.read_csv(args.root/"analysis/summary.csv")
    effects = pd.read_csv(args.root/"analysis/paired_effects.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#414B53", "xtick.color": "#414B53", "ytick.color": "#414B53"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, (mask, title) in enumerate(FAMILIES.items()):
        table = summary[summary["mask"].eq(mask)].set_index("model_name")
        for x, name in enumerate(COLORS):
            value = table.loc[name, "mae"]
            axes[0, column].bar(x, value, color=COLORS[name], width=.6)
            axes[0, column].text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        axes[0, column].set_xticks(range(3), ["Current", "Trees", "Upgrade"])
        axes[0, column].set(title=title, ylabel=r"MAE (mg L$^{-1}$)", ylim=(0, table.loc[list(COLORS), "mae"].max()*1.2))
        rows = effects[effects["mask"].eq(mask) & effects.candidate.eq("upgraded_fusion")].set_index("reference")
        for y, reference in enumerate(("current_fusion", "station_hidden_trees", "original_hybrid")):
            row = rows.loc[reference]
            axes[1, column].errorbar(row.relative_gain_pct, y,
                xerr=np.array([row.relative_gain_pct-row.gain_ci_low_pct,
                    row.gain_ci_high_pct-row.relative_gain_pct]).reshape(2, 1),
                fmt="s", color=COLORS["upgraded_fusion"], capsize=4, markersize=7)
        axes[1, column].axvline(0, color="#737E87", linewidth=1)
        axes[1, column].set_yticks(range(3), ["vs current full", "vs matched trees", "vs original hybrid"])
        axes[1, column].set(xlabel="MAE reduction (%)", ylim=(2.7, -.7))
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.14, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=15)
        ax.grid(axis="y", color="#E7ECEF", linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Temporal compatibility of the station-hidden DOC upgrade", fontsize=17, fontweight="bold", y=.97)
    fig.subplots_adjust(left=.20, right=.97, bottom=.21, top=.88, hspace=.43, wspace=.72)
    handles = [plt.Line2D([], [], color=COLORS[name], linewidth=4, label=LABELS[name]) for name in COLORS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .08), frameon=False, ncol=3, fontsize=9)
    fig.text(.5, .045, "Fresh fit per time split · three training seeds · mean seed errors · 95% paired station-bootstrap intervals",
        ha="center", fontsize=9, color="#646F79")
    fig.text(.5, .015, "Monitored-station temporal task: legitimate past DOC/context remains visible. This is a separate refit, not external deployment.",
        ha="center", fontsize=8.7, color="#646F79")
    output = args.root/"figures"
    output.mkdir(exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output/f"doc_temporal_compatibility.{extension}", dpi=240)
    write_json(output/"sources.json", {"summary_sha256": sha256_file(args.root/"analysis/summary.csv"),
        "effects_sha256": sha256_file(args.root/"analysis/paired_effects.csv"), "plotter_sha256": sha256_file(__file__)})
    print(output/"doc_temporal_compatibility.png", flush=True)


if __name__ == "__main__":
    main()
