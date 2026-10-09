"""Plot fixed external DOC comparisons, adaptation, tail error and intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_external_replication_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

COLORS = {"current_model": "#737E87", "station_hidden_trees": "#2F7F71",
          "unmonitored_integrated": "#CE7B2E"}
LABELS = {"current_model": "Current full model", "station_hidden_trees": "Matched unmonitored trees",
          "unmonitored_integrated": "Upgraded full model"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    metrics = pd.read_csv(root/"analysis/metrics.csv")
    effects = pd.read_csv(root/"analysis/paired_effects.csv")
    primary = metrics[metrics.population.eq("all_observed_k0")].set_index("model_name")
    curves = metrics[metrics.population.eq("fixed_query_curve")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#414B53", "xtick.color": "#414B53", "ytick.color": "#414B53"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    positions = np.arange(len(COLORS))
    for position, name in enumerate(COLORS):
        axes[0, 0].bar(position, primary.loc[name, "mae"], color=COLORS[name], width=.6)
        axes[0, 0].text(position, primary.loc[name, "mae"], f"{primary.loc[name, 'mae']:.3f}",
                        ha="center", va="bottom", fontsize=10)
        curve = curves[curves.model_name.eq(name)].sort_values("k")
        axes[1, 0].plot(curve.k, curve.mae, "o-", color=COLORS[name], label=LABELS[name])
        width, coverage = primary.loc[name, "width_median"], primary.loc[name, "coverage"]
        axes[1, 1].errorbar(width, coverage, yerr=np.array([
            coverage-primary.loc[name, "coverage_ci_low"], primary.loc[name, "coverage_ci_high"]-coverage
        ]).reshape(2, 1), fmt="o", color=COLORS[name], capsize=4, markersize=7)
    axes[0, 0].set_xticks(positions, ["Current", "Trees", "Upgrade"])
    axes[0, 0].set(ylabel=r"K0 MAE (mg L$^{-1}$)", title="Independent unmonitored stations")
    selected = effects[effects.population.eq("all_observed_k0") & effects.zone.eq("overall")].set_index("reference")
    for position, reference in enumerate(("current_model", "station_hidden_trees")):
        row = selected.loc[reference]
        axes[0, 1].errorbar(row.relative_gain_pct, position,
            xerr=np.array([row.relative_gain_pct-row.gain_ci_low_pct,
                           row.gain_ci_high_pct-row.relative_gain_pct]).reshape(2, 1),
            color=COLORS["unmonitored_integrated"], fmt="s", capsize=4, markersize=7)
    axes[0, 1].axvline(0, color="#737E87", linewidth=1)
    axes[0, 1].set_yticks([0, 1], ["vs current full", "vs matched trees"])
    axes[0, 1].set(xlabel="K0 MAE reduction (%)", title="Paired station-bootstrap gain")
    axes[0, 1].set_ylim(1.7, -.7)
    axes[1, 0].set_xticks([0, 1, 3, 5])
    axes[1, 0].set(xlabel="DOC support observations per station",
                   ylabel=r"Fixed-query MAE (mg L$^{-1}$)", title="Adaptation on the same query cells")
    axes[1, 1].axhline(.9, color="#A7ADB3", linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel=r"Median 90% interval width (mg L$^{-1}$)",
                   ylabel="External coverage", title="Source-calibrated uncertainty")
    axes[1, 1].set_ylim(0, 1.04)
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.14, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=15)
        ax.grid(axis="y", color="#E7ECEF", linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("DOC reconstruction in an independent river basin", fontsize=17, fontweight="bold", y=.97)
    fig.subplots_adjust(left=.09, right=.97, bottom=.22, top=.88, hspace=.47, wspace=.38)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .1), frameon=False, ncol=3)
    fig.text(.5, .063, "HUC02040104 · 130 stations · fixed five-seed ensemble · 95% station-clustered intervals",
             ha="center", color="#646F79", fontsize=9)
    fig.text(.5, .027, "K0: no external DOC calibration. K curves: designated support only. Intervals: source validation only.",
             ha="center", color="#646F79", fontsize=9)
    output = root/"figures"
    output.mkdir(exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output/f"doc_external_replication.{extension}", dpi=240)
    write_json(output/"sources.json", {"metrics_sha256": sha256_file(root/"analysis/metrics.csv"),
        "effects_sha256": sha256_file(root/"analysis/paired_effects.csv"), "plotter_sha256": sha256_file(__file__)})
    print(output/"doc_external_replication.png", flush=True)


if __name__ == "__main__":
    main()
