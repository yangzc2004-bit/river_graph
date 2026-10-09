"""Draw reviewed source-development comparisons for the small DOC correction."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from run_doc_concentration_hydro_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

COLORS = {"unmonitored_integrated": "#6b7b83", "station_hidden_trees": "#2e7c73",
          "trees_concentration_hydro": "#2e7c73", "integrated_log_affine": "#416f94",
          "integrated_concentration_hydro": "#ce7e35"}
LABELS = {"unmonitored_integrated": "Retained full", "station_hidden_trees": "Strong trees",
          "trees_concentration_hydro": "Trees + conditional", "integrated_log_affine": "Full + affine",
          "integrated_concentration_hydro": "Full + conditional"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    args = parser.parse_args()
    summary = pd.read_csv(args.root/"analysis/summary.csv").set_index("model_name")
    effects = pd.read_csv(args.root/"analysis/paired_effects.csv")
    hydro = pd.read_csv(args.root/"analysis/hydro_strata.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10., 7.2))
    fig.subplots_adjust(left=.2, right=.97, top=.9, bottom=.2, wspace=.85, hspace=.6)
    names = list(LABELS)
    for i, name in enumerate(names):
        row = summary.loc[name]
        axes[0, 0].barh(i, row.mae, color=COLORS[name], alpha=.65 if name.startswith("trees_") else 1)
        axes[0, 0].text(row.mae+.012, i, f"{row.mae:.3f}", va="center", fontsize=8)
    axes[0, 0].set(yticks=range(len(names)), yticklabels=[LABELS[n] for n in names],
                   xlabel="K0 MAE (mg L$^{-1}$)", xlim=(0, 2.1), title="Source-validation reconstruction")
    axes[0, 0].invert_yaxis()
    contrasts = [("integrated_concentration_hydro", "unmonitored_integrated", "Conditional vs retained"),
                 ("integrated_log_affine", "unmonitored_integrated", "Affine vs retained"),
                 ("integrated_concentration_hydro", "trees_concentration_hydro", "Conditional vs calibrated trees")]
    for i, (candidate, reference, label) in enumerate(contrasts):
        row = effects[(effects.candidate.eq(candidate)) & effects.reference.eq(reference) & effects.region.eq("overall")].iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=COLORS[candidate], lw=1.5)
        axes[0, 1].plot(row.relative_gain_pct, i, "s", color=COLORS[candidate], ms=5)
    axes[0, 1].axvline(0, color="#777777", lw=.8)
    axes[0, 1].set(yticks=range(3), yticklabels=[x[2].replace(" vs ", "\nvs ") for x in contrasts],
                   ylim=(2.5, -.5), xlabel="Paired MAE reduction (%)", title="Paired station-bootstrap intervals")
    for name in ("unmonitored_integrated", "station_hidden_trees", "integrated_concentration_hydro", "trees_concentration_hydro"):
        part = hydro[hydro.model_name.eq(name)].groupby(["split_seed", "hydro_availability"], as_index=False).agg(
            mae=("mae", "mean"), bias=("bias", "mean"))
        part = part.groupby("hydro_availability", as_index=False)[["mae", "bias"]].mean()
        for metric, ax in (("mae", axes[1, 0]), ("bias", axes[1, 1])):
            ax.plot(part.hydro_availability, part[metric], marker="o", ms=4,
                    linestyle="--" if name.startswith("trees_") else "-", color=COLORS[name], label=LABELS[name])
    for ax in axes[1]:
        ax.set(xticks=[0, 1, 2], xticklabels=["None", "One", "Both"], xlabel="Available hydro channels")
    axes[1, 0].set(ylabel="K0 MAE (mg L$^{-1}$)", title="Hydrological input availability")
    axes[1, 1].axhline(0, color="#777777", lw=.8)
    axes[1, 1].set(ylabel="Mean prediction bias (mg L$^{-1}$)", title="Signed reconstruction error")
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax in axes[0] else "y", color="#eeeeee", lw=.6)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.55, .047))
    fig.suptitle("Concentration–hydrology correction on source-validation stations", fontsize=12, y=.97)
    fig.text(.55, .018, "3 development partitions × 3 seeds · 5,000 paired station draws · no geographical/external re-evaluation",
             ha="center", fontsize=8, color="#6b7b83")
    output = args.root/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output/f"doc_concentration_hydro.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__), "source_role": "source_validation_only",
        "csvs": {name: sha256_file(args.root/"analysis"/name) for name in
                 ("summary.csv", "paired_effects.csv", "hydro_strata.csv")}})


if __name__ == "__main__":
    main()
