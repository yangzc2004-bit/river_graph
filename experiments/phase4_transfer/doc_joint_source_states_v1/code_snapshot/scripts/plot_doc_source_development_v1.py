"""Plot fixed DOC development comparisons without changing model selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

KINDS = {
    "joint_source": ("doc_joint_source_states_v1", "Joint source chemistry supervision", "joint_source"),
    "extended": ("doc_extended_optimization_v1", "Longer optimization of the retained DOC model", "extended"),
    "log_concentration": ("doc_log_concentration_residual_v2", "Log-concentration residual", "log_concentration"),
    "regional": ("doc_regional_source_training_v2", "Region-hidden source training", "regional"),
    "nonlinear": ("doc_nonlinear_native_residual_v1", "Nonlinear native residual", "nonlinear_native"),
    "composition": ("doc_ecological_composition_v1", "Detailed ecological composition", "composition"),
    "composition_encoder": ("doc_composition_encoder_v1", "Detailed ecology in the neural encoder", "composition_encoder"),
    "antecedent": ("doc_antecedent_hydro_v1", "Antecedent hydro-climate states", "hydro_state"),
    "antecedent_residual": ("doc_antecedent_residual_v1", "Hydro-climate states in the native residual", "hydro_state"),
    "source_auxiliary": ("doc_source_auxiliary_states_v1", "Source water-quality representation learning", "source_auxiliary"),
    "reference_trajectory": ("doc_reference_trajectory_v1", "Environmental trajectories in the DOC GRU", "reference_history"),
    "leaf_median": ("doc_leaf_median_residual_v1", "Conditional-median reference in the DOC residual", "leaf_median"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    directory, title, prefix = KINDS[args.kind]
    root = args.root or Path("experiments/phase4_transfer")/directory
    summary = pd.read_csv(root/"analysis/summary.csv").set_index("model_name")
    effects = pd.read_csv(root/"analysis/paired_effects.csv")
    parts = pd.read_csv(root/"analysis/partition_metrics.csv")
    tree_kind = args.kind in {"composition", "antecedent"}
    candidate = f"{prefix}_trees" if tree_kind else f"{prefix}_integrated"
    reference_tree = "regional_trees" if args.kind == "regional" else "station_hidden_trees"
    labels = {"current_model": "Preceding full", "station_hidden_trees": "Strong trees",
              "unmonitored_integrated": "Retained full", f"{prefix}_residual": "New neural residual",
              candidate: "New complete model"}
    if args.kind == "regional":
        labels["regional_trees"] = "Region-hidden trees"
    if args.kind == "composition":
        labels = {"station_hidden_trees": "Retained broad-total trees",
                  "unmonitored_integrated": "Retained complete model",
                  "aggregate_expanded_trees": "Matched broad-total trees",
                  candidate: "Detailed-composition trees"}
    if args.kind == "composition_encoder":
        labels["aggregate_encoder_integrated"] = "Matched broad-total encoder"
        labels[candidate] = "Detailed-ecology complete"
    if args.kind == "antecedent":
        labels = {"station_hidden_trees": "Retained strong trees",
                  "unmonitored_integrated": "Retained complete model",
                  "hydro_availability_trees": "Matched availability trees",
                  candidate: "Antecedent-state trees"}
    if args.kind == "antecedent_residual":
        labels["hydro_availability_integrated"] = "Matched availability residual"
        labels[candidate] = "Hydro-state complete"
    if args.kind == "source_auxiliary":
        labels["source_shuffle_integrated"] = "Source-label shuffle"
        labels[candidate] = "Source-supervised complete"
    if args.kind == "joint_source":
        labels["joint_shuffle_integrated"] = "Source-label shuffle"
        labels[candidate] = "Jointly supervised complete"
    if args.kind == "reference_trajectory":
        labels["reference_current_integrated"] = "Current-reference complete"
        labels[candidate] = "Reference-history complete"
    if args.kind == "leaf_median":
        labels["leaf_median_reference"] = "Conditional-median reference"
        labels[candidate] = "Median-reference complete"
    colors = {name: "#77858d" for name in labels}
    colors.update({"station_hidden_trees": "#347d72", "regional_trees": "#61a399",
                   f"{prefix}_residual": "#6595b7", candidate: "#c57a3e"})
    if args.kind == "composition":
        colors["aggregate_expanded_trees"] = "#6595b7"
    if args.kind == "composition_encoder":
        colors["aggregate_encoder_integrated"] = "#9eb7c2"
    if args.kind == "antecedent":
        colors["hydro_availability_trees"] = "#6595b7"
    if args.kind == "antecedent_residual":
        colors["hydro_availability_integrated"] = "#9eb7c2"
    if args.kind == "source_auxiliary":
        colors["source_shuffle_integrated"] = "#9eb7c2"
    if args.kind == "joint_source":
        colors["joint_shuffle_integrated"] = "#9eb7c2"
    if args.kind == "reference_trajectory":
        colors["reference_current_integrated"] = "#9eb7c2"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10., 7.2))
    fig.subplots_adjust(left=.2, right=.97, top=.9, bottom=.16, wspace=.8, hspace=.65)
    names = list(labels)
    for i, name in enumerate(names):
        for ax, metric in ((axes[0, 0], "mae"), (axes[1, 0], "q90_mae")):
            value = summary.loc[name, metric]
            ax.barh(i, value, color=colors[name])
            ax.text(value+.01*summary.loc[names, metric].max(), i, f"{value:.3f}", va="center", fontsize=8)
    for ax, metric, heading in ((axes[0, 0], "mae", "All-observation K0"), (axes[1, 0], "q90_mae", "High-DOC reconstruction")):
        ax.set(yticks=range(len(names)), yticklabels=[labels[n] for n in names],
               xlim=(0, summary.loc[names, metric].max()*1.18), xlabel="MAE (mg L$^{-1}$)", title=heading)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#eeeeee", lw=.6)
    references = list(dict.fromkeys(["unmonitored_integrated", "station_hidden_trees", reference_tree]))
    if args.kind == "composition":
        references.append("aggregate_expanded_trees")
    if args.kind == "composition_encoder":
        references.append("aggregate_encoder_integrated")
    if args.kind == "antecedent":
        references.append("hydro_availability_trees")
    if args.kind == "antecedent_residual":
        references.append("hydro_availability_integrated")
    if args.kind == "source_auxiliary":
        references.append("source_shuffle_integrated")
    if args.kind == "joint_source":
        references.append("joint_shuffle_integrated")
    if args.kind == "reference_trajectory":
        references.append("reference_current_integrated")
    if args.kind == "leaf_median":
        references.append("leaf_median_reference")
    for i, reference in enumerate(references):
        row = effects[(effects.candidate.eq(candidate)) & effects.reference.eq(reference) & effects.region.eq("overall")].iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=colors[candidate], lw=1.5)
        axes[0, 1].plot(row.relative_gain_pct, i, "s", color=colors[candidate], ms=5)
    axes[0, 1].axvline(0, color="#777777", lw=.8)
    subject = "Detailed trees" if args.kind == "composition" else "New complete"
    if args.kind == "antecedent":
        subject = "Hydro-state trees"
    axes[0, 1].set(yticks=range(len(references)), yticklabels=[f"{subject}\nvs {labels[n]}" for n in references],
                   ylim=(len(references)-.5, -.5), xlabel="Paired MAE reduction (%)", title="95% paired station intervals")
    partition_ids = sorted(parts.split_seed.unique())
    for offset, name in zip((-.12, 0., .12), ("unmonitored_integrated", "station_hidden_trees", candidate), strict=True):
        group = parts[parts.model_name.eq(name)].set_index("split_seed")
        axes[1, 1].scatter(np.arange(len(partition_ids))+offset, group.loc[partition_ids, "mae"],
                           color=colors[name], s=30, label=labels[name])
    axes[1, 1].set(xticks=range(len(partition_ids)), xticklabels=partition_ids,
                   xlabel="Source-development partition", ylabel="MAE (mg L$^{-1}$)", title="Matched partition averages")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", color="#eeeeee", lw=.6)
    for letter, ax in zip("abcd", axes.ravel(), strict=True):
        ax.text(-.18, 1.09, letter, transform=ax.transAxes, fontweight="bold")
        ax.set_axisbelow(True)
    fig.suptitle(f"{title}: matched source-development results", fontsize=12, y=.97)
    fig.text(.54, .035, "3 partitions × 3 seeds · 5,000 paired station draws · Q90 thresholds from source training",
             ha="center", fontsize=8, color="#6b7b83")
    fig.text(.54, .017, "Development comparisons; previous geographical and independent-basin queries remain untouched",
             ha="center", fontsize=8, color="#6b7b83")
    output = root/"figures"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output/f"source_development_comparison.{suffix}", dpi=200)
    plt.close(fig)
    write_json(output/"sources.json", {"plotter_sha256": sha256_file(__file__), "kind": args.kind,
        "source_role": "source_validation_development", "csvs": {name: sha256_file(root/"analysis"/name)
            for name in ("summary.csv", "paired_effects.csv", "partition_metrics.csv")}})


if __name__ == "__main__":
    main()
