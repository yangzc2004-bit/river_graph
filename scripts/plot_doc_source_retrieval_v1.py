"""Plot source-validation DOC development, without selecting a model."""
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

ROOT = Path("experiments/phase4_transfer/doc_source_retrieval_v1")
MODELS = ("current_model", "matched_daily_trees", "static_memory", "retrieval",
          "hydro_pretrained", "hydro_pretrained_retrieval")
LABELS = ("Current model", "Matched trees", "Static memory", "Source retrieval",
          "Hydro pretraining", "Pretraining + retrieval")
COLORS = ("#727B83", "#36796C", "#A0A5AA", "#265D82", "#C87932", "#7097B0")


def draw(root):
    directory = root / "analysis"
    summary = pd.read_csv(directory / "summary.csv").set_index("model_name")
    parts = pd.read_csv(directory / "partition_metrics.csv")
    effects = pd.read_csv(directory / "paired_effects.csv")
    contrast = "retrieval_contrast" in summary.index
    models = (("current_model", "matched_daily_trees", "static_memory", "retrieval_contrast",
               "retrieval_contrast_uniform", "retrieval_contrast_zero_source_values") if contrast else MODELS)
    labels = (("Current model", "Matched trees", "Static memory", "Source contrast",
               "Uniform source weights", "Donor values removed") if contrast else LABELS)
    if not set(models) <= set(summary.index) or not summary.k.eq(0).all():
        raise ValueError("Need all six predeclared K0 procedures")
    for metric in ("mae", "q90_mae"):
        observed = parts.groupby("model_name")[metric].mean().loc[list(models)]
        np.testing.assert_allclose(summary.loc[list(models), metric], observed, rtol=1e-10)
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.1))
    fig.subplots_adjust(left=.1, right=.97, top=.77, bottom=.32, wspace=.65)
    for ax, metric, title in zip(axes[:2], ("mae", "q90_mae"),
                               ("All DOC observations", "High-DOC observations"), strict=True):
        for i, model in enumerate(models):
            ax.plot(i, summary.loc[model, metric], "s", color=COLORS[i], ms=5, zorder=4)
            values = parts[parts.model_name.eq(model)][metric].to_numpy()
            ax.scatter(i+np.linspace(-.15, .15, len(values)), values, s=12,
                       facecolors="white", edgecolors=COLORS[i], linewidths=.7, zorder=3)
        ax.set(xticks=np.arange(len(models)), xticklabels=np.arange(1, len(models)+1),
               xlabel="Procedure", ylabel="MAE (mg L$^{-1}$)", title=title)
        ax.grid(axis="y", color="#E8EBED", linewidth=.5)
    ax = axes[2]
    candidates = (("retrieval_contrast",)*3 if contrast else ("retrieval", "hydro_pretrained", "hydro_pretrained_retrieval"))
    references = (("current_model", "retrieval_contrast_uniform", "retrieval_contrast_zero_source_values")
                  if contrast else ("current_model",)*3)
    selected_rows = []
    for i, (model, reference) in enumerate(zip(candidates, references, strict=True)):
        row = effects[effects.candidate.eq(model) & effects.reference.eq(reference)
                      & effects.region.eq("overall")]
        if len(row) != 1:
            raise ValueError("Missing/duplicate paired comparison")
        row = row.iloc[0]
        selected_rows.append(row)
        color = COLORS[models.index(model)]
        ax.hlines(i, row.gain_ci_low_pct, row.gain_ci_high_pct, color=color, lw=1.5)
        ax.plot(row.relative_gain_pct, i, "s", ms=5, color=color)
    ax.axvline(0, color="#596168", lw=.7)
    paired = pd.DataFrame(selected_rows)
    limits = np.r_[paired.gain_ci_low_pct, paired.gain_ci_high_pct, 0]
    padding = .1*max(np.ptp(limits), .1)
    ax.set_xlim(limits.min()-padding, limits.max()+padding)
    ax.set(yticks=range(3), yticklabels=("vs current", "vs uniform", "vs no donors") if contrast
           else ("Retrieval", "Pretraining", "Both"), ylim=(2.5, -.5),
           xlabel="MAE reduction vs comparator (%)" if contrast else "MAE reduction vs current (%)",
           title="Paired development effects")
    ax.grid(axis="x", color="#E8EBED", linewidth=.5)
    for ax, letter in zip(axes, "abc", strict=True):
        ax.text(-.23, 1.09, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.suptitle("DOC reconstruction at unmonitored validation stations", y=.96, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], marker="s", ls="none", color=color, label=f"{i+1}. {name}")
               for i, (name, color) in enumerate(zip(labels, COLORS, strict=True))]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.51, .12),
               frameon=False, fontsize=7.5, columnspacing=1.1)
    packages = pd.read_csv(directory / "run_metrics.csv")[["split_seed", "seed"]].drop_duplicates()
    fig.text(.5, .075, f"Source validation only · {len(packages)} completed runs · model selection uses these stations",
             ha="center", fontsize=7, color="#596168")
    fig.text(.5, .036, "Squares: equal-partition means; open circles: partitions. Paired bars: station-bootstrap 95% intervals.",
             ha="center", fontsize=7, color="#596168")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    fig = draw(args.root)
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = out / f"doc_source_retrieval.{extension}"
        fig.savefig(path, dpi=300)
        outputs.append(path)
    plt.close(fig)
    inputs = [args.root / "analysis" / f"{name}.csv" for name in
              ("summary", "partition_metrics", "run_metrics", "paired_effects")]
    manifest = {"scope": "selected source-validation development; not independent confirmation",
                "inputs": {str(path): sha256_file(path) for path in inputs},
                "outputs": {str(path): sha256_file(path) for path in outputs}}
    (out / "sources.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(out / "doc_source_retrieval.png")


if __name__ == "__main__":
    main()
