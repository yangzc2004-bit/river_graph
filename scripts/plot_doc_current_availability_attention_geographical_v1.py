"""Draw the fixed DOC geographical comparison from verified analysis tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_doc_chemistry_confirmation_v1 import set_style

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_current_availability_attention_geographical_v1")
MODELS = ("unmonitored_integrated", "relative_real_integrated", "level_real_integrated",
          "station_hidden_trees", "available_seasonal_integrated", "available_real_integrated")
LABELS = ("Retained complete", "Matched anomaly-only", "Matched full source",
          "Strong station-hidden trees", "Expanded seasonal only", "All current source DOC")
COLORS = ("#727B83", "#80ABA0", "#7896AF", "#ACBDBA", "#BFAD7C", "#307F74")
REGIONS = (1013, 1019, 708, 1030, 1101)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    folder = args.root / "analysis"
    parts = pd.read_csv(folder / "primary_region_metrics.csv")
    summary = pd.read_csv(folder / "primary_summary.csv")
    curves = pd.read_csv(folder / "curves_summary.csv")
    effects = pd.read_csv(folder / "paired_effects.csv")
    if set(parts.split_seed) != set(REGIONS) or not set(MODELS) <= set(parts.model_name):
        raise ValueError("all five fixed geographical regions are required")
    for name in ("mae", "q90_mae"):
        grouped = parts.groupby("model_name")[name].mean()
        for row in summary.itertuples():
            if np.isfinite(getattr(row, name)):
                np.testing.assert_allclose(getattr(row, name), grouped[row.model_name])
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.7))
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.22, wspace=.40, hspace=.52)
    for ax, metric, title in ((axes[0, 0], "mae", "Unmonitored geographical regions"),
                              (axes[1, 1], "q90_mae", "High-DOC reconstruction")):
        for j, model in enumerate(MODELS):
            series = parts[parts.model_name.eq(model)].set_index("split_seed").reindex(REGIONS)[metric]
            ax.plot(np.arange(5)+(j-(len(MODELS)-1)/2)*.12, series, "o", ls="none", color=COLORS[j], ms=5)
        ax.set(xticks=range(5), xticklabels=[str(r).zfill(4) for r in REGIONS],
               xlabel="Held-out HUC4 region", ylabel="MAE (mg L$^{-1}$)", title=title)
        ax.grid(axis="y", color="#E8EBED", lw=.5)
    ax = axes[0, 1]
    rows = []
    for i, reference in enumerate(MODELS[:-1]):
        selected = effects[effects.population.eq("all_observed_k0") & effects.zone.eq("overall")
            & effects.k.eq(0) & effects.reference.eq(reference) & effects.candidate.eq(MODELS[-1])]
        if len(selected) != 1:
            raise ValueError("missing paired geographical effect")
        row = selected.iloc[0]
        rows.append(row)
        ax.hlines(i, row.gain_ci_low_pct, row.gain_ci_high_pct, color=COLORS[-1], lw=1.5)
        ax.plot(row.relative_gain_pct, i, "s", color=COLORS[-1], ms=5)
        ax.text(.98, i, f"{int(row.improving_regions)}/5 regions", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7, color="#596168")
    extents = np.r_[[r.gain_ci_low_pct for r in rows], [r.gain_ci_high_pct for r in rows], 0]
    pad = max(np.ptp(extents), .1)*.12
    ax.set(xlim=(extents.min()-pad, extents.max()+pad), ylim=(4.7, -.7), yticks=range(5),
           yticklabels=("vs retained full", "vs anomaly-only", "vs matched full", "vs strong trees", "vs seasonal only"), xlabel="MAE reduction (%)",
           title="Current availability: paired K0 gain")
    ax.axvline(0, color="#596168", lw=.7)
    ax.grid(axis="x", color="#E8EBED", lw=.5)
    ax = axes[1, 0]
    for model, label, color in zip(MODELS, LABELS, COLORS, strict=True):
        rows = curves[curves.model_name.eq(model)].set_index("k").reindex([0, 1, 3, 5])
        if len(rows) != 4 or rows.mae.isna().any():
            raise ValueError("incomplete fixed-query support curve")
        ax.plot(rows.index, rows.mae, "o-", color=color, lw=1.2, ms=4, label=label)
    ax.set(xticks=[0, 1, 3, 5], xlabel="DOC support observations per station",
           ylabel="MAE (mg L$^{-1}$)", title="Matched fixed-query adaptation")
    ax.grid(axis="y", color="#E8EBED", lw=.5)
    for letter, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-.19, 1.06, letter, fontsize=11, fontweight="bold", transform=ax.transAxes)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.52, .12), ncol=3,
               frameon=False, fontsize=7.5)
    fig.suptitle("Available source experience across river regions", y=.97, fontsize=12, fontweight="bold")
    seeds = pd.read_csv(folder / "primary_run_metrics.csv").seed.nunique()
    fig.text(.5, .065, f"Retrospective ST357 replication · {seeds} seeds · equal region weights · station-clustered 95% intervals",
             ha="center", fontsize=7, color="#596168")
    fig.text(.5, .033, "Main K0: every observed test cell. K curves: identical query with five candidate supports reserved. Q90: source-derived threshold.",
             ha="center", fontsize=7, color="#596168")
    output = args.root / "figures"
    output.mkdir(exist_ok=True)
    products = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"doc_current_availability_attention_geographical_replication.{extension}"
        fig.savefig(path, dpi=300)
        products.append(path)
    plt.close(fig)
    inputs = [folder / name for name in ("primary_region_metrics.csv", "primary_summary.csv",
        "curves_summary.csv", "paired_effects.csv", "primary_run_metrics.csv")]
    (output / "sources.json").write_text(json.dumps({"inputs": {str(p): sha256_file(p) for p in inputs},
        "outputs": {str(p): sha256_file(p) for p in products}}, indent=2)+"\n")
    print(output / "doc_current_availability_attention_geographical_replication.png")


if __name__ == "__main__":
    main()
