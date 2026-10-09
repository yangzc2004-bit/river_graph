"""Research figures for river typology, DOC response and adjusted associations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_structure_clustering_v1")
COLORS = {1: "#267F88", 2: "#C77C48", 3: "#626D9E", 4: "#9B8C43", 5: "#B06982", 6: "#46586A"}
FEATURE_LABELS = {"stream_order": "River order", "log_slope": "Slope", "log_area": "Drainage area",
    "junction_density_20": "Junctions / length (20 km)", "junction_density_5": "Junctions / length (5 km)",
    "log_length_20": "Upstream length (20 km)", "log_length_5": "Upstream length (5 km)",
    "log_length_50": "Upstream length (50 km)", "major_fraction_20": "Major junction fraction",
    "tributary_balance_5": "Tributary area balance", "storage_fraction_20km": "Storage (20 km)",
    "storage_fraction_5km": "Storage (5 km)", "storage_fraction_50km": "Storage (50 km)"}
FINE_NAMES = {1: "Small steep tributaries", 2: "Confluent small tributaries",
              3: "Confluent intermediate rivers", 4: "Few-confluence intermediate rivers",
              5: "Low-gradient integrated rivers", 6: "Storage-influenced large rivers"}


def save(fig, out, name, note):
    fig.text(.02, .015, note, fontsize=8, va="bottom", color="#46586A")
    fig.subplots_adjust(bottom=.16, top=.91, left=.09, right=.97, wspace=.38, hspace=.55)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"{name}.{ext}", dpi=220, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def typology(s, centroid, candidates, stability, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("River organization across 357 monitoring reaches", fontsize=17, x=.03, ha="left")
    for c, sub in s.groupby("cluster"):
        axes[0, 0].scatter(sub.longitude, sub.latitude, s=25, color=COLORS[c], edgecolors="white", linewidths=.3,
                           label=f"Type {c} (n={len(sub)})", alpha=.85)
    axes[0, 0].set(xlabel="Longitude (°)", ylabel="Latitude (°)", title="a  Station geography")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    ax = axes[0, 1]
    im = ax.imshow(centroid.to_numpy(), cmap="BrBG", vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set(yticks=np.arange(len(centroid)), yticklabels=[f"Type {i}" for i in centroid.index],
           xticks=np.arange(len(centroid.columns)), xticklabels=[FEATURE_LABELS[n] for n in centroid.columns],
           title="b  Physical feature centroids")
    ax.tick_params(axis="x", labelrotation=55, labelsize=8)
    fig.colorbar(im, ax=ax, shrink=.75, label="Block-weighted standardized feature")
    # Centroid labels intentionally have their own full-height row; move the
    # lower panels down so long feature names cannot touch them.
    ax = axes[1, 0]
    ax.plot(candidates.k, candidates.silhouette, "o-", color=COLORS[1])
    ax.set(xticks=candidates.k, xlabel="Number of classes", ylabel="Silhouette score", title="c  Geometry-only class selection")
    for r in candidates.itertuples():
        ax.annotate(f"Smallest n={r.minimum_class}", (r.k, r.silhouette), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    low, high = candidates.silhouette.min(), candidates.silhouette.max()
    ax.set_ylim(low-.012, high+.016)
    ax = axes[1, 1]
    data = [stability.loc[stability.k.eq(k), "ari"] for k in candidates.k]
    box = ax.boxplot(data, patch_artist=True, tick_labels=candidates.k, widths=.55)
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS[1])
        patch.set_alpha(.65)
    ax.set(xlabel="Number of classes", ylabel="Adjusted Rand index", title="d  Station-subset stability", ylim=(0, 1.05))
    save(fig, out, "river_structure_classification",
         "Structure-only Ward hierarchy; 3–6 candidate classes. Storage fractions use log1p(100 × fraction).\n"
         "Stability: 100 refits on 80% station subsets; preprocessing refit each time. Geography is a station-location plot.")


def response(stats, season, panel, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("DOC concentration, variability and seasonal response", fontsize=17, x=.03, ha="left")
    metrics = [("doc_median", "a  Concentration", "Station median DOC (mg/L)"),
               ("doc_cv", "b  Temporal variability", "Station DOC coefficient of variation"),
               ("q90_fraction", "c  High DOC frequency", "Fraction above source-pool Q90"),
               ("harmonic_amplitude", "d  Annual seasonality", "Peak-to-trough log1p DOC amplitude"),
               ("cq_slope", "e  Concentration–flow response", "Season/year-adjusted log1p C–Q slope")]
    for ax, (metric, title, label) in zip(axes.ravel(), metrics):
        for row in stats[stats.metric.eq(metric)].itertuples():
            ax.plot([row.cluster, row.cluster], [row.ci_low, row.ci_high], color=COLORS[row.cluster], lw=2)
            ax.scatter(row.cluster, row.median, s=65, color=COLORS[row.cluster], zorder=3)
            ax.annotate(f"n={row.n_stations}", (row.cluster, row.ci_high), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        ax.set(xticks=sorted(panel.cluster.unique()), xticklabels=[f"Type {c}" for c in sorted(panel.cluster.unique())],
               xlabel="River structure class", ylabel=label, title=title)
        ax.margins(y=.22, x=.3)
        if metric == "cq_slope":
            ax.axhline(0, color="#A8B2B8", lw=1, ls="--")
    ax = axes[1, 2]
    for c, sub in season.groupby("cluster"):
        ax.plot(sub.month, sub.doc_station_mean, color=COLORS[c], label=f"Type {c}", marker="o", ms=3)
        ax.fill_between(sub.month, sub.ci_low, sub.ci_high, color=COLORS[c], alpha=.12)
    ax.set(xticks=[1, 4, 7, 10, 12], xlim=(1, 12), xlabel="Calendar month", ylabel="Mean of station monthly medians (mg/L)",
           title="f  Calendar-month pattern")
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, "doc_response_by_river_type",
         "Deduplicated source-training DOC cells (142 / 143 / 144); each eligible station contributes once.\n"
         "Points: class medians; intervals: 95% station-bootstrap intervals (2,000 draws). Seasonal lines: station means with pointwise intervals.\n"
         "Station eligibility: ≥12 DOC months and ≥6 calendar months. C–Q additionally needs ≥24 paired DOC–flow months.")


def contrasts(table, cv, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle("River structure associations after environmental adjustment", fontsize=17, x=.03, ha="left")
    for ax, metric, title, label in zip(axes.ravel()[:3], ["doc_median", "doc_cv", "cq_slope"],
        ["a  Concentration", "b  Variability", "c  Concentration–flow response"],
        ["Class contrast in log(1 + median DOC)", "Class contrast in log(1 + CV)", "Class contrast in C–Q slope"], strict=True):
        for offset, mode, mark in [(-.08, "class_hydro_adjusted", "o"), (.08, "class_no_local_hydro", "D")]:
            sub = table[table.metric.eq(metric) & table.model.eq(mode)]
            for i, row in enumerate(sub.itertuples()):
                c = int(row.term.split("_")[1])
                ax.plot([row.ci_low, row.ci_high], [c+offset, c+offset], color=COLORS[c], lw=1.6)
                if mark == "o":
                    ax.plot([row.huc4_ci_low, row.huc4_ci_high], [c+offset, c+offset], color=COLORS[c], lw=.7, alpha=.6)
                ax.scatter(row.estimate, c+offset, color=COLORS[c], marker=mark, facecolors="none" if mark == "D" else COLORS[c], s=42,
                           label=("Including local hydrology" if mark == "o" else "Without local hydrology") if i == 0 else None)
        ax.axvline(0, color="#A8B2B8", lw=1)
        groups = sorted(int(t.split("_")[1]) for t in table[table.metric.eq(metric) & table.model.eq("class_hydro_adjusted")].term)
        ax.set(yticks=groups, yticklabels=[f"Type {c} vs Type 1" for c in groups], ylim=(min(groups)-.5, max(groups)+.5), xlabel=label, title=title)
        if metric == "doc_median":
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 1]
    models = ["environment", "environment_classes", "environment_structure"]
    means = cv.groupby("model").mae_log1p.mean().reindex(models)
    ax.bar(range(3), means, color=["#A8B2B8", COLORS[1], COLORS[2]], width=.55)
    for fold, sub in cv.groupby("fold"):
        vals = sub.set_index("model").mae_log1p.reindex(models)
        ax.plot(range(3), vals, "o-", color="#46586A", alpha=.4, lw=.8, ms=3)
    ax.set(xticks=range(3), xticklabels=["Environment", "+ River classes", "+ Structure"], ylabel="Held-out station log1p MAE",
           title="d  Five HUC4-blocked folds", ylim=(0, max(cv.mae_log1p)*1.15))
    save(fig, out, "adjusted_structure_associations",
         "Station-equal regression; controls include ecology, climate, sampling, geography and HUC2. Bootstrap intervals condition on the fixed classification.\n"
         "Thin intervals additionally resample HUC4 basins. Open diamonds omit local hydrology. Bars: equal-fold mean; connected dots: folds, ridge α=10.\n"
         "Classes use all available physical geometry, without DOC; this is exploratory association analysis, not independent external validation.")


def curves_figure(curves, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Continuous river structure and adjusted DOC concentration", fontsize=17, x=.03, ha="left")
    for ax, (name, sub) in zip(axes.ravel(), curves.groupby("feature", sort=False)):
        x = sub.x_transformed.to_numpy()
        if name == "log_area":
            x, label = np.expm1(x), "Drainage area (km²)"
            ax.set_xscale("log")
        elif name == "stream_order":
            label = "Strahler river order"
        elif name == "junction_density_20":
            x, label = np.expm1(x), "Junctions per 100 km of upstream network"
        elif name == "storage_fraction_20km":
            x, label = np.expm1(x), "Upstream lake / reservoir path length (%)"
        else:
            label = "Minor-branch drainage-area share (5 km)"
        ax.plot(x, sub.adjusted_log1p_doc, color=COLORS[1], lw=2)
        ax.fill_between(x, sub.ci_low, sub.ci_high, color=COLORS[1], alpha=.18)
        ax.set(xlabel=label, ylabel="Adjusted log(1 + median DOC)", title=FEATURE_LABELS[name])
    save(fig, out, "continuous_structure_response",
         "Additive quadratic splines, five quantile knots; remaining structure and environmental covariates included.\n"
         "Curves span source-station 10th–90th percentiles. Other design columns held at their observed mean; bands: pointwise station-bootstrap 95% intervals.\n"
         "Conditional associations describe the fitted model; they are not physical interventions or transport coefficients.")


def fine_hierarchy(physical, response, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    fig.suptitle("Six physical subtypes within the river hierarchy", fontsize=17, x=.03, ha="left")
    order = physical.fine_class.tolist()
    ypos = np.arange(len(order))[::-1]
    labels = [f"{FINE_NAMES[c]}\n(n={int(physical.loc[physical.fine_class.eq(c), 'n_stations'].iloc[0])})" for c in order]
    # Colour encodes the three coarse parent classes, not DOC concentration.
    for ax, field, title, xlabel in [(axes[0], "storage", "a  Upstream storage", "Lake / reservoir path length (%)"),
        (axes[1], "doc_median", "b  DOC concentration", "Station median DOC (mg/L)"),
        (axes[2], "harmonic_amplitude", "c  Seasonality", "Peak-to-trough log1p DOC amplitude")]:
        for c, y in zip(order, ypos, strict=True):
            row = physical[physical.fine_class.eq(c)].iloc[0]
            color = COLORS[int(row.parent_class)]
            if field == "storage":
                ax.scatter(100*row.storage, y, color=color, s=55)
            else:
                r = response[response.cluster.eq(c) & response.metric.eq(field)].iloc[0]
                ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=1.6)
                ax.scatter(r["median"], y, color=color, s=55)
        ax.set(yticks=ypos, yticklabels=labels if field == "storage" else [], xlabel=xlabel,
               ylim=(-.6, len(order)-.4), title=title)
        ax.margins(x=.15)
    fig.text(.02, .015,
        "The six-class cut is a structural sensitivity, nested within the selected three coarse classes; colour denotes the coarse parent.\n"
        "Names describe structural medians. All-reach n appears at left; DOC eligibility and response n are recorded in the underlying table.\n"
        "DOC points are station medians with 95% station-bootstrap intervals. The six-class cut has median subset ARI 0.707, versus 0.329 at three classes.",
        fontsize=8, va="bottom", color="#46586A")
    fig.subplots_adjust(left=.27, right=.98, bottom=.19, top=.88, wspace=.34)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"six_subtype_doc_response.{ext}", dpi=220, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    a, out = args.root/"analysis", args.root/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#B6C2C8", "pdf.fonttype": 42, "svg.fonttype": "none"})
    s = pd.read_csv(a/"station_classification.csv", dtype={"station": str})
    typology(s, pd.read_csv(a/"weighted_centroids.csv", index_col=0), pd.read_csv(a/"cluster_candidates.csv"),
             pd.read_csv(a/"cluster_stability.csv"), out)
    response(pd.read_csv(a/"class_doc_response.csv"), pd.read_csv(a/"class_seasonal_response.csv"),
             pd.read_csv(a/"station_doc_response.csv"), out)
    contrasts(pd.read_csv(a/"adjusted_associations.csv"), pd.read_csv(a/"huc4_blocked_comparison.csv"), out)
    curves_figure(pd.read_csv(a/"adjusted_structure_curves.csv"), out)
    fine_hierarchy(pd.read_csv(a/"six_class_physical_sensitivity.csv"), pd.read_csv(a/"six_class_doc_sensitivity.csv"), out)
    receipt = {str(p): sha256_file(p) for p in [*sorted(a.glob("*")), Path(__file__)] if p.is_file()}
    (args.root/"figure_sources.json").write_text(json.dumps({"input_hashes": receipt,
        "figure_hashes": {str(p): sha256_file(p) for p in sorted(out.glob("*"))}}, indent=2)+"\n")
    print("Five figure families generated (PNG / PDF / SVG).")


if __name__ == "__main__":
    main()
