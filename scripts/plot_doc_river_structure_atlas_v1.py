"""Publication figures for the physical river and source DOC atlas."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.topology.river_structure import PROFILES

ROOT = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1")
TEAL, ORANGE, SLATE = "#267F88", "#C77C48", "#46586A"
LABELS = {"overall": "All stations", "low_order": "Low-order tributary",
          "chain": "Chain conveyance", "confluence": "Major confluence",
          "mainstem": "Integrated mainstem", "storage": "Lake / reservoir path"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": SLATE, "text.color": SLATE, "xtick.color": SLATE,
        "ytick.color": SLATE, "axes.edgecolor": "#C4CED2", "pdf.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white"})


def finish(fig, path, caption, *, left=.16, bottom=.18, wspace=.40):
    fig.text(.02, .02, caption, fontsize=9, va="bottom", color=SLATE)
    fig.subplots_adjust(left=left, bottom=bottom, top=.88, wspace=wspace, hspace=.48)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=230, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def topology_figure(structures, edges, counts, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Physical river settings behind the monitoring graph", fontsize=17, x=.02, ha="left")
    ax = axes[0, 0]
    orders = structures.stream_order.value_counts().sort_index()
    ax.bar(orders.index, orders.values, color=TEAL, width=.78)
    ax.set(xticks=range(1, 11), xlabel="Strahler stream order", ylabel="Stations", title="a  Network position")
    for order, n in orders.items():
        ax.text(order, n+1, str(n), ha="center", fontsize=9)
    ax.set_ylim(0, orders.max()*1.18)
    ax = axes[0, 1]
    values = [int(structures.sampled_in_degree.eq(0).sum()), int(structures.physical_headwater.sum())]
    ax.barh([1, 0], values, color=[ORANGE, TEAL], height=.6)
    ax.set(yticks=[1, 0], yticklabels=["No sampled upstream station", "Physical headwater reach"],
           xlabel="Stations", title="b  Monitoring gaps and physical headwaters")
    for y, value in zip([1, 0], values, strict=True):
        ax.text(value+4, y, str(value), va="center")
    ax.set_xlim(0, max(values)*1.20)
    ax = axes[1, 0]
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.n_all_stations, color=TEAL, height=.62)
    ax.set(yticks=y, yticklabels=[LABELS[n] for n in counts.profile], xlabel="Stations", title="c  Overlapping process settings")
    for i, row in enumerate(counts.itertuples()):
        ax.text(row.n_all_stations+3, y[i], str(row.n_all_stations), va="center")
    ax.set_xlim(0, counts.n_all_stations.max()*1.16)
    ax = axes[1, 1]
    length = edges.loc[edges.mainstem_connected, "path_length_km"].to_numpy()
    ax.hist(length, bins=np.geomspace(length.min(), length.max()*1.01, 17), color=TEAL, edgecolor="white")
    ax.axvline(np.median(length), color=ORANGE, lw=1.8, ls="--", label=f"Median {np.median(length):.0f} km")
    ax.set(xscale="log", xlabel="Physical distance between linked stations (km)", ylabel="Monitored edges", title="d  What one sampled graph edge spans")
    ax.legend(frameon=False)
    finish(fig, output/"river_structure_overview",
           "ST357, 357 station reaches and 324 monitored edges. Physical topology comes from the full NHDPlus VAA cache.\n"
           "Profiles overlap; a chain describes the nearest 5 km. Lake / reservoir paths use a 20 km upstream window.", left=.22, wspace=.65)


def doc_figure(source, output):
    # A station contributes once, after equal averaging its available source
    # partition summaries. This view is descriptive and station weighted.
    metrics = ["doc_median", "doc_cv", "cq_log1p_slope"]
    station = source.groupby("station", as_index=False).agg(
        **{name: (name, "mean") for name in metrics},
        **{name: (name, "first") for name in PROFILES},
        stream_order=("stream_order", "first"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8))
    fig.suptitle("DOC behaviour across physical river settings", fontsize=17, x=.02, ha="left")
    for ax, metric, title, ylabel in zip(axes, metrics,
        ["a  Station concentration", "b  Temporal variability", "c  Hydrologic association"],
        ["Station median DOC (mg/L)", "DOC coefficient of variation", "Season-adjusted log1p C–Q slope"], strict=True):
        groups = [station.loc[station[name], metric].dropna().to_numpy() for name in PROFILES]
        box = ax.boxplot(groups, patch_artist=True, widths=.58, showfliers=True,
            medianprops={"color": "white", "linewidth": 1.6},
            boxprops={"edgecolor": TEAL}, whiskerprops={"color": TEAL}, capprops={"color": TEAL},
            flierprops={"markersize": 3, "markeredgecolor": SLATE, "alpha": .45})
        for item in box["boxes"]:
            item.set_facecolor(TEAL)
            item.set_alpha(.8)
        ax.set(xticks=np.arange(1, 6), xticklabels=[f"{label}\n(n={len(values)})" for label, values in
            zip(["Tributary", "Chain", "Confluence", "Mainstem", "Storage"], groups, strict=True)],
            ylabel=ylabel, title=title)
        ax.tick_params(axis="x", rotation=35)
        if metric == "doc_median":
            ax.set_yscale("log")
        if metric == "cq_log1p_slope":
            ax.axhline(0, color=ORANGE, ls="--", lw=1)
    finish(fig, output/"doc_structure_behaviour",
           "Source-training observations only, partitions 142 / 143 / 144. Each station appears once after averaging its available partition summaries.\n"
           "Profiles overlap. C–Q slopes require 24 paired DOC–flow months and adjust for annual sine/cosine; associations are observational.", left=.08, bottom=.32)


def effect_panel(ax, rows, *, title, color):
    groups = ["overall", *PROFILES]
    ax.axvline(0, color="#ABB8BC", lw=1)
    ax.set_title(title, loc="left", fontsize=11)
    for i, name in enumerate(groups):
        sub = rows[rows.group.eq(name)]
        if len(sub) != 1:
            continue
        r = sub.iloc[0]
        y = len(groups)-1-i
        ax.plot([r.gain_ci_low_pct, r.gain_ci_high_pct], [y, y], color=color, lw=1.7)
        ax.scatter([r.relative_gain_pct], [y], s=42, edgecolors=color,
                   facecolors="white" if r.few_stations else color, zorder=3)
    ax.set(yticks=np.arange(len(groups))[::-1], yticklabels=[LABELS[n] for n in groups],
           xlabel="MAE reduction (%) — positive is better", ylim=(-.7, len(groups)-.3))
    for i, name in enumerate(groups):
        row = rows[rows.group.eq(name)]
        if len(row):
            ax.text(.99, len(groups)-1-i+.15, f"n={int(row.n_stations_unique.iloc[0])}",
                    transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=SLATE)


def model_figure(current, historical, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Where model information adds value", fontsize=17, x=.02, ha="left")
    current = current[~current["tail"]]
    historical = historical[~historical["tail"] & historical.candidate.eq("residual_upstream")]
    effect_panel(axes[0, 0], current[current.reference.eq("station_hidden_trees")],
                 title="a  Current complete model vs strong trees", color=TEAL)
    effect_panel(axes[0, 1], current[current.reference.eq("available_seasonal_integrated")],
                 title="b  Actual-source vs seasonal-source model", color=TEAL)
    effect_panel(axes[1, 0], historical[historical["mask"].eq("e2a_strict")],
                 title="c  Historical upstream messages: temporal validation", color=ORANGE)
    effect_panel(axes[1, 1], historical[historical["mask"].eq("e3_spatial_seed42")],
                 title="d  Historical upstream messages: spatial-task validation", color=ORANGE)
    finish(fig, output/"model_gain_by_river_structure",
           "Points and 95% paired station-bootstrap intervals (5,000 draws); n denotes distinct stations. Losses are averaged across training seeds.\n"
           "Top: selected source-validation roles, no physical river messages. Bottom: separate older validation tasks, upstream vs no-message residual.\n"
           "Historical spatial-task validation includes source stations and cannot establish new-station transfer. Panels use independent x scales.", left=.20, wspace=.75)


def pair_figure(correlation, balanced, output):
    # Each physical edge contributes once at each lag after source-partition
    # averaging. Shared stations/edges still prevent independent-pair inference.
    qualified = correlation[correlation.rho_seasonal_anomaly.notna()]
    pair = qualified.groupby(["source", "target", "lag_months"], as_index=False).agg(
        rho=("rho_seasonal_anomaly", "mean"), n_pairs=("n_pairs", "mean"),
        path_length_km=("path_length_km", "first"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.suptitle("Upstream DOC association at monthly resolution", fontsize=17, x=.02, ha="left")
    lags = [0, 1, 3, 6, 12]
    groups = [pair.loc[pair.lag_months.eq(lag), "rho"].to_numpy() for lag in lags]
    box = axes[0].boxplot(groups, patch_artist=True, widths=.6,
        medianprops={"color": "white"}, flierprops={"markersize": 3, "markeredgecolor": SLATE, "alpha": .4})
    for patch in box["boxes"]:
        patch.set_facecolor(TEAL)
    fixed = balanced.groupby(["source", "target", "lag_months"], as_index=False).rho_seasonal_anomaly.mean()
    medians = fixed.groupby("lag_months").rho_seasonal_anomaly.median().reindex(lags)
    fixed_count = fixed[["source", "target"]].drop_duplicates().shape[0]
    axes[0].scatter(range(1, 6), medians, color=ORANGE, marker="D", s=27, zorder=4,
                    label=f"Fixed-edge subset median (n={fixed_count})")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    axes[0].set(xticks=range(1, 6), xticklabels=[f"{lag}\n(n={len(g)})" for lag, g in zip(lags, groups, strict=True)],
                xlabel="Source lag (months)", ylabel="Seasonal-anomaly Spearman correlation",
                ylim=(-1.08, 1.08), title="a  All prespecified lags")
    axes[0].axhline(0, color=ORANGE, ls="--", lw=1)
    same = pair[pair.lag_months.eq(0)]
    axes[1].scatter(same.path_length_km, same.rho, color=TEAL, s=20, alpha=.7)
    axes[1].axhline(0, color=ORANGE, ls="--", lw=1)
    axes[1].set(xscale="log", xlabel="Along-river station separation (km)", ylabel="Same-month seasonal-anomaly correlation",
                ylim=(-1.08, 1.08), title="b  Physical separation and association")
    finish(fig, output/"upstream_doc_associations",
           "Source-training readings only; at least 12 paired months per edge and partition. Edge summaries average available source partitions.\n"
           "Pair counts vary by lag. Shared stations and irregular sampling limit comparison; monthly correlations do not identify travel time or causal effects.", left=.13, bottom=.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    style()
    analysis, output = args.root/"analysis", args.root/"figures"
    output.mkdir(parents=True, exist_ok=True)
    def load(name):
        return pd.read_csv(analysis/name, dtype={"station": str, "source": str, "target": str})
    topology_figure(load("station_structure.csv"), load("station_edge_paths.csv"), load("profile_counts.csv"), output)
    doc_figure(load("source_doc_station_summaries.csv"), output)
    model_figure(load("current_model_structure_effects.csv"), load("historical_message_structure_effects.csv"), output)
    pair_figure(load("source_upstream_doc_associations.csv"), load("balanced_source_upstream_doc_associations.csv"), output)
    print(f"Four figures exported as PNG, PDF and SVG: {output}", flush=True)


if __name__ == "__main__":
    main()
