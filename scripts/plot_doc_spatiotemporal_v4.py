"""Create the five main paper figures from consolidated, saved DOC evidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from consolidate_doc_spatiotemporal_v4 import MAIN_MODELS, OUTPUT, TASKS
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path("docs/paper/latex/figures/spatiotemporal_v4")
COLORS = {"trees": "#71838D", "previous": "#C3A56B", "current": "#247F83", "native": "#6585AF"}
INK = "#253746"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#88939D", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})


def save(fig, name):
    ROOT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = ROOT / f"{name}.{extension}"
        fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=.08)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def panels(axes):
    for letter, ax in zip("abcdef", np.asarray(axes).ravel(), strict=False):
        ax.text(-.12, 1.075, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")


def interval(ax, row, y, color):
    lo, hi, point = row.gain_ci_low_pct, row.gain_ci_high_pct, row.relative_gain_pct
    ax.plot([lo, hi], [y, y], color=color, lw=1.7)
    ax.plot([lo, lo], [y-.065, y+.065], color=color, lw=1.2)
    ax.plot([hi, hi], [y-.065, y+.065], color=color, lw=1.2)
    ax.plot(point, y, "o", color=color, ms=6)


def architecture():
    fig, ax = plt.subplots(figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.025, right=.985, bottom=.025, top=.985)
    ax.set(xlim=(0, 12), ylim=(0, 8.8))
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.06,rounding_size=.10",
            edgecolor=color, facecolor=color+"10", lw=1.25))
        ax.text(x+w/2, y+h-.22, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x+w/2, y+h*.39, body, ha="center", va="center", fontsize=8.8, linespacing=1.35)

    def arrow(a, b, color="#74818B", dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
            color=color, lw=1.25, linestyle="--" if dashed else "-"))

    ax.text(.05, 8.55, "Two missing-data tasks, one information-fusion framework", fontweight="bold", fontsize=14)
    for x, title, local in ((.12, "Temporal reconstruction", True), (6.12, "Spatial reconstruction", False)):
        ax.text(x, 8.10, title, fontsize=10.5, fontweight="bold")
        for row in range(3):
            for month in range(9):
                visible = (month < 5 or (row == 1 and month == 7)) if local else row < 2
                ax.add_patch(Rectangle((x+month*.34, 7.26+row*.16), .29, .12,
                    facecolor=COLORS["current"] if visible else "#EDF0F2", edgecolor="white", lw=.3))
        ax.text(x+3.25, 7.66, "Observed history + gaps" if local else "Source stations + unseen site", fontsize=8.5)
        ax.text(x+3.25, 7.34, "Local DOC visible where allowed" if local else "No local DOC / pH / conductance", fontsize=8.1)
    ax.text(.13, 6.98, "Filled cells: permitted DOC observations    Pale cells: hidden DOC", fontsize=8.2, color="#64727C")
    box(.2, 4.86, 2.85, 1.66, "Receiving-site inputs", "Ecology + coordinates\nTemperature + discharge\nSeason + daily hydrology\nObservation masks + age", COLORS["trees"])
    box(4.0, 5.41, 3.1, 1.11, "Environmental baseline", "Station-blocked ExtraTrees\nBackground DOC estimate C", COLORS["trees"])
    box(4.0, 3.40, 3.1, 1.53, "Ecological temporal state", "Ecology encoder: 32 dimensions\nObservation-aware GRU: hidden 64\nPast 12 months, including current", COLORS["native"])
    box(.2, 1.41, 2.85, 2.75, "Monitored source bank", "Ecology + daily hydrology\nPermitted current DOC\nDouble-held OOF residual values\nUp to 20 ecological candidates\nAvailability mask + zero prior", COLORS["previous"])
    box(4.0, 1.14, 3.1, 1.60, "Source residual attention", "Query: receiving state + environment\nKeys: source environment\nValues: OOF log-DOC residuals\nTwo heads, 32 dimensions per head", COLORS["current"])
    box(8.05, 3.51, 3.52, 2.08, "Residual reconstruction", "Local temporal correction d\nSource-attended correction g\nZero-initialized residual readout\n"+r"$P=\max\{0,C+s(d+g)\}$", COLORS["current"])
    box(8.05, .92, 3.52, 1.79, "Task-specific output fusion", "Temporal: validation-selected fusion\nSpatial: selected source-memory blend\nDOC reconstruction in mg/L\nSeparate fitted weights for each task", COLORS["trees"])
    arrow((3.08, 6.02), (3.88, 6.02))
    arrow((1.62, 4.25), (1.62, 4.76), dashed=True)
    ax.text(1.83, 4.50, "Source context", fontsize=8, va="center", color="#64727C")
    arrow((3.08, 5.04), (3.88, 4.47))
    arrow((3.08, 2.08), (3.88, 2.08))
    arrow((5.55, 3.32), (5.55, 2.84))
    ax.text(5.73, 3.02, "Q", fontsize=9, color=COLORS["native"])
    ax.text(3.35, 2.26, "K, V", fontsize=9, color=COLORS["previous"])
    arrow((7.18, 6.0), (7.92, 5.09))
    arrow((7.18, 4.22), (7.92, 4.22))
    arrow((7.18, 1.94), (7.92, 3.71))
    arrow((9.80, 3.40), (9.80, 2.83))
    ax.text(.2, .53, "Shared architecture; separately fitted temporal and spatial procedures.", fontsize=8.7)
    ax.text(.2, .18, "Source links encode ecological / hydrological similarity. Current model uses no neural river-edge messages.", fontsize=8.3, color="#64727C")
    save(fig, "fig1_framework")


def temporal(summary, effects):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    fig.subplots_adjust(left=.20, right=.98, bottom=.13, top=.88, hspace=.60, wspace=.55)
    for column, task in enumerate(("e2a_strict", "e2b_partial")):
        data = summary[summary.task_id.eq(task)].set_index("model_name")
        names = ["station_hidden_trees", "upgraded_fusion", "available_native", "available_fusion"]
        ax = axes[0, column]
        values = data.loc[names, "mae"]
        ax.bar(range(4), values, color=[COLORS[k] for k in ("trees", "previous", "native", "current")], width=.6)
        for x, value in enumerate(values):
            ax.text(x, value+.015, f"{value:.3f}", ha="center", fontsize=9)
        ax.set(xticks=range(4), xticklabels=["Trees", "Previous\ncomplete", "Current\nnative", "Current\ncomplete"],
               ylabel=r"MAE (mg L$^{-1}$)", ylim=(0, 1.13), title=TASKS[task])
        ax = axes[1, column]
        rows = effects[effects.task_id.eq(task)]
        for y, (candidate, reference) in enumerate((("available_fusion", "station_hidden_trees"),
                ("available_fusion", "upgraded_fusion"), ("available_native", "station_hidden_trees"))):
            row = rows[rows.candidate.eq(candidate) & rows.reference.eq(reference)].iloc[0]
            interval(ax, row, y, COLORS["current"] if candidate == "available_fusion" else COLORS["native"])
        ax.axvline(0, color="#8F989F", lw=.8)
        ax.set(yticks=range(3), yticklabels=["Complete vs trees", "Complete vs previous", "Native vs trees"],
               xlabel="MAE reduction (%)", ylim=(2.5, -.5), xlim=(-2, 18))
    panels(axes)
    fig.suptitle("Temporal DOC reconstruction", x=.11, ha="left", fontsize=15, fontweight="bold")
    save(fig, "fig2_temporal")


def spatial(regions, effects):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.25))
    fig.subplots_adjust(left=.085, right=.98, bottom=.24, top=.83, wspace=.63)
    data = regions[regions.task_id.eq("geographical")]
    order = (1013, 1019, 708, 1030, 1101)
    labels = ("Strong trees", "Previous complete", "Current complete")
    for index, (name, label, key) in enumerate(zip(MAIN_MODELS["geographical"], labels, ("trees", "previous", "current"), strict=True)):
        points = data[data.model_name.eq(name)].set_index("split_seed").reindex(order)
        axes[0].plot(np.arange(5)+(index-1)*.15, points.mae, "o", ls="none", color=COLORS[key], label=label, ms=6)
    axes[0].set(xticks=range(5), xticklabels=[str(x).zfill(4) for x in order],
        ylabel=r"MAE (mg L$^{-1}$)", xlabel="Held-out HUC4 region", title="Reconstruction without local chemistry", ylim=(0, 6.2))
    rows = effects[effects.task_id.eq("geographical") & effects.candidate.eq("available_real_integrated")]
    for y, reference in enumerate(("station_hidden_trees", "unmonitored_integrated")):
        row = rows[rows.reference.eq(reference)].iloc[0]
        interval(axes[1], row, y, COLORS["current"])
        axes[1].text(row.relative_gain_pct, y-.18, f"{row.relative_gain_pct:.2f}%", ha="center", fontsize=9)
    axes[1].axvline(0, color="#8F989F", lw=.8)
    axes[1].set(yticks=(0, 1), yticklabels=("vs strong trees", "vs previous complete"),
                xlabel="MAE reduction (%)", title="Equal-region paired improvement", ylim=(1.65, -.65), xlim=(-.4, 7.8))
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.6)
    panels(axes)
    fig.suptitle("Spatial DOC reconstruction", x=.085, ha="left", fontsize=15, fontweight="bold")
    save(fig, "fig3_spatial")


def information(summary, effects):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(left=.25, right=.98, bottom=.18, top=.81, wspace=.48)
    data = summary[summary.task_id.eq("geographical")].set_index("model_name")
    names = ("available_seasonal_integrated", "level_real_integrated", "available_real_integrated")
    labels = ("Seasonal source values\nexpanded availability", "Current source values\nearlier-support requirement", "Current source values\nactual availability")
    for y, name in enumerate(names):
        value = data.loc[name, "mae"]
        axes[0].plot(value, y, "o", color=COLORS["current"] if y == 2 else COLORS["previous"], ms=7)
        axes[0].text(value, y-.19, f"{value:.4f}", ha="center", fontsize=9)
    axes[0].set(yticks=range(3), yticklabels=labels, ylim=(2.55, -.55), xlim=(2.240, 2.278),
                xlabel=r"Equal-region MAE (mg L$^{-1}$)", title="Matched source-information controls")
    rows = effects[effects.task_id.eq("geographical") & effects.candidate.eq("available_real_integrated")]
    for y, name in enumerate(("available_seasonal_integrated", "level_real_integrated", "unmonitored_integrated")):
        row = rows[rows.reference.eq(name)].iloc[0]
        interval(axes[1], row, y, COLORS["current"])
    axes[1].axvline(0, color="#8F989F", lw=.8)
    axes[1].set(yticks=range(3), yticklabels=["vs seasonal values", "vs restricted access", "vs previous complete"],
                xlabel="MAE reduction (%)", ylim=(2.55, -.55), xlim=(-.15, 2.95), title="Increment and accumulated benefit")
    panels(axes)
    fig.suptitle("What contemporaneous source information adds", x=.08, ha="left", fontsize=15, fontweight="bold")
    save(fig, "fig4_information")


def external(summary, curves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fig.subplots_adjust(left=.09, right=.98, bottom=.21, top=.82, wspace=.35)
    data = summary[summary.task_id.eq("external")].set_index("model_name")
    for index, (name, key) in enumerate(zip(MAIN_MODELS["external"], ("trees", "previous", "current"), strict=True)):
        value = data.loc[name, "mae"]
        axes[0].bar(index, value, color=COLORS[key], width=.58)
        axes[0].text(index, value+.014, f"{value:.3f}", ha="center", fontsize=9)
        rows = curves[curves.task_id.eq("external") & curves.model_name.eq(name)].sort_values("k")
        axes[1].plot(rows.k, rows.mae, "o-", color=COLORS[key], lw=1.5, ms=5,
                     label=("Strong trees", "Previous complete", "Current complete")[index])
    axes[0].set(xticks=range(3), xticklabels=["Strong\ntrees", "Previous\ncomplete", "Current\ncomplete"],
                ylim=(0, 1.09), ylabel=r"MAE (mg L$^{-1}$)", title="All-observation reconstruction")
    axes[1].set(xticks=(0, 1, 3, 5), xlabel="Designated DOC observations per station", ylabel=r"MAE (mg L$^{-1}$)",
                title="Adaptation on the same query cells", ylim=(.65, 1.02))
    axes[1].legend(frameon=False, fontsize=8.8, loc="lower left")
    panels(axes)
    fig.suptitle("External application in HUC02040104", x=.09, ha="left", fontsize=15, fontweight="bold")
    save(fig, "fig5_external")


def supplementary(curves):
    diag = pd.read_csv(OUTPUT / "external_diagnostics.csv")
    names = MAIN_MODELS["external"]
    data = diag[diag.population.eq("all_observed_k0") & diag.model_name.isin(names)].set_index("model_name")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.27, top=.81, wspace=.45)
    for index, (name, key) in enumerate(zip(names, ("trees", "previous", "current"), strict=True)):
        row = data.loc[name]
        axes[0].bar(index, row.coverage*100, color=COLORS[key], width=.55)
        axes[0].errorbar(index, row.coverage*100,
            yerr=np.array([(row.coverage-row.coverage_ci_low)*100, (row.coverage_ci_high-row.coverage)*100]).reshape(2, 1),
            color=INK, capsize=3, lw=1)
        axes[1].bar(index, row.width_median, color=COLORS[key], width=.55)
        axes[2].bar(index, row.q90_mae, color=COLORS[key], width=.55)
    for ax in axes:
        ax.set_xticks(range(3), ["Trees", "Previous", "Current"])
    axes[0].axhline(90, color=INK, ls="--", lw=.8)
    axes[0].set(ylabel="Coverage (%)", ylim=(0, 105), title="Empirical intervals")
    axes[1].set(ylabel=r"Median width (mg L$^{-1}$)", title="Interval width", ylim=(0, 4))
    axes[2].set(ylabel=r"Tail MAE (mg L$^{-1}$)", title="High DOC: 16 unique cells", ylim=(0, 12))
    panels(axes)
    fig.suptitle("External uncertainty and high-DOC diagnostics", x=.08, ha="left", fontsize=13, fontweight="bold")
    save(fig, "supp1_uncertainty_tail")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.subplots_adjust(left=.16, right=.97, bottom=.18, top=.85)
    for name, key, label in zip(MAIN_MODELS["geographical"], ("trees", "previous", "current"),
                              ("Strong trees", "Previous complete", "Current complete"), strict=True):
        rows = curves[curves.task_id.eq("geographical") & curves.model_name.eq(name)].sort_values("k")
        ax.plot(rows.k, rows.mae, "o-", color=COLORS[key], label=label, lw=1.5)
    ax.set(xticks=(0, 1, 3, 5), xlabel="Designated DOC observations per station", ylabel=r"MAE (mg L$^{-1}$)",
           title="Internal fixed-query adaptation")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "supp2_internal_support")


def environmental_strata():
    """Retain the saved stratum estimates and bins, using the manuscript palette."""
    strata = (("hydro_channels_available", "Available hydro channels", ("None", "One", "Both")),
              ("ecological_novelty", "Ecological novelty", ("Low", "Middle", "High")),
              ("source_similarity_distance", "Nearest-source distance", ("Low", "Middle", "High")))
    geo = pd.read_csv(OUTPUT / "geographical_strata.csv")
    geo = geo[geo.k.eq(0) & geo.stratum.isin([s[0] for s in strata])].copy()
    geo["group"] = pd.to_numeric(geo["group"], errors="raise").astype(int)
    external_data = pd.read_csv(OUTPUT / "external_strata.csv")
    external_data["stratum"] = external_data.stratum.str.replace("_group", "", regex=False)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.1))
    for row, table in enumerate((geo, external_data)):
        for column, (stratum, title, tick_names) in enumerate(strata):
            ax = axes[row, column]
            finite = []
            for name, label, key in zip(MAIN_MODELS["geographical"],
                    ("Strong trees", "Previous complete", "Current complete"),
                    ("trees", "previous", "current"), strict=True):
                part = table[table.model_name.eq(name) & table.stratum.eq(stratum)].set_index("group")
                if not part.index.is_unique:
                    raise ValueError("stratum summaries contain duplicate groups")
                values = part.mae.reindex(range(3)).to_numpy()
                finite.extend(values[np.isfinite(values)])
                ax.plot(range(3), values, "o-", color=COLORS[key], label=label, lw=1.5, ms=5)
            count = table[table.model_name.eq("station_hidden_trees") & table.stratum.eq(stratum)].set_index("group")
            ticks = []
            for group, name in enumerate(tick_names):
                note = "empty" if group not in count.index else (
                    f"{int(count.loc[group, 'n_regions'])} regions" if row == 0 else
                    f"{int(count.loc[group, 'n_stations'])} stations")
                ticks.append(f"{name}\n{note}")
            ax.set(xticks=range(3), xticklabels=ticks, title=title,
                   ylabel=r"MAE (mg L$^{-1}$)", ylim=(0, max(finite)*1.23))
            ax.tick_params(axis="x", labelsize=8.5)
            ax.grid(axis="y", color="#E7ECEF", lw=.6)
            ax.set_axisbelow(True)
    panels(axes)
    fig.suptitle("DOC reconstruction across environmental strata", x=.07, ha="left", y=.98,
                 fontsize=14, fontweight="bold")
    fig.text(.07, .885, "Geographical withholding", fontsize=10.5, fontweight="bold")
    fig.text(.07, .475, "External application", fontsize=10.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .038), frameon=False, ncol=3)
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.15, hspace=.70, wspace=.43)
    save(fig, "supp3_environment")


def main():
    style()
    summary = pd.read_csv(OUTPUT / "all_model_metrics.csv")
    effects = pd.read_csv(OUTPUT / "paired_effects.csv")
    regions = pd.read_csv(OUTPUT / "internal_region_metrics.csv")
    curves = pd.read_csv(OUTPUT / "support_curves.csv")
    architecture()
    temporal(summary, effects)
    spatial(regions, effects)
    information(summary, effects)
    external(summary, curves)
    supplementary(curves)
    environmental_strata()
    print(ROOT)


if __name__ == "__main__":
    main()
