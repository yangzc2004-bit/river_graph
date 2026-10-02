"""Publication figures from the completed unified DOC analysis, without refitting.

Run after analyze_unified_doc_spatial.py. Partial and smoke analyses are refused.
All plotted estimates and intervals are read from the existing analysis tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DEFAULT_ANALYSIS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/analysis")
BLUE = "#265D82"
ORANGE = "#C87932"
GREY = "#727B83"
MM = 1 / 25.4
K_VALUES = (0, 1, 3, 5)
MODELS = ("extra_trees", "extra_trees_calibrated", "hybrid", "hybrid_calibrated")
COMPARISONS = (
    ("hybrid_vs_et_k5_calibrated", "Hybrid vs trees\nK = 5, both calibrated", ORANGE),
    ("hybrid_vs_et_k0", "Hybrid vs trees\nK = 0", ORANGE),
    ("et_calibration_k5_vs_k0", "Trees\nK = 5 vs K = 0", BLUE),
    ("hybrid_calibration_k5_vs_k0", "Hybrid\nK = 5 vs K = 0", ORANGE),
)
SOURCES = ("status.json", "k_curves.csv", "split_metrics.csv", "primary_comparisons.csv",
           "station_heterogeneity.csv")


def load_analysis(directory: Path) -> dict:
    status = json.loads((directory / "status.json").read_text())
    if status.get("complete") is not True or status.get("confirmation_budget") is not True:
        raise ValueError("Manuscript figures require complete confirmation analysis; smoke/partial refused")
    if status.get("available_runs") != 9 or status.get("expected_runs") != 9:
        raise ValueError("Expected all nine partition-by-training-seed fits")
    curves = pd.read_csv(directory / "k_curves.csv")
    splits = pd.read_csv(directory / "split_metrics.csv")
    effects = pd.read_csv(directory / "primary_comparisons.csv")
    stations = pd.read_csv(directory / "station_heterogeneity.csv", dtype={"station": str})
    expected = pd.MultiIndex.from_product([K_VALUES, MODELS], names=["k", "model_name"])
    if not curves.set_index(["k", "model_name"]).index.sort_values().equals(expected.sort_values()):
        raise ValueError("K curves must contain each of the four arms at every K exactly once")
    if set(splits.split_seed) != {142, 143, 144} or not splits.n_seeds.eq(3).all():
        raise ValueError("Split metrics must describe the three intended partitions and three seeds")
    for _, group in splits.groupby("split_seed"):
        if not group.set_index(["k", "model_name"]).index.sort_values().equals(expected.sort_values()):
            raise ValueError("Each partition must contain the complete four-arm K curve")
    if set(effects.comparison) != {row[0] for row in COMPARISONS} or len(effects) != 4:
        raise ValueError("Expected all four paired comparisons")
    if not curves.n_splits.eq(3).all() or not effects.n_splits.eq(3).all():
        raise ValueError("Aggregate tables must include all three partitions")
    if not effects.bootstrap_draws.eq(5000).all():
        raise ValueError("Publication intervals require the planned 5,000 bootstrap draws")
    for table, columns in ((curves, ["mae"]), (splits, ["mae"]),
                           (effects, ["relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"])):
        if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite plotted estimates or intervals")
    if (effects.gain_ci_low_pct > effects.gain_ci_high_pct).any():
        raise ValueError("Interval lower bounds must not exceed upper bounds")
    return {"status": status, "curves": curves, "splits": splits,
            "effects": effects, "stations": stations}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "axes.labelsize": 8.3, "axes.titlesize": 9, "axes.titleweight": "regular",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#596168", "axes.linewidth": .6,
        "axes.labelcolor": "#252A2E", "text.color": "#252A2E",
        "xtick.color": "#444A50", "ytick.color": "#444A50",
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save_figure(fig, output: Path, name: str) -> None:
    # Fixed 170-mm page box: no tight bounding box that changes print width.
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=400)
    plt.close(fig)


def panel_label(ax, letter: str, x: float = -.18) -> None:
    ax.text(x, 1.07, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left")


def main_figure(data: dict, output: Path, show_partitions: bool) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(170 * MM, 96 * MM),
                                     gridspec_kw={"width_ratios": [1.14, 1]})
    fig.subplots_adjust(left=.09, right=.98, bottom=.29, top=.89, wspace=.92)
    curves, splits = data["curves"], data["splits"]
    legend = []
    for stem, label, color, marker in (("extra_trees", "Trees", BLUE, "o"),
                                      ("hybrid", "Hybrid", ORANGE, "s")):
        calibrated = curves[curves.model_name.eq(f"{stem}_calibrated")].sort_values("k")
        if show_partitions:
            for _, part in splits[splits.model_name.eq(f"{stem}_calibrated")].groupby("split_seed"):
                part = part.sort_values("k")
                left.plot(part.k, part.mae, color=color, alpha=.22, linewidth=.7, zorder=1)
        reference = float(curves[curves.model_name.eq(stem) & curves.k.eq(0)].mae.iloc[0])
        left.axhline(reference, color=color, linewidth=1.15, linestyle=(0, (4, 2.5)), zorder=2)
        left.plot(calibrated.k, calibrated.mae, color=color, linewidth=1.65,
                  marker=marker, markersize=4, markeredgewidth=.7,
                  markeredgecolor="white", zorder=3)
        legend.extend([
            Line2D([], [], color=color, linewidth=1.65, marker=marker, markersize=4,
                   label=f"{label} + calibration"),
            Line2D([], [], color=color, linewidth=1.15, linestyle=(0, (4, 2.5)),
                   label=f"{label}, K = 0"),
        ])
    left.set(xlabel="Observations per station (K)", ylabel="DOC MAE (mg L$^{-1}$)",
             xticks=K_VALUES, xlim=(-.2, 5.2), title="Reconstruction accuracy")
    left.yaxis.set_major_locator(MaxNLocator(5))
    left.grid(axis="y", linewidth=.5, color="#E8EBED", zorder=0)
    # Two rows per expert keep the line/marker encoding explicit, without
    # treating the faint partition trajectories as confidence intervals.
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.085, .015),
               frameon=False, ncol=2, fontsize=7.5, handlelength=2.5,
               labelspacing=.45, columnspacing=1.5, borderaxespad=0)
    panel_label(left, "a", -.23)

    effects = data["effects"].set_index("comparison")
    for position, (name, _, color) in enumerate(COMPARISONS):
        row = effects.loc[name]
        right.plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [position, position],
                   color=color, linewidth=1.2, solid_capstyle="round", zorder=2)
        right.plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [position, position],
                   linestyle="none", marker="|", color=color, markersize=5, markeredgewidth=.8)
        right.scatter([row.relative_gain_pct], [position], s=24, color=color,
                      marker="o" if name.startswith("et_") else "s",
                      edgecolor="white", linewidth=.5, zorder=3)
    right.axvline(0, color=GREY, linewidth=.8, zorder=0)
    lo = min(0, float(effects.gain_ci_low_pct.min()))
    hi = max(0, float(effects.gain_ci_high_pct.max()))
    pad = max(1, (hi - lo) * .09)
    right.set(xlim=(lo - pad, hi + pad), ylim=(3.55, -.55),
              yticks=range(4), yticklabels=[row[1] for row in COMPARISONS],
              xlabel="Relative MAE reduction (%)", title="Paired comparisons")
    right.xaxis.set_major_locator(MaxNLocator(4))
    right.tick_params(axis="y", length=0, pad=6, labelsize=7.8)
    right.spines["left"].set_visible(False)
    right.grid(axis="x", color="#EEF0F2", linewidth=.45)
    panel_label(right, "b", -.57)
    save_figure(fig, output, "unified_doc_spatial_main")


def station_figure(data: dict, output: Path) -> None:
    comparisons = (
        ("hybrid_vs_et_k5_calibrated", "Hybrid vs trees, K = 5", ORANGE),
        ("hybrid_calibration_k5_vs_k0", "Hybrid, K = 5 vs K = 0", BLUE),
    )
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 95 * MM), sharey=True)
    fig.subplots_adjust(left=.105, right=.975, bottom=.17, top=.87, wspace=.25)
    for ax, (name, title, color), letter in zip(axes, comparisons, "ab", strict=True):
        part = data["stations"].loc[data["stations"].comparison.eq(name)]
        if part.empty or not np.isfinite(part.relative_gain_pct).all():
            raise ValueError("Station responses must be present and finite")
        part = part.sort_values(["relative_gain_pct", "split_seed", "station"])
        rank = np.arange(1, len(part) + 1)
        for split, marker in ((142, "o"), (143, "s"), (144, "^")):
            selected = part.split_seed.eq(split).to_numpy()
            ax.scatter(part.relative_gain_pct.to_numpy()[selected], rank[selected],
                       s=12, marker=marker, color=color, alpha=.72, linewidth=0)
        ax.axvline(0, color=GREY, linewidth=.8)
        ax.set(title=title, xlabel="Station MAE reduction (%)", ylim=(-2, len(part) + 3))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.grid(axis="x", color="#EEF0F2", linewidth=.45)
        panel_label(ax, letter, -.18)
    axes[0].set_ylabel("Station–partition cases (ordered)")
    handles = [Line2D([], [], color=GREY, marker=marker, linestyle="none", markersize=4,
                      label=f"Partition {number}")
               for number, marker in enumerate(("o", "s", "^"), 1)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(.5, -.005))
    save_figure(fig, output, "unified_doc_station_responses")


def write_captions(data: dict, output: Path, show_partitions: bool) -> None:
    row = data["effects"].set_index("comparison").loc["hybrid_vs_et_k5_calibrated"]
    counts = data["splits"].loc[lambda df: df.k.eq(0) & df.model_name.eq("extra_trees")]
    populations = "; ".join(
        f"partition {index + 1}: {int(item.n_stations):,} stations, {int(item.n_query_cells):,} query cells"
        for index, item in enumerate(counts.sort_values("split_seed").itertuples()))
    split_note = (" Thin, pale lines show the three seed-averaged partition trajectories, "
                  "not confidence intervals." if show_partitions else "")
    caption = (
        "**Figure 3. Integrated DOC reconstruction with sparse station calibration.** "
        "**a,** Solid lines show calibrated ExtraTrees (blue circles) and hybrid (orange squares) "
        "predictions as target observations increase from K = 0 to 5. Dashed lines show each "
        "predictor's uncalibrated K = 0 reference. Bold trajectories average training seeds within "
        "each partition, then weight partitions equally." + split_note + " **b,** Paired relative "
        "MAE reductions; positive values favor the first predictor or K = 5, while negative values "
        "indicate higher error. Points and 95% intervals "
        "are the existing paired estimates and 5,000-resample joint station-bootstrap intervals. "
        "The same station receives the same resampling multiplicity across partitions, retaining "
        "repeated-station dependence and cell weighting within each partition. "
        f"The study includes three partitions, three training seeds per partition, and "
        f"{int(row.n_stations_unique):,} unique test stations ({populations}). Query cells remain "
        "fixed across arms and K within each partition. Support observations may follow query "
        "dates: the task is retrospective reconstruction within the existing river cohort. "
        "Trees denotes the validation-selected ExtraTrees context predictor; the hybrid may "
        "select its context-only fallback.\n\n"
        "**Supplementary figure. Station variation in reconstruction benefit.** "
        "**a,** Calibrated hybrid versus calibrated ExtraTrees at K = 5. **b,** Calibrated hybrid "
        "at K = 5 versus its uncalibrated K = 0 output. Each point is one station–partition case "
        "using seed-averaged station MAEs from the existing analysis. Cases are ordered separately "
        "in each panel; their rank positions do not identify matching stations between panels. "
        "Horizontal scales differ between panels. "
        "Positive values indicate lower error for the first predictor or K = 5. Shapes identify "
        "partitions 142, 143, and 144 in order. A station can appear in multiple partitions, so "
        "points are descriptive responses rather than independent replicates.\n"
    )
    (output / "captions.md").write_text(caption)


def render(directory: Path, *, show_partitions: bool = True) -> Path:
    data = load_analysis(directory)
    output = directory / "manuscript_figures"
    output.mkdir(parents=True, exist_ok=True)
    set_style()
    main_figure(data, output, show_partitions)
    station_figure(data, output)
    write_captions(data, output, show_partitions)
    sources = {str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in SOURCES}
    (output / "figure_sources.json").write_text(json.dumps({
        "source_files": sources, "figure_width_mm": 170,
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "show_partition_trajectories": show_partitions,
        "estimates": "Unchanged values read from completed analysis tables; no new inferential analysis",
    }, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--hide-partitions", action="store_true",
                        help="Omit pale partition trajectories from the mean K curves")
    args = parser.parse_args()
    print(render(args.analysis_dir, show_partitions=not args.hide_partitions))


if __name__ == "__main__":
    main()
