"""Three-panel publication figure from the completed fresh DOC analysis only.

No predictions, training data or fitted models are loaded. Aggregate estimates,
partition means and paired intervals come from the fixed analysis CSV tables.
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

ROOT = Path("experiments/phase4_transfer/doc_chemistry_confirmation_v1")
SPLITS, SEEDS, KS = (242, 243, 244), (42, 43, 44), (0, 1, 3, 5)
ACCEPTED = "neural_chemistry_integrated_selected"
GENERAL = "point_integrated_legacy"
TREE = "tree_chemistry_selected"
MODELS = (ACCEPTED, GENERAL, TREE)
# Same blue/orange/grey roots as plot_unified_doc_manuscript.py.
STYLE = {
    ACCEPTED: {"label": "Chemistry-aware neural", "color": "#C87932", "marker": "s", "linestyle": "-"},
    GENERAL: {"label": "General model", "color": "#727B83", "marker": "o", "linestyle": (0, (3, 2))},
    TREE: {"label": "Chemistry-aware trees", "color": "#265D82", "marker": "D", "linestyle": "-"},
}
PARTITION_MARKERS = ("o", "^", "D")
CSV_NAMES = ("k_curves", "metrics_by_run", "metrics_by_partition", "paired_effects",
             "directions_by_partition", "availability_by_partition", "query_population",
             "deployment_availability")
AVAILABILITY = ("both", "ph_only", "ec_only", "neither")
MM = 1 / 25.4
FIGURE_NAME = "doc_chemistry_confirmation"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(frame, keys, expected):
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError(f"Incomplete or duplicate figure population for {keys}")


def load_analysis(root):
    directory = root / "analysis"
    manifest_path = directory / "sources.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("complete") is not True or manifest.get("packages") != 9
            or tuple(manifest.get("partitions", ())) != SPLITS
            or tuple(manifest.get("training_seeds", ())) != SEEDS
            or manifest.get("bootstrap_draws") != 5000):
        raise ValueError("Publication figure requires all nine fresh production packages and 5000-draw analysis")
    data = {name: pd.read_csv(directory / f"{name}.csv") for name in CSV_NAMES}
    data["manifest"] = manifest
    data["source_files"] = {str(manifest_path): sha256(manifest_path),
        **{str(directory / f"{name}.csv"): sha256(directory / f"{name}.csv") for name in CSV_NAMES}}
    curves = data["k_curves"][data["k_curves"].model_name.isin(MODELS)].copy()
    parts = data["metrics_by_partition"][data["metrics_by_partition"].model_name.isin(MODELS)].copy()
    runs = data["metrics_by_run"][data["metrics_by_run"].model_name.isin(MODELS)].copy()
    _unique(curves, ["model_name", "k"], {(model, k) for model in MODELS for k in KS})
    _unique(parts, ["split_seed", "model_name", "k"],
            {(split, model, k) for split in SPLITS for model in MODELS for k in KS})
    _unique(runs, ["split_seed", "seed", "model_name", "k"],
            {(split, seed, model, k) for split in SPLITS for seed in SEEDS for model in MODELS for k in KS})
    if not curves.n_splits.eq(3).all() or not parts.n_seeds.eq(3).all():
        raise ValueError("Displayed means must contain three seeds and three partitions")
    for table in (curves, parts, runs):
        if not np.isfinite(table[["mae", "q90_mae"]].to_numpy()).all():
            raise ValueError("Nonfinite main or tail estimate; inspect analysis before rendering")
        if (table[["mae", "q90_mae"]] < 0).any().any():
            raise ValueError("Negative absolute-error estimate")
    if not curves.q90_nonempty_partitions.eq(3).all():
        raise ValueError("Tail curve cannot silently omit a partition")
    # Reconcile the existing summaries, without recomputing any cell error.
    for metric in ("mae", "q90_mae"):
        expected_parts = runs.groupby(["split_seed", "model_name", "k"])[metric].mean()
        actual_parts = parts.set_index(["split_seed", "model_name", "k"])[metric].loc[expected_parts.index]
        np.testing.assert_allclose(actual_parts, expected_parts, rtol=1e-12, atol=1e-12)
        expected_curves = expected_parts.groupby(["model_name", "k"]).mean()
        actual_curves = curves.set_index(["model_name", "k"])[metric].loc[expected_curves.index]
        np.testing.assert_allclose(actual_curves, expected_curves, rtol=1e-12, atol=1e-12)
    comparisons = tuple(f"accepted_vs_{GENERAL}_k{k}" for k in (0, 3, 5))
    effects = data["paired_effects"]
    effects = effects[effects.comparison.isin(comparisons) & effects.region.eq("overall") & effects.metric.eq("mae")].copy()
    _unique(effects, ["comparison"], {(name,) for name in comparisons})
    if (not effects.status.eq("estimated").all() or not effects.bootstrap_draws.eq(5000).all()
            or not effects.n_splits.eq(3).all()
            or not np.isfinite(effects[["delta_value", "delta_ci_low", "delta_ci_high"]]).all().all()
            or (effects.delta_ci_low > effects.delta_ci_high).any()):
        raise ValueError("Missing or invalid paired station-bootstrap intervals")
    directions = data["directions_by_partition"]
    directions = directions[directions.comparison.isin(comparisons) & directions.region.eq("overall")].copy()
    _unique(directions, ["comparison", "split_seed"], {(name, split) for name in comparisons for split in SPLITS})
    if not directions.n_seeds.eq(3).all():
        raise ValueError("Partition dots must average three seeds")
    for k in (0, 3, 5):
        name = f"accepted_vs_{GENERAL}_k{k}"
        delta = directions[directions.comparison.eq(name)].set_index("split_seed").delta_mae
        point = effects[effects.comparison.eq(name)].iloc[0]
        np.testing.assert_allclose(point.delta_value, delta.mean(), rtol=1e-12, atol=1e-12)
        for split in SPLITS:
            scores = parts[parts.split_seed.eq(split) & parts.k.eq(k)].set_index("model_name").mae
            np.testing.assert_allclose(delta.loc[split], scores.loc[ACCEPTED] - scores.loc[GENERAL],
                                       rtol=1e-12, atol=1e-12)
    availability = data["availability_by_partition"]
    availability = availability[availability.model_name.eq(GENERAL) & availability.k.eq(0)].copy()
    _unique(availability, ["split_seed", "aux_available"], {(split, group) for split in SPLITS for group in AVAILABILITY})
    populations = data["query_population"]
    _unique(populations, ["aux_available"], {(group,) for group in AVAILABILITY})
    deployment = data["deployment_availability"]
    _unique(deployment, ["cohort", "aux_available"],
            {(cohort, group) for cohort in ("all_station_months", "doc_observed", "doc_genuinely_missing")
             for group in AVAILABILITY})
    counts = parts[parts.model_name.eq(GENERAL) & parts.k.eq(0)].set_index("split_seed")
    for split in SPLITS:
        group = availability[availability.split_seed.eq(split)]
        if int(group.n_query_cells.sum()) != int(counts.loc[split, "n_query_cells"]):
            raise ValueError("Auxiliary-availability denominator differs from the plotted query population")
    if int(populations.n_split_cell_occurrences.sum()) != int(counts.n_query_cells.sum()):
        raise ValueError("Unique/query occurrence accounting differs")
    for _, group in deployment.groupby("cohort"):
        if group.cohort_denominator.nunique() != 1 or int(group.n_station_months.sum()) != int(group.cohort_denominator.iloc[0]):
            raise ValueError("Deployment availability denominator differs")
    data.update(curves=curves, parts=parts, effects=effects, directions=directions,
                availability=availability, counts=counts)
    return data


def set_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "axes.titleweight": "regular",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#596168", "axes.linewidth": .6,
        "axes.labelcolor": "#252A2E", "text.color": "#252A2E",
        "xtick.color": "#444A50", "ytick.color": "#444A50",
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white"})


def draw_figure(data):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 104 * MM))
    fig.subplots_adjust(left=.075, right=.986, top=.765, bottom=.255, wspace=.64)
    fig.suptitle("DOC reconstruction on new station assignments", y=.965, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], **{key: value for key, value in STYLE[model].items()
                               if key != "label"}, markersize=4, linewidth=1.45,
                      label=STYLE[model]["label"]) for model in MODELS]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.52, .895),
               frameon=False, fontsize=7.6, handlelength=2.1, columnspacing=1.5)
    for axis, metric, title in ((axes[0], "mae", "All query observations"),
                                (axes[2], "q90_mae", "High-DOC observations")):
        for model in MODELS:
            points = data["curves"][data["curves"].model_name.eq(model)].sort_values("k")
            style = STYLE[model]
            axis.plot(points.k, points[metric], color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.45, markersize=4, markeredgewidth=.45,
                markeredgecolor="white", zorder=3)
        axis.set(xlabel="Support observations (K)", xticks=KS, xlim=(-.2, 5.2),
                 ylim=(0, None), title=title)
        axis.set_ylabel("MAE (mg L$^{-1}$)" if metric == "mae" else "Q90-tail MAE (mg L$^{-1}$)")
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.grid(axis="y", color="#E8EBED", linewidth=.5, zorder=0)
    ax = axes[1]
    effects = data["effects"].set_index("comparison")
    for position, k in enumerate((0, 3, 5)):
        name = f"accepted_vs_{GENERAL}_k{k}"
        row = effects.loc[name]
        ax.hlines(position, row.delta_ci_low, row.delta_ci_high, color=STYLE[ACCEPTED]["color"], linewidth=1.5)
        ax.plot([row.delta_ci_low, row.delta_ci_high], [position, position], "|",
                color=STYLE[ACCEPTED]["color"], markersize=5, markeredgewidth=.8)
        ax.plot(row.delta_value, position, "s", color=STYLE[ACCEPTED]["color"],
                markersize=5, markeredgewidth=.6, markeredgecolor="white", zorder=4)
        dots = data["directions"][data["directions"].comparison.eq(name)].set_index("split_seed")
        for split, marker, offset in zip(SPLITS, PARTITION_MARKERS, (-.22, -.12, .14), strict=True):
            ax.plot(dots.loc[split, "delta_mae"], position + offset, marker=marker,
                    linestyle="none", color="#727B83", markerfacecolor="white",
                    markersize=3.4, markeredgewidth=.7, zorder=3)
    ax.axvline(0, color="#596168", linewidth=.7, zorder=0)
    all_x = np.r_[data["effects"].delta_ci_low, data["effects"].delta_ci_high,
                  data["directions"].delta_mae, 0.]
    span = max(float(np.ptp(all_x)), .01)
    ax.set(xlim=(all_x.min() - .1 * span, all_x.max() + .1 * span), ylim=(2.5, -.5),
           yticks=range(3), yticklabels=[f"K = {k}" for k in (0, 3, 5)],
           xlabel="Neural − general MAE\n(mg L$^{-1}$)", title="Paired model difference")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#EEF0F2", linewidth=.45, zorder=0)
    for axis, letter in zip(axes, "abc", strict=True):
        axis.text(-.24, 1.095, letter, transform=axis.transAxes, fontsize=11, fontweight="bold", va="bottom")
    partition_handles = [Line2D([], [], color="#727B83", marker=marker, markerfacecolor="white",
        markersize=3.8, markeredgewidth=.7, linestyle="none", label=f"Partition {i + 1}")
        for i, marker in enumerate(PARTITION_MARKERS)]
    partition_handles.append(Line2D([], [], color=STYLE[ACCEPTED]["color"], marker="s",
        linewidth=1.5, markersize=4, label="Overall + paired 95% CI"))
    fig.legend(handles=partition_handles, ncol=4, loc="lower center", bbox_to_anchor=(.525, .095),
               frameon=False, fontsize=7.1, handlelength=1.7, columnspacing=1.3)
    fig.text(.5, .065, "Panel b: partition dots average three seeds; they are not confidence intervals.",
             ha="center", fontsize=7, color="#596168")
    fig.text(.5, .027, "Curves: equal partition means  ·  Paired intervals: 5,000 joint station-bootstrap draws",
             ha="center", fontsize=7, color="#596168")
    return fig


def write_readme(data, output):
    counts, availability = data["counts"], data["availability"]
    population = data["query_population"]
    unique = int(population.n_station_months_unique.sum())
    occurrences = int(population.n_split_cell_occurrences.sum())
    active_unique = int(population.loc[~population.aux_available.eq("neither"), "n_station_months_unique"].sum())
    lines = ["# Fresh station-assignment DOC confirmation figure", "",
        "## Figure caption", "",
        ("**DOC reconstruction using hydrological history, auxiliary chemistry and sparse station calibration.** "
        "**a,** Native-scale mean absolute error (MAE) over the fixed held-station query observations, "
        "as the retrospective DOC support set increases from K = 0 to 5. "
        "**b,** Paired MAE difference between the chemistry-aware neural procedure and the retained general model "
        "at K = 0, 3 and 5; negative values favor the chemistry-aware model. Filled squares and horizontal bars "
        "show the aggregate difference and its paired 95% percentile interval from 5,000 joint whole-station "
        "bootstrap draws. Open markers show individual partition estimates, each averaged over three training "
        "seeds. They describe partition variation, not uncertainty intervals. "
        "**c,** MAE conditional on observed DOC being at or above the source-training 90th percentile. "
        "Tail thresholds are defined separately by source partition. Panels a and c show seed-averaged metrics "
        "within each partition and equal weights across the three partitions; these curves carry no confidence bands."), "",
        ("The three partitions use new station-role assignments on the existing ST357 Mississippi cohort. "
        "They are not external-basin validation or new independent measurements. K counts retrospective support "
        "observations per held station. All five reserved support candidates are excluded from the query set at "
        "every K, so its population remains fixed. Training seeds do not increase ecological sample size."), "",
        "## Curves and information scope", "",
        "| Legend | Frozen analysis model |", "|---|---|"]
    lines.extend(f"| {STYLE[model]['label']} | `{model}` |" for model in MODELS)
    lines += ["", ("The chemistry-aware neural and tree procedures select their support representation using source "
        "validation only, among legacy temporal, availability-augmented and chemistry-augmented coordinates. "
        "Each uses the same held-station support/query identities. Auxiliary pH and specific conductance are "
        "observed calendar-month means, not necessarily measurements taken at the same sampling instant. "
        "When both are absent, predictions retain the corresponding general neural/tree fallback after adaptation."), "",
        "## Evaluation denominators", "",
        "| Partition | Partition seed | Held stations | Fixed query cells | Auxiliary observed | Q90 cells |",
        "|---|---:|---:|---:|---:|---:|"]
    for i, split in enumerate(SPLITS):
        row = counts.loc[split]
        part = availability[availability.split_seed.eq(split)]
        active = int(part.loc[~part.aux_available.eq("neither"), "n_query_cells"].sum())
        total = int(row.n_query_cells)
        lines.append(f"| {i + 1} | {split} | {int(row.n_stations):,} | {total:,} | "
                     f"{active:,}/{total:,} ({100 * active / total:.1f}%) | {int(row.q90_n):,} |")
    lines += ["", (f"The evaluation contains **{unique:,} unique station-months** and **{occurrences:,} partition-cell "
        f"occurrences**. Of the unique query cells, {active_unique:,}/{unique:,} "
        f"({100 * active_unique / unique:.1f}%) have at least one auxiliary chemistry observation. "
        "The occurrence count retains cells that appear in multiple partitions; the bootstrap samples such "
        "stations jointly. It does not treat seed repeats as new observations."), "",
        "| Unique query availability | Station-months | Stations (may overlap groups) | Partition-cell occurrences |",
        "|---|---:|---:|---:|"]
    for group in AVAILABILITY:
        row = population[population.aux_available.eq(group)].iloc[0]
        lines.append(f"| {group.replace('_', ' ')} | {int(row.n_station_months_unique):,} | "
                     f"{int(row.n_stations_unique):,} | {int(row.n_split_cell_occurrences):,} |")
    lines += ["", "## Availability beyond observed DOC queries", "",
        "This table describes covariate coverage, not predictive accuracy at missing DOC cells.", "",
        "| Full-grid population | Auxiliary observed | Population denominator | Fraction |",
        "|---|---:|---:|---:|"]
    for cohort in ("all_station_months", "doc_observed", "doc_genuinely_missing"):
        deployment = data["deployment_availability"]
        group = deployment[deployment.cohort.eq(cohort)]
        active = int(group.loc[~group.aux_available.eq("neither"), "n_station_months"].sum())
        denominator = int(group.cohort_denominator.iloc[0])
        lines.append(f"| {cohort.replace('_', ' ')} | {active:,} | {denominator:,} | {100 * active / denominator:.1f}% |")
    if counts.q90_unstable.any():
        lines += ["", "At least one partition contains fewer than 20 Q90 query cells; its tail estimate is unstable."]
    lines += ["", "## Files and reproduction", "",
        ("The PDF and SVG are vector figures; the PNG is 300 dpi. White backgrounds, DejaVu Sans typography "
        "and the manuscript's blue/orange/grey palette are used. Shape and line style provide redundant model "
        "encodings. Panel b has a visible zero reference; absolute-error axes start at zero."), "",
        "Run from the repository root after the complete analysis:", "", "```bash",
        "uv run python scripts/plot_doc_chemistry_confirmation_v1.py", "```", "",
        ("The source manifest records every input CSV, the complete-analysis manifest and the plotting script. "
        "All numbers are read from those analysis products. This command does not fit models, read target "
        "prediction parquet files, recalculate intervals or choose models from plotted outcomes."), ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true", help="Validate complete analysis tables without rendering")
    args = parser.parse_args()
    data = load_analysis(args.root)
    if args.check_only:
        print("Complete nine-package analysis tables reconcile; figure inputs ready")
        return
    output = args.root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure = draw_figure(data)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(output / f"{FIGURE_NAME}.{suffix}", dpi=300)
    plt.close(figure)
    write_readme(data, output)
    artifacts = [output / f"{FIGURE_NAME}.{suffix}" for suffix in ("pdf", "svg", "png")] + [output / "README.md"]
    manifest = {"figure": FIGURE_NAME, "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "sources": data["source_files"], "outputs": {path.name: sha256(path) for path in artifacts},
        "models": {model: STYLE[model]["label"] for model in MODELS}, "partitions": list(SPLITS),
        "training_seeds": list(SEEDS), "k_values": list(KS), "bootstrap_draws": 5000,
        "paired_interval": "95% percentile joint whole-station bootstrap; repeated stations sampled jointly",
        "partition_dots": "three-seed means, not confidence intervals",
        "render": {"width_mm": 183, "height_mm": 104, "png_dpi": 300,
                   "vector_formats": ["pdf", "svg"], "font": "DejaVu Sans"}}
    (output / "sources_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {FIGURE_NAME}.pdf/.svg/.png, README and source manifest in {output}")


if __name__ == "__main__":
    main()
