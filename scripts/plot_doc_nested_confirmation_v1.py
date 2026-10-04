"""Render fixed nested-DOC confirmation curves and paired intervals as vectors.

Only source-bound analysis tables are read. No fitting, target selection or
interval calculation occurs in this plotting command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_nested_confirmation_v1 import (
    GENERAL,
    JOINT,
    KS,
    LEGACY,
    MASKS,
    MODELS,
    NESTED,
    ROOT,
    SEEDS,
    SPLITS,
    TREE,
)
from analyze_unified_doc_spatial import sha256
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from plot_doc_chemistry_confirmation_v1 import _unique, set_style

FIGURE_NAME = "doc_nested_confirmation"
CSV_NAMES = ("k_curves", "metrics_by_run", "metrics_by_partition", "paired_effects",
             "directions_by_partition", "query_population", "deployment_availability")
# Fixed semantic order and redundant line/marker encodings; no outcome ranking.
PLOT_MODELS = (NESTED, JOINT, LEGACY, MASKS, GENERAL, TREE)
STYLE = {
    NESTED: {"label": "Separate chemical increment", "color": "#265D82", "marker": "s", "linestyle": "-"},
    JOINT: {"label": "Joint-selected calibration", "color": "#C87932", "marker": "o", "linestyle": "-"},
    LEGACY: {"label": "Legacy calibration", "color": "#727B83", "marker": "^", "linestyle": "--"},
    MASKS: {"label": "Availability-only increment", "color": "#7097B0", "marker": "v", "linestyle": ":"},
    GENERAL: {"label": "General model", "color": "#A0A5AA", "marker": "D", "linestyle": "-."},
    TREE: {"label": "Chemical trees", "color": "#36796C", "marker": "P", "linestyle": "--"},
}
PARTITION_MARKERS = ("o", "^", "D")
MM = 1 / 25.4


def load_analysis(root):
    out = root / "analysis"
    manifest_path = out / "sources.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("complete") is not True or manifest.get("packages") != 9
            or tuple(manifest.get("partitions", ())) != SPLITS
            or tuple(manifest.get("training_seeds", ())) != SEEDS
            or tuple(manifest.get("models", ())) != MODELS
            or manifest.get("bootstrap_draws") != 5000):
        raise ValueError("Figure requires the complete six-model, nine-package 5000-draw analysis")
    for name in CSV_NAMES:
        path = out / f"{name}.csv"
        if manifest.get("outputs", {}).get(path.name) != sha256(path):
            raise ValueError(f"Changed or unbound figure input: {path}")
    data = {name: pd.read_csv(out / f"{name}.csv") for name in CSV_NAMES}
    data["source_files"] = {str(path): sha256(path) for path in
                            (manifest_path, *(out / f"{name}.csv" for name in CSV_NAMES))}
    data["manifest"] = manifest
    curves, parts, runs = data["k_curves"], data["metrics_by_partition"], data["metrics_by_run"]
    _unique(curves, ["model_name", "k"], {(m, k) for m in MODELS for k in KS})
    _unique(parts, ["split_seed", "model_name", "k"], {(s, m, k) for s in SPLITS for m in MODELS for k in KS})
    _unique(runs, ["split_seed", "seed", "model_name", "k"],
            {(s, r, m, k) for s in SPLITS for r in SEEDS for m in MODELS for k in KS})
    if not curves.n_splits.eq(3).all() or not parts.n_seeds.eq(3).all():
        raise ValueError("Displayed means must include all seeds and partitions")
    for metric in ("mae", "q90_mae"):
        if not all(np.isfinite(table[metric]).all() and table[metric].ge(0).all()
                   for table in (runs, parts, curves)):
            raise ValueError("Nonfinite error curve; missing tail partitions must be disclosed")
        p = runs.groupby(["split_seed", "model_name", "k"])[metric].mean()
        np.testing.assert_allclose(parts.set_index(["split_seed", "model_name", "k"]).loc[p.index, metric], p,
                                   rtol=1e-12, atol=1e-12)
        c = p.groupby(["model_name", "k"]).mean()
        np.testing.assert_allclose(curves.set_index(["model_name", "k"]).loc[c.index, metric], c,
                                   rtol=1e-12, atol=1e-12)
    comparisons = {f"nested_vs_joint_k{k}" for k in (3, 5)}
    effects = data["paired_effects"]
    effects = effects[effects.comparison.isin(comparisons) & effects.metric.eq("mae") & effects.region.eq("overall")]
    directions = data["directions_by_partition"]
    directions = directions[directions.comparison.isin(comparisons) & directions.region.eq("overall")]
    _unique(effects, ["comparison"], {(name,) for name in comparisons})
    _unique(directions, ["comparison", "split_seed"], {(name, s) for name in comparisons for s in SPLITS})
    if (not effects.status.eq("estimated").all() or not effects.bootstrap_draws.eq(5000).all()
            or not effects.n_splits.eq(3).all() or not directions.n_seeds.eq(3).all()
            or not np.isfinite(effects[["delta_value", "delta_ci_low", "delta_ci_high"]]).all().all()
            or (effects.delta_ci_low > effects.delta_ci_high).any()):
        raise ValueError("Invalid paired intervals or partition means")
    for k in (3, 5):
        name = f"nested_vs_joint_k{k}"
        ds = directions[directions.comparison.eq(name)].set_index("split_seed").delta_mae
        estimate = effects[effects.comparison.eq(name)].iloc[0]
        np.testing.assert_allclose(estimate.delta_value, ds.mean(), rtol=1e-12, atol=1e-12)
        for split in SPLITS:
            p = parts[parts.split_seed.eq(split) & parts.k.eq(k)].set_index("model_name").mae
            np.testing.assert_allclose(ds.loc[split], p.loc[NESTED] - p.loc[JOINT], rtol=1e-12, atol=1e-12)
    data.update(curves=curves, effects=effects, directions=directions)
    return data


def draw_figure(data):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 108 * MM))
    fig.subplots_adjust(left=.075, right=.986, top=.715, bottom=.285, wspace=.64)
    fig.suptitle("Separating chemical and temporal station calibration", y=.967, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], **STYLE[m], markersize=3.8, linewidth=1.35) for m in PLOT_MODELS]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(.52, .906),
               frameon=False, fontsize=7.2, columnspacing=1.65, handlelength=2.1, labelspacing=.7)
    for axis, metric, title in ((axes[0], "mae", "All query observations"),
                                (axes[2], "q90_mae", "High-DOC observations")):
        for model in PLOT_MODELS:
            curve = data["curves"][data["curves"].model_name.eq(model)].sort_values("k")
            style = {key: value for key, value in STYLE[model].items() if key != "label"}
            axis.plot(curve.k, curve[metric], **style, linewidth=1.35, markersize=3.8,
                      markeredgewidth=.4, markeredgecolor="white", zorder=3)
        values = data["curves"][metric].to_numpy()
        padding = .13 * max(float(np.ptp(values)), .01)
        axis.set(xlabel="Support observations (K)", xticks=KS, xlim=(-.2, 5.2), title=title,
                 ylim=(max(0., float(values.min()) - padding), float(values.max()) + padding))
        axis.set_ylabel("MAE (mg L$^{-1}$)" if metric == "mae" else "Q90-tail MAE (mg L$^{-1}$)")
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(axis="y", color="#E8EBED", linewidth=.5)
    axis = axes[1]
    for y, k in enumerate((3, 5)):
        name = f"nested_vs_joint_k{k}"
        row = data["effects"].set_index("comparison").loc[name]
        color = STYLE[NESTED]["color"]
        axis.hlines(y, row.delta_ci_low, row.delta_ci_high, color=color, linewidth=1.5)
        axis.plot([row.delta_ci_low, row.delta_ci_high], [y, y], "|", color=color, markersize=5)
        axis.plot(row.delta_value, y, "s", color=color, markersize=5, zorder=4)
        dots = data["directions"][data["directions"].comparison.eq(name)].set_index("split_seed")
        for split, marker, offset in zip(SPLITS, PARTITION_MARKERS, (-.22, -.12, .14), strict=True):
            axis.plot(dots.loc[split, "delta_mae"], y+offset, marker=marker, linestyle="none",
                      color="#727B83", markerfacecolor="white", markersize=3.4, markeredgewidth=.7)
    axis.axvline(0, color="#596168", linewidth=.7)
    values = np.r_[data["effects"].delta_ci_low, data["effects"].delta_ci_high,
                   data["directions"].delta_mae, 0.]
    pad = .1 * max(float(np.ptp(values)), .01)
    axis.set(xlim=(values.min()-pad, values.max()+pad), ylim=(1.55, -.55), yticks=(0, 1),
             yticklabels=("K = 3", "K = 5"), xlabel="Separate − joint MAE\n(mg L$^{-1}$)", title="Paired calibration difference")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#EEF0F2", linewidth=.45)
    for axis, letter in zip(axes, "abc", strict=True):
        axis.text(-.24, 1.09, letter, transform=axis.transAxes, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], color="#727B83", marker=m, markerfacecolor="white", markersize=3.8,
                     linestyle="none", label=f"Partition {i+1}") for i, m in enumerate(PARTITION_MARKERS)]
    handles.append(Line2D([], [], color=STYLE[NESTED]["color"], marker="s", linewidth=1.5,
                          markersize=4, label="Overall + paired 95% CI"))
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.525, .095),
               frameon=False, fontsize=7, columnspacing=1.2, handlelength=1.5)
    fig.text(.5, .065, "Negative paired differences favor the separate chemical increment.",
             ha="center", fontsize=7, color="#596168")
    fig.text(.5, .027, "Equal partition means  ·  5,000 joint station-bootstrap draws  ·  Dots are partition means",
             ha="center", fontsize=7, color="#596168")
    return fig


def write_caption(out):
    lines = ["# Fresh nested chemistry confirmation figure", "",
        "**Separating chemical and temporal station calibration.** a, Native DOC MAE across all fixed queries",
        "as retrospective support increases from K0 to K5. b, Separate chemical increment minus joint-selected",
        "calibration MAE at K3/K5; negative differences favor the separate increment. Filled squares show the",
        "equal-partition estimate; bars show 95% percentile intervals from 5,000 joint whole-station bootstrap",
        "draws. Open symbols show partition means over three training seeds, not additional confidence intervals.",
        "c, MAE for DOC at or above each source-training Q90 threshold. Curves average seeds within partitions,",
        "then weight partitions equally. All six predeclared procedures appear in a fixed semantic order.", "",
        "Panels a/c use focused absolute-error axes with 13% padding; panel b retains zero. K0/K1 chemical",
        "increments are exactly zero by construction. These are fresh roles on the same ST357 cohort, not",
        "an independent external basin. Auxiliary pH/conductance are allowed covariates; support may postdate",
        "queries. Five reserved support candidates are excluded from every query set. Repeated seeds do not",
        "increase ecological sample size. Tail groups below 20 cells are marked unstable in analysis tables.", "",
        "The figure does not select a model or splice procedures by K. See the complete findings and availability",
        "tables for ordinary-DOC errors, bias, classification, and the difference between covariate availability",
        "on observed test queries and at genuinely missing DOC cells.", "",
        "| Legend | Fixed model |", "|---|---|"]
    lines.extend(f"| {STYLE[m]['label']} | `{m}` |" for m in PLOT_MODELS)
    lines += ["", "PDF and SVG are vector outputs; PNG is 300 dpi. Input and output hashes are recorded in",
              "sources_manifest.json. Reproduce with:", "", "```bash",
              "uv run python scripts/plot_doc_nested_confirmation_v1.py", "```", ""]
    (out / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    data = load_analysis(args.root)
    if args.check_only:
        print("Complete source-bound figure tables reconcile")
        return
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    figure = draw_figure(data)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(out / f"{FIGURE_NAME}.{extension}", dpi=300)
    plt.close(figure)
    write_caption(out)
    outputs = [out / f"{FIGURE_NAME}.{suffix}" for suffix in ("pdf", "svg", "png")] + [out / "README.md"]
    manifest = {"figure": FIGURE_NAME, "models": list(PLOT_MODELS), "partitions": list(SPLITS),
        "training_seeds": list(SEEDS), "k_values": list(KS), "bootstrap_draws": 5000,
        "sources": data["source_files"], "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "style_helper_sha256": sha256(Path("scripts/plot_doc_chemistry_confirmation_v1.py")),
        "outputs": {path.name: sha256(path) for path in outputs},
        "render": {"width_mm": 183, "height_mm": 108, "png_dpi": 300, "vector_formats": ["pdf", "svg"]}}
    (out / "sources_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {FIGURE_NAME}.pdf/.svg/.png and caption/source manifest in {out}")


if __name__ == "__main__":
    main()
