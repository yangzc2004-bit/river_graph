"""Figure 1 (data-accurate): method evolution ladder from the frozen tables.

G0 comes from the frozen single-run table ``benchmark.csv``; H1/H2/H2X come
from the multi-seed table ``benchmark_multiseed.csv`` (5 training seeds each).

What this figure claims
-----------------------
It describes how the stored models perform across three scenarios and shows
the H2X improvement on temporal reconstruction, while keeping the fact that
the directed models do NOT improve on E2b/E3. It is NOT a strict one-factor
mechanism ablation: H2X adds ecological information AND an encoder at once,
and there is no same-protocol H2E control.

Scenario values are the equal-weight mean over the masks in that scenario
(E1 and E3 average 3 masks each; E2a/E2b are single masks). A mean of per-mask
R2 is not the R2 of all cells pooled. No error bars are drawn, because the two
input tables have different aggregation levels: G0 is a single historical run
without a training-seed spread, so plotting a spread for H1/H2/H2X only would
invite a comparison the data cannot support (see
``scripts/figure2_multiseed_variation.py`` for the training-seed spread).

Panels instead of one crowded axis
----------------------------------
An earlier revision put all three series on one axis and offset every label by
the same 8 points, which pushed the G0/E3 label onto the E2b marker. Per the
plan this version uses one panel per scenario, so no two labels can collide,
and each panel gets its own y-range so the small differences stay readable.

Usage:
    python scripts/figure1_evolution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FROZEN = Path("experiments/frozen_results/benchmark.csv")
MULTISEED = Path("experiments/frozen_results/benchmark_multiseed.csv")
OUT = Path("experiments/figures")

MODELS = ["G0", "H1", "H2", "H2X"]
LABELS = ["G0\nriver graph", "H1\n+ direction", "H2\n+ transport gates",
          "H2X\n+ ecological encoder"]

# G0 is only present in the frozen single-run table.
G0_MODEL_NAME = "G0_gcn_river"
MULTISEED_MODELS = ["H1", "H2", "H2X"]

E1_MASKS = ["e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44"]
E3_MASKS = ["e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44"]

SERIES = {
    "E1 random mask (r20)": E1_MASKS,
    "E2b future reconstruction": ["e2b_partial"],
    "E3 unseen stations": E3_MASKS,
}
COLORS = {"E1 random mask (r20)": "#1f4e79",
          "E2b future reconstruction": "#2e75b6",
          "E3 unseen stations": "#2e8b57"}
METRIC = "r2"
Y_LABEL = "R² (higher is better)"


class Figure1Error(RuntimeError):
    """Raised when the input tables cannot support the figure's claims."""


def _exactly_one(df: pd.DataFrame, model: str, mask: str, metric: str,
                 table: str) -> float:
    sub = df[(df["model"] == model) & (df["mask"] == mask)]
    if len(sub) == 0:
        raise Figure1Error(f"{table}: no row for {model} @ {mask}")
    if len(sub) > 1:
        raise Figure1Error(
            f"{table}: {len(sub)} rows for {model} @ {mask}; refusing to "
            "average silently")
    value = sub[metric].iloc[0]
    if value is None or not np.isfinite(value):
        raise Figure1Error(f"{table}: {model} @ {mask} has non-finite {metric}")
    return float(value)


def build_source_data(frozen: pd.DataFrame,
                      multi: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, series, mask) with the table it came from."""
    rows = []
    for model in MODELS:
        for series, masks in SERIES.items():
            for mask in masks:
                if model == "G0":
                    value = _exactly_one(frozen, G0_MODEL_NAME, mask, METRIC,
                                         "benchmark.csv")
                    source = "benchmark.csv (G0_gcn_river, single run)"
                    seeds = 1
                else:
                    value = _exactly_one(multi, model, mask, METRIC,
                                         "benchmark_multiseed.csv")
                    source = "benchmark_multiseed.csv"
                    seeds = 5
                rows.append({
                    "model": model, "series": series, "mask": mask,
                    "metric": METRIC, "value": value,
                    "training_seeds": seeds, "source": source,
                })
    df = pd.DataFrame(rows)
    # scenario value = equal-weight mean over that scenario's masks
    grouped = (df.groupby(["model", "series"], as_index=False)
               .agg(value=("value", "mean"), n_masks=("mask", "count"),
                    training_seeds=("training_seeds", "first"),
                    source=("source", "first")))
    return df, grouped


def draw(grouped: pd.DataFrame, out_base: Path) -> tuple[object, list[Path]]:
    """Render the three scenario panels and export png/svg/pdf.

    One panel per scenario: with a single series per axis no two labels can
    collide, which is what the previous shared-axis revision got wrong.
    """
    # No sharey: the three scenarios have different R2 ranges, so a shared axis
    # would either clip E1 (its values sit above the E3 band) or flatten the
    # differences that the figure exists to show.
    fig, axes = plt.subplots(1, len(SERIES), figsize=(16.0, 5.4))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.22,
                        wspace=0.16)
    xs = np.arange(len(MODELS))
    series_values = {}
    # Pass 1: fix every axis range BEFORE placing labels, because changing the
    # y-limits afterwards re-scales the panel and invalidates the pixel boxes
    # the placement used.
    for ax, series in zip(axes, SERIES):
        sub = grouped[grouped["series"] == series].set_index("model")
        ys = [float(sub.loc[m, "value"]) for m in MODELS]
        series_values[series] = ys
        ax.plot(xs, ys, "o-", color=COLORS[series], lw=2, ms=8)
        ax.set_xticks(xs, LABELS, fontsize=9)
        ax.set_title(series, fontsize=12, color=COLORS[series])
        ax.grid(axis="y", alpha=0.3)
        lo, hi = min(ys), max(ys)
        pad = max(0.02, (hi - lo) * 0.45)
        ax.set_ylim(lo - pad, hi + pad)
        seeds = int(sub["training_seeds"].max())
        n_masks = int(sub["n_masks"].max())
        ax.set_xlabel(f"{n_masks} mask(s), {seeds} training seed(s)"
                      if seeds > 1 else f"{n_masks} mask(s), single run")

    # Pass 2: place the value labels against the final axis geometry.
    for ax, series in zip(axes, SERIES):
        place_labels(ax, xs, series_values[series], COLORS[series])
    axes[0].set_ylabel(Y_LABEL)

    fig.suptitle(
        "Figure 1 | Stored model performance across scenarios "
        "(R², equal-weight mean over each scenario's masks)",
        fontsize=13, y=0.975)
    # The caveats live in a footnote instead of the title: they are longer than
    # the figure is wide, and a clipped caption would hide exactly the limits
    # that make this figure honest.
    fig.text(
        0.5, 0.075,
        "G0: frozen single run. H1/H2/H2X: mean over 5 training seeds. "
        "A mean of per-mask R² is not the R² of all cells pooled. No error bars "
        "are shown because G0 has no training-seed spread to compare against.\n"
        "H2X adds ecological information AND an encoder at once, so the H2X gain "
        "is not attributable to the encoder alone. This is not a strict "
        "single-factor ablation.",
        ha="center", va="center", fontsize=9.5, color="#333333", wrap=True)

    paths = []
    for suffix, kwargs in (("png", {"dpi": 200}), ("svg", {}), ("pdf", {})):
        path = out_base.with_suffix(f".{suffix}")
        fig.savefig(path, **kwargs)
        paths.append(path)
    return fig, paths


def place_labels(ax, xs, ys, color: str) -> list:
    """Annotate every point, nudging labels until none of them overlap.

    A fixed offset for all points is what broke the previous revision (the G0
    E3 label landed on the E2b marker). Here each label starts above its point
    and moves up in small steps until its pixel box is clear of the labels
    already placed in this panel.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    placed: list = []
    annotations = []
    for x, y in zip(xs, ys):
        step = 0
        while True:
            ann = ax.annotate(
                f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 10 + step * 6), ha="center", va="bottom",
                fontsize=10, color=color, fontweight="bold")
            box = ann.get_window_extent(renderer=renderer)
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                annotations.append(ann)
                break
            ann.remove()
            step += 1
            if step > 24:
                raise Figure1Error(
                    f"cannot place the label for value {y:.3f} without a "
                    "collision; widen the panel or reduce the label size")
    return annotations


def check_layout(fig, out_base: Path) -> None:
    """Fail loudly if text is clipped or two texts overlap.

    Readability was the R2 defect, so it is checked instead of assumed: every
    text artist must sit inside its own axes (or the figure, for the suptitle),
    and no two texts inside the same axes may share pixels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox

    for index, ax in enumerate(fig.axes):
        value_labels = {t.get_text()[:28] for t in ax.texts}
        texts = ([ax.title] if ax.title.get_text() else []) + \
                ([ax.xaxis.label] if ax.xaxis.label.get_text() else []) + \
                ([ax.yaxis.label] if ax.yaxis.label.get_text() else []) + \
                list(ax.texts)
        boxes = []
        for text in texts:
            box = text.get_window_extent(renderer=renderer)
            if box.width == 0 or box.height == 0:
                continue
            boxes.append((text.get_text()[:28], box))
        # Report collisions first: a label that both collides and would be
        # clipped is more usefully described as a collision.
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][1].overlaps(boxes[j][1]):
                    raise Figure1Error(
                        f"panel {index}: labels {boxes[i][0]!r} and "
                        f"{boxes[j][0]!r} overlap")
        # value labels must stay inside the panel, otherwise they are clipped
        # by the axes boundary when the figure is rendered
        axes_box = ax.get_window_extent(renderer=renderer)
        for label, box in boxes:
            inside_panel = (box.x0 >= axes_box.x0 - 1
                            and box.x1 <= axes_box.x1 + 1
                            and box.y0 >= axes_box.y0 - 1
                            and box.y1 <= axes_box.y1 + 1)
            if label in value_labels and not inside_panel:
                raise Figure1Error(
                    f"panel {index}: value label {label!r} extends outside "
                    f"the panel ({box} vs axes {axes_box}) and would be "
                    "clipped")
            if not (box.x0 >= 0 and box.y0 >= 0
                    and box.x1 <= fig_bbox.width and box.y1 <= fig_bbox.height):
                raise Figure1Error(
                    f"panel {index}: text {label!r} falls outside the figure "
                    f"({box}); the caption would be clipped")

    for text in fig.texts:
        if not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer=renderer)
        if not (box.x0 >= -1 and box.x1 <= fig_bbox.width + 1
                and box.y0 >= -1 and box.y1 <= fig_bbox.height + 1):
            raise Figure1Error(
                f"figure-level caption text is clipped ({box}); shrink it or "
                "raise the margins")
    print(f"layout check passed for {out_base.name}: no overlapping or clipped text")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--basename", default="figure1_method_evolution_data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(FROZEN)
    multi = pd.read_csv(MULTISEED)

    per_mask, grouped = build_source_data(frozen, multi)

    source_csv = out_dir / f"{args.basename}_source_data.csv"
    per_mask.to_csv(out_dir / f"{args.basename}_source_data_per_mask.csv",
                    index=False)
    grouped.to_csv(source_csv, index=False)
    print(f"saved {source_csv}")

    fig, paths = draw(grouped, out_dir / args.basename)
    check_layout(fig, out_dir / args.basename)
    plt.close(fig)
    for path in paths:
        print(f"saved {path}")

    print()
    print("scenario values (equal-weight mean over masks):")
    pivot = grouped.pivot(index="model", columns="series", values="value")
    print(pivot.reindex(MODELS).round(6).to_string())
    print()
    e1 = pivot.loc["H2", "E1 random mask (r20)"]
    e1x = pivot.loc["H2X", "E1 random mask (r20)"]
    print(f"E1 R2: H2={e1:.6f} H2X={e1x:.6f} "
          f"-> both display as {e1:.2f}; do not narrate as H2X improving R2")


if __name__ == "__main__":
    main()
