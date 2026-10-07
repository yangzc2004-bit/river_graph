"""Actual whole-network form and local junction geometry, with paired contrasts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_confluence_mixing_v1 import CACHE, CLASSES, PREVIOUS, ROOT
from matplotlib.font_manager import FontProperties, fontManager
from plot_doc_river_planform_v1 import draw_network

from river_graph.experiments.provenance import sha256_file

COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}
BRANCHES = {"branch_a": "#257F88", "branch_b": "#C78169", "common": "#304B58"}
DTYPES = {"station": str, "huc4": str}


def local_map(ax, frame, row, cn):
    groups = {segment: f.sort_values("vertex") for segment, f in frame.groupby("segment")}
    common = groups["common"]
    origin = common[["x_m", "y_m"]].iloc[0].to_numpy()
    for segment, group in groups.items():
        xy = group[["x_m", "y_m"]].to_numpy()-origin
        ax.plot(xy[:, 0], xy[:, 1], color=BRANCHES[segment], linewidth=2.2)
        if len(xy) >= 2:
            from shapely.geometry import LineString

            line = LineString(xy)
            a, b = (90., 140.) if segment == "common" else (140., 90.)
            start, end = line.interpolate(a), line.interpolate(b)
            ax.annotate("", xy=end.coords[0], xytext=start.coords[0],
                        arrowprops={"arrowstyle": "->", "lw": 1.3, "color": BRANCHES[segment]})
    ax.scatter(0., 0., color="#D89942", edgecolor="white", s=45, zorder=5)
    ax.set(xlim=(-295, 295), ylim=(-295, 295))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.plot([-255, -155], [-255, -255], color="#304B58", linewidth=1.4)
    ax.text(-205, -278, "100 m", ha="center", va="top", fontsize=9)
    ax.annotate("N", xy=(.93, .93), xytext=(.93, .76), xycoords="axes fraction", ha="center",
                arrowprops={"arrowstyle": "->", "color": "#76868B"}, fontsize=9)
    if cn:
        detail = f"支流夹角 {row.incoming_angle_deg:.1f}°\n下游弯曲比 {row.downstream_sinuosity:.3f}"
    else:
        detail = f"Incoming angle {row.incoming_angle_deg:.1f}°\nDownstream sinuosity {row.downstream_sinuosity:.3f}"
    ax.text(.02, -.05, detail, transform=ax.transAxes, fontsize=9, va="top")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    out, analysis = ROOT/"figures", ROOT/"analysis"
    out.mkdir(exist_ok=True)
    inputs = [analysis/"local_junction_geometry.csv", analysis/"local_centerlines.parquet",
              analysis/"matched_form_contrasts.csv", analysis/"summary.json",
              PREVIOUS/"representatives.csv", CLASSES, Path(__file__), CACHE,
              Path("scripts/plot_doc_river_planform_v1.py")]
    geometry = pd.read_csv(inputs[0], dtype=DTYPES)
    lines = pd.read_parquet(inputs[1])
    contrasts = pd.read_csv(inputs[2])
    summary = json.loads(inputs[3].read_text())
    representatives = pd.read_csv(inputs[4], dtype=DTYPES)
    classes = pd.read_csv(CLASSES, dtype=DTYPES)
    names = {1: "细长、多支流", 2: "稀疏、主干占优", 3: "宽阔、多支流"} if cn else {
        1: "Elongated / tributary-rich", 2: "Sparse / mainstem-dominated", 3: "Broad / tributary-rich"}
    suffix, written = ("_cn" if cn else ""), []

    def save(fig, name):
        for ext in ("png", "pdf"):
            p = out/f"{name}{suffix}.{ext}"
            fig.savefig(p, dpi=220, facecolor="white", bbox_inches="tight")
            written.append(p)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.5))
    reps = representatives[representatives.example_group.le(3)].sort_values("cluster")
    primary = geometry[geometry.scale_m.eq(250.) & geometry.status.eq("measured")]
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for col, r in enumerate(reps.itertuples()):
            mapped = classes[classes.station.eq(r.station)].iloc[0]
            draw_network(axes[0, col], connection, mapped, COLORS[int(r.cluster)])
            axes[0, col].set_title(f"{chr(97+col)}  {names[int(r.cluster)]}\n"
                                   + ((f"整体河网 · 站点 {r.station}") if cn else f"Whole network · station {r.station}"),
                                   loc="left", fontsize=11)
            local = primary[primary.station.eq(r.station)]
            if len(local) != 1:
                raise ValueError("original representative requires measurable local geometry")
            selected_lines = lines[lines.station.eq(r.station)]
            joint = selected_lines[selected_lines.segment.eq("common")].sort_values("vertex").iloc[0]
            axes[0, col].scatter(joint.x_m/1000, joint.y_m/1000, marker="*", s=60,
                                 color="#D89942", edgecolor="#304B58", linewidth=.6, zorder=7)
            local_map(axes[1, col], selected_lines, local.iloc[0], cn)
            axes[1, col].set_title(f"{chr(100+col)}  "+("同一河网的实际汇流点 · 250 m" if cn else "Its actual selected junction · 250 m"),
                                   loc="left", fontsize=11)
    fig.suptitle("整体形态与局地汇流结构：真实河道的两个尺度" if cn else
                 "Whole-network form and local confluence geometry", x=.035, ha="left", fontsize=17)
    note = ("上排保留真实完整河网，各图比例尺独立；下排统一测量汇流点周围 250 m 的实际中心线。\n"
            "上排金色星号为放大的汇流点；下排绿色与橙色为进入支流，深色为下游主干，箭头为水流方向。代表选择未用 DOC。") if cn else (
            "Top: original real network representatives, independent physical extents. Bottom: exact 250 m mapped centreline windows.\n"
            "Gold star: selected junction. Below: teal/coral incoming branches, dark downstream channel, flow arrows. Representatives selected without DOC.")
    fig.text(.035, .025, note, fontsize=9)
    fig.subplots_adjust(top=.87, bottom=.17, hspace=.24, wspace=.12)
    save(fig, "whole_forms_and_actual_junctions")

    fields = ("incoming_angle_deg", "area_weighted_deflection_deg", "downstream_sinuosity")
    labels = ("支流夹角（°）", "面积加权流向转角（°）", "下游弯曲比") if cn else (
        "Incoming angle (°)", "Area-weighted deflection (°)", "Downstream sinuosity")
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.4))
    rng = np.random.default_rng(42)
    for col, (metric, label) in enumerate(zip(fields, labels, strict=True)):
        ax = axes[0, col]
        data = [primary[primary.cluster.eq(c)][metric].to_numpy() for c in (1, 2, 3)]
        short_names = ["细长", "稀疏", "宽阔"] if cn else ["Elongated", "Sparse", "Broad"]
        tick_labels = [f"{name}\nn={len(values)}" for name, values in zip(short_names, data, strict=True)]
        boxes = ax.boxplot(data, tick_labels=tick_labels,
                           widths=.45, patch_artist=True, showfliers=False,
                           medianprops={"color": "#263B42", "linewidth": 1.4},
                           whiskerprops={"color": "#76868B"}, capprops={"color": "#76868B"})
        for c, patch, values in zip((1, 2, 3), boxes["boxes"], data, strict=True):
            patch.set(facecolor=COLORS[c]+"35", edgecolor=COLORS[c], linewidth=1.1)
            ax.scatter(c+rng.uniform(-.16, .16, len(values)), values, s=8, alpha=.26, color=COLORS[c], zorder=2)
        ax.set_ylabel(label)
        ax.set_title(f"{chr(97+col)}  "+("原三类整体河网 · 250 m" if cn else "Original network forms · 250 m"), loc="left", fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", color="#EDF0F0", zorder=0)
        if metric.endswith("deg"):
            ax.set_ylim(0, 185)
        else:
            ax.axhline(1., color="#B9C4C7", linewidth=.8)
        ax = axes[1, col]
        f = contrasts[contrasts.metric.eq(metric)].sort_values("scale_m")
        for i, row in enumerate(f.itertuples()):
            color = "#D89942" if row.scale_m == 250. else "#607E88"
            ax.plot([row.ci_low, row.ci_high], [i, i], color=color, linewidth=2)
            ax.scatter(row.mean_difference, i, s=42, color=color, zorder=3)
            ax.text(.98, (i+.35)/3, f"n={row.n_pairs}", transform=ax.transAxes, ha="right", fontsize=9)
        ax.axvline(0., color="#B9C4C7", linewidth=1., linestyle="--")
        ax.set(yticks=range(len(f)), yticklabels=[f"{int(s)} m" for s in f.scale_m], ylim=(-.55, len(f)-.5))
        ax.set_xlabel(("宽阔类 − 细长类\n" if cn else "Broad − elongated\n")+label)
        ax.set_title(f"{chr(100+col)}  "+("原有配对比较 · 三个测量尺度" if cn else "Original matches · three scales"), loc="left", fontsize=11)
        ax.grid(axis="x", color="#EDF0F0", zorder=0)
    fig.suptitle("整体河网类别能否代表局地汇流几何？" if cn else
                 "Does whole-network form distinguish local junction geometry?", x=.035, ha="left", fontsize=17)
    note = (f"上排 {summary['n_measured_primary_scale']} 个可测河网，涉及 {summary['n_unique_primary_junctions']} 个不同物理汇流点；嵌套河网与重复汇流点保留并注明。\n"
            "下排沿用原 22 对细长—宽阔河网；点为配对均值差，线为按 HUC4 整组重采样的 95% 区间（5,000 次）。面积权重不是实测流量。\n"
            "本图测量实际结构，尚未将它等同于观测混合速度或 DOC 去除。") if cn else (
            f"Top: {summary['n_measured_primary_scale']} measured network instances, {summary['n_unique_primary_junctions']} distinct physical junctions; nested/repeated junctions retained.\n"
            "Bottom: original 22 elongated–broad pairs; mean difference and 95% HUC4-group bootstrap interval (5,000 draws). Area share is not measured flow.\n"
            "These are mapped structural measurements; observed mixing distance and DOC removal are not estimated here.")
    fig.text(.035, .024, note, fontsize=9)
    fig.subplots_adjust(top=.88, bottom=.19, hspace=.53, wspace=.34)
    save(fig, "junction_geometry_by_network_form")
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "scope": "actual whole-network and local mapped geometry; unchanged morphology labels",
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "outputs": {str(p): sha256_file(p) for p in written},
    }, indent=2)+"\n")
    print("\n".join(map(str, written)))


if __name__ == "__main__":
    main()
