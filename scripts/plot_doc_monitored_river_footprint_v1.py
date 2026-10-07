"""Actual monitored river corridors, matched pulses and DOC sensitivities."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D

from river_graph.analysis.river_monitored_footprint import cropped_corridor
from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import project_geometry, read_cached_lines

ROOT = Path("experiments/phase4_transfer/doc_monitored_river_footprint_v1")
CACHE = Path("data/raw/river_planform_v1/flowlines.sqlite")
COLORS = {"branch_a": "#267F88", "branch_b": "#C78169", "common": "#304B58"}
OUTLINES = {1: "#267F88", 2: "#C5814A", 3: "#66749F"}
DTYPES = {"target": str, "source_a": str, "source_b": str, "huc4": str}


def mapped_corridor(connection, reaches):
    ids = reaches.comid.tolist()
    lines = read_cached_lines(connection, ids)
    if set(lines) != set(ids):
        raise ValueError("actual geometry required for every mapped corridor reach")
    h = hashlib.sha256()
    for cid in sorted(lines):
        h.update(str(cid).encode()+b"\0"+lines[cid].wkb)
    pieces, gap = cropped_corridor(reaches, {cid: project_geometry(g) for cid, g in lines.items()})
    return pieces, gap, h.hexdigest()


def draw_corridor(ax, reaches, parts, row, cn):
    all_xy = []
    for r in reaches.itertuples():
        p = parts[r.comid]
        xy = np.asarray(p.coords)
        all_xy.append(xy)
        if len(xy) >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=COLORS[r.segment], linewidth=1.8, solid_capstyle="round")
    coords = np.concatenate(all_xy)
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    width = max(upper-lower)
    ax.set(xlim=(lower[0]-.14*width, upper[0]+.14*width),
           ylim=(lower[1]-.16*width, upper[1]+.18*width))
    for segment, station, marker, start in (("branch_a", row.source_a, "o", True),
        ("branch_b", row.source_b, "s", True), ("common", row.target, "D", False)):
        s = reaches[reaches.segment.eq(segment)].sort_values("sequence")
        point = parts[int(s.comid.iloc[0 if start else -1])]
        x, y = point.coords[0 if start else -1]
        color = COLORS[segment] if start else "#D89942"
        ax.scatter(x, y, color=color, edgecolor="white", linewidth=.7, marker=marker, s=38, zorder=5)
        ax.annotate(station, (x, y), xytext=(7, 7 if segment != "branch_b" else -13), textcoords="offset points", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    options = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])*1000
    size = options[options <= .25*width][-1] if np.any(options <= .25*width) else .2*width
    x, y = lower[0], lower[1]-.10*width
    ax.plot([x, x+size], [y, y], color="#304B58", linewidth=1.6)
    ax.text(x+size/2, y-.025*width, f"{size/1000:g} km", ha="center", va="top", fontsize=8)
    ax.annotate("N", xy=(.035, .94), xytext=(.035, .78), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", arrowprops={"arrowstyle": "->", "color": "#76868B"}, fontsize=8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    detail = ((f"独立 A {row.branch_a_km:.1f} km · 独立 B {row.branch_b_km:.1f} km · 共同主干 {row.common_km:.1f} km\n"
               f"支路差异 CV {row.independent_branch_cv:.3f} · 主干占比 {row.common_fraction:.1%}") if cn else
              (f"Branch A {row.branch_a_km:.1f} km · B {row.branch_b_km:.1f} km · common trunk {row.common_km:.1f} km\n"
               f"Independent-branch CV {row.independent_branch_cv:.3f} · common share {row.common_fraction:.1%}"))
    ax.text(.02, -.045, detail, transform=ax.transAxes, va="top", fontsize=8.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        p = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(p)
        plt.rcParams["font.family"] = FontProperties(fname=p).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 220, "pdf.fonttype": 42})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    sources = [a/f"{name}.csv" for name in ("footprints", "corridor_reaches", "representatives", "pulse_scenarios",
        "observed_receivers", "structural_associations", "paired_error_comparisons")]+[a/"representative_pulses.parquet", ROOT/"config.json"]
    geometry, reaches, reps, pulses, observed, assoc, errors = [pd.read_csv(p, dtype=DTYPES) for p in sources[:7]]
    traces = pd.read_parquet(a/"representative_pulses.parquet")
    selected = reps.merge(geometry, on="pair_id", validate="one_to_one").sort_values("example_group")
    suffix, written = ("_cn" if cn else ""), []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, facecolor="white", bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    titles = {1: "支路差异较小 · 共同主干较短", 2: "支路差异较大 · 共同主干较短",
              3: "支路差异较小 · 共同主干较长", 4: "支路差异较大 · 共同主干较长"} if cn else {
              1: "Lower branch dispersion · lower common share", 2: "Higher branch dispersion · lower common share",
              3: "Lower branch dispersion · higher common share", 4: "Higher branch dispersion · higher common share"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    flow_hashes, gaps = {}, {}
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for ax, row in zip(axes.flat, selected.itertuples(), strict=True):
            f = reaches[reaches.pair_id.eq(row.pair_id)]
            parts, gap, gh = mapped_corridor(connection, f)
            flow_hashes[row.pair_id], gaps[row.pair_id] = gh, gap
            draw_corridor(ax, f, parts, row, cn)
            ax.set_title(f"{chr(96+row.example_group)}  {titles[row.example_group]}", loc="left", fontsize=12, pad=14)
    legend = [Line2D([0], [0], color=COLORS[s], linewidth=2.5, label=label) for s, label in zip(
        ("branch_a", "branch_b", "common"), ("独立支路 A", "独立支路 B", "汇合后的共同主干") if cn else
        ("Independent A", "Independent B", "Shared downstream trunk"), strict=True)]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.035, .945), ncol=3, frameon=False)
    fig.suptitle("DOC 监测实际覆盖了哪段河网？" if cn else "The river footprint actually connected by DOC monitors",
                 x=.035, ha="left", y=.99, fontsize=17)
    fig.text(.035, .025, ("真实 NHDPlus 河道，EPSG:5070；按站点在河段内的位置截取，比例尺独立，投影网格北向为图上方。\n"
        "59 个连接中按几何中位分界选择四个展示例；不是四种新分类。主干较长／较短表示占平均路径比例的高低。") if cn else
        ("Actual NHDPlus channels in EPSG:5070, cropped at station linear measures; independent scale bars, grid north up.\n"
         "Four geometry-selected examples from 59 connections, not new classes. Higher/lower common share refers to its fraction of the mean path."), fontsize=8.5)
    fig.subplots_adjust(left=.06, right=.97, top=.85, bottom=.13, wspace=.23, hspace=.34)
    save(fig, "real_monitored_footprints")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    end = max(2., traces.relative_time.max())
    for ax, row in zip(axes.flat, selected.itertuples(), strict=True):
        t = traces[traces.pair_id.eq(row.pair_id)]
        for name, label, color, style in (("actual", "实际支路＋主干" if cn else "Actual branches + trunk", "#267F88", "-"),
            ("equal_branches", "支路等长＋原主干" if cn else "Equal branches + same trunk", "#A4AEB1", ":"),
            ("no_common_trunk", "原支路，移除共同主干" if cn else "Same branches, no shared trunk", "#C78169", "--")):
            f = t[t.scenario.eq(name)]
            ax.plot(f.relative_time, f.outlet_anomaly, label=label, color=color, linestyle=style, linewidth=1.8)
        p = pulses[pulses.pair_id.eq(row.pair_id)].set_index("scenario")
        reduction = 100*(1-p.loc["actual", "pulse_peak"])
        ax.set_title(f"{chr(96+row.example_group)}  {titles[row.example_group]}", loc="left", fontsize=11, pad=14)
        ax.text(.62, .90, (f"路径差异削峰 {reduction:.1f}%\n共同延后 {row.common_fraction:.2f}") if cn else
                (f"Peak reduction {reduction:.1f}%\nShared shift {row.common_fraction:.2f}"), transform=ax.transAxes, fontsize=9, va="top")
        ax.set(xlim=(-.6, end), ylim=(0, 1.07), xlabel="相对情景时间" if cn else "Relative scenario time",
               ylabel="下游峰值 / 同一输入峰值" if cn else "Outlet anomaly / identical input peak")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(.04, .938), fontsize=9)
    fig.suptitle("支路错峰，共同主干整体延后" if cn else "Branch differences disperse peaks; the common trunk shifts both signals",
                 x=.04, ha="left", y=.99, fontsize=16)
    fig.text(.04, .025, ("两路输入完全相同，权重固定；实际平均延后归一为 1。移除共同主干只平移曲线，峰高、宽度和总量不变。\n"
        "59 个连接均满足总量守恒、恒定信号不变；相对时间不是天数。情景只隔离路径效应，没有模拟沿程 DOC 去除。") if cn else
        ("Identical source pulses, fixed area shares; actual mean delay = 1. Removing the trunk translates the curve without changing peak, width or mass.\n"
         "Mass and steady concentration preserved in all 59 connections. Relative time is not days; this routing scenario has no along-channel DOC removal."), fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.975, top=.83, bottom=.16, hspace=.48, wspace=.22)
    save(fig, "branch_and_common_trunk_signals")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    shape_names = ("细长、多支流型", "主干主导、稀支流型", "宽展、多支流型") if cn else (
        "Elongated / tributary-rich", "Mainstem / sparse", "Broad / tributary-rich")
    for ax, term, letter, title in ((axes[0, 0], "independent_branch_cv", "a", "独立支路长短差异" if cn else "Independent branch dispersion"),
                                  (axes[0, 1], "common_fraction", "b", "共同主干占比" if cn else "Shared downstream trunk fraction")):
        for k in (1, 2, 3):
            f = observed[observed.cluster.eq(k)]
            ax.scatter(f[term], np.exp(f.outlet_mixture_log_sd_ratio), color=OUTLINES[k], s=42, edgecolor="white", linewidth=.5,
                       label=shape_names[k-1], alpha=.85)
        ax.axhline(1., color="#9DAEB4", linestyle="--", linewidth=1.)
        ax.set_yscale("log")
        ax.set(xlabel=title, ylabel="下游 / 两支流混合的波动 SD 比" if cn else "Outlet / gauged-mixture SD ratio")
        ax.set_title(f"{letter}  "+("实测 DOC 波动" if cn else "Measured DOC variability"), loc="left", pad=12)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(.035, .945), ncol=3, frameon=False, fontsize=9)
    ax = axes[1, 0]
    terms = ("branch_balance", "independent_branch_cv", "common_fraction")
    names = ("支路均衡度", "独立支路差异", "共同主干占比") if cn else ("Branch balance", "Branch dispersion", "Common share")
    for i, term in enumerate(terms):
        for offset, outcome, color, marker, label in ((-.08, "outlet_mixture_log_sd_ratio", "#267F88", "o", "原浓度" if cn else "Native DOC"),
                (.08, "logscale_outlet_mixture_log_sd_ratio", "#C78169", "s", "log1p 浓度" if cn else "Log1p DOC")):
            r = assoc[assoc.focal.eq(term) & assoc.outcome.eq(outcome)].iloc[0]
            ax.plot([r.ci_low, r.ci_high], [i+offset]*2, color=color, linewidth=1.5)
            ax.scatter(r.estimate_per_receiver_sd, i+offset, color=color, marker=marker, s=28, label=label if i == 0 else None)
    ax.axvline(0., color="#9DAEB4", linestyle="--", linewidth=1.)
    ax.set(yticks=range(3), yticklabels=names, ylim=(2.5, -.5),
           xlabel="每 1 SD 结构变化对应的 log(SD 比) 变化" if cn else "Change in log(SD ratio) per 1 SD of structure")
    ax.set_title("c  "+("控制其余结构、平均路径与监测覆盖" if cn else "Adjusted structural associations"), loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1, 1]
    orders = ("same_month", "common_trunk_only", "actual_branches")
    labels = ("同月输入", "只保留共同延后", "两条支路分别延后") if cn else (
        "Same-month input", "Shared delay only", "Branch-specific delays")
    for i, operator in enumerate(orders):
        r = errors[errors.operator.eq(operator) & errors.metric.eq("mae")].iloc[0]
        color = "#267F88" if operator == "actual_branches" else "#C78169"
        ax.plot([r.relative_ci_low, r.relative_ci_high], [i]*2, color=color, linewidth=1.7)
        ax.scatter(r.relative_gain_pct, i, color=color, s=38)
    ax.axvline(0., color="#9DAEB4", linestyle="--", linewidth=1.)
    ax.set(yticks=range(3), yticklabels=labels, ylim=(2.5, -.5),
           xlabel="相对完整平均延后模型的 MAE 改善 (%)" if cn else "MAE improvement vs complete mean-delay model (%)")
    ax.set_title("d  "+("同一个固定模型，只替换上游输入" if cn else "Same fixed model, alternative upstream inputs"), loc="left", pad=12)
    fig.suptitle("结构机制接上真实月度 DOC 后怎样表现？" if cn else "Connecting matched river structure to actual monthly DOC",
                 x=.035, ha="left", y=.992, fontsize=16)
    fig.text(.035, .025, ("22 个接收站、59 个连接、11 个相连监测系统；按站点等权，5,000 次整系统 bootstrap。颜色沿用完整河网的三种外形。\n"
        "c 全部六个结构关系；d 参照为已有完整模型，正值表示改善，未重新拟合。已知上游 DOC 的机制诊断，不是零观测新站点任务。") if cn else
        ("22 receivers, 59 connections, 11 monitoring systems; receiver-equal, 5,000 system-bootstrap draws. Colors retain whole-network outline classes.\n"
         "c: all six adjusted associations. d: saved complete model reference, positive = improvement, no refitting; known-upstream diagnostic, not unmonitored K0."), fontsize=8.5)
    fig.subplots_adjust(left=.08, right=.975, top=.83, bottom=.15, hspace=.48, wspace=.55)
    save(fig, "matched_structure_and_doc")
    manifest = {"source_hashes": {str(p): sha256_file(p) for p in sources},
        "generator_sha256": sha256_file(Path(__file__)),
        "geometry_helper_sha256": sha256_file(Path("src/river_graph/analysis/river_monitored_footprint.py")),
        "figure_hashes": {str(p): sha256_file(p) for p in written}, "flowline_hashes": flow_hashes,
        "max_endpoint_gaps_m": gaps, "map_projection": "EPSG:5070", "station_crop": "NHD measure, inferred downstream coordinate orientation"}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"figures": list(map(str, written)), "endpoint_gaps_m": gaps}, indent=2))


if __name__ == "__main__":
    main()
