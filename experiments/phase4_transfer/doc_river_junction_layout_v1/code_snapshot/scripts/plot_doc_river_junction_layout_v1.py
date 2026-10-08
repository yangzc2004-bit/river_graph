"""Real whole networks, weighted tributary entry layout and controlled arrivals."""

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
from analyze_doc_river_junction_layout_v1 import PANEL, ROOT, TYPES
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import MaxNLocator
from plot_doc_river_planform_v1 import boundary, segments
from shapely.geometry import shape
from shapely.ops import unary_union

from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import project_geometry, read_cached_lines

CACHE = Path("data/raw/river_planform_v1")
COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def real_map(ax, connection, row, main, entries, color):
    with np.load(CACHE/"members_full"/f"comid_{int(row.comid)}.npz") as z:
        ids = z["comids"]
    lines = {c: project_geometry(g) for c, g in read_cached_lines(connection, ids).items()}
    saved = json.loads((CACHE/"basins"/f"comid_{int(row.comid)}.json").read_text())
    basin = project_geometry(unary_union([shape(f["geometry"]) for f in saved["features"]]))
    boundary(ax, basin)
    ax.add_collection(LineCollection(segments(lines.values()), colors=color, linewidth=.3, alpha=.65, zorder=2))
    ax.add_collection(LineCollection(segments(lines[int(c)] for c in main), colors="#273F48", linewidth=1.1, zorder=3))
    points, previous = {}, None
    for k, cid in enumerate(main):
        coordinates = np.asarray(lines[int(cid)].coords, float)[[0, -1], :2]
        if previous is not None:
            down = int(np.argmin(np.linalg.norm(coordinates-previous, axis=1)))
            gap = float(np.linalg.norm(coordinates[down]-previous))
            up = 1-down
        elif len(main) > 1:
            following = np.asarray(lines[int(main[k+1])].coords, float)[[0, -1], :2]
            up = int(np.argmin(np.linalg.norm(coordinates[:, None]-following[None], axis=2).min(axis=1)))
            gap = 0.
        else:
            outlet = np.array([row.outlet_x, row.outlet_y])
            up = 1-int(np.argmin(np.linalg.norm(coordinates-outlet, axis=1)))
            gap = 0.
        previous = coordinates[up]
        points[int(cid)] = (previous, gap)
    mapped = []
    for r in entries.itertuples():
        xy, gap = points[int(r.entry_comid)]
        ax.scatter(xy[0]/1000, xy[1]/1000, s=10+900*r.lateral_area_fraction,
                   facecolor="#DBAB52", edgecolor="#4B5261", linewidth=.4, alpha=.8, zorder=4)
        mapped.append({"station": row.station, "entry_comid": int(r.entry_comid), "x_m": xy[0], "y_m": xy[1],
                       "lateral_area_fraction": r.lateral_area_fraction, "downstream_link_gap_m": gap})
    ax.scatter(row.outlet_x/1000, row.outlet_y/1000, s=28, color="#CC7155", edgecolor="white", linewidth=.7, zorder=5)
    x0, y0, x1, y1 = np.asarray(basin.bounds)/1000
    span = max(x1-x0, y1-y0)*1.14
    xm, ym = (x0+x1)/2, (y0+y1)/2
    ax.set(xlim=(xm-span/2, xm+span/2), ylim=(ym-span/2, ym+span/2), aspect="equal")
    ax.set_axis_off()
    target = span*.22
    power = 10**np.floor(np.log10(target))
    scale = max(v*power for v in (1, 2, 5) if v*power <= target)
    sx, sy = xm-.42*span, ym-.49*span
    ax.plot([sx, sx+scale], [sy, sy], color="#34464C", linewidth=1.5)
    ax.text(sx+scale/2, sy+.015*span, f"{scale:g} km", ha="center", fontsize=8)
    return mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    parser.add_argument("--geometric-mainstem", action="store_true")
    args = parser.parse_args()
    cn = args.chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    root = ROOT/"original_mainstem_sensitivity" if args.geometric_mainstem else ROOT
    out, data = root/"figures", root/"analysis"
    out.mkdir(exist_ok=True)
    names = {1: "细长、多支流", 2: "稀疏、主干占优", 3: "宽阔、多支流"} if cn else {
        1: "Elongated / tributary-rich", 2: "Sparse / mainstem-dominated", 3: "Broad / tributary-rich"}
    reps = pd.read_csv(data/"representatives.csv", dtype=TYPES).sort_values("cluster")
    panel = pd.read_csv(PANEL, dtype=TYPES).set_index("station", drop=False)
    mains = pd.read_csv(data/"route_mainstems.csv", dtype=TYPES)
    entries = pd.read_csv(data/"entry_junctions.csv", dtype=TYPES)
    curves = pd.read_parquet(data/"representative_curves.parquet")
    descriptor = pd.read_csv(data/"station_junction_layout.csv", dtype=TYPES)
    matched = pd.read_csv(data/"matched_form_contrasts.csv")
    suffix, written = ("_cn" if cn else ""), []

    def save(fig, name):
        for ext in ("png", "pdf"):
            p = out/f"{name}{suffix}.{ext}"
            fig.savefig(p, dpi=220, facecolor="white", bbox_inches="tight")
            written.append(p)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13.6, 13.1), gridspec_kw={"height_ratios": [1.8, 1., 1.1]})
    markers = []
    labelmap = {"actual_paths": "真实路径" if cn else "Actual paths",
        "within_unit_collapsed": "支流内部路径等长" if cn else "Within-unit paths equalized",
        "unit_means_aligned": "各单元平均到达同步" if cn else "Unit mean arrivals aligned"}
    lineprops = {"actual_paths": {"color": "#344C5A", "linestyle": "-"},
        "within_unit_collapsed": {"color": "#257F88", "linestyle": "--"},
        "unit_means_aligned": {"color": "#C5814A", "linestyle": "-."}}
    with sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True) as connection:
        for col, row in enumerate(reps.itertuples()):
            main = mains[mains.station.eq(row.station)].sort_values("sequence_outlet_to_head").mainstem_comid.to_numpy()
            e = entries[entries.station.eq(row.station)]
            markers.extend(real_map(axes[0, col], connection, panel.loc[row.station], main, e, COLORS[row.cluster]))
            axes[0, col].set_title(f"{chr(97+col)}  {names[row.cluster]}\n"
                                   + (f"站点 {row.station} · 实际汇入点" if cn else f"Station {row.station} · actual entries"),
                                   loc="left", fontsize=11)
            mass, bins = np.histogram(e.entry_position, bins=np.linspace(0, 1, 11), weights=e.lateral_only_weight)
            ax = axes[1, col]
            ax.bar((bins[:-1]+bins[1:])/2, mass, width=.084, color=COLORS[row.cluster], alpha=.8)
            ax.set(xlim=(0, 1), ylim=(0, max(.55, mass.max()*1.12)))
            ax.set_xlabel("汇入位置：0 出口 → 1 主干上端" if cn else "Entry position: 0 outlet → 1 trunk head")
            ax.set_ylabel("侧支流面积份额" if cn else "Lateral area share")
            ax.set_title(f"{chr(100+col)}  "+("侧支流沿主干的布局" if cn else "Where lateral tributaries enter"), loc="left", fontsize=11)
            ax.grid(axis="y", color="#EDF0F0")
            ax = axes[2, col]
            t = curves[curves.station.eq(row.station)]
            for scenario, props in lineprops.items():
                c = t[t.scenario.eq(scenario)]
                ax.plot(c.centered_time, c.outlet_anomaly, label=labelmap[scenario], linewidth=1.75, **props)
            ax.set_xlabel("相对到达时间（各曲线均值置 0）" if cn else "Relative arrival time\n(centered at its own mean)")
            ax.set_ylabel("出口异常信号（情景）" if cn else "Outlet anomaly")
            ax.set_title(f"{chr(103+col)}  "+("出口到达时序比较" if cn else "Outlet arrival comparison"), loc="left", fontsize=11)
            ax.grid(axis="y", color="#EDF0F0")
            ax.set_xlim(-max(1., 2.5*row.path_cv), max(1.5, 3.8*row.path_cv))
    handles, labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .106), ncol=3, frameon=False, fontsize=10)
    headline = "整张河网的汇流布局与出口信号叠加" if cn else "Whole-network tributary layout and outlet signal overlap"
    route_note = (("原几何主干情景" if cn else "Original mapped-mainstem scenario") if args.geometric_mainstem else
                  ("保存的最短流路情景" if cn else "Saved shortest-route scenario"))
    fig.text(.035, .936, route_note, fontsize=10, color="#607E88")
    fig.suptitle(headline, x=.035, ha="left", fontsize=17)
    note = ("上排：真实完整河网；深线为本次连通流路的主干，金点越大表示该汇入点的侧支流面积越大，红点为出口；各图比例尺独立。\n"
            "中排：全部侧支流的面积加权汇入位置。下排：同一输入的保守传输诊断；时间按原网络平均路径归一，并分别居中。\n"
            "每个局地增量汇水区只计一次；到达单元包括侧支流和主干局地输入。整条曲线共同平移与原曲线居中后完全重叠。\n"
            "图中峰值是几何控制情景，不是实测 DOC 事件，也不表示 DOC 被去除。沿用原形态代表，未按 DOC 选择。") if cn else (
            "Top: complete real networks; dark line is this analysis's contiguous routing mainstem; larger gold entry markers denote more lateral catchment area.\n"
            "Middle: area-weighted lateral entry positions. Bottom: conservative same-input diagnostic, original mean-path units; curves centered separately.\n"
            "Each incremental catchment is counted once. Arrival units include lateral tributaries and mainstem-local inputs. Common translation overlays the original.\n"
            "These peaks are controlled geometry scenarios, not observed DOC events or DOC loss. Original outline representatives retained; no DOC-based selection.")
    fig.text(.035, .018, note, fontsize=9)
    fig.subplots_adjust(top=.92, bottom=.20, hspace=.39, wspace=.34)
    save(fig, "real_tributary_layout_and_arrivals")
    pd.DataFrame(markers).to_csv(out/"representative_entry_locations.csv", index=False)

    fields = ("lateral_entry_mean", "lateral_entry_sd", "between_unit_variance_fraction")
    labels = ("平均汇入位置（0 出口，1 上端）", "汇入位置的面积加权标准差", "单元之间占总路径差异的份额") if cn else (
        "Mean entry position (0 outlet, 1 head)", "Area-weighted SD of entry position", "Between-unit share of path variance")
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.1))
    rng = np.random.default_rng(42)
    for col, (metric, label) in enumerate(zip(fields, labels, strict=True)):
        ax = axes[0, col]
        values = [descriptor[descriptor.cluster.eq(c)][metric].dropna().to_numpy() for c in (1, 2, 3)]
        short = ["细长", "稀疏", "宽阔"] if cn else ["Elongated", "Sparse", "Broad"]
        box = ax.boxplot(values, tick_labels=[f"{n}\nn={len(v)}" for n, v in zip(short, values, strict=True)],
                         widths=.46, patch_artist=True, showfliers=False,
                         medianprops={"color": "#263B42", "linewidth": 1.3})
        for c, patch, v in zip((1, 2, 3), box["boxes"], values, strict=True):
            patch.set(facecolor=COLORS[c]+"35", edgecolor=COLORS[c])
            ax.scatter(c+rng.uniform(-.15, .15, len(v)), v, color=COLORS[c], s=8, alpha=.27)
        ax.set_ylabel(label)
        ax.set_title(f"{chr(97+col)}  "+("原三类整体形态" if cn else "Original outline forms"), loc="left", fontsize=11)
        ax.grid(axis="y", color="#EDF0F0")
        ax.set_ylim(-.02, 1.02 if metric != "lateral_entry_sd" else .52)
        ax = axes[1, col]
        r = matched[matched.metric.eq(metric)].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [0, 0], color="#607E88", linewidth=2.2)
        ax.scatter(r.difference_b_minus_a, 0, color="#D89942", s=65, zorder=3)
        ax.axvline(0, color="#BBC5C7", linestyle="--", linewidth=1.)
        ax.set(yticks=[], ylim=(-.6, .6))
        ax.set_xlabel(("宽阔 − 细长\n" if cn else "Broad − elongated\n")+label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{chr(100+col)}  "+("原 22 对配对河网" if cn else "Original 22 matched networks"), loc="left", fontsize=11)
        ax.text(.02, .1, f"{r.difference_b_minus_a:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", transform=ax.transAxes, fontsize=10)
        ax.grid(axis="x", color="#EDF0F0")
    fig.suptitle("形态类别与汇流布局：分布及原有配对比较" if cn else
                 "Network forms and tributary layout: distributions and original matches", x=.035, ha="left", fontsize=17)
    fig.text(.035, .916, route_note, fontsize=10, color="#607E88")
    note = ("上排保留原 297 个河网；没有正面积侧支流的河网不计算汇入位置。\n"
            "下排沿用原 22 对细长—宽阔网络，按 12 个 HUC4 整组重采样 5,000 次；点为等配对均值差，线为 95% 区间。\n"
            "单元之间的差异包含侧支流平均到达时间和主干局地输入；河网形态、汇入位置及路径时序并非同一个量。") if cn else (
            "Top: original 297 networks; entry-position statistics exclude networks without positive-area lateral tributaries.\n"
            "Bottom: original 22 elongated–broad pairs; equal-pair mean difference, 95% whole-HUC4 bootstrap interval, 12 blocks and 5,000 draws.\n"
            "Between-unit variance includes tributary mean arrivals and mainstem-local inputs. Outline form, entry position and arrival timing are distinct.")
    fig.text(.035, .025, note, fontsize=9)
    fig.subplots_adjust(top=.88, bottom=.18, hspace=.49, wspace=.33)
    save(fig, "layout_and_arrival_variance_by_form")
    inputs = [PANEL, data/"representatives.csv", data/"route_mainstems.csv", data/"entry_junctions.csv",
              data/"representative_curves.parquet", data/"station_junction_layout.csv", data/"matched_form_contrasts.csv",
              Path(__file__), Path("scripts/plot_doc_river_planform_v1.py"), CACHE/"flowlines.sqlite"]
    for row in reps.itertuples():
        inputs.extend([CACHE/"basins"/f"comid_{int(row.comid)}.json",
                       CACHE/"members_full"/f"comid_{int(row.comid)}.npz"])
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "outputs": {str(p): sha256_file(p) for p in [*written, out/"representative_entry_locations.csv"]},
        "scope": "original real outlines; new contiguous routing mainstem; controlled same-input response"}, indent=2)+"\n")
    print("\n".join(map(str, written)))


if __name__ == "__main__":
    main()
