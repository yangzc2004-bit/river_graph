"""Real-form figures: source clocks, shared transit and distributed mixing."""
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
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from plot_doc_monitored_river_footprint_v1 import mapped_corridor
from plot_doc_river_planform_v1 import draw_network, segments

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_junction_dynamics_v1")
CACHE = Path("data/raw/river_planform_v1")
PLANFORM = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/classes.csv")
CORRIDORS = Path("experiments/phase4_transfer/doc_river_storage_placement_v1/analysis/corridor_reaches.csv")
COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}
PHASE_COLORS = {-1.: "#C5814A", 0.: "#257F88", 1.: "#66749F"}


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
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    f = pd.read_parquet(folder/"scenario_metrics.parquet")
    curves = pd.read_parquet(folder/"example_curves.parquet")
    cohort = pd.read_csv(folder/"cohort_summary.csv")
    field = pd.read_csv(folder/"field_transit_proxies.csv")
    reps = pd.read_csv(folder/"representatives.csv", dtype={"case_id": str, "station": str}).merge(
        pd.read_csv(PLANFORM, dtype={"station": str}).drop(columns="centroid_distance", errors="ignore"),
        on=["station", "comid", "cluster"], suffixes=("", "_shape"), validate="one_to_one")
    phase_names = {-1.: "汇流处同步" if cn else "Aligned at junction",
                    0.: "源头同步" if cn else "Simultaneous sources",
                    1.: "进一步错峰" if cn else "Reinforced separation"}
    forms = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated / tributary-rich", "Mainstem dominated / sparse", "Broad / tributary-rich")
    inputs = list(folder.iterdir())+[Path(__file__), PLANFORM, CORRIDORS,
        Path("scripts/plot_doc_river_planform_v1.py"), Path("scripts/plot_doc_monitored_river_footprint_v1.py"),
        Path("src/river_graph/topology/river_planform.py"), Path("src/river_graph/analysis/river_monitored_footprint.py"),
        CACHE/"flowlines.sqlite", Path("cache/nldplus_vaa.parquet")]
    outputs, mapped = [], []
    suffix = "_cn" if cn else ""

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    example = reps.sort_values("cluster").iloc[0]
    trace = curves.loc[curves.case_id.eq(example.case_id)]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.1))
    fig.subplots_adjust(left=.09, right=.96, top=.84, bottom=.13, hspace=.45, wspace=.32)
    ax = axes[0, 0]
    for phase in (-1., 0., 1.):
        t = trace.loc[trace.source_phase_factor.eq(phase) & trace.volume_factor.eq(1) &
                      trace.flow_factor.eq(1) & trace.mixing_fraction.eq(0)]
        ax.plot(t.relative_time, t.outlet_anomaly, color=PHASE_COLORS[phase], linewidth=1.8, label=phase_names[phase])
    ax.set(xlim=(.1, 2.2), ylim=(0, 1.08), xlabel="相对时间" if cn else "Relative time",
           ylabel="出口峰形 / 单支流输入高度" if cn else "Outlet anomaly / branch input height")
    ax.set_title("a  相同河网，输入时间能改变峰形" if cn else "a  Input clocks change the same network's pulse", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    settings = [(1., 0., "原通行时间" if cn else "Reference transit", "#86979C", "-"),
                (2., 0., "仅延长通行时间" if cn else "Longer transit only", "#C5814A", "--"),
                (2., .5, "延时 + 分布式混合" if cn else "Longer transit + mixing", "#257F88", "-")]
    for volume, mixing, label, color, style in settings:
        t = trace.loc[trace.source_phase_factor.eq(0) & trace.volume_factor.eq(volume) &
                      trace.flow_factor.eq(1) & trace.mixing_fraction.eq(mixing)]
        ax.plot(t.relative_time, t.outlet_anomaly, color=color, linestyle=style, label=label, linewidth=1.8)
    ax.set(xlim=(.1, 3.2), ylim=(0, 1.08), xlabel="相对时间" if cn else "Relative time",
           ylabel="出口 DOC 异常（归一值）" if cn else "Normalized outlet DOC anomaly")
    ax.set_title("b  晚到与削峰是不同作用" if cn else "b  Later arrival differs from peak attenuation", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    selected = cohort.loc[cohort.cohort.eq("whole_confluence") & cohort.flow_rule.eq("area_proxy") &
                          cohort.input_sd.eq(.15) & cohort.volume_factor.eq(1) & cohort.flow_factor.eq(1) &
                          cohort.mixing_fraction.eq(.5) & cohort.metric.eq("mixing_peak_reduction_pct")].sort_values("source_phase_factor")
    x = np.arange(3)
    means = selected["mean"].to_numpy()
    ax.bar(x, means, color=[PHASE_COLORS[p] for p in (-1., 0., 1.)], width=.55)
    ax.errorbar(x, means, yerr=np.vstack([means-selected.ci_low, selected.ci_high-means]),
                fmt="none", color="#354950", capsize=4, linewidth=1.2)
    for i, value in enumerate(means):
        ax.text(i, value+2.6, f"{value:.1f}%", ha="center", fontsize=10)
    ax.set(xticks=x, xticklabels=[phase_names[p] for p in (-1., 0., 1.)], ylim=(0, 34),
           ylabel="混合相对纯延时的额外削峰 (%)" if cn else "Extra attenuation from mixing (%)")
    ax.set_title("c  相同混合强度，来水时序改变收益" if cn else "c  Input alignment changes the mixing gain", loc="left", fontsize=11)

    ax = axes[1, 1]
    for row in field.itertuples():
        ax.scatter(row.same_length_transit_proxy_ratio, row.reported_tracer_transit_ratio,
                   s=50, color="#257F88", edgecolor="white", linewidth=.5)
        ax.annotate(row.confluence, (row.same_length_transit_proxy_ratio, row.reported_tracer_transit_ratio),
                    xytext=(5, 7), textcoords="offset points", fontsize=9)
    ax.plot([.75, 2.85], [.75, 2.85], color="#A4B7BC", linestyle="--", linewidth=1)
    ax.axhline(1, color="#D0DADC", linestyle=":", linewidth=1)
    ax.axvline(1, color="#D0DADC", linestyle=":", linewidth=1)
    ax.set(xlim=(.75, 2.85), ylim=(.75, 2.85),
           xlabel="断面代理 / 流量推得的时间比" if cn else "Area-proxy / flow transit ratio",
           ylabel="示踪测得的 100 m 通行时间比" if cn else "Reported tracer 100-m transit ratio")
    ax.set_title("d  实测几何不能直接替代示踪时间" if cn else "d  Field geometry does not replace tracer timing", loc="left", fontsize=11)
    fig.suptitle("河网结构如何作用：来水错峰 × 汇流后混合" if cn else "How structure acts: incoming clocks × post-junction mixing",
                 x=.07, y=.98, ha="left", fontsize=16)
    fig.text(.07, .925, (f"a/b 使用真实河网 {example.station} 的路径；c 汇总 295 个真实汇流结构；d 为 5 个实测汇流点。" if cn else
                        f"a/b: Real paths at station {example.station}; c: 295 real junction corridors; d: Five measured confluences."), fontsize=10)
    fig.text(.07, .035, ("a–c 为恒定流量、保守 DOC 脉冲对照；混合份额固定为共享时间的 50%，不是实测削峰率。\n"
                        "c 线为 HUC4 分块 95% 区间；d 用断面宽×平均实测深度作为代理，保留缺测与测量差异。" if cn else
                        "a–c: Constant-flow conservative concentration scenarios; 50% shared-time mixing is prescribed, not measured DOC attenuation.\n"
                        "c: HUC4-block 95% intervals. d: Width × observed mean depth is a rectangular proxy, not exact wetted area."), fontsize=9, color="#53696E")
    save(fig, "junction_timing_and_mixing")

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.2), gridspec_kw={"height_ratios": [1.2, 1., 1.]})
    fig.subplots_adjust(left=.075, right=.97, top=.87, bottom=.13, hspace=.55, wspace=.34)
    corridors = pd.read_csv(CORRIDORS, dtype={"station": str})
    connection = sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True)
    for col, row in enumerate(reps.sort_values("cluster").itertuples()):
        color = COLORS[int(row.cluster)]
        ax = axes[0, col]
        draw_network(ax, connection, row, "#A8BBBE")
        reaches = corridors.loc[corridors.example_group.eq(row.example_group)]
        parts, gap, digest = mapped_corridor(connection, reaches)
        for segment, part in reaches.groupby("segment"):
            ink = "#C5814A" if segment == "common" else color
            ax.add_collection(LineCollection(segments(parts[int(cid)] for cid in part.comid),
                                            colors=ink, linewidths=2., zorder=6))
        mapped.append({"station": row.station, "geometry_sha256": digest, "maximum_connectivity_gap_m": gap})
        inputs.extend([CACHE/"members_full"/f"comid_{int(row.comid)}.npz", CACHE/"basins"/f"comid_{int(row.comid)}.json"])
        ax.set_title(f"{'abc'[col]}  {forms[int(row.cluster)-1]}\n{row.station}", loc="left", fontsize=11, pad=10)
        ax = axes[1, col]
        trace = curves.loc[curves.case_id.eq(row.case_id) & curves.volume_factor.eq(1) &
                           curves.flow_factor.eq(1) & curves.mixing_fraction.eq(0)]
        for phase in (-1., 0., 1.):
            t = trace.loc[trace.source_phase_factor.eq(phase)]
            ax.plot(t.relative_time, t.outlet_anomaly, color=PHASE_COLORS[phase], linewidth=1.6,
                    label=phase_names[phase])
        ax.set(xlim=(0, 2.4), ylim=(0, 1.08), xlabel="相对时间" if cn else "Relative time",
               ylabel="出口 DOC 异常" if cn else "Outlet DOC anomaly")
        ax.set_title(f"{'def'[col]}  "+("保持该河网，只改输入时间" if cn else "Same river, different source clocks"), loc="left", fontsize=10)
        if col == 0:
            ax.legend(frameon=False, fontsize=8)
        ax = axes[2, col]
        selected = f.loc[f.cohort.eq("whole_confluence") & f.cluster.eq(row.cluster) & f.flow_rule.eq("area_proxy") &
                         f.input_sd.eq(.15) & f.volume_factor.eq(1) & f.flow_factor.eq(1) & f.mixing_fraction.eq(.5)]
        values = [selected.loc[selected.source_phase_factor.eq(p), "pulse_peak"].to_numpy() for p in (-1., 0., 1.)]
        box = ax.boxplot(values, positions=[0, 1, 2], widths=.5, showfliers=False, patch_artist=True,
                         medianprops={"color": "#263B42"})
        for p, patch, array, pos in zip((-1., 0., 1.), box["boxes"], values, (0, 1, 2), strict=True):
            patch.set(facecolor=PHASE_COLORS[p], alpha=.22, edgecolor=PHASE_COLORS[p])
            jitter = np.random.default_rng(17+int(row.cluster)).uniform(-.15, .15, len(array))
            ax.scatter(pos+jitter, array, s=7, color=PHASE_COLORS[p], alpha=.35, edgecolor="none")
        short = ("汇流同步", "源头同步", "再错峰") if cn else ("Aligned", "Same source", "Separated")
        ax.set(xticks=[0, 1, 2], xticklabels=short, ylim=(0, 1.05), ylabel="出口 / 输入峰值" if cn else "Outlet / input peak")
        ax.set_title(f"{'ghi'[col]}  "+(f"类内分布 · n={len(values[0])}" if cn else f"Within-form distribution · n={len(values[0])}"), loc="left", fontsize=10)
    connection.close()
    fig.suptitle("三类真实河网：结构与输入时间共同决定响应" if cn else "Three real river forms: structure and input clocks act together",
                 x=.06, y=.99, ha="left", fontsize=16)
    fig.text(.06, .94, "保持原来的三类形态与代表河网；选定汇流路径叠加在真实全河网之上。" if cn else
             "Original classes and representatives retained; selected junction paths overlaid on the real complete networks.", fontsize=10)
    fig.text(.06, .038, ("地图保持真实方向与比例尺，面板范围独立；橙色路径为共享下游河段。d–f 为代表路径的保守脉冲对照。\n"
                        "g–i：点为选定真实汇流结构，盒为四分位范围；固定共享混合份额 50%，两处无合格汇流的稀疏河网仍保留排除记录。" if cn else
                        "Maps keep native orientation and scale bars with independent extents; orange marks the shared downstream reach. d–f: Controlled corridor responses.\n"
                        "g–i: Points are selected real junctions, boxes are interquartile ranges; prescribed 50% mixing. Two sparse no-junction exclusions remain recorded."), fontsize=9, color="#53696E")
    save(fig, "real_forms_incoming_clocks")
    receipt = {"input_hashes": {str(p): sha256_file(p) for p in sorted(set(inputs))},
               "output_hashes": {str(p): sha256_file(p) for p in outputs}, "mapped_geometry": mapped,
               "scope": "real maps plus controlled conservative concentration responses; field proxies separately identified"}
    (out/f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Saved {len(outputs)} figures ({'Chinese' if cn else 'English'}).")


if __name__ == "__main__":
    main()
