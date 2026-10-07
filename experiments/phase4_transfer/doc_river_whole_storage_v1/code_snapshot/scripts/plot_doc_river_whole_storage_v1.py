"""Scientific figures of real river forms and conditional storage responses."""
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
from plot_doc_river_planform_v1 import draw_network, segments

from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import project_geometry, read_cached_lines

ROOT = Path("experiments/phase4_transfer/doc_river_whole_storage_v1")
PLANFORM = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis")
CACHE = Path("data/raw/river_planform_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")
GOLD = "#D2AC69"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 230, "text.color": "#263B42", "axes.labelcolor": "#263B42"})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    f = pd.read_csv(a/"scenario_metrics.csv", dtype={"station": str, "huc4": str})
    geometry = pd.read_csv(a/"network_descriptors.csv", dtype={"station": str, "huc4": str})
    summary = pd.read_csv(a/"class_summary.csv")
    matched = pd.read_csv(a/"matched_contrasts.csv")
    traces = pd.read_parquet(a/"representative_responses.parquet")
    reps = pd.read_csv(a/"representatives.csv", dtype={"station": str}).merge(
        pd.read_csv(PLANFORM/"classes.csv", dtype={"station": str}), on=["station", "comid", "cluster"], validate="one_to_one")
    vaa = pd.read_parquet("cache/nldplus_vaa.parquet", columns=["comid", "wbareatype"]).set_index("comid")
    source_files = list(a.iterdir())+[PLANFORM/"classes.csv", Path("cache/nldplus_vaa.parquet"),
        CACHE/"flowlines.sqlite", Path("scripts/plot_doc_river_planform_v1.py"),
        Path("src/river_graph/topology/river_planform.py"), Path(__file__)]
    names = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated / tributary-rich", "Mainstem dominated / sparse", "Broad / tributary-rich")
    suffix, written = "_cn" if cn else "", []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13.6, 11.5))
    example_metrics = f[f.station.isin(reps.station) & f.input_sd.eq(.15) & f.source_points.eq(1)]
    curve_end = example_metrics.t90.max()+.8
    curve_top = min(1.04, example_metrics.pulse_peak.max()*1.22)
    effect_interval = summary[summary.input_sd.eq(.15) & summary.source_points.eq(1)
                              & summary.metric.eq("peak_reduction_pct")]
    effect_top = effect_interval.ci_high.max()*1.1
    connection = sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True)
    for col, r in enumerate(reps.sort_values("cluster").itertuples()):
        color = COLORS[col]
        draw_network(axes[0, col], connection, r, color)
        member = CACHE/"members_full"/f"comid_{int(r.comid)}.npz"
        comids = np.load(member)["comids"]
        stored = vaa.reindex(comids).wbareatype.isin(["LakePond", "Reservoir"]).to_numpy()
        lines = [project_geometry(v) for v in read_cached_lines(connection, comids[stored]).values()]
        axes[0, col].add_collection(LineCollection(segments(lines), colors=GOLD, linewidths=1.6, zorder=3))
        n = int(f[f.cluster.eq(r.cluster) & f.storage_fraction.eq(0) & f.input_sd.eq(.15) & f.source_points.eq(1)].shape[0])
        axes[0, col].set_title(f"{'abc'[col]}  {names[col]}\nn = {n} · {r.station}", loc="left", fontsize=11, pad=9)
        source_files.extend([member, CACHE/"basins"/f"comid_{int(r.comid)}.json"])
        ax = axes[1, col]
        tr = traces[traces.station.eq(r.station) & traces.input_sd.eq(.15)]
        s = f[f.station.eq(r.station) & f.input_sd.eq(.15) & f.source_points.eq(1)]
        for fraction, linecolor, style in ((0., "#9CAAAF", "-"), (.5, color, "-"), (1., GOLD, "--")):
            t = tr[tr.storage_fraction.eq(fraction)]
            ax.plot(t.time, t.outlet_anomaly, color=linecolor, linestyle=style, linewidth=1.8,
                    label=(f"湖库时间分配 f={fraction:g}" if cn else f"Storage allocation f={fraction:g}"))
        ax.set(xlim=(-.5, curve_end), ylim=(0, curve_top),
            xlabel="归一化到达时间" if cn else "Normalized arrival time",
            ylabel="出口脉冲 / 输入峰值" if cn else "Outlet pulse / input peak")
        ax.set_title(f"{'def'[col]}  "+("相同输入、固定平均时距" if cn else "Identical forcing, unchanged mean"), loc="left", pad=12)
        if col == 0:
            ax.legend(frameon=False, fontsize=7.5)
        ax = axes[2, col]
        rows = summary[summary.cluster.eq(r.cluster) & summary.input_sd.eq(.15) & summary.source_points.eq(1)
                       & summary.metric.eq("peak_reduction_pct")]
        for rr in rows.itertuples():
            ax.plot([rr.storage_fraction]*2, [rr.ci_low, rr.ci_high], color=color, linewidth=1.4)
            ax.scatter(rr.storage_fraction, rr.mean, color=color, s=30, zorder=3)
        ax.axhline(0, color="#BAC5C7", linewidth=.7)
        ax.set(ylim=(-.25, effect_top), xticks=[0, .25, .5, 1.], xlabel="湖库时间分配 f" if cn else "Storage mean allocation f",
               ylabel="相对无滞留情景的峰值降低（%）" if cn else "Peak reduction vs translation (%)")
        ax.set_title(f"{'ghi'[col]}  "+("该类全部河网：均值与 HUC4 区间" if cn else "All networks: mean and HUC4 interval"), loc="left", pad=12, fontsize=10)
    connection.close()
    fig.suptitle("完整真实河网：路径分散与湖库滞留怎样改变 DOC 信号？" if cn else
                 "Complete real networks: path dispersion and mapped storage reshape DOC signals",
                 x=.045, ha="left", fontsize=16, y=.99)
    fig.text(.045, .947, ("原三类形态，297 个站点河网 / 295 个出口。金色河段为地图中的湖泊或水库；代表河网按几何选定。") if cn else
             "Three existing forms; 297 station-network instances / 295 outlets. Gold reaches mark mapped lakes/reservoirs; geometry-selected examples.", fontsize=9)
    fig.text(.045, .018, ("所有局地汇水区接收相同 DOC 输入，流量权重按增量面积分配。时间 = 河道长度 / √流域面积；不是实测天数。\n"
             "f 将同一路径的部分平移时间改为守恒滞留，不改变平均到达时间。下排为 HUC4 区组 bootstrap 95% 区间；地图独立比例尺，曲线统一坐标。") if cn else
             "Identical DOC input in every incremental catchment; constant flow shares proportional to area. Time = channel length / √basin area, not measured days.\n"
             "f redistributes translation into conservative storage at unchanged path mean. Bottom: HUC4-block bootstrap 95%; maps: independent scales, curves: shared axes.", fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.90, bottom=.11, wspace=.34, hspace=.46)
    save(fig, "whole_real_forms_and_storage")

    mid = f[f.input_sd.eq(.15) & f.source_points.eq(1) & f.storage_fraction.eq(.5)].merge(
        geometry[["station", "storage_exposed_flow_share", "storage_mean_share", "serial_storage_variance", "lumped_storage_variance"]],
        on="station", validate="one_to_one")
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.7))
    for col, metric in enumerate(("pulse_peak", "duration_80")):
        ax = axes[0, col]
        for group in (1, 2, 3):
            for points, marker, offset in ((1, "o", -.10), (5, "s", .10)):
                s = summary[summary.cluster.eq(group) & summary.input_sd.eq(.15)
                            & summary.storage_fraction.eq(.5) & summary.source_points.eq(points) & summary.metric.eq(metric)].iloc[0]
                x = group+offset
                ax.plot([x, x], [s.ci_low, s.ci_high], color=COLORS[group-1], linewidth=1.3)
                ax.scatter(x, s["mean"], marker=marker, color=COLORS[group-1], s=28)
        ax.set(xticks=[1, 2, 3], xticklabels=[n.replace(" / ", "\n").replace("、", "\n") for n in names])
        ax.set_ylabel(("出口脉冲峰值" if col == 0 else "中央 80% 脉冲持续时间") if cn else
                      ("Outlet peak / input peak" if col == 0 else "Central 80% pulse duration"))
        ax.set_title(f"{'ab'[col]}  "+("源位置敏感性：○ 中点，■ 沿段五点" if cn else "Source placement: ○ midpoint, ■ five points"), loc="left", pad=12, fontsize=10)
    ax = axes[0, 2]
    for group in (1, 2, 3):
        s = mid[mid.cluster.eq(group)]
        jitter = np.random.default_rng(42).uniform(-.23, .23, len(s))
        ax.scatter(group+jitter, s.peak_reduction_pct, s=13, color=COLORS[group-1], alpha=.55, linewidth=0)
        ax.plot([group-.27, group+.27], [s.peak_reduction_pct.median()]*2, color="#263B42", linewidth=1.6)
    ax.axhline(0, color="#A7B5B9", linewidth=.8)
    ax.set(xticks=[1, 2, 3], xticklabels=[n.replace(" / ", "\n").replace("、", "\n") for n in names],
           ylabel="额外峰值降低（%）" if cn else "Additional peak reduction (%)")
    ax.set_title("c  "+("同一形态内部也有很大差异" if cn else "Within-form response distributions"), loc="left", pad=12)
    for col, x in enumerate(("storage_exposed_flow_share", "storage_share_of_structural_variance")):
        ax = axes[1, col]
        for group in (1, 2, 3):
            s = mid[mid.cluster.eq(group)]
            ax.scatter(100*s[x], s.peak_reduction_pct, s=17, color=COLORS[group-1], alpha=.65, linewidth=0)
        ax.axhline(0, color="#A7B5B9", linewidth=.8)
        ax.set(xlabel=("经过湖库的流量份额（%）" if col == 0 else "湖库占结构时间方差的份额（%）") if cn else
                      ("Flow share traversing mapped storage (%)" if col == 0 else "Storage share of structural time variance (%)"),
               ylabel="额外峰值降低（%）" if cn else "Additional peak reduction (%)")
        ax.set_title(f"{'de'[col]}  "+("湖库位置和路径关系" if cn else "Storage location and path arrangement"), loc="left", pad=12)
    ax = axes[1, 2]
    p = matched[matched.class_a.eq(1) & matched.class_b.eq(3) & matched.input_sd.eq(.15)
                & matched.metric.eq("pulse_peak")]
    settings = ((0., 1), (.5, 1), (.5, 5))
    for j, (fraction, points) in enumerate(settings):
        r = p[p.storage_fraction.eq(fraction) & p.source_points.eq(points)].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [j, j], color=COLORS[2], linewidth=2)
        ax.scatter(r.difference_b_minus_a, j, color=COLORS[2], s=34)
    ax.axvline(0, color="#A7B5B9", linewidth=.8)
    ax.set(yticks=range(3), yticklabels=("无滞留 / 中点", "f=0.5 / 中点", "f=0.5 / 沿段五点") if cn else
           ("No storage / midpoint", "f=0.5 / midpoint", "f=0.5 / five points"),
           ylim=(2.6, -.6), xlabel="宽展 − 细长：出口峰值差" if cn else "Broad − elongated: outlet peak difference")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("f  "+("原固定配对：形态差异是否保留？" if cn else "Original matched pairs: form contrast"), loc="left", pad=12, fontsize=10)
    fig.suptitle("形态之间的差异，与同一形态内部的差异" if cn else "Between-form contrasts and within-form heterogeneity",
                 x=.045, ha="left", fontsize=16, y=.99)
    fig.text(.045, .934, ("统一中等脉冲（SD=0.15）；湖库时间分配 f=0.5。不将经过湖库的长度比例视为实测水力滞留时间。") if cn else
             "Middle pulse SD=0.15; storage allocation f=0.5. Mapped storage length is not a measured hydraulic residence time.", fontsize=9)
    fig.text(.045, .018, ("a、b、f：HUC4 区组 bootstrap 95%；c–e：全部河网的确定性情景响应，c 的横线为中位数。\n"
             "固定配对不按结果重新匹配；负削峰率表示峰值上升。信号总量守恒，曲线变化不表示 DOC 被化学移除。") if cn else
             "a/b/f: HUC4-block bootstrap 95%; c–e: deterministic scenario responses, c lines are medians. No outcome-based rematching.\n"
             "Negative reductions indicate peak increases. Identical forcing and conserved integrated anomaly; response changes are not chemical DOC removal.", fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.87, bottom=.16, wspace=.43, hspace=.52)
    save(fig, "whole_form_storage_heterogeneity")
    receipt = {"inputs": {str(p): sha256_file(p) for p in source_files},
               "outputs": {str(p): sha256_file(p) for p in written}}
    (out/f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
