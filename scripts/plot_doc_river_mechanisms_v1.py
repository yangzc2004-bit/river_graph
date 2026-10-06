"""Render observed mixing evidence, actual confluence geometry and simulations."""
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
from analyze_doc_river_mechanisms_v1 import primary_route
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from shapely.geometry import Point

from river_graph.topology.river_planform import project_geometry, read_cached_lines
from river_graph.topology.river_structure import VAA_COLUMNS, ReachNetwork

ROOT = Path("experiments/phase4_transfer/doc_river_mechanisms_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    args = parser.parse_args()
    cn = args.chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 230})
    out = ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix = "_cn" if cn else ""
    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{name}{suffix}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    results = pd.read_csv(ROOT/"analysis/mixing_results.csv")
    receivers = pd.read_csv(ROOT/"analysis/mixing_receivers.csv", dtype={"target": str})
    cases = pd.read_csv(ROOT/"analysis/mixing_cases.csv", dtype={"pair_id": str, "target": str})
    groups = [("monthly", "area"), ("monthly", "flow"), ("same_day", "area"), ("same_day", "flow")]
    titles = ["共同月份·面积代理", "共同月份·流量代理", "同日采样·面积代理", "同日采样·流量代理"] if cn else [
        "Common months · area", "Common months · flow", "Same-day samples · area", "Same-day samples · flow"]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.6))
    for i, ((population, weighting), label) in enumerate(zip(groups, titles, strict=True)):
        s = results[(results.population == population) & (results.weighting == weighting)
                    & results.group.eq("all") & results.metric.eq("mae_gain_pct")]
        if s.empty:
            axes[0].text(0, i, "样本不足" if cn else "Insufficient overlap", va="center", fontsize=9)
            continue
        for unit, offset, filled in (("target", -.09, True), ("huc4", .09, False)):
            row = s[s.resampling_unit == unit].iloc[0]
            axes[0].plot([row.ci_low, row.ci_high], [i+offset]*2, color=COLORS[0], linewidth=1.5)
            axes[0].scatter(row.estimate, i+offset, s=35, facecolor=COLORS[0] if filled else "white", edgecolor=COLORS[0], zorder=3)
        axes[0].text(1.02, i, f"n={int(row.n_receivers)}", transform=axes[0].get_yaxis_transform(), va="center", fontsize=8)
    axes[0].axvline(0, color="#6F7D80", linewidth=.8, linestyle="--")
    axes[0].set(yticks=range(4), yticklabels=titles, ylim=(3.5, -.6),
                xlabel="混合代理的 MAE 改善（%）" if cn else "Mixture-proxy MAE reduction (%)")
    axes[0].set_title("a  支流整合的信息价值" if cn else "a  Information from tributary integration", loc="left", fontweight="bold", pad=45)
    s = receivers[receivers.population.eq("monthly") & receivers.weighting.eq("area")]
    for c, label in zip((1, 2, 3), ("细长多支流", "主干少支流", "宽展多支流") if cn else
                        ("Elongated", "Mainstem", "Broad"), strict=True):
        v = s[s.cluster.eq(c)]
        axes[1].scatter(v.branch_rho, v.downstream_cv_change, color=COLORS[c-1], s=30, alpha=.75, label=f"{label} (n={len(v)})")
    u = s[s.cluster.isna()]
    if len(u):
        axes[1].scatter(u.branch_rho, u.downstream_cv_change, color="#929C9F", marker="x", s=30, label="未分类" if cn else "Unclassified")
    axes[1].axhline(0, color="#6F7D80", linewidth=.8, linestyle="--")
    axes[1].set(xlabel="支流信号相关（去季节／年趋势）" if cn else "Branch correlation (calendar/year adjusted)",
                ylabel="下游 CV − 支流平均 CV" if cn else "Downstream CV − mean branch CV")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.02))
    axes[1].set_title("b  同步性与下游波动" if cn else "b  Synchrony and downstream variation", loc="left", fontweight="bold", pad=45)
    for i, (population, weighting) in enumerate(groups):
        s = cases[(cases.population == population) & (cases.weighting == weighting)]
        counts = [len(s), s.target.nunique()]
        axes[2].barh(i-.15, counts[0], height=.27, color=COLORS[0], alpha=.75)
        axes[2].barh(i+.15, counts[1], height=.27, color=COLORS[1])
        for value, offset in zip(counts, (-.15, .15), strict=True):
            axes[2].text(value+1, i+offset, str(value), va="center", fontsize=9)
    axes[2].set(yticks=range(4), yticklabels=titles, ylim=(3.5, -.6), xlabel="组合／下游站点数" if cn else "Combinations / receiving stations")
    axes[2].plot([], [], linewidth=6, color=COLORS[0], label="组合" if cn else "Combinations")
    axes[2].plot([], [], linewidth=6, color=COLORS[1], label="下游站点" if cn else "Receivers")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("c  真正可用的观测组合" if cn else "c  Observation overlap", loc="left", fontweight="bold", pad=45)
    fig.suptitle("真实支流整合与 DOC" if cn else "Observed tributary integration and DOC", fontsize=17, x=.035, ha="left")
    fig.text(.035, .03, "参考：两支流单独预测下游的平均误差。实心／空心区间：下游站点／HUC4 bootstrap 95%；每个下游站点等权。" if cn else
             "Reference: mean error of the two individual tributaries. Filled/open intervals: receiver/HUC4 bootstrap 95%; receivers equally weighted.", fontsize=9)
    fig.subplots_adjust(left=.15, right=.98, top=.78, bottom=.19, wspace=.75)
    save(fig, "tributary_integration")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for ax, population, title in zip(axes, ("monthly", "same_day"),
                                    ("共同月份", "同日采样") if cn else ("Common months", "Same-day samples"), strict=True):
        for i, weighting in enumerate(("area", "flow")):
            s = results[results.population.eq(population) & results.weighting.eq(weighting)
                        & results.group.eq("all") & results.metric.eq("weighting_gain_pct")]
            if s.empty:
                continue
            for unit, offset, marker, filled in (("target", -.14, "o", True), ("huc4", 0., "o", False), ("component", .14, "D", False)):
                r = s[s.resampling_unit.eq(unit)].iloc[0]
                ax.plot([r.ci_low, r.ci_high], [i+offset]*2, color=COLORS[i], linewidth=1.5)
                ax.scatter(r.estimate, i+offset, facecolor=COLORS[i] if filled else "white", edgecolor=COLORS[i], marker=marker, s=35, zorder=3)
            ax.text(1.02, i, f"n={int(r.n_receivers)}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.axvline(0, color="#6F7D80", linestyle="--", linewidth=.8)
        ax.set(yticks=[0, 1], yticklabels=["面积权重", "流量权重"] if cn else ["Area weights", "Flow weights"], ylim=(1.5, -.5),
               xlabel="相对等权浓度平均的 MAE 改善（%）" if cn else "MAE reduction vs equal-concentration average (%)")
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle("水文权重是否超越简单平均？" if cn else "Do hydrological weights improve on simple averaging?", fontsize=16, x=.07, ha="left")
    fig.text(.07, .025, "参考：支流浓度等权平均。实心圆／空心圆／菱形：站点／HUC4／共享站点组 bootstrap 95%；单组不画区间。" if cn else
             "Reference: equal source concentration average. Filled/open circles/diamonds: receiver/HUC4/shared-station-group bootstrap 95%; single-group intervals omitted.", fontsize=9)
    fig.subplots_adjust(left=.15, right=.91, top=.78, bottom=.2, wspace=.6)
    save(fig, "hydrological_weighting_increment")

    # Representative is chosen by geometry and sampling overlap, never DOC fit.
    pool = cases[cases.population.eq("same_day") & cases.weighting.eq("flow")]
    if pool.empty:
        pool = cases[cases.population.eq("monthly") & cases.weighting.eq("flow")]
    near = pool[pool.near_complete_confluence]
    if not near.empty:
        pool = near
    example = pool.sort_values(["n_records", "pair_id"], ascending=[False, True]).iloc[0]
    inventory = pd.read_csv(ROOT/"analysis/confluence_inventory.csv", dtype={"source_a": str, "source_b": str, "target": str})
    info = inventory[inventory.pair_id.eq(example.pair_id)].iloc[0]
    records = pd.read_parquet(ROOT/"analysis/mixing_records.parquet")
    f = records[records.pair_id.eq(example.pair_id) & records.population.eq(example.population) & records.weighting.eq(example.weighting)].sort_values("date")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.6), gridspec_kw={"width_ratios": [1., 1.5, 1.]})
    nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str}).set_index("site_no")
    network = ReachNetwork(pd.read_parquet("cache/nldplus_vaa.parquet", columns=list(VAA_COLUMNS)))
    with sqlite3.connect("data/raw/river_planform_v1/flowlines.sqlite") as connection:
        for station, color, text in ((info.source_a, COLORS[0], "A"), (info.source_b, COLORS[1], "B")):
            route = primary_route(network, nodes.loc[station, "comid"], nodes.loc[info.target, "comid"])
            lines = read_cached_lines(connection, network.comid[route])
            parts = []
            for line in lines.values():
                projected = project_geometry(line)
                for piece in (projected.geoms if projected.geom_type == "MultiLineString" else [projected]):
                    parts.append(np.asarray(piece.coords)/1000)
            axes[0].add_collection(LineCollection(parts, colors=color, linewidths=1.3, alpha=.9))
            point = project_geometry(Point(nodes.loc[station, "dec_long_va"], nodes.loc[station, "dec_lat_va"]))
            axes[0].scatter(point.x/1000, point.y/1000, color=color, s=40, edgecolor="white", zorder=3)
            axes[0].annotate(f"{text}: {station}", (point.x/1000, point.y/1000), xytext=(4, 5), textcoords="offset points", fontsize=8)
    point = project_geometry(Point(nodes.loc[info.target, "dec_long_va"], nodes.loc[info.target, "dec_lat_va"]))
    axes[0].scatter(point.x/1000, point.y/1000, color="#263B42", s=45, marker="s", zorder=4)
    axes[0].annotate(f"D: {info.target}", (point.x/1000, point.y/1000), xytext=(4, -12), textcoords="offset points", fontsize=8)
    axes[0].autoscale()
    axes[0].set_aspect("equal")
    axes[0].margins(.2)
    axes[0].set(xlabel="投影横坐标（km）" if cn else "Projected easting (km)",
                ylabel="投影纵坐标（km）" if cn else "Projected northing (km)")
    axes[0].set_title("a  真实河道路径" if cn else "a  Actual NHD river paths", loc="left", fontweight="bold")
    for column, color, label, style in (("doc_a", COLORS[0], "支流 A" if cn else "Tributary A", ":"),
                                       ("doc_b", COLORS[1], "支流 B" if cn else "Tributary B", ":"),
                                       ("mixture_doc", COLORS[2], "流量混合代理" if cn else "Flow mixture proxy", "-"),
                                       ("doc_target", "#263B42", "下游观测" if cn else "Downstream observed", "-")):
        axes[1].plot(f.date, f[column], color=color, linewidth=1.1, linestyle=style, marker="o", markersize=2.3, label=label)
    axes[1].set(xlabel="同日采样日期" if cn and example.population == "same_day" else "Date", ylabel="DOC（mg/L）" if cn else "DOC (mg/L)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("b  共同观测序列" if cn else "b  Common observations", loc="left", fontweight="bold")
    axes[2].scatter(f.mixture_doc, f.doc_target, color=COLORS[2], s=28, alpha=.7)
    limit = float(max(f.mixture_doc.max(), f.doc_target.max())*1.1)
    axes[2].plot([0, limit], [0, limit], linestyle="--", linewidth=.9, color="#6F7D80")
    axes[2].set(xlim=(0, limit), ylim=(0, limit), xlabel="混合代理 DOC（mg/L）" if cn else "Mixture proxy DOC (mg/L)",
                ylabel="下游 DOC（mg/L）" if cn else "Downstream DOC (mg/L)")
    axes[2].set_title("c  混合与实际下游" if cn else "c  Mixture vs downstream", loc="left", fontweight="bold")
    fig.suptitle("支流汇合观测案例" if cn else "A monitored tributary-integration example", fontsize=17, x=.035, ha="left")
    fig.text(.035, .03, (f"{len(f)} 对记录；源区覆盖 {info.source_drainage_coverage:.0%}；汇合后至下游站 {info.junction_receiver_km:.1f} km。案例按采样重叠选择；流量为日均／月均代理。" if cn else
              f"{len(f)} paired records; source drainage coverage {info.source_drainage_coverage:.0%}; junction-to-receiver {info.junction_receiver_km:.1f} km. Selected by sampling overlap, not fit."), fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, top=.8, bottom=.2, wspace=.42)
    save(fig, "real_confluence_example")

    associations = pd.read_csv(ROOT/"analysis/pathway_associations.csv")
    edges = pd.read_csv(ROOT/"analysis/pathway_edges.csv")
    terms = ["wetland_change_pct", "forest_change_pct", "added_drainage_fraction", "log_path_km", "storage_fraction", "temperature_change", "log_flow_ratio"]
    labels = ["湿地覆盖差", "森林覆盖差", "新增汇水区比例", "河程长度", "湖库路径比例", "温度变化", "流量比"] if cn else [
        "Wetland-cover contrast", "Forest-cover contrast", "Added drainage fraction", "River-path length", "Storage-path fraction", "Temperature change", "Flow ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 6.4))
    for ax, outcome, title in zip(axes[:2], ("log_doc_change", "rho_calendar_adjusted"),
                                  ("a  沿程 DOC 浓度变化", "b  上下游信号相似性") if cn else
                                  ("a  Longitudinal DOC change", "b  Upstream–downstream similarity"), strict=True):
        for i, term in enumerate(terms):
            r = associations[(associations.outcome == outcome) & (associations.term == term)].iloc[0]
            for low, high, offset, filled in ((r.huc4_ci_low, r.huc4_ci_high, -.08, True),
                                              (r.component_ci_low, r.component_ci_high, .08, False)):
                ax.plot([low, high], [i+offset]*2, color=COLORS[0], linewidth=1.5)
                ax.scatter(r.estimate_per_receiver_sd, i+offset, facecolor=COLORS[0] if filled else "white", edgecolor=COLORS[0], s=35, zorder=3)
        ax.axvline(0, linestyle="--", color="#6F7D80", linewidth=.8)
        ax.set(yticks=range(len(terms)), yticklabels=labels, ylim=(len(terms)-.5, -.5),
               xlabel="每一标准差的调整后系数" if cn else "Adjusted coefficient per receiver SD")
        ax.set_title(title, loc="left", fontweight="bold")
    s = edges.dropna(subset=["rho_hydro_adjusted", "rho_calendar_common_hydro"])
    axes[2].scatter(s.rho_calendar_common_hydro, s.rho_hydro_adjusted, color=COLORS[2], s=26, alpha=.7)
    axes[2].plot([-1, 1], [-1, 1], linestyle="--", linewidth=.9, color="#6F7D80")
    axes[2].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="去季节／年趋势后的相关" if cn else "Calendar/year-adjusted correlation",
                ylabel="进一步去局地水文后的相关" if cn else "Also local-hydro-adjusted correlation")
    axes[2].set_title(f"c  共同水文驱动（{len(s)} 条连接）" if cn else f"c  Shared hydro forcing ({len(s)} edges)", loc="left", fontweight="bold")
    fig.suptitle("陆地来源、传输路径与 DOC 变化" if cn else "Terrestrial source contrasts, transport paths and DOC", fontsize=16, x=.035, ha="left")
    fig.text(.035, .03, "实心／空心：HUC4／共享站点组 bootstrap 95%。覆盖差为流域平均值；浓度变化含侧向输入与处理，河程不等于停留时间。" if cn else
             "Filled/open: HUC4/shared-station-group bootstrap 95%. Watershed-average cover contrasts; concentration change includes lateral input and processing.", fontsize=9)
    fig.subplots_adjust(left=.15, right=.98, top=.78, bottom=.2, wspace=.7)
    save(fig, "source_transport_pathways")

    simulation = pd.read_csv(ROOT/"analysis/simulation_timeseries.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    names = ["分布式长链", "均衡分支", "伸展式分支"] if cn else ["Distributed chain", "Balanced branches", "Elongated branches"]
    for row, removal in enumerate((0., .04)):
        for col, forcing in enumerate(("synchronous", "asynchronous")):
            ax = axes[row, col]
            for topology, color, name in zip(("chain", "balanced", "elongated"), COLORS, names, strict=True):
                s = simulation[(simulation.topology == topology) & (simulation.forcing == forcing) & simulation.removal_per_step.eq(removal)]
                ax.plot(s.step, s.doc_normalized, color=color, linewidth=1.5, label=name)
            title = ("同步来源" if forcing == "synchronous" else "异步来源") if cn else ("Synchronous sources" if forcing == "synchronous" else "Asynchronous sources")
            title += (" · 保守混合" if removal == 0 else " · 一阶去除") if cn else (" · conservative" if removal == 0 else " · first-order removal")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set(xlabel="模拟时间步" if cn else "Synthetic time step", ylabel="归一化 DOC 信号" if cn else "Normalized DOC signal", ylim=(0, 11), xlim=(14, 479))
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("保持来源总量不变，只改变河网组织" if cn else "Fixed source budgets, different network organization", fontsize=17, x=.065, ha="left")
    fig.text(.065, .02, "8 个相同流量来源、14 段等长河段；输入序列相同。延迟与去除为假设值，图为机制演示，不是现场拟合或真实停留时间。" if cn else
             "Eight equal-flow sources, 14 equal-length reaches, identical input series. Assumed delays/removal; illustrative mechanisms, not field-fitted residence times.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.97, top=.85, bottom=.12, wspace=.28, hspace=.4)
    save(fig, "controlled_network_simulation")
    (ROOT/f"figure_sources{suffix}.json").write_text(json.dumps({
        "analysis_files": [str(p) for p in (ROOT/"analysis").glob("*")],
        "representative_pair": str(example.pair_id), "selection": "near-complete same-day flow overlap, then record count; no DOC-fit selection",
        "language": "Chinese" if cn else "English", "simulation_is_observed_data": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
