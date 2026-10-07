"""Real confluence maps and conservative DOC placement mechanisms, EN/CN."""
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
from matplotlib.lines import Line2D
from plot_doc_monitored_river_footprint_v1 import mapped_corridor
from plot_doc_river_planform_v1 import draw_network, segments

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_storage_placement_v1")
PLANFORM = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/classes.csv")
CACHE = Path("data/raw/river_planform_v1")
COLORS = {"early_branch": "#257F88", "late_branch": "#C5814A", "shared_trunk": "#66749F", "translation": "#869499"}
ORDER = ("early_branch", "late_branch", "shared_trunk")
SHAPE_COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 210, "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    f = pd.read_parquet(a/"scenario_metrics.parquet")
    s = pd.read_csv(a/"cohort_summary.csv", dtype={"scope": str})
    differences = pd.read_parquet(a/"placement_contrasts.parquet")
    reps = pd.read_csv(a/"representatives.csv", dtype={"case_id": str, "station": str}).merge(
        pd.read_csv(PLANFORM, dtype={"station": str}).drop(columns="centroid_distance", errors="ignore"),
        on=["station", "comid", "cluster"], suffixes=("", "_shape"), validate="many_to_one")
    corridors = pd.read_csv(a/"corridor_reaches.csv", dtype={"station": str})
    traces = pd.read_parquet(a/"representative_responses.parquet")
    names = dict(zip(ORDER, ("较早支流", "较晚支流", "共同主干") if cn else ("Early branch", "Late branch", "Shared trunk")))
    forms = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated / tributary-rich", "Mainstem dominated / sparse", "Broad / tributary-rich")
    inputs = list(a.iterdir())+[PLANFORM, CACHE/"flowlines.sqlite", Path(__file__),
        Path("scripts/plot_doc_river_planform_v1.py"), Path("scripts/plot_doc_monitored_river_footprint_v1.py"),
        Path("src/river_graph/analysis/river_monitored_footprint.py"), Path("src/river_graph/topology/river_planform.py")]
    outputs, mapped = [], []
    suffix = "_cn" if cn else ""

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    def geometry_map(ax, row):
        draw_network(ax, connection, row, "#B5C6C6")
        reaches = corridors[corridors.example_group.eq(row.example_group)]
        parts, gap, digest = mapped_corridor(connection, reaches)
        early_a = row.branch_a_km <= row.branch_b_km
        arm_names = {"branch_a": "early_branch" if early_a else "late_branch",
                     "branch_b": "late_branch" if early_a else "early_branch", "common": "shared_trunk"}
        for segment, r in reaches.groupby("segment"):
            color = COLORS[arm_names[segment]]
            ax.add_collection(LineCollection(segments(parts[int(cid)] for cid in r.comid), colors=color,
                                            linewidths=2., zorder=6))
            path = r.sort_values("sequence")
            point = parts[int(path.comid.iloc[-1 if segment == "common" else 0])]
            xy = np.asarray(point.coords)[-1 if segment == "common" else 0]/1000
            ax.scatter(*xy, color=color, s=29, edgecolor="white", linewidth=.6, zorder=7,
                       marker="D" if segment == "common" else "o")
        inputs.extend([CACHE/"members_full"/f"comid_{int(row.comid)}.npz", CACHE/"basins"/f"comid_{int(row.comid)}.json"])
        mapped.append({"example_group": row.example_group, "station": row.station,
                       "geometry_sha256": digest, "maximum_connectivity_gap_m": gap})

    def selected_traces(row):
        return traces[traces.case_id.eq(row.case_id) & traces.cohort.eq("whole_confluence") &
            ((traces.strength_match.eq("variance_matched") & traces.fraction.eq(.5)) | traces.strength_match.eq("baseline"))]

    def response_axes(ax, ymax, xmax):
        ax.set(xlim=(-.3, xmax), ylim=(0, ymax), xlabel="相对到达时间（均值=1）" if cn else "Relative arrival time (mean = 1)",
               ylabel="出口 DOC 脉冲 / 输入峰值" if cn else "Outlet DOC pulse / input peak")
        ax.axvline(1, color="#D4DCDD", linestyle=":", linewidth=.9, zorder=0)

    connection = sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.3))
    shapes = reps[reps.example_group.le(3)].sort_values("example_group")
    selected = f[f.case_id.isin(shapes.case_id) & f.cohort.eq("whole_confluence") & f.flow_rule.eq("area_proxy") & f.input_sd.eq(.15)]
    xend = max(2.1, selected.t90.max()+.5)
    top = min(1.04, selected.pulse_peak.max()*1.15)
    for col, r in enumerate(shapes.itertuples()):
        geometry_map(axes[0, col], r)
        n = f[(f.cohort.eq("whole_confluence")) & f.cluster.eq(r.cluster) & f.strength_match.eq("baseline") &
              f.flow_rule.eq("area_proxy") & f.input_sd.eq(.15)].case_id.nunique()
        axes[0, col].set_title(f"{'abc'[col]}  {forms[int(r.cluster)-1]}\n{r.station} · n={n}", loc="left", pad=10, fontsize=11)
        share = (f"选定支流覆盖面积 {r.selected_pair_area_share:.0%}" if cn else f"Selected pair area share {r.selected_pair_area_share:.0%}")
        axes[0, col].text(.02, -.04, share, transform=axes[0, col].transAxes, fontsize=9)
        tr = selected_traces(r)
        for place in ("translation", *ORDER):
            t = tr[tr.placement.eq(place)]
            axes[1, col].plot(t.relative_time, t.outlet_anomaly, color=COLORS[place], linewidth=1.9,
                             linestyle="--" if place == "translation" else "-")
        response_axes(axes[1, col], top, xend)
        axes[1, col].set_title(f"{'def'[col]}  "+("相同额外分散，位置不同" if cn else "Equal added variance, different placement"), loc="left", pad=12, fontsize=11)
    handles = [Line2D([], [], color=COLORS[k], linewidth=2, linestyle="--" if k == "translation" else "-",
        label=("纯传输对照" if cn else "Translation control") if k == "translation" else names[k]) for k in ("translation", *ORDER)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .92), ncol=4, frameon=False)
    fig.suptitle("真实河网：同样的滞留强度放在哪里？" if cn else "Real river geometry: where does matched storage act?",
                 x=.04, ha="left", fontsize=17)
    note = ("真实 NHDPlus 河道；地图各用独立比例尺。两条代表支流输入相同 DOC 脉冲；灰色为完整河网，彩色为选定传输路径。\n"
            "滞留位置是受控情景，并非搬移实测湖库；路径平均时间、总输入与新增方差不变。相对时间不是实测天数。" if cn else
            "Actual NHDPlus flowlines; maps use independent physical scales. Identical forcing at two real representative branch locations.\n"
            "Controlled storage locations, not relocated mapped waterbodies. Fixed path means, unit gain and added variance; relative time is not measured days.")
    fig.text(.04, .025, note, fontsize=9, color="#56696F")
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.16, hspace=.33, wspace=.26)
    save(fig, "real_forms_storage_placement")

    primary = f.query("flow_rule == 'area_proxy' and input_sd == .15 and fraction == .5 and cohort == 'whole_confluence'")
    fig, axes = plt.subplots(2, 3, figsize=(14.9, 10.0))
    ax = axes[0, 0]
    for k, match in enumerate(("variance_matched", "mean_budget_matched")):
        for pos, place in enumerate(ORDER):
            r = s.query("scope == 'all' and cohort == 'whole_confluence' and flow_rule == 'area_proxy' and input_sd == .15 and fraction == .5 and metric == 'peak_reduction_pct'")
            r = r[r.strength_match.eq(match) & r.placement.eq(place)].iloc[0]
            x = pos+(k-.5)*.16
            ax.plot([x, x], [r.ci_low, r.ci_high], color=COLORS[place], linewidth=1.5)
            ax.scatter(x, r["mean"], color=COLORS[place] if k == 0 else "white", edgecolor=COLORS[place], s=42, zorder=3)
    ax.axhline(0, color="#C1CCCE", linewidth=.7)
    ax.set(xticks=range(3), xticklabels=[names[k] for k in ORDER], ylabel="平均峰值降低（%）" if cn else "Mean peak reduction (%)")
    ax.set_title("a  "+("两种强度匹配方式" if cn else "Two ways to match strength"), loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#566F77", markeredgecolor="#566F77", label="新增方差相同" if cn else "Equal added variance"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#566F77", label="分配时间相同" if cn else "Equal allocated time")], frameon=False, fontsize=8)
    ax = axes[0, 1]
    for c, offset in ((1, -.16), (2, 0), (3, .16)):
        for pos, place in enumerate(ORDER):
            r = s[(s.scope.eq(str(c))) & s.cohort.eq("whole_confluence") & s.flow_rule.eq("area_proxy") & s.input_sd.eq(.15) &
                s.fraction.eq(.5) & s.strength_match.eq("variance_matched") & s.metric.eq("peak_reduction_pct") & s.placement.eq(place)].iloc[0]
            x = pos+offset
            ax.plot([x, x], [r.ci_low, r.ci_high], color=SHAPE_COLORS[c], linewidth=1.2)
            ax.scatter(x, r["mean"], color=SHAPE_COLORS[c], s=28, marker=("o", "s", "^")[c-1], zorder=3)
    ax.axhline(0, color="#C1CCCE", linewidth=.7)
    ax.set(xticks=range(3), xticklabels=[names[k] for k in ORDER], ylabel="平均峰值降低（%）" if cn else "Mean peak reduction (%)")
    ax.set_title("b  "+("三类河网内部比较" if cn else "Within the three river forms"), loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker=("o", "s", "^")[c-1], color="none", markerfacecolor=SHAPE_COLORS[c], markeredgecolor=SHAPE_COLORS[c], label=forms[c-1]) for c in (1, 2, 3)], frameon=False, fontsize=7.5)
    ax = axes[0, 2]
    d = differences.query("cohort == 'whole_confluence' and flow_rule == 'area_proxy' and input_sd == .15 and fraction == .5 and strength_match == 'variance_matched' and contrast == 'late_branch_minus_early_branch'")
    edges = np.linspace(min(-1, d.peak_reduction_pct.min()), max(1, d.peak_reduction_pct.max()), 24)
    ax.hist(d.peak_reduction_pct, bins=edges, color="#728C95", edgecolor="white", linewidth=.5)
    ax.axvline(0, color="#344D56", linewidth=1.)
    ax.set(xlabel="较晚 − 较早：峰值降低差（百分点）" if cn else "Late − early: reduction difference (pp)",
        ylabel="真实汇流河网数" if cn else "Real confluence instances")
    ax.set_title("c  "+("同一河网的位置差异" if cn else "Paired placement differences"), loc="left", pad=12)
    ax.text(.96, .94, f"n={len(d)}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax = axes[1, 0]
    for place in ORDER:
        g = primary[primary.strength_match.eq("variance_matched") & primary.placement.eq(place)]
        ax.scatter(-g.log_envelope_change, g.log_alignment_change, color=COLORS[place], s=14, alpha=.55, label=names[place], linewidths=0)
    limit = max(-primary.log_envelope_change.min(), primary.log_alignment_change.max())*1.04
    ax.plot([0, limit], [0, limit], linestyle="--", color="#6F7C81", linewidth=1.)
    ax.set(xlim=(-.025, limit), ylim=(min(-.025, primary.log_alignment_change.min()*1.05), limit),
        xlabel="单支流峰值削弱：−log(E/E₀)" if cn else "Component attenuation: −log(E/E₀)",
        ylabel="叠加变化：log(A/A₀)" if cn else "Overlap change: log(A/A₀)")
    ax.set_title("d  "+("峰值增加需要更强叠加" if cn else "Peak increase requires greater overlap"), loc="left", pad=12)
    ax.text(.04, .94, "虚线上方：总峰值增加" if cn else "Above dashed line: combined peak increases", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1, 1]
    for k, flow in enumerate(("area_proxy", "equal_flow")):
        for pos, place in enumerate(ORDER):
            r = s[(s.scope.eq("all")) & s.cohort.eq("whole_confluence") & s.flow_rule.eq(flow) & s.input_sd.eq(.15) & s.fraction.eq(.5) &
                s.strength_match.eq("variance_matched") & s.metric.eq("peak_reduction_pct") & s.placement.eq(place)].iloc[0]
            x = pos+(k-.5)*.16
            ax.plot([x, x], [r.ci_low, r.ci_high], color=COLORS[place], linewidth=1.2)
            ax.scatter(x, r["mean"], color=COLORS[place] if k == 0 else "white", edgecolor=COLORS[place], marker="o" if k == 0 else "s", s=35)
    ax.set(xticks=range(3), xticklabels=[names[k] for k in ORDER], ylabel="平均峰值降低（%）" if cn else "Mean peak reduction (%)")
    ax.axhline(0, color="#C1CCCE", linewidth=.7)
    ax.set_title("e  "+("支流水量权重的影响" if cn else "Sensitivity to branch flow shares"), loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#566F77", markeredgecolor="#566F77", label="面积代理" if cn else "Area proxy"),
        Line2D([], [], marker="s", color="none", markerfacecolor="white", markeredgecolor="#566F77", label="两支流等量" if cn else "Equal branch flows")], frameon=False, fontsize=8)
    ax = axes[1, 2]
    for k, cohort in enumerate(("whole_confluence", "monitored_footprint")):
        for pos, place in enumerate(ORDER):
            r = s[(s.scope.eq("all")) & s.cohort.eq(cohort) & s.flow_rule.eq("area_proxy") & s.input_sd.eq(.15) & s.fraction.eq(.5) &
                s.strength_match.eq("variance_matched") & s.metric.eq("peak_reduction_pct") & s.placement.eq(place)].iloc[0]
            x = pos+(k-.5)*.16
            ax.plot([x, x], [r.ci_low, r.ci_high], color=COLORS[place], linewidth=1.2)
            ax.scatter(x, r["mean"], color=COLORS[place] if k == 0 else "white", edgecolor=COLORS[place], marker="o" if k == 0 else "D", s=35)
    ax.set(xticks=range(3), xticklabels=[names[k] for k in ORDER], ylabel="平均峰值降低（%）" if cn else "Mean peak reduction (%)")
    ax.axhline(0, color="#C1CCCE", linewidth=.7)
    ax.set_title("f  "+("监测支流路径复算" if cn else "Replication on monitored footprints"), loc="left", pad=12)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#566F77", markeredgecolor="#566F77", label="完整河网选定汇流" if cn else "Whole-network confluences"),
        Line2D([], [], marker="D", color="none", markerfacecolor="white", markeredgecolor="#566F77", label="22 个出口等权" if cn else "22 receivers, equal weight")], frameon=False, fontsize=7.5)
    # Comparable effect panels use exactly the same vertical scale.
    limits = [axes[r, c].get_ylim() for r, c in ((0, 0), (0, 1), (1, 1), (1, 2))]
    common_limits = (min(v[0] for v in limits), max(v[1] for v in limits))
    for r, c in ((0, 0), (0, 1), (1, 1), (1, 2)):
        axes[r, c].set_ylim(common_limits)
    fig.suptitle("滞留位置怎样改变 DOC 峰值？" if cn else "How storage placement changes DOC pulse peaks", x=.055, ha="left", fontsize=17)
    fig.text(.055, .025, ("中等脉冲 SD=0.15；可行强度的 50%。完整河网为 295 个汇流结构、按站点等权；59 条监测路径先在出口内平均。\n"
        "区间为 5,000 次 HUC4 / 监测系统块自助法；不是实测峰值的不确定性。峰值 = 单支流峰值包络 E × 波形叠加比例 A。" if cn else
        "Middle pulse SD=0.15; 50% of feasible strength. 295 confluences, instance equal; 59 monitored footprints averaged within 22 receivers.\n"
        "Intervals: 5,000 HUC4 / monitoring-system block resamples of geometry means, not uncertainty of field peaks. Peak = component envelope E × overlap ratio A."), fontsize=9, color="#56696F")
    fig.subplots_adjust(left=.065, right=.985, top=.89, bottom=.16, hspace=.46, wspace=.31)
    save(fig, "placement_effects_and_overlap")

    # Largest geometry-defined capacity × arrival dispersion, selected before results.
    extra = next(reps[reps.example_group.eq(4)].itertuples())
    tr = selected_traces(extra)
    rows = f[f.case_id.eq(extra.case_id) & f.cohort.eq("whole_confluence") & f.flow_rule.eq("area_proxy") & f.input_sd.eq(.15)]
    fig = plt.figure(figsize=(14.4, 8.7))
    gs = fig.add_gridspec(2, 3)
    map_ax = fig.add_subplot(gs[:, 0])
    geometry_map(map_ax, extra)
    map_ax.set_title("a  "+("几何条件选定的真实案例" if cn else "Geometry-selected real example")+f"\n{extra.station}", loc="left", fontsize=11)
    title = (f"相对到达差 {abs(extra.branch_a_km-extra.branch_b_km)/extra.mean_total_km:.2f}\n支流面积权重 {extra.weight_a:.0%} / {1-extra.weight_a:.0%}" if cn else
             f"Relative arrival gap {abs(extra.branch_a_km-extra.branch_b_km)/extra.mean_total_km:.2f}\nBranch area shares {extra.weight_a:.0%} / {1-extra.weight_a:.0%}")
    map_ax.text(.02, -.14, title, transform=map_ax.transAxes, fontsize=10)
    ymax = rows.pulse_peak.max()*1.17
    xmax = max(2.2, rows.t90.max()+.5)
    for pos, place in enumerate(("translation", *ORDER)):
        ax = fig.add_subplot(gs[pos//2, 1+pos%2])
        t = tr[tr.placement.eq(place)]
        if place != "translation":
            base = tr[tr.placement.eq("translation")]
            ax.plot(base.relative_time, base.outlet_anomaly, color="#AAB6B9", linewidth=1.3, linestyle=":")
        ax.plot(t.relative_time, t.weighted_a, color=COLORS["early_branch"] if extra.branch_a_km <= extra.branch_b_km else COLORS["late_branch"], linestyle="--", linewidth=1.3)
        ax.plot(t.relative_time, t.weighted_b, color=COLORS["early_branch"] if extra.branch_b_km < extra.branch_a_km else COLORS["late_branch"], linestyle="--", linewidth=1.3)
        ax.plot(t.relative_time, t.outlet_anomaly, color=COLORS[place], linewidth=2.)
        response_axes(ax, ymax, xmax)
        title = ("纯传输对照" if cn else "Translation control") if place == "translation" else names[place]
        rr = rows[rows.placement.eq(place) & ((rows.strength_match.eq("variance_matched") & rows.fraction.eq(.5)) | rows.strength_match.eq("baseline"))].iloc[0]
        change = -rr.peak_reduction_pct
        ax.set_title(f"{'bcde'[pos]}  {title}"+(f" · {'峰值' if cn else 'peak'} {change:+.1f}%" if place != "translation" else ""), loc="left", pad=10, fontsize=10)
    fig.suptitle("单支流削平与汇流叠加：同一结构的两种作用" if cn else "Component attenuation and overlap in one real confluence", x=.055, ha="left", fontsize=17)
    fig.legend(handles=[Line2D([], [], color=COLORS[k], linestyle="--", linewidth=1.5, label=names[k]) for k in ORDER[:2]]+
        [Line2D([], [], color="#3A5360", linewidth=2., label="出口叠加" if cn else "Combined outlet")],
        loc="upper center", bbox_to_anchor=(.66, .93), ncol=3, frameon=False, fontsize=9)
    fig.text(.055, .03, ("虚线：两条支流的加权响应；实线：出口叠加响应；灰色点线：纯传输对照。三种位置平均到达时间与新增方差相同。\n"
        "案例由到达差与可行干预量选定；未按 DOC 结果或峰值增幅挑选。受控机制情景，不是实测浓度曲线。" if cn else
        "Dashed: weighted individual branches; solid: combined outlet response; grey dotted: translation control. Fixed means and added variance.\n"
        "Example selected from arrival dispersion × feasible capacity before simulated outcomes; not a measured concentration time series."), fontsize=9, color="#56696F")
    fig.subplots_adjust(left=.055, right=.98, top=.87, bottom=.17, hspace=.37, wspace=.36)
    save(fig, "real_confluence_component_decomposition")
    connection.close()
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in set(inputs)},
        "outputs": {str(p): sha256_file(p) for p in outputs}, "mapped_examples": mapped,
        "scope": "Conservative geometry scenarios; no measured DOC peak or residence-time inference"}, indent=2)+"\n")
    print(f"Wrote {len(outputs)} EN/CN scientific files; checked real corridor geometry.")


if __name__ == "__main__":
    main()
