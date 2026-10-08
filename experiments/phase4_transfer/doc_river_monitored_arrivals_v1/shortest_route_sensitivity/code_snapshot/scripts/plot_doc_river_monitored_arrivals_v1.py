"""Plot complete monitored river catchments and actually sampled DOC signals."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_monitored_arrivals_v1 import (
    PANEL,
    ROOT,
    TYPES,
    load_context,
    routed_network,
)
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from plot_doc_river_planform_v1 import boundary, segments
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from river_graph.analysis.river_monitored_arrivals import covered_geometry
from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import project_geometry, read_cached_lines

CACHE = Path("data/raw/river_planform_v1")
COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def real_coverage(ax, connection, row, gauges, context, shortest):
    paths, layout, hydro, _ = routed_network(row, context[-1], shortest)
    _, _, owner = covered_geometry(paths, hydro, gauges.comid.to_numpy(int), layout)
    with np.load(CACHE/"members_full"/f"comid_{int(row.comid)}.npz") as z:
        all_ids = z["comids"]
    lines = {c: project_geometry(g) for c, g in read_cached_lines(connection, all_ids).items()}
    raw = json.loads((CACHE/"basins"/f"comid_{int(row.comid)}.json").read_text())
    basin = project_geometry(unary_union([shape(f["geometry"]) for f in raw["features"]]))
    boundary(ax, basin)
    ax.add_collection(LineCollection(segments(lines.values()), colors="#B9C5C8", linewidth=.32,
                                      alpha=.7, rasterized=True))
    covered = paths.comids[owner >= 0]
    ax.add_collection(LineCollection(segments(lines[int(c)] for c in covered), colors=COLORS[int(row.cluster)],
                                      linewidth=.4, alpha=.8, rasterized=True))
    ax.add_collection(LineCollection(segments(lines[int(c)] for c in layout.mainstem_comids), colors="#273F48", linewidth=.9))
    locations = []
    nodes = context[1]
    for r in gauges.itertuples():
        n = nodes.loc[r.station]
        point = project_geometry(Point(n.dec_long_va, n.dec_lat_va))
        ax.scatter(point.x/1000, point.y/1000, color="#DAAB52", edgecolor="#4B5261", linewidth=.5, s=40, zorder=5)
        ax.annotate(str(int(r.source_order)+1), (point.x/1000, point.y/1000), xytext=(5, 3), textcoords="offset points", fontsize=8)
        locations.append({"target": row.station, "station": r.station, "source_order": r.source_order,
                          "projected_x_m": point.x, "projected_y_m": point.y, "kind": "source"})
    n = nodes.loc[row.station]
    point = project_geometry(Point(n.dec_long_va, n.dec_lat_va))
    ax.scatter(point.x/1000, point.y/1000, marker="s", color="#CC7155", edgecolor="white", linewidth=.6, s=40, zorder=6)
    locations.append({"target": row.station, "station": row.station, "source_order": -1,
                      "projected_x_m": point.x, "projected_y_m": point.y, "kind": "receiver"})
    x0, y0, x1, y1 = np.asarray(basin.bounds)/1000
    span = max(x1-x0, y1-y0)*1.14
    xm, ym = (x0+x1)/2, (y0+y1)/2
    ax.set(xlim=(xm-span/2, xm+span/2), ylim=(ym-span/2, ym+span/2), aspect="equal")
    ax.set_axis_off()
    power = 10**np.floor(np.log10(span*.22))
    scale = max(v*power for v in (1, 2, 5) if v*power <= span*.22)
    sx, sy = xm-.42*span, ym-.49*span
    ax.plot([sx, sx+scale], [sy, sy], color="#34464C", linewidth=1.5)
    ax.text(sx+scale/2, sy+.013*span, f"{scale:g} km", ha="center", fontsize=8)
    return locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    parser.add_argument("--shortest-routes", action="store_true")
    args = parser.parse_args()
    cn, suffix = args.chinese, "_cn" if args.chinese else ""
    if cn:
        p = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(p)
        plt.rcParams["font.family"] = FontProperties(fname=p).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    data, out = root/"analysis", root/"figures"
    out.mkdir(exist_ok=True)
    inventory = pd.read_csv(data/"network_inventory.csv", dtype=TYPES)
    stats = pd.read_csv(data/"receiver_signals.csv", dtype=TYPES)
    full = stats[stats.version.eq("full_monthly")]
    summary = pd.read_csv(data/"signal_summary.csv")
    names = ["细长多支流", "主干主导", "宽阔多支流"] if cn else ["Elongated", "Mainstem dominated", "Broad"]
    written = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            p = out/f"{name}{suffix}.{ext}"
            fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
            written.append(p)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.7))
    ax = axes[0, 0]
    total = [int((inventory.cluster.eq(c) & inventory.physical_receiver_representative).sum()) for c in (1, 2, 3)]
    included = [int(full.cluster.eq(c).sum()) for c in (1, 2, 3)]
    pos = np.arange(3)
    ax.bar(pos, total, color="#DEE5E5", width=.6, label="全部不同河网" if cn else "All unique networks")
    ax.bar(pos, included, color=[COLORS[c] for c in (1, 2, 3)], width=.6, label="上游—出口共同观测" if cn else "Common source/outlet observations")
    for i, (n, m) in enumerate(zip(total, included, strict=True)):
        ax.text(i, n+3, f"{m}/{n}", ha="center", fontsize=10)
    ax.set(xticks=pos, xticklabels=names, ylim=(0, max(total)*1.23))
    ax.set_ylabel("不同出口河网数量" if cn else "Unique receiving networks")
    ax.set_title("a  整张河网的观测覆盖" if cn else "a  Whole-network observation coverage", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[0, 1]
    rng = np.random.default_rng(42)
    for c in (1, 2, 3):
        f = full[full.cluster.eq(c)]
        ax.scatter(c+rng.uniform(-.12, .12, len(f)), f.covered_area_fraction*100, color=COLORS[c], s=28, alpha=.7)
        if len(f):
            ax.plot([c-.19, c+.19], [f.covered_area_fraction.median()*100]*2, color="#273F48", linewidth=1.7)
    ax.axhline(80, color="#BBC5C7", linestyle="--")
    ax.set(xticks=[1, 2, 3], xticklabels=[f"{n}\nn={included[i]}" for i, n in enumerate(names)], ylim=(0, 104))
    ax.set_ylabel("上游监测覆盖的汇水面积 (%)" if cn else "Catchment area represented by gauges (%)")
    ax.set_title("b  独立上游输入覆盖程度" if cn else "b  Non-overlapping upstream inputs", loc="left")
    ax = axes[1, 0]
    labels = {"source_coherence": "支流波动同步程度" if cn else "Source coherence",
        "mixture_buffer_fraction": "混合波动减小比例" if cn else "Mixture variance reduction",
        "outlet_mix_correlation": "上游混合—出口相关" if cn else "Mix / outlet correlation",
        "real_minus_shuffle_correlation": "同期关联超过打乱的部分" if cn else "Real minus calendar shuffle"}
    focal = summary[summary.group.eq("all") & summary.version.eq("full_monthly")].set_index("metric")
    for i, (metric, label) in enumerate(labels.items()):
        r = focal.loc[metric]
        if np.isfinite(r.ci_low+r.ci_high):
            ax.plot([r.ci_low, r.ci_high], [i, i], color="#607E88", linewidth=2)
        ax.scatter(r.estimate, i, color="#C5814A", s=45, zorder=3)
    ax.axvline(0, color="#BBC5C7", linestyle="--")
    ax.set(yticks=np.arange(4), yticklabels=list(labels.values()), ylim=(3.6, -.6))
    ax.set_xlabel("相关系数 / 方差比例（各量分别定义）" if cn else "Correlation / variance fraction (separate metrics)")
    ax.set_title("c  实际 DOC 波动摘要" if cn else "c  Observed DOC fluctuation summaries", loc="left")
    ax = axes[1, 1]
    for c in (1, 2, 3):
        f = full[full.cluster.eq(c)]
        ax.scatter(f.full_path_cv, f.outlet_mix_log_sd_ratio, color=COLORS[c], s=28, alpha=.75, label=names[c-1])
    ax.axhline(0, color="#BBC5C7", linestyle="--")
    ax.set_xlabel("整张河网路径差异（SD / 均值）" if cn else "Complete-network path SD / mean")
    ax.set_ylabel("log(出口波动 / 上游混合波动)" if cn else "log(outlet SD / source-mixture SD)")
    ax.set_title("d  形态机制与实测波动" if cn else "d  Complete geometry and observed response", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("整张河网形态与实际 DOC 信号：监测范围及共同月份" if cn else
                 "Whole river form and observed DOC: coverage and common sampling months", x=.035, ha="left", fontsize=17)
    note = (f"每个不同出口河网一票；{len(full)} 个出口，{full.component.nunique()} 个共享流域系统。点线为均值和系统整组重采样 5,000 次的 95% 区间。\n"
        "汇水面积只作为混合权重代理。分析去除共同季节与年份趋势；散点描述关系，未估计事件峰值或 DOC 去除率。") if cn else (
        f"One vote per distinct receiving network; {len(full)} receivers in {full.component.nunique()} overlapping-catchment systems. Mean and 95% whole-system bootstrap intervals, 5,000 draws.\n"
        "Catchment area is a flow-share proxy. Common calendar/year effects removed; scatter is descriptive, not measured event attenuation or DOC removal.")
    fig.text(.035, .028, note, fontsize=9)
    fig.subplots_adjust(top=.9, bottom=.15, left=.17, hspace=.45, wspace=.5)
    save(fig, "whole_network_observed_overlap")

    context = load_context()
    panel = pd.read_csv(PANEL, dtype=TYPES).set_index("station", drop=False)
    reps = pd.read_csv(data/"representatives.csv", dtype=TYPES).sort_values("cluster")
    gauges = pd.read_csv(data/"candidate_gauges.csv", dtype=TYPES)
    series = pd.read_parquet(data/"monthly_series.parquet")
    selected = pd.read_parquet(data/"selected_activities.parquet")
    fig, axes = plt.subplots(2, len(reps), figsize=(4.8*len(reps), 9.5), squeeze=False,
                             gridspec_kw={"height_ratios": [1.5, 1.]})
    mapped = []
    examples = []
    with sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True) as connection:
        for col, r in enumerate(reps.itertuples()):
            g = gauges[gauges.target.eq(r.station) & gauges.selected].sort_values("source_order")
            mapped.extend(real_coverage(axes[0, col], connection, panel.loc[r.station], g, context, args.shortest_routes))
            axes[0, col].set_title(f"{chr(97+col)}  {names[r.cluster-1]}\n{r.station} · "
                +(f"面积覆盖 {r.covered_area_fraction:.0%}" if cn else f"area coverage {r.covered_area_fraction:.0%}"), loc="left", fontsize=11)
            f = series[series.target.eq(r.station)].sort_values("date")
            spans = [(int(((f.date.dt.year >= year) & (f.date.dt.year <= year+3)).sum()), -int(year))
                     for year in f.date.dt.year.unique()]
            _, neg_year = max(spans)
            year = -neg_year
            f = f[f.date.dt.year.between(year, year+3)]
            examples.append({"target": r.station, "first_year": year, "last_year": year+3,
                             "selection": "maximum common months in a four-calendar-year window; no DOC values"})
            ax = axes[1, col]
            ax.scatter(f.date, f.doc_mixture, facecolors="none", edgecolors="#257F88", s=28,
                       label="上游面积加权混合" if cn else "Source area-weighted mixture")
            ax.scatter(f.date, f.doc_receiver, color="#C5814A", s=21,
                       label="实测出口 DOC" if cn else "Observed receiver DOC")
            actual = selected[selected.target.eq(r.station) & selected.source_order.eq(-1)
                              & selected.month.isin(f.date)]
            ax.set_title(f"{chr(97+len(reps)+col)}  "+("实际共同观测月份" if cn else "Actual common sampling months")
                +"\n"+(f"{len(g)} 条独立输入；{len(f)} 个共同月；采样跨度中位 {actual.sample_span_days.median():.0f} 天" if cn else
                f"{len(g)} inputs; {len(f)} months; median sample span {actual.sample_span_days.median():.0f} days"),
                loc="left", fontsize=10, pad=12)
            ax.set_ylabel("DOC (mg/L)")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlim(pd.Timestamp(year, 1, 1), pd.Timestamp(year+4, 1, 1))
            ax.grid(axis="y", color="#EDF0F0")
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.12), borderaxespad=0)
    pd.DataFrame(mapped).to_csv(out/"representative_gauge_locations.csv", index=False)
    pd.DataFrame(examples).to_csv(out/"representative_calendar_windows.csv", index=False)
    fig.suptitle("真实河网中监测到的支流与出口 DOC" if cn else "Actually monitored tributaries and receiving DOC", x=.035, ha="left", fontsize=17)
    note = ("彩色河道：选定监测站上游的非重叠汇水区；灰色河道：未覆盖；金色点：上游监测站；橙色方点：出口监测站。各图比例尺独立。\n"
        "例子按面积覆盖及共同月份选择；下排只画真实月观测点，缺测不连接、不插值。混合权重为面积代理，非逐时流量。") if cn else (
        "Colored channels: disjoint upstream gauged catchments; grey: uncovered; gold: source gauges; orange square: receiver gauge. Independent physical scale bars.\n"
        "Examples selected by coverage and calendar availability. Bottom panels show actual monthly points; no gap filling, continuous event curve or measured-flow weights.")
    fig.text(.035, .025, note, fontsize=9)
    fig.subplots_adjust(top=.87, bottom=.2, hspace=.48, wspace=.3)
    save(fig, "real_monitored_networks_doc")

    cases = pd.read_csv(data/"weekly_case_stations.csv", dtype=TYPES)
    weeks = pd.read_parquet(data/"weekly_case_selected_activities.parquet")
    weekly = pd.read_parquet(data/"weekly_case_series.parquet")
    ledger = pd.read_csv(data/"weekly_case_years.csv", dtype=TYPES)
    chosen_year = int(ledger.sort_values(["n_joint_days", "year"], ascending=[False, True]).iloc[0].year)
    f = weeks[weeks.date.dt.year.eq(chosen_year)].pivot(index="date", columns="site_no", values="doc")
    full_case = weekly[weekly.version.eq("metadata_selected_activity") & weekly.adjustment.eq("calendar_year")]
    short = weekly[weekly.version.eq("metadata_selected_activity") & weekly.adjustment.eq("within_month")]
    fig = plt.figure(figsize=(11.2, 8.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.])
    ax = fig.add_subplot(grid[0, :])
    source_names = {"401707105395000": "Icy Brook", "401723105400000": "Andrews Creek"}
    source_color = ["#257F88", "#66749F"]
    receiver = cases.loc[cases.source_order.eq(-1), "station"].iloc[0]
    times = mdates.date2num(f.index.to_pydatetime())
    linked = np.diff(times) <= 14

    def actual_trace(values, label, color, dashed=False):
        values = np.asarray(values)
        ax.scatter(f.index, values, color=color, s=21 if not dashed else 12, label=label, zorder=3)
        pieces = [np.array([[times[i], values[i]], [times[i+1], values[i+1]]])
                  for i in np.flatnonzero(linked)]
        ax.add_collection(LineCollection(pieces, colors=color, linewidth=1., alpha=.65,
                                         linestyles="--" if dashed else "solid"))

    for i, r in enumerate(cases[cases.source_order.ge(0)].sort_values("source_order").itertuples()):
        actual_trace(f[r.station], source_names.get(r.station, r.station), source_color[i % len(source_color)])
    mix = full_case.set_index("date").doc_mixture.reindex(f.index)
    actual_trace(mix, "面积加权混合" if cn else "Area-share mixture", "#7E8C90", True)
    actual_trace(f[receiver], "The Loch 湖泊出口" if cn else "The Loch outlet", "#C5814A")
    ax.set_title(f"a  {chosen_year} · "+(f"{len(f)} 组同一天的真实采样" if cn else f"{len(f)} actual same-day sample sets"), loc="left")
    ax.set_ylabel("DOC (mg/L)")
    ax.set_xlim(pd.Timestamp(chosen_year, 1, 1), pd.Timestamp(chosen_year+1, 1, 1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m" if cn else "%b"))
    ax.grid(axis="y", color="#EDF0F0")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.19), borderaxespad=0)
    ax = fig.add_subplot(grid[1, 0])
    ax.scatter(short.mixture_anomaly, short.receiver_anomaly, s=17, color="#257F88", alpha=.65)
    ax.axhline(0, color="#BBC5C7", linewidth=.8)
    ax.axvline(0, color="#BBC5C7", linewidth=.8)
    ax.set_xlabel("上游混合：相对该月均值的变化 (mg/L)" if cn else "Source mixture: within-month departure (mg/L)")
    ax.set_ylabel("出口：相对该月均值的变化 (mg/L)" if cn else "Outlet: within-month departure (mg/L)")
    signals = pd.read_csv(data/"weekly_case_signals.csv", dtype=TYPES)
    r = signals[signals.version.eq("metadata_selected_activity") & signals.adjustment.eq("within_month")].iloc[0]
    ax.set_title("b  "+(f"月内波动 · {len(short)} 个共同日，相关 {r.outlet_mix_correlation:.2f}" if cn else
                 f"Within-month fluctuations · n={len(short)}, r={r.outlet_mix_correlation:.2f}"), loc="left", fontsize=10)
    ax = fig.add_subplot(grid[1, 1])
    reference = r.source_reference_variance
    values = [100., r.mixture_variance/reference*100, np.mean(short.receiver_anomaly**2)/reference*100]
    ax.bar(np.arange(3), values, color=["#B9C5C8", "#257F88", "#C5814A"], width=.6)
    for i, v in enumerate(values):
        ax.text(i, v+max(values)*.025, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set(xticks=np.arange(3), xticklabels=["上游各支流\n波动参照", "计算混合", "实测出口"] if cn else
           ["Source variance\nreference", "Calculated mix", "Observed outlet"], ylim=(0, max(values)*1.18))
    ax.set_ylabel("相对上游参照的方差 (%)" if cn else "Variance / source reference (%)")
    ax.set_title("c  月内波动的传递" if cn else "c  Transmission of within-month fluctuations", loc="left", fontsize=10)
    fig.suptitle("Loch Vale：两条支流与湖泊出口的实测 DOC" if cn else
                 "Loch Vale: observed tributary and lake-outlet DOC", x=.045, ha="left", fontsize=17)
    note = (f"全案例 {len(full_case)} 组同日样本；同日不等于同一时刻。示例年份只按采样数量选择；线仅连接相隔不超过 14 天的实际点。\n"
        "月内分析仅纳入至少 3 个共同采样日的月份；一个湖泊影响的河网，不代表形态类型的总体效应，也未测量 DOC 去除率。") if cn else (
        f"Complete case: {len(full_case)} same-day sets; not necessarily simultaneous. Year selected by sample count; lines join actual points only at <=14-day gaps.\n"
        "Within-month analysis requires >=3 common days/month. One lake-influenced network, not replication of form classes or measured DOC removal.")
    fig.text(.045, .025, note, fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.16, left=.09, right=.97, hspace=.42, wspace=.36)
    save(fig, "loch_vale_actual_weekly_doc")
    window = out/"weekly_case_calendar_window.csv"
    pd.DataFrame([{"target": receiver, "year": chosen_year, "n_joint_days": len(f),
                   "selection": "maximum number of same-day sample sets, then earliest year; no DOC values"}]).to_csv(window, index=False)
    inputs = [PANEL, root/"analysis_sources.json", *data.glob("*.csv"), data/"monthly_series.parquet", data/"selected_activities.parquet",
              data/"weekly_case_series.parquet", data/"weekly_case_selected_activities.parquet",
              Path(__file__), Path("scripts/plot_doc_river_planform_v1.py"), CACHE/"flowlines.sqlite"]
    inputs.extend(CACHE/"basins"/f"comid_{int(r.comid)}.json" for r in reps.itertuples())
    outputs = [*written, out/"representative_gauge_locations.csv", out/"representative_calendar_windows.csv", window]
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in inputs},
        "output_hashes": {str(p): sha256_file(p) for p in outputs},
        "scope": "actual mapped channels and sampled station-months; no synthetic DOC event curves"}, indent=2)+"\n")
    print("\n".join(map(str, written)))


if __name__ == "__main__":
    main()
