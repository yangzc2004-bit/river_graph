"""Scientific figures of observed tributary coordination and river form."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_observed_synchrony_v1 import ROOT, TYPES
from matplotlib.font_manager import FontProperties, fontManager

from river_graph.experiments.provenance import sha256_file

COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    parser.add_argument("--shortest-routes", action="store_true")
    args = parser.parse_args()
    cn, suffix = args.chinese, "_cn" if args.chinese else ""
    if cn:
        fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        plt.rcParams["font.family"] = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf").get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    source, out = root/"analysis", root/"figures"
    out.mkdir(exist_ok=True)
    network = pd.read_csv(source/"network_metrics.csv", dtype=TYPES)
    summary = pd.read_csv(source/"signal_summary.csv")
    periods = pd.read_csv(source/"within_network_periods.csv", dtype=TYPES)
    case = pd.read_csv(source/"dense_case_metrics.csv", dtype=TYPES)
    annual = pd.read_csv(source/"dense_case_years.csv", dtype=TYPES)
    trace = pd.read_parquet(source/"dense_case_series.parquet")
    primary = network.query("version == 'monthly' and adjustment == 'calendar_year' and excursion_quantile == .75")
    names = {1: "细长多支流" if cn else "Elongated", 2: "主干主导" if cn else "Mainstem dominated",
             3: "宽阔多支流" if cn else "Broad"}
    written = []
    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0))
    ax = axes[0, 0]
    for c in (1, 2, 3):
        for high in (False, True):
            g = primary[primary.cluster.eq(c) & primary.covered_area_fraction.ge(.8).eq(high)]
            ax.scatter(g.source_coherence, np.exp(g.outlet_sync_log_sd_ratio),
                marker="s" if high else "o", s=42, facecolor=COLORS[c] if high else "white",
                edgecolor=COLORS[c], linewidth=1., label=names[c] if not high else None)
    ax.axhline(1, color="#93A1A6", linestyle="--", linewidth=1)
    ax.set(xlabel="支流波动的同步程度" if cn else "Observed tributary coordination",
        ylabel="出口波动 / 完全同步支流参考 (SD)" if cn else "Receiving / perfectly coordinated source SD",
        xlim=(-1.05, 1.05), yscale="log")
    ax.set_title("a  真实支流协同与出口波动" if cn else "a  Observed input coordination and receiving signal", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]
    scopes = [("monthly", -1, 0.), ("monthly", -1, .8), ("activity", 0, 0.)]
    labels = ["全部月尺度河网", "监测覆盖 ≥80%", "严格同日观测"] if cn else ["All monthly networks", "Area coverage >=80%", "Strict same-day activities"]
    for i, (version, cut, coverage) in enumerate(scopes):
        g = summary.query("metric == 'peak_risk_difference' and group == 'all' and adjustment == 'calendar_year' and excursion_quantile == .75")
        g = g[g.version.eq(version) & g.sample_cut_days.eq(cut) & g.minimum_coverage.eq(coverage)]
        if g.empty:
            ax.text(0, i, "无足够样本" if cn else "Insufficient groups", va="center")
            continue
        row = g.iloc[0]
        ax.plot(np.array([row.ci_low, row.ci_high])*100, [i, i], color="#257F88", linewidth=2)
        ax.scatter(row.estimate*100, i, color="#257F88", s=42)
        labels[i] += f" · n={int(row.n_receivers)}"
    ax.axvline(0, color="#93A1A6", linestyle="--", linewidth=1)
    ax.set(yticks=range(3), yticklabels=labels, ylim=(2.6, -.6),
        xlabel="出口高波动概率差（共同高点 − 单支流高点，百分点）" if cn else "Receiving-excursion difference (coincident minus solo, pp)")
    ax.set_title("b  支流共同高点是否对应出口高点？" if cn else "b  Receiving excursions with coincident source highs", loc="left")
    ax = axes[1, 0]
    for c in (1, 2, 3):
        g = primary[primary.cluster.eq(c)].sort_values("target")
        jitter = np.linspace(-.12, .12, len(g))
        for j, row in enumerate(g.itertuples()):
            high = row.covered_area_fraction >= .8
            ax.scatter(c+jitter[j], row.source_coherence, marker="s" if high else "o", s=40,
                facecolor=COLORS[c] if high else "white", edgecolor=COLORS[c])
    ax.set(xticks=[1, 2, 3], xticklabels=[names[c] for c in (1, 2, 3)], ylim=(-1.05, 1.05),
        ylabel="实际支流同步程度" if cn else "Observed tributary coordination")
    ax.axhline(0, color="#C2CDD1", linewidth=.8)
    ax.set_title("c  保留原来的整片河网形态" if cn else "c  Original complete-network forms", loc="left")
    ax = axes[1, 1]
    for c in (1, 2, 3):
        g = periods[periods.cluster.eq(c)]
        for _, r in g.groupby("target"):
            r = r.sort_values("period_start_year")
            ax.plot(r.source_coherence_within, r.outlet_sync_log_sd_ratio_within,
                marker="o", color=COLORS[c], linewidth=1., alpha=.8)
    ax.axhline(0, color="#C2CDD1", linewidth=.8)
    ax.axvline(0, color="#C2CDD1", linewidth=.8)
    ax.set(xlabel="同步程度相对本站均值的变化" if cn else "Coordination minus the network mean",
        ylabel="出口相对波动的变化（log SD）" if cn else "Receiving-reference change (log SD)")
    ax.set_title("d  同一河网的不同时期" if cn else "d  Different periods within the same river", loc="left")
    fig.text(.06, .015, "方块：覆盖 ≥80%；空心圆：其余。高波动为各自异常值上四分位；区间按整片系统重抽 5,000 次。" if cn else
        "Squares: area coverage >=80%; open circles: remaining. Excursions are upper-quartile anomalies; intervals resample whole systems 5,000 times.", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1), w_pad=2, h_pad=2.5)
    save(fig, "observed_coordination_and_whole_form")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6), gridspec_kw={"width_ratios": [1., 1.05, 1.6]})
    f = annual.query("version == 'selected_activity' and adjustment == 'within_month' and eligible")
    ax = axes[0]
    ax.scatter(f.source_coherence, np.exp(f.outlet_sync_log_sd_ratio), color="#257F88", s=34)
    ax.axhline(1, color="#93A1A6", linestyle="--", linewidth=1)
    ax.set(xlabel="每年支流同步程度" if cn else "Annual tributary coordination",
        ylabel="出口 / 同步支流参考 (SD)" if cn else "Receiving / coordinated-source SD", yscale="log")
    ax.set_title("a  同一真实路径，不同年份" if cn else "a  Fixed actual paths, different years", loc="left")
    ax = axes[1]
    labels = ["季节与趋势调整", "同月内波动"] if cn else ["Season/trend adjusted", "Within-month changes"]
    for i, adjustment in enumerate(("calendar_year", "within_month")):
        for version, offset, marker, color in (("selected_activity", -.12, "o", "#257F88"), ("daily_mean", .12, "s", "#C5814A")):
            g = case[(case.version.eq(version)) & case.adjustment.eq(adjustment) & case.excursion_quantile.eq(.75)].iloc[0]
            ax.plot(np.array([g.peak_ci_low, g.peak_ci_high])*100, [i+offset]*2, color=color, linewidth=2)
            ax.scatter(g.peak_risk_difference*100, i+offset, color=color, marker=marker, s=35,
                label=("活动值" if cn else "Selected activity") if i == 0 and version == "selected_activity" else
                      ("日均值" if cn else "Daily mean") if i == 0 else None)
    ax.axvline(0, color="#93A1A6", linestyle="--", linewidth=1)
    ax.set(yticks=[0, 1], yticklabels=labels, ylim=(1.6, -.6),
        xlabel="出口高点概率差（百分点）" if cn else "Receiving-excursion difference (pp)")
    ax.set_title("b  同日记录中的共同高点" if cn else "b  Coincident highs in same-day records", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2)
    t = trace.query("version == 'selected_activity' and adjustment == 'within_month' and excursion_quantile == .75").copy()
    # Sampling density alone selects the displayed year, never wave magnitude.
    n = t.groupby(t.date.dt.year).size()
    year = min(n.index, key=lambda y: (-int(n[y]), y))
    t = t[t.date.dt.year.eq(year)].sort_values("date")
    ax = axes[2]
    for field, label, color, style in (("source_0_anomaly", "Icy Brook", "#257F88", "--"),
            ("source_1_anomaly", "Andrews Creek", "#C5814A", ":"),
            ("receiver_anomaly", "Loch 出口" if cn else "Loch outlet", "#66749F", "-")):
        # Break the line across a long observation gap; never interpolate events.
        for _, chunk in t.groupby(t.date.diff().dt.days.gt(30).cumsum()):
            ax.plot(chunk.date, chunk[field], color=color, linestyle=style, marker="o", markersize=3, linewidth=1.2,
                label=label if chunk.index[0] == t.index[0] else None)
    ax.axhline(0, color="#C2CDD1", linewidth=.8)
    first, last = t.date.min(), t.date.max()
    calendar_ticks = pd.date_range(first.to_period("M").to_timestamp(), last, freq="2MS")
    ticks = [first, *[date for date in calendar_ticks if (date-first).days > 25 and (last-date).days > 25], last]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set(xlabel=f"{year} · 月-日" if cn else f"Month–day, {year}",
        ylabel="同月内 DOC 偏差 (mg/L)" if cn else "Within-month DOC deviation (mg/L)")
    ax.set_title("c  实际支流与出口波动" if cn else "c  Actual tributary and receiving records", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.23), ncol=3)
    fig.text(.05, .018, "Loch Vale 是一个受湖泊影响的案例；同日不等于同一时刻。区间重抽年份，不能替代多个河网形态的比较。" if cn else
        "Loch Vale is one lake-influenced case; same day is not the same instant. Year-block intervals do not replicate river forms.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1), w_pad=2)
    save(fig, "same_river_observed_source_timing")
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "input_hashes": {str(path): sha256_file(path) for path in (
            source/"network_metrics.csv", source/"signal_summary.csv", source/"within_network_periods.csv",
            source/"dense_case_metrics.csv", source/"dense_case_years.csv", source/"dense_case_series.parquet")},
        "output_hashes": {str(path): sha256_file(path) for path in written},
        "example_selection": f"Most observed within-month dates, then earliest year: {year}",
        "scope": "Actual DOC observations; no synthetic forcing, no fitted travel time"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
