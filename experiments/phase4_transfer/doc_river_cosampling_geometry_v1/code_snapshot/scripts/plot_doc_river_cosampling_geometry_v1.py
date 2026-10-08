"""Bilingual scientific figures of co-sampling opportunities and real geometry."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_cosampling_geometry_v1 import ROOT, TYPES
from matplotlib.font_manager import FontProperties, fontManager

from river_graph.experiments.provenance import sha256_file

COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chinese", action="store_true")
    args = p.parse_args()
    cn, suffix = args.chinese, "_cn" if args.chinese else ""
    if cn:
        fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        plt.rcParams["font.family"] = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf").get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    folder = ROOT/"analysis"
    inventory = pd.read_csv(folder/"network_inventory.csv", dtype=TYPES)
    geo = pd.read_csv(folder/"network_geometry.csv", dtype=TYPES)
    signals = pd.read_csv(folder/"network_signals.csv", dtype=TYPES)
    summaries = pd.read_csv(folder/"signal_summary.csv")
    trace = pd.read_parquet(folder/"signal_series.parquet")
    pairs = pd.read_csv(folder/"same_region_form_pairs.csv", dtype={"huc4": str})
    main = signals.query("version == 'selected_activity' and adjustment == 'within_month' and excursion_quantile == .75")
    names = {1: "细长多支流" if cn else "Elongated", 2: "主干主导" if cn else "Mainstem dominated",
             3: "宽阔多支流" if cn else "Broad"}
    out = ROOT/"figures"
    out.mkdir(exist_ok=True)
    written, selections = [], []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.5), constrained_layout=True)
    ax = axes[0, 0]
    populations = [inventory[inventory.status.eq("included")], inventory[inventory.within_month_eligible],
                   inventory[inventory.within_month_eligible & inventory.covered_area_fraction.ge(.8)]]
    labels = ["同日共同记录", "月内密集记录", "密集且覆盖≥80%"] if cn else ["Common\nsame days", "Dense within\nmonths", "Dense & area\n>=80%"]
    for j, c in enumerate((1, 2, 3)):
        values = [int(g.cluster.eq(c).sum()) for g in populations]
        positions = np.arange(3)+(j-1)*.23
        ax.bar(positions, values, width=.21, color=COLORS[c], label=names[c])
        for x, value in zip(positions, values, strict=True):
            ax.text(x, value+.25, str(value), ha="center", fontsize=9)
    ax.set(xticks=range(3), xticklabels=labels, ylabel="不同出口河网数量" if cn else "Unique receiving networks")
    ax.set_ylim(0, max(int(g.groupby("cluster").size().max()) for g in populations if len(g))*1.3+1)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("a  真实同日采样覆盖" if cn else "a  Actual co-sampling coverage", loc="left")
    ax = axes[0, 1]
    for c in (1, 2, 3):
        for dense in (False, True):
            f = geo[geo.cluster.eq(c) & geo.within_month_eligible.eq(dense)]
            ax.scatter(f.monitored_path_cv, f.monitored_common_fraction, marker="s" if dense else "o",
                s=44, facecolor=COLORS[c] if dense else "white", edgecolor=COLORS[c])
    ax.set(xlabel="上游到出口路径长度的相对差异" if cn else "Source-to-receiver path CV",
        ylabel="共同河段 / 平均路径长度" if cn else "Shared corridor / mean path length", ylim=(-.03, 1.03))
    ax.set_title("b  保留真实路径与共同汇流段" if cn else "b  Real paths and shared corridors", loc="left")
    ax = axes[1, 0]
    eligible = summaries.query("group == 'all' and metric == 'peak_risk_difference' and version == 'selected_activity' and excursion_quantile == .75 and minimum_coverage == 0")
    ylabels = []
    for i, adjustment in enumerate(("within_month", "calendar_year", "raw")):
        f = eligible[eligible.adjustment.eq(adjustment)]
        label = {"within_month": "月内波动" if cn else "Within-month anomalies",
            "calendar_year": "季节与趋势调整" if cn else "Calendar/trend adjustment",
            "raw": "原始浓度居中" if cn else "Centered native DOC"}[adjustment]
        if len(f):
            row = f.iloc[0]
            ax.plot(np.array([row.ci_low, row.ci_high])*100, [i, i], color=COLORS[1], linewidth=2)
            ax.scatter(row.estimate*100, i, color=COLORS[1], s=40)
            label += f" · n={int(row.n_receivers)}"
            if not np.isfinite(row.ci_low):
                label += "\n单河网，无跨河网区间" if cn else "\nOne river; no between-river interval"
        else:
            ax.text(0, i, "没有足够的对照日期" if cn else "Insufficient excursion groups", va="center")
        ylabels.append(label)
    ax.axvline(0, color="#A8B5B9", linewidth=.8, linestyle="--")
    ax.set(yticks=range(3), yticklabels=ylabels, ylim=(2.6, -.6),
        xlabel="出口高点概率差（共同 − 单支流，百分点）" if cn else "Receiving-excursion difference (coincident minus solo, pp)")
    ax.set_title("c  同日记录中的高点对应" if cn else "c  Co-excursions on actual same days", loc="left")
    ax = axes[1, 1]
    for c in (1, 2, 3):
        f = main[main.cluster.eq(c)].sort_values("target")
        for j, row in enumerate(f.itertuples()):
            ax.scatter(c+(j-(len(f)-1)/2)*.05, np.exp(row.outlet_sync_log_sd_ratio), s=42,
                marker="s" if row.covered_area_fraction >= .8 else "o", edgecolor=COLORS[c],
                facecolor=COLORS[c] if row.covered_area_fraction >= .8 else "white")
    ax.axhline(1, color="#A8B5B9", linewidth=.8, linestyle="--")
    ax.set(xticks=[1, 2, 3], xticklabels=[names[c] for c in (1, 2, 3)], yscale="log",
        ylabel="出口波动 / 完全同步支流参考 (SD)" if cn else "Receiving / coordinated source reference SD")
    ax.set_title("d  各形态实际月内波动" if cn else "d  Observed within-month response by form", loc="left")
    caption = (f"同区域、面积相近形态对：{int((pairs.area_comparable & pairs.both_observed).sum())}对有同日记录，"
        f"{int((pairs.area_comparable & pairs.both_within_month_eligible).sum())}对有月内密集记录。方块：密集或覆盖≥80%（各面板定义）。" if cn else
        f"Same-region, comparable-area pairs: {int((pairs.area_comparable & pairs.both_observed).sum())} co-sampled; "
        f"{int((pairs.area_comparable & pairs.both_within_month_eligible).sum())} dense. Squares: dense (b) or area >=80% (d).")
    fig.supxlabel(caption, fontsize=9)
    save(fig, "cosampling_opportunities_and_geometry")

    selected = []
    for c in (1, 3, 2):
        f = main[main.cluster.eq(c)].sort_values(["n_dates", "target"], ascending=[False, True])
        if len(f):
            selected.append(f.iloc[0])
    if selected:
        fig, axes = plt.subplots(len(selected), 1, figsize=(11.5, 3.0*len(selected)), squeeze=False, constrained_layout=True)
        for ax, row in zip(axes[:, 0], selected, strict=True):
            f = trace[trace.target.eq(row.target) & trace.version.eq("selected_activity") & trace.adjustment.eq("within_month") & trace.excursion_quantile.eq(.75)]
            counts = f.groupby(f.date.dt.to_period("M")).size()
            month = min(counts.index, key=lambda m: (-int(counts[m]), str(m)))
            f = f[f.date.dt.to_period("M").eq(month)].sort_values("date")
            for j in range(int(row.n_sources)):
                ax.plot(f.date, f[f"source_{j}_anomaly"], color="#A5B0B8", linewidth=1,
                    marker="o", markersize=3, label=("上游支流" if cn else "Upstream source") if j == 0 else None)
            ax.plot(f.date, f.receiver_anomaly, color=COLORS[int(row.cluster)], linewidth=1.8,
                marker="s", markersize=4, label="出口" if cn else "Receiver")
            ax.axhline(0, color="#CBD3D6", linewidth=.8)
            ax.set(ylabel="月内 DOC 偏离 (mg/L)" if cn else "Within-month DOC (mg/L)")
            ax.set_title(f"{names[int(row.cluster)]} · {row.target} · {month} · n={len(f)}", loc="left")
            ax.set_xticks(pd.DatetimeIndex([f.date.iloc[0], f.date.iloc[-1]]).unique())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.legend(frameon=False, fontsize=9)
            selections.append({"target": row.target, "year_month": str(month), "n_dates": len(f),
                "selection": "most within-month dates per form, then target ID; densest month, then earliest month"})
        save(fig, "observed_monthly_campaign_examples")
    inputs = [folder/f"{name}.csv" for name in ("network_inventory", "network_geometry", "network_signals", "signal_summary", "same_region_form_pairs")]
    inputs.append(folder/"signal_series.parquet")
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in inputs},
        "output_hashes": {str(p): sha256_file(p) for p in written}, "waveform_selection": selections}, indent=2)+"\n")


if __name__ == "__main__":
    main()
