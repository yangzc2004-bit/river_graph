"""Dates, matched DOC variability and real river-footprint sampling examples."""
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
from matplotlib.font_manager import FontProperties, fontManager
from plot_doc_monitored_river_footprint_v1 import CACHE, draw_corridor, mapped_corridor

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_sampling_resolution_v1")
GEOMETRY = Path("experiments/phase4_transfer/doc_monitored_river_footprint_v1/analysis")
DTYPES = {"site_no": str, "source_a": str, "source_b": str, "target": str, "huc4": str}
COLORS = ("#267F88", "#C78169", "#304B58")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chinese", action="store_true")
    cn = p.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42, "savefig.dpi": 200})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    triplets = pd.read_parquet(a/"sample_triplets.parquet")
    cadence = pd.read_csv(a/"station_cadence.csv", dtype=DTYPES)
    ledger = pd.read_csv(a/"date_cut_ledger.csv")
    summaries = pd.read_csv(a/"signal_summary.csv")
    hydro = pd.read_csv(a/"hydro_date_summary.csv")
    suffix, files = ("_cn" if cn else ""), []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, facecolor="white", bbox_inches="tight")
            files.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    ax = axes[0, 0]
    counts = triplets.sample_span_days.value_counts().sort_index()
    ax.bar(counts.index, 100*counts/len(triplets), color=COLORS[0], width=.85)
    ax.set(xlabel="三站采样的最大日期间隔（天）" if cn else "Three-station sampling span (calendar days)",
           ylabel="连接—月份占比（%）" if cn else "Connection-months (%)", xlim=(-.8, 29.8))
    ax.set_title("a  同月记录常常不同日" if cn else "a  Same month often means different days", loc="left", fontsize=12)
    ax.text(.44, .82, (f"同日 {triplets.sample_span_days.eq(0).mean():.1%}\n中位间隔 {triplets.sample_span_days.median():g} 天") if cn else
        (f"Same day {triplets.sample_span_days.eq(0).mean():.1%}\nMedian span {triplets.sample_span_days.median():g} days"), transform=ax.transAxes)
    ax = axes[0, 1]
    ax.hist(cadence.median_gap_days, bins=np.arange(0, 51, 2.5), color=COLORS[1], edgecolor="white")
    median = cadence.median_gap_days.median()
    ax.axvline(median, color=COLORS[2], linestyle="--", linewidth=1)
    ax.set(xlabel="各站采样间隔中位数（天）" if cn else "Station median between-sample interval (days)",
           ylabel="站点数量" if cn else "Stations", xlim=(0, 50))
    ax.set_title("b  典型站点按月采样" if cn else "b  Typical sampling is monthly", loc="left", fontsize=12)
    ax.text(.68, .82, (f"71 站等权中位数\n{median:g} 天") if cn else f"Median across 71 stations\n{median:g} days", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    x = np.arange(len(ledger))
    ax.bar(x, ledger.n_eligible_receivers, color=COLORS[0], width=.62)
    for row, xx in zip(ledger.itertuples(), x, strict=True):
        ax.text(xx, row.n_eligible_receivers+.5, f"{row.n_eligible_receivers} / {row.n_eligible_systems}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=["同日", "≤1", "≤3", "≤7", "全部"] if cn else ["Same day", "≤1", "≤3", "≤7", "All"],
           ylabel="可分析接收站数量" if cn else "Receivers with eligible connections", ylim=(0, 26),
           xlabel="最大采样间隔（天）；每连接至少 24 个月" if cn else "Maximum span (days); ≥24 months per pair")
    ax.set_title("c  日期越严格，河网覆盖越少" if cn else "c  Closer dates retain fewer river systems", loc="left", fontsize=12)
    ax.text(.02, .89, "柱上数字：接收站 / 独立监测系统" if cn else "Labels: receivers / monitoring systems", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 1]
    for name, label, color, offset in (("fraction_offset_flow_ratio_ge1_5", "≥1.5 倍" if cn else "Ratio ≥1.5", COLORS[0], -.16),
        ("fraction_offset_flow_ratio_ge2", "≥2 倍" if cn else "Ratio ≥2", COLORS[1], .16)):
        valid = hydro[hydro.max_span_days.gt(0)]
        xx = np.arange(len(valid))
        ax.bar(xx+offset, valid[name]*100, width=.3, color=color, label=label)
    ax.set(xticks=np.arange(4), xticklabels=["≤1", "≤3", "≤7", "全部" if cn else "All"],
        xlabel="最大采样间隔（天）" if cn else "Maximum sampling span (days)",
        ylabel="间隔内流量变化比例（%）" if cn else "Measured flow-ratio exceedance (%)")
    ax.set_title("d  采样间隔内水文可能变化" if cn else "d  Flow can change between sampling dates", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("先看采样时钟，再研究河网传输" if cn else "The sampling clock matters for river transport", x=.06, ha="left", fontsize=17)
    fig.text(.06, .025, ("59 个支流连接，3,026 个连接—月份；采样日期选择不读取 DOC。流量比较去除重复站点—日期组合，缺测不填补。\n"
        "d 仅包含源站采样日不同于接收站、两日均有正流量的组合；比值为两日较大流量 / 较小流量。") if cn else
        ("59 branch pairs, 3,026 connection-months; date selection never reads DOC. Flow comparisons deduplicate station/date pairs; no gap filling.\n"
         "d: source dates differ from receiver dates; both flows must be measured and positive. Ratio = larger daily flow / smaller daily flow."), fontsize=8.5)
    fig.subplots_adjust(left=.08, right=.98, top=.90, bottom=.15, hspace=.53, wspace=.30)
    save(fig, "sampling_clock_and_hydrology")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.7))
    for ax, metric, factor, title in zip(axes, ("mixture_buffer_fraction", "asynchronous_buffer_fraction", "outlet_mixture_log_sd_ratio"),
        (100, 100, 1), ("混合波动降低" if cn else "Mixture variance reduction", "错峰贡献" if cn else "Asynchrony contribution",
                       "下游 / 混合波动" if cn else "Outlet / mixture variability"), strict=True):
        for j, version in enumerate(("monthly_mean", "date_selected_activity")):
            f = summaries[summaries.metric.eq(metric) & summaries.comparison.eq(version)].sort_values("max_span_days")
            xx = np.arange(len(f))+(j-.5)*.18
            color = COLORS[j]
            ax.scatter(xx, f.estimate*factor, color=color, s=32, zorder=3,
                label=("原月均值" if cn else "Original monthly mean") if j == 0 else ("日期接近的实测值" if cn else "Date-selected samples"))
            valid = f.ci_low.notna()
            ax.vlines(xx[valid], f.loc[valid, "ci_low"]*factor, f.loc[valid, "ci_high"]*factor, color=color, linewidth=1.6)
        ax.set(xticks=np.arange(5), xticklabels=["同日", "≤1", "≤3", "≤7", "全部"] if cn else ["Same\nday", "≤1", "≤3", "≤7", "All"],
            xlabel="最大采样间隔（天）" if cn else "Maximum sampling span (days)")
        ax.set_title(f"{chr(97+list(axes).index(ax))}  {title}", loc="left", fontsize=12)
        ax.set_ylabel(("相对支流加权方差（%）" if cn else "Fraction of weighted source variance (%)") if factor == 100 else
            ("log（下游 SD / 混合 SD）" if cn else "log(outlet SD / mixture SD)"))
        if factor == 1:
            ax.axhline(0, linestyle="--", color="#A5B1B5", linewidth=1)
        ax.grid(axis="y", alpha=.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(.05, .935))
    fig.suptitle("采样日期接近时，支流混合的波动降低仍存在" if cn else "Mixture variability reduction persists in date-aligned samples", x=.05, ha="left", fontsize=16)
    fig.text(.05, .035, ("每个日期阈值内，两种估计使用完全相同的连接和月份；各阈值之间的河网样本不同，不能视作剂量响应。\n"
        "接收站等权，5,000 次整个监测系统 bootstrap；同日组只有一个系统，不画总体区间。混合降低不代表 DOC 被去除。") if cn else
        ("Within each cut, both estimators use identical pairs/months. Different cuts retain different rivers; this is not a dose-response curve.\n"
         "Receivers equally weighted; 5,000 whole-system bootstrap draws. Same-day group has one system and no population CI. Mixing reduction is not DOC removal."), fontsize=8.5)
    fig.subplots_adjust(left=.06, right=.98, top=.79, bottom=.22, wspace=.32)
    save(fig, "matched_date_signal_sensitivity")

    events = pd.read_parquet(a/"doc_activities.parquet")
    daily = pd.read_parquet(a/"daily_flow_context.parquet")
    examples = pd.read_csv(a/"calendar_examples.csv", dtype=DTYPES)
    geometry = pd.read_csv(GEOMETRY/"footprints.csv", dtype=DTYPES)
    reaches = pd.read_csv(GEOMETRY/"corridor_reaches.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 8.8), gridspec_kw={"width_ratios": [1.2, 1.5, 1.5]})
    geometry_hashes = {}
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for i, example in enumerate(examples.itertuples()):
            row = geometry[geometry.pair_id.eq(example.pair_id)].iloc[0]
            r = reaches[reaches.pair_id.eq(example.pair_id)]
            parts, _, geometry_hashes[example.pair_id] = mapped_corridor(connection, r)
            draw_corridor(axes[i, 0], r, parts, row, cn)
            axes[i, 0].set_title(("蓄水河段较多" if cn else "Storage-rich shared trunk") if example.mapped_common_storage else
                                 ("未标记蓄水主干" if cn else "No mapped shared-trunk storage"), loc="left", fontsize=12)
            for j, site in enumerate((example.source_a, example.source_b, example.target)):
                f = events[events.site_no.eq(site) & events.date.dt.year.eq(example.year)]
                axes[i, 1].scatter(f.date, f.doc, color=COLORS[j], marker=("o", "s", "D")[j], s=25, alpha=.8, label=site)
                q = daily[daily.site_no.eq(site) & daily.date.dt.year.eq(example.year)].set_index("date").discharge_cfs
                # Calendar reindex preserves gaps in the line instead of bridging them.
                q = q.reindex(pd.date_range(f"{example.year}-01-01", f"{example.year}-12-31", freq="D"))
                if q.gt(0).any():
                    axes[i, 2].plot(q.index, q.where(q > 0), color=COLORS[j], linewidth=1.2, label=site)
                else:
                    axes[i, 2].plot([], [], color="#A9B0B4", linewidth=0, marker="x",
                                    label=site+("（缺测）" if cn else " (no daily flow)"))
                for date in f.date.unique():
                    if date in q.index and q.loc[date] > 0:
                        axes[i, 2].scatter(date, q.loc[date], marker=("o", "s", "D")[j], s=17, color=COLORS[j], zorder=3)
            axes[i, 1].set_title(f"{example.year}  " + ("实际 DOC 采样" if cn else "Actual DOC sampling"), loc="left", fontsize=12)
            axes[i, 2].set_title("采样日期上的实测日流量" if cn else "Daily flow and DOC sampling dates", loc="left", fontsize=12)
            axes[i, 1].set_ylabel("DOC（mg/L）" if cn else "DOC (mg/L)")
            axes[i, 2].set_ylabel("流量（cfs，对数轴）" if cn else "Discharge (cfs, log scale)")
            axes[i, 2].set_yscale("log")
            for ax in axes[i, 1:]:
                ax.set_xlim(pd.Timestamp(f"{example.year}-01-01"), pd.Timestamp(f"{example.year}-12-31"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
                ax.set_xlabel("月份" if cn else "Month")
                ax.grid(axis="y", alpha=.16)
                ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.94,
                          fontsize=8, ncol=1, loc="upper right")
    fig.suptitle("真实河网可见，DOC 峰值过程仍未被连续采到" if cn else "Real river corridors, sparse DOC event observations", x=.04, ha="left", fontsize=17)
    fig.text(.04, .025, ("几何和采样覆盖选例，不按 DOC 响应挑选。DOC 只画实测点；流量线保留缺测断点。地图为实际 NHDPlus 河道，各图独立比例尺。\n"
        "蓄水标记来自 NHDPlus 河段属性，不等于已测得停留时间；全样本五个蓄水连接没有三站月内各 ≥3 天的采样序列。") if cn else
        ("Examples selected by geometry and sampling coverage, not DOC response. DOC points are observations; flow lines retain gaps. Actual NHDPlus routes, separate scales.\n"
         "Mapped storage is a reach attribute, not a measured residence time. None of the five storage connections has ≥3 sample days per station in a common month."), fontsize=8.5)
    fig.subplots_adjust(left=.045, right=.98, top=.88, bottom=.17, hspace=.54, wspace=.34)
    save(fig, "real_river_sampling_windows")
    sources = [a/"sample_triplets.parquet", a/"doc_activities.parquet", a/"daily_flow_context.parquet", a/"signal_summary.csv",
               a/"station_cadence.csv", a/"date_cut_ledger.csv", a/"hydro_date_summary.csv", a/"calendar_examples.csv",
               GEOMETRY/"footprints.csv", GEOMETRY/"corridor_reaches.csv", Path(__file__),
               Path("scripts/plot_doc_monitored_river_footprint_v1.py"), Path("src/river_graph/analysis/river_monitored_footprint.py"),
               Path("src/river_graph/topology/river_planform.py")]
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({"inputs": {str(p): sha256_file(p) for p in sources},
        "geometry_hashes": geometry_hashes, "outputs": {str(p): sha256_file(p) for p in files}}, indent=2)+"\n")
    print("Wrote", len(files), "scientific figures")


if __name__ == "__main__":
    main()
