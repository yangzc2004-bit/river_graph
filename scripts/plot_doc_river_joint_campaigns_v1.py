"""Show observed DOC response, common calendars and real confluence arrangements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_joint_campaigns_v1")
ORDER = ("C12", "C16", "C7", "C9")
COLORS = {"all_calendar": "#297B8B", "joint_calendar": "#CA8748"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263A40", "axes.labelcolor": "#263A40", "pdf.fonttype": 42})
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(folder/"configuration_comparison.csv")
    details = pd.read_parquet(folder/"adjusted_campaigns.parquet")
    summary = json.loads((folder/"summary.json").read_text())
    inputs = [folder/"configuration_comparison.csv", folder/"adjusted_campaigns.parquet", folder/"summary.json"]
    paths = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out/f"{name}{'_cn' if cn else ''}.{extension}"
            fig.savefig(path, dpi=220, facecolor="white")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.9, 8.9))
    fig.subplots_adjust(left=.095, right=.975, bottom=.16, top=.865, wspace=.4, hspace=.5)
    fig.suptitle("真实汇合布局与实测 DOC 波动" if cn else "Actual confluence arrangement and measured DOC variation",
                 x=.095, ha="left", fontsize=17, y=.97)
    n = summary["n_joint_calendar_dates"]
    fig.text(.095, .92, f"四处汇合点 · 2014–2017 年 {n} 个共同采样日期 · 同一相连流域" if cn else
             f"Four confluences · {n} common sampling dates in 2014–2017 · One connected catchment", fontsize=11)
    geometry = comparison.loc[comparison.population.eq("joint_calendar")].set_index("receiver").reindex(ORDER)

    ax = axes[0, 0]
    for i, site in enumerate(ORDER):
        row = geometry.loc[site]
        for offset, path, source in ((-.14, row.path_a_km, row.source_a), (.14, row.path_b_km, row.source_b)):
            independent = path - row.common_path_km
            ax.barh(i+offset, independent, height=.23, color="#A8C7CD")
            ax.barh(i+offset, row.common_path_km, left=independent, height=.23, color="#344F5A")
            ax.text(path+.10, i+offset, source, va="center", fontsize=8)
    ax.set(yticks=range(4), yticklabels=ORDER, xlabel="地图河道路径长度（km）" if cn else "Mapped source-to-outlet path (km)",
           xlim=(0, geometry[["path_a_km", "path_b_km"]].max().max()+1.0))
    ax.invert_yaxis()
    ax.set_title("a  支流路径与共同下游河段" if cn else "a  Branch paths and common downstream corridor", loc="left", fontsize=11)
    ax.text(.02, -.24, "浅色：独立支流路径    深色：汇合后共同河段" if cn else
            "Light: separate branch path    Dark: common corridor", transform=ax.transAxes, fontsize=9)

    ax = axes[0, 1]
    for population, offset in (("all_calendar", -.12), ("joint_calendar", .12)):
        selected = comparison.loc[comparison.population.eq(population)].set_index("receiver").reindex(ORDER)
        y = np.arange(4)+offset
        x = selected.adjusted_outlet_mix_sd_ratio.to_numpy()
        low = selected.adjusted_outlet_mix_sd_ratio_lo.to_numpy()
        high = selected.adjusted_outlet_mix_sd_ratio_hi.to_numpy()
        label = ("全部日期" if population == "all_calendar" else "四处共同日期") if cn else (
            "All available dates" if population == "all_calendar" else "Joint calendar")
        # Percentile intervals need not contain the plug-in estimate; draw bounds independently.
        ax.hlines(y, low, high, color=COLORS[population], linewidth=1.3)
        ax.scatter(x, y, color=COLORS[population], s=34, label=label, zorder=3)
    ax.axvline(1, color="#58696F", linestyle="--", linewidth=1)
    ax.set(yticks=range(4), yticklabels=ORDER, xlabel="出口 / 支流混合的 DOC 波动比" if cn else "Outlet / branch-mixture DOC SD ratio", xlim=(0, None))
    ax.invert_yaxis()
    ax.set_title("b  去除季节与年份趋势后的波动" if cn else "b  Variation after season/time adjustment", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax = axes[1, 0]
    for i, site in enumerate(ORDER):
        row = geometry.loc[site]
        ax.plot([0, 1], [row.raw_outlet_mix_sd_ratio, row.adjusted_outlet_mix_sd_ratio],
                color="#9BAEB3", linewidth=1.1)
        ax.scatter([0, 1], [row.raw_outlet_mix_sd_ratio, row.adjusted_outlet_mix_sd_ratio],
                   color=["#297B8B", "#CA8748"], s=34)
        ax.annotate(site, (1, row.adjusted_outlet_mix_sd_ratio), xytext=(9, (i-1.5)*8),
                    textcoords="offset points", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": "#819499", "linewidth": .6})
    ax.axhline(1, color="#58696F", linestyle="--", linewidth=1)
    ax.set(xticks=[0, 1], xticklabels=("原始波动", "去除季节趋势") if cn else
           ("Raw variation", "Season/time adjusted"), xlim=(-.2, 1.4), ylim=(0, 1.8),
           ylabel="出口 / 混合 SD 比" if cn else "Outlet / mixture SD ratio")
    ax.set_title("c  相同日期下的尺度检查" if cn else "c  Same-date scale check", loc="left", fontsize=11)

    ax = axes[1, 1]
    for i, site in enumerate(ORDER):
        values = details.loc[details.population.eq("joint_calendar") & details.receiver.eq(site), "known_upstream_flow_share"]
        jitter = np.random.default_rng(42+i).uniform(-.13, .13, len(values))
        ax.scatter(values, i+jitter, s=13, color="#A8C7CD", alpha=.8)
        ax.scatter(values.median(), i, marker="|", s=155, color="#344F5A", zorder=3)
    ax.axvline(1, color="#58696F", linestyle="--", linewidth=1)
    ax.set(yticks=range(4), yticklabels=ORDER, xlim=(0, None),
           xlabel="已测支流流量 / 出口流量" if cn else "Measured branch flow / outlet flow")
    ax.invert_yaxis()
    ax.set_title("d  同时保留未监测来水的缺口" if cn else "d  Preserve the unmonitored water contribution", loc="left", fontsize=11)
    fig.text(.095, .085, "比值 <1：出口波动更小；>1：出口波动更大。区间按整年重抽样，不代表不同流域的重复。" if cn else
             "Ratio <1: smaller outlet variation; >1: larger. Intervals resample whole years, not independent catchments.", fontsize=9)
    fig.text(.095, .052, "每日流量只用于混合权重；离散 DOC 采样不能给出连续峰值或实际到达时间。" if cn else
             "Daily flow supplies mixing weights. Discrete DOC samples do not resolve continuous peaks or actual arrival times.", fontsize=9)
    save(fig, "joint_calendar_structure_and_doc")

    joint = details.loc[details.population.eq("joint_calendar")].sort_values("date_local")
    fig, axes = plt.subplots(4, 1, figsize=(11.9, 10.5), sharex=True)
    fig.subplots_adjust(left=.095, right=.975, bottom=.13, top=.885, hspace=.42)
    fig.suptitle("共同日期上的实际 DOC 与支流混合" if cn else "Observed outlet DOC and branch mixture on common dates",
                 x=.095, ha="left", fontsize=16, y=.965)
    fig.text(.095, .925, "只绘制真实采样点；四处使用完全相同的日期" if cn else
             "Actual sampled points only; all four configurations use identical dates", fontsize=11)
    for ax, site in zip(axes, ORDER, strict=True):
        sub = joint.loc[joint.receiver.eq(site)]
        ax.scatter(sub.date_local, sub.doc_dynamic_mix, s=22, color="#297B8B", label="支流混合" if cn else "Branch mixture")
        ax.scatter(sub.date_local, sub.doc_receiver, s=26, marker="x", color="#CA8748", label="实际出口" if cn else "Observed outlet")
        ax.set_ylabel("DOC (mg C/L)")
        ax.set_title(site, loc="left", fontsize=10)
        ax.grid(axis="y", color="#E4E9EA", linewidth=.5)
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    first, last = joint.date_local.min(), joint.date_local.max()
    ticks = [first, pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01"), pd.Timestamp("2017-01-01"), last]
    axes[-1].set_xticks(ticks, [p.strftime("%Y-%m-%d") for p in ticks])
    fig.text(.095, .065, "SITES 实验流域的四处相连汇合布局；支流混合只包含已测的两路来水。" if cn else
             "Four connected SITES confluences; each calculated mixture includes only its two measured branches.", fontsize=9)
    save(fig, "joint_calendar_actual_doc")
    (out/f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in inputs},
        "output_hashes": {str(p): sha256_file(p) for p in paths},
        "code_hash": sha256_file(Path(__file__)), "illustrative_generated_images": False,
    }, indent=2)+"\n")
    print(f"Saved {len(paths)} figure files", flush=True)


if __name__ == "__main__":
    main()
