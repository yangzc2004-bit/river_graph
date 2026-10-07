"""Source-backed figures separating branch integration from path timing."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_signal_mechanisms_v1")
COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 220,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    sources = [a/f"{name}.csv" for name in ("mixing_receivers", "mixing_summary", "arrival_receivers",
               "arrival_summary", "budget_availability", "apparent_departure_summary")]
    mix, ms, arrival, ars, budget, departure = [pd.read_csv(path, dtype={"target": str}) for path in sources]
    suffix, written = ("_cn" if cn else ""), []
    classes = ("细长、多支流型", "主干主导、稀支流型", "宽展、多支流型") if cn else (
        "Elongated / tributary-rich", "Mainstem / sparse", "Broad / tributary-rich")

    def summary(frame, metric):
        return frame[frame.population.eq("all") & frame.unit.eq("component") & frame.metric.eq(metric)].iloc[0]

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))
    ax = axes[0, 0]
    for k in (1, 2, 3):
        f = mix[mix.cluster.eq(k)]
        ax.scatter(f.branch_balance, f.source_rho, s=28+230*f.equal_amplitude_buffer_fraction,
                   color=COLORS[k], alpha=.8, edgecolor="white", linewidth=.7, label=classes[k-1]+f" (n={len(f)})")
    ax.axhline(0, linewidth=.8, color="#D6E0E2")
    ax.set(xlim=(-.02, 1.06), ylim=(-.48, 1.),
           xlabel="两支流贡献均衡程度（面积代理）" if cn else "Branch balance (area-share proxy)",
           ylabel="季节调整后的两支流 DOC 同步性" if cn else "Calendar-adjusted source DOC correlation")
    ax.set_title("a  "+("真实支流的汇合条件" if cn else "Integration conditions in real tributaries"), loc="left", pad=15)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(.02, -.43, "圆越大：等幅输入的混合缓冲越强" if cn else "Larger circle: stronger equal-amplitude mixing buffer", fontsize=8)

    ax = axes[0, 1]
    scenarios = ("mixture_buffer_fraction", "equal_amplitude_buffer_fraction", "balanced_equal_amplitude_buffer_fraction")
    labels = ("实测信号＋原支流贡献", "两支流波动幅度相同", "再设两支流贡献各一半") if cn else (
        "Observed signals + existing shares", "Equal source amplitudes", "+ balanced branch shares")
    for i, metric in enumerate(scenarios):
        r = summary(ms, metric)
        if i == 0:
            amplitude = 100*summary(ms, "amplitude_buffer_fraction").estimate
            asynchronous = 100*summary(ms, "asynchronous_buffer_fraction").estimate
            ax.barh(i, amplitude, color="#CFA979", height=.45)
            ax.barh(i, asynchronous, left=amplitude, color=COLORS[1], height=.45)
        else:
            ax.barh(i, 100*r.estimate, color=COLORS[1] if i == 1 else COLORS[3], height=.45)
        ax.plot(100*np.array([r.ci_low, r.ci_high]), [i, i], color="#263B42", linewidth=1.1)
        ax.scatter(100*r.estimate, i, color="#263B42", s=14, zorder=3)
        ax.text(100*r.ci_high+1, i, f"{100*r.estimate:.1f}%", va="center", fontsize=9)
    ax.set(yticks=range(3), yticklabels=labels, ylim=(2.7, -.6), xlim=(0, 49),
           xlabel="两支流混合的方差削减 (%)" if cn else "Two-source mixture variance reduction (%)")
    ax.set_title("b  "+("改变汇合条件，保持源信号同步性" if cn else "Integration scenarios with source synchrony fixed"), loc="left", pad=15)
    ax.text(.02, .04, "首行：棕色为幅度差异，青色为不同步变化" if cn else "First bar: tan = amplitude imbalance; teal = asynchronous changes",
            transform=ax.transAxes, fontsize=8)

    ax = axes[1, 0]
    for k in (1, 2, 3):
        f = arrival[arrival.cluster.eq(k)]
        ax.scatter(100*f.arrival_gap_relative, np.full(len(f), .12*(k-2)), color=COLORS[k], s=30, alpha=.8)
    for i, metric in enumerate(("arrival_gap_relative", "prediction_gap_relative")):
        r = summary(ars, metric)
        y = i*.9+.75
        ax.plot(100*np.array([r.ci_low, r.ci_high]), [y, y], color="#263B42", linewidth=1.6)
        ax.scatter(100*r.estimate, y, color=COLORS[1], s=45)
        ax.annotate(f"{100*r.estimate:.2f}%", (100*r.estimate, y), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set(yticks=[0, .75, 1.65], yticklabels=("各接收站：输入变化", "输入变化均值", "固定模型输出变化") if cn else (
        "Input: individual receivers", "Mean input change", "Fixed-model output change"),
        ylim=(2.18, -.6), xlim=(0, max(.85, 100*arrival.arrival_gap_relative.max()*1.15)),
        xlabel="绝对变化 / (1＋参考 DOC)，%" if cn else "Absolute change / (1 + reference DOC), %")
    ax.set_title("c  "+("分别表示两条路径：月尺度改变量" if cn else "Separate paths: monthly input sensitivity"), loc="left", pad=15)
    r = summary(ars, "arrival_gap_over_1pct")
    ax.text(.02, .04, (f"仅 {100*r.estimate:.1f}% 的日期超过 1% 输入变化（分层等权）" if cn else
        f"Only {100*r.estimate:.1f}% of dates exceed 1% input change (hierarchically weighted)"), transform=ax.transAxes, fontsize=8)

    ax = axes[1, 1]
    labels = ("共同 DOC 月份", "三站均有实测流量", "上游面积覆盖≥80%", "面积＋流量平衡筛查") if cn else (
        "Common DOC months", "Three measured flows", "Source area coverage ≥80%", "Area + monthly flow screen")
    colors = ["#AABDC1", COLORS[3], "#7AA6AC", COLORS[1]]
    for i, row in budget.iterrows():
        ax.barh(i, row.n_unique_receiver_months, color=colors[i], height=.5)
        ax.text(row.n_unique_receiver_months+18, i, f"{row.n_unique_receiver_months:,} / {row.n_receivers} "+("站" if cn else "sites"), va="center", fontsize=9)
    ax.set(yticks=range(4), yticklabels=labels, ylim=(3.7, -.6), xlim=(0, 1360),
        xlabel="不重复的下游站点—月份" if cn else "Unique receiver-months")
    ax.set_title("d  "+("沿程损失需要更完整的水量预算" if cn else "Channel loss requires a fuller water budget"), loc="left", pad=15)
    fig.suptitle("河网通过什么过程改变 DOC 波动？" if cn else "How can river-network operations modify DOC variability?",
                 x=.035, ha="left", fontsize=17, y=.995)
    fig.text(.035, .025, ("59 对真实支流 / 22 个接收站 / 11 个监测系统；先日期、再支流对、再接收站等权。线为系统 bootstrap 95% 区间。\n"
        "b 为固定混合公式及情景，不等于实测整条河的削减率；c 使用既定路径插值，不估计真实流速。长条型和稀支流型缺少独立系统重复。") if cn else
        ("59 real tributary pairs / 22 receivers / 11 monitoring systems; dates, pairs and receivers weighted hierarchically. Lines: system bootstrap 95%.\n"
         "b: fixed mixing identity/scenarios, not whole-river removal. c: existing path interpolation, not measured velocity. Elongated/sparse forms lack independent systems."), fontsize=8)
    fig.subplots_adjust(left=.12, right=.98, top=.91, bottom=.12, wspace=.73, hspace=.55)
    save(fig, "tributary_integration_and_arrival")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    ax = axes[0]
    for i, (metric, label) in enumerate((("outlet_mixture_log_sd_ratio", "原浓度" if cn else "Native concentration"),
        ("logscale_outlet_mixture_log_sd_ratio", "log1p 浓度" if cn else "log1p concentration"))):
        r = summary(ms, metric)
        ax.plot(np.exp([r.ci_low, r.ci_high]), [i, i], color=COLORS[3], linewidth=1.8)
        ax.scatter(np.exp(r.estimate), i, color=COLORS[3], s=42)
        ax.annotate(f"{np.exp(r.estimate):.2f}", (np.exp(r.estimate), i), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.axvline(1, color="#9FAFB4", linestyle="--", linewidth=.9)
    ax.set(yticks=[0, 1], yticklabels=("原浓度", "log1p 浓度") if cn else ("Native concentration", "log1p concentration"),
        ylim=(1.6, -.6), xlabel="下游 / 两支流混合的波动 SD 比值（几何均值）" if cn else "Outlet / mixture SD ratio (geometric mean)")
    ax.set_title("a  "+("下游波动：比较尺度也很重要" if cn else "Outlet variability depends on scale"), loc="left", pad=16)
    ax = axes[1]
    s = departure[departure.unit.eq("component") & departure.metric.eq("apparent_departure_mg_l")]
    for i, row in s.iterrows():
        y = 0 if row.population == "three_measured_flows" else 1
        ax.plot([row.ci_low, row.ci_high], [y, y], color=COLORS[1], linewidth=1.8)
        ax.scatter(row.estimate, y, color=COLORS[1], s=42)
        ax.annotate(f"{row.estimate:+.2f}", (row.estimate, y), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.axvline(0, color="#9FAFB4", linestyle="--", linewidth=.9)
    ax.set(yticks=[0, 1], yticklabels=("实测三站流量（18 站）", "面积＋水量筛查（5 站）") if cn else (
        "Three measured flows (18 sites)", "Area + water screen (5 sites)"), ylim=(1.6, -.6),
        xlabel="下游 DOC − 流量加权支流 DOC (mg/L)" if cn else "Outlet DOC − flow-weighted source DOC (mg/L)")
    ax.set_title("b  "+("表观浓度差尚不能说明沿程损失" if cn else "Apparent departure does not establish channel loss"), loc="left", pad=16)
    fig.suptitle("波动变小，是否意味着 DOC 被河道去除了？" if cn else "Does lower variability imply DOC removal along the channel?",
                 x=.035, ha="left", fontsize=16, y=.995)
    fig.text(.035, .025, ("a 使用完全相同月份，两尺度并报；保留全部高值和异常比值。b 是月平均的表观浓度差，不是同时刻质量预算。\n"
        "原浓度下游／源方差比的算术均值为 1.89，受个别高比值影响；较小的典型波动并不表示所有河段都在缓冲。") if cn else
        ("a: identical months, both scales; all high values and unusual ratios retained. b: apparent monthly concentration departure, not a contemporaneous mass budget.\n"
         "Native outlet/source arithmetic variance ratio is 1.89, influenced by large ratios; smaller typical variability does not imply every reach buffers."), fontsize=8)
    fig.subplots_adjust(left=.15, right=.985, top=.79, bottom=.22, wspace=.75)
    save(fig, "variability_and_channel_budget")
    manifest = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in sources},
        "generator_sha256": sha256_file(Path(__file__)), "figure_hashes": {str(p): sha256_file(p) for p in written}}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
