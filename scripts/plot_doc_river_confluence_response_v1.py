"""Make scientific figures for measured incoming water and confluence DOC."""

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
from matplotlib.lines import Line2D

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_confluence_response_v1")
WATER_COLORS = {"C7": "#277F89", "C9": "#B77A46", "C16": "#827CA7", "C12": "#66838E"}
SITE_COLORS = {f"Con-{i}": c for i, c in enumerate(("#277F89", "#D49B55", "#758CB6", "#A07799", "#809A72"), 1)}
SIGNAL_COLORS = ["#277F89", "#8F82AD", "#C78349"]


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
    analysis, out = ROOT / "analysis", ROOT / "figures"
    out.mkdir(exist_ok=True)
    suffix, outputs = ("_cn" if cn else ""), []
    water = pd.read_csv(analysis / "water_window_comparison.csv")
    main = water[water.primary_coverage_subset].sort_values(["receiver", "year"]).reset_index(drop=True)
    chemistry = pd.read_csv(analysis / "confluence_chemistry.csv")
    geometry = pd.read_csv(analysis / "channel_geometry.csv")
    fall = pd.read_csv(analysis / "fall_geometry_doc.csv")
    relations = pd.read_csv(analysis / "descriptive_geometry_relations.csv").set_index("geometry_metric")

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out / f"{name}{suffix}.{ext}"
            fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.8))
    fig.subplots_adjust(left=.11, right=.96, top=.84, bottom=.14, hspace=.45, wspace=.37)
    y = np.arange(len(main))
    ax = axes[0, 0]
    for i, row in enumerate(main.itertuples()):
        lo, hi = row.b_minus_a_lower_hours/24, row.b_minus_a_upper_hours/24
        ax.errorbar((lo+hi)/2, i, xerr=[[(hi-lo)/2], [(hi-lo)/2]], fmt="o", capsize=3,
                    color=WATER_COLORS[row.receiver], markersize=5)
    ax.axvline(0, color="#BCC9CC", linestyle=":", linewidth=1)
    ax.set(yticks=y, yticklabels=[f"{r.source_a}/{r.source_b} → {r.receiver} · {r.year}" for r in main.itertuples()],
           xlabel="支流 B－A 季节最大值时差（天）" if cn else "Branch B − A seasonal maximum clock (days)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("a  来水最大值不总在同一天" if cn else "a  Incoming maxima need not share a day", loc="left", fontsize=11)

    ax = axes[0, 1]
    for i, row in enumerate(main.itertuples()):
        ax.plot([row.wave_overlap, row.wave_overlap_p10], [i, i], color="#C6D1D3", linewidth=2)
        ax.scatter(row.wave_overlap, i, s=35, color=WATER_COLORS[row.receiver])
        ax.scatter(row.wave_overlap_p10, i, s=28, facecolors="white", edgecolors=WATER_COLORS[row.receiver])
    ax.set(yticks=y, yticklabels=[f"{r.receiver} · {r.year}" for r in main.itertuples()], xlim=(0, 1.05),
           xlabel="两支流归一化水波重合度（0–1）" if cn else "Overlap of normalized incoming water profiles (0–1)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("b  直接量化来水过程的重合" if cn else "b  Compare the incoming waveform overlap", loc="left", fontsize=11)
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", color="#277F89", label="窗口最小值基线" if cn else "Window-minimum baseline"),
                       Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", color="#277F89", label="10% 分位基线" if cn else "10th-percentile baseline")],
              frameon=False, fontsize=8, loc="lower left")

    ax = axes[1, 0]
    for receiver, frame in main.groupby("receiver"):
        ax.scatter(frame.incoming_peak_coincidence, frame.amplitude_balanced_peak_coincidence,
                   color=WATER_COLORS[receiver], s=40, label=receiver, edgecolor="white", linewidth=.4)
    ax.plot([.45, 1.02], [.45, 1.02], color="#ADBCC1", linestyle="--", linewidth=1)
    ax.set(xlim=(.45, 1.02), ylim=(.45, 1.02), xlabel="保留真实流量幅度的峰值重合比" if cn else "Peak coincidence with observed flow amplitudes",
           ylabel="两支流等幅后的峰值重合比" if cn else "Peak coincidence after balancing amplitudes")
    ax.set_title("c  大支流主导不等于两波同步" if cn else "c  Flow dominance differs from synchrony", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    for receiver, frame in main.groupby("receiver"):
        x = frame.branch_flow_share_median.to_numpy()
        ax.scatter(x, frame.wave_overlap, color=WATER_COLORS[receiver], s=40,
                   label=receiver, edgecolor="white", linewidth=.4)
    ax.axvline(1, color="#ADBCC1", linestyle="--", linewidth=1)
    ax.set(xlabel="两支流之和／接收河段流量（窗口中位数）" if cn else "Branch-sum / receiver flow (window median)",
           ylabel="来水水波重合度" if cn else "Incoming water-profile overlap", ylim=(0, 1.05))
    ax.set_title("d  保留未监测来水的份额" if cn else "d  Preserve partial input coverage", loc="left", fontsize=11)
    fig.suptitle("真实汇流点：支流水波怎样重合" if cn else "Real confluences: how incoming water waves overlap", x=.07, y=.98, ha="left", fontsize=16)
    fig.text(.07, .924, (f"保留全部 20 个原春季窗口；{len(main)} 个满足 ≥95% 同时观测并保留各站最大值。不是 20 次独立暴雨。" if cn else
                        f"All 20 original spring windows retained; {len(main)} have ≥95% joint coverage and preserve site maxima. These are not 20 independent storms."), fontsize=10)
    fig.text(.07, .035, ("流量为半小时水位换算值；时间差是窗口最大值时钟，不是河段行程时间。a 的横线为并列最大值的时间范围。\n"
                        "b 保留同一时钟，不搜索最佳平移。c 比较来水叠加，不是实测出口削峰；DOC 没有被插值或由流量估算。" if cn else
                        "Flow is stage-derived at 30-min resolution; peak-clock differences are not hydraulic travel times. a: Bars span tied sampled maxima.\n"
                        "b: Same-clock comparison without lag search. c: Incoming-signal combination, not observed outlet attenuation. DOC is neither filled nor inferred from flow."), fontsize=9, color="#53696E")
    save(fig, "incoming_water_overlap")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
    fig.subplots_adjust(left=.09, right=.96, top=.84, bottom=.14, hspace=.46, wspace=.37)
    ax = axes[0, 0]
    x = np.arange(len(geometry))
    ax.bar(x-.17, geometry.receiver_main_width_ratio, width=.32, color="#277F89", label="河宽" if cn else "Width")
    ax.bar(x+.17, geometry.receiver_main_depth_ratio, width=.32, color="#C78349", label="水深" if cn else "Depth")
    ax.set_ylim(0, max(geometry.receiver_main_width_ratio.max(), geometry.receiver_main_depth_ratio.max())*1.24)
    ax.axhline(1, color="#A7B8BB", linestyle="--", linewidth=1)
    ax.set(xticks=x, xticklabels=geometry.confluence, ylabel="下游／上游主干（秋季实测）" if cn else "Downstream / mainstem (fall survey)")
    ax.set_title("a  汇流区几何存在真实差异" if cn else "a  Measured geometry differs among junctions", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    for i, site in enumerate(geometry.confluence):
        frame = chemistry[chemistry.confluence.eq(site)].set_index("season")
        for season, dy, marker in (("summer", -.12, "o"), ("fall", .12, "s")):
            row = frame.loc[season]
            mid = row.doc_receiver_minus_mix_pct
            lo = 100*(row.doc_receiver_min/row.doc_mix_reference-1)
            hi = 100*(row.doc_receiver_max/row.doc_mix_reference-1)
            ax.errorbar(mid, i+dy, xerr=[[mid-lo], [hi-mid]], marker=marker, color=SITE_COLORS[site],
                        linestyle="none", capsize=3, markersize=5)
    ax.axvline(0, color="#ADBCC1", linestyle="--", linewidth=1)
    ax.set(yticks=x, yticklabels=geometry.confluence,
           xlabel="下游 DOC 相对支流混合参考值的差别（%）" if cn else "Downstream DOC relative to branch-mixture reference (%)")
    ax.invert_yaxis()
    ax.set_title("b  同一结构下，DOC 响应可换方向" if cn else "b  DOC direction can change at one junction", loc="left", fontsize=11)
    ax.legend(handles=[Line2D([0], [0], color="#506B72", marker="o", linestyle="none", label="夏季" if cn else "Summer"),
                       Line2D([0], [0], color="#506B72", marker="s", linestyle="none", label="秋季" if cn else "Fall")], frameon=False, fontsize=8, loc="lower left")

    ax = axes[1, 0]
    for row in fall.itertuples():
        ax.scatter(row.receiver_main_depth_ratio, row.doc_receiver_minus_mix_pct, color=SITE_COLORS[row.confluence], s=55)
        ax.annotate(row.confluence, (row.receiver_main_depth_ratio, row.doc_receiver_minus_mix_pct),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#ADBCC1", linestyle="--", linewidth=1)
    ax.axvline(1, color="#CCD6D8", linestyle=":", linewidth=1)
    ax.margins(.22)
    ax.set(xlabel="下游／主干水深（同一次秋季调查）" if cn else "Downstream / mainstem depth (same fall campaign)",
           ylabel="DOC 混合偏差（%）" if cn else "DOC mixture deviation (%)")
    ax.set_title("c  水深变化提供了具体线索" if cn else "c  Depth change supplies a specific lead", loc="left", fontsize=11)
    depth_r = relations.loc["receiver_main_depth_ratio", "spearman"]
    ax.text(.03, .95, f"秩相关 {depth_r:.2f}（5 个汇流点）" if cn else f"Rank correlation {depth_r:.2f} (five junctions)",
            transform=ax.transAxes, va="top", fontsize=9)

    ax = axes[1, 1]
    for row in fall.itertuples():
        ax.scatter(row.receiver_main_width_ratio, row.doc_receiver_minus_mix_pct, color=SITE_COLORS[row.confluence], s=55)
        ax.annotate(row.confluence, (row.receiver_main_width_ratio, row.doc_receiver_minus_mix_pct),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#ADBCC1", linestyle="--", linewidth=1)
    ax.axvline(1, color="#CCD6D8", linestyle=":", linewidth=1)
    ax.margins(.22)
    ax.set(xlabel="下游／主干河宽（同一次秋季调查）" if cn else "Downstream / mainstem width (same fall campaign)",
           ylabel="DOC 混合偏差（%）" if cn else "DOC mixture deviation (%)")
    ax.set_title("d  河宽变化的对应关系较弱" if cn else "d  Width change has a weaker correspondence", loc="left", fontsize=11)
    width_r = relations.loc["receiver_main_width_ratio", "spearman"]
    ax.text(.03, .95, f"秩相关 {width_r:.2f}（5 个汇流点）" if cn else f"Rank correlation {width_r:.2f} (five junctions)",
            transform=ax.transAxes, va="top", fontsize=9)
    fig.suptitle("真实汇流结构与 DOC 的混合响应" if cn else "Confluence geometry and DOC mixture response", x=.07, y=.98, ha="left", fontsize=16)
    fig.text(.07, .923, "一个河网中的 5 个连续汇流点；2021 年夏、秋两次采样。河宽／深按调查断面等权汇总。" if cn else
             "Five serial confluences in one network, sampled in summer and fall 2021. Width/depth summaries give each surveyed transect equal weight.", fontsize=10)
    fig.text(.07, .035, ("b：圆点／方点为三个横向位置的平均，横线是空间范围，不是置信区间。混合值按两支流实测流量重新归一化。\n"
                        "下游三点均值不是流量加权断面浓度；相对混合值的差别不能单独认定为 DOC 去除或生成。秋季几何不外推为夏季实测。" if cn else
                        "b: Symbols are three-position lateral means; bars are spatial ranges, not confidence intervals. Mixture weights are renormalized from branch flow.\n"
                        "A lateral mean is not a flux-weighted concentration. Deviations alone do not establish DOC removal/production; fall geometry is not a summer measurement."), fontsize=9, color="#53696E")
    save(fig, "confluence_geometry_doc")

    clocks = pd.read_parquet(analysis / "water_clocks.parquet")
    lab = pd.read_parquet(analysis / "laboratory_doc_points.parquet")
    examples = main.groupby("receiver", sort=True).head(2)
    fig, axes = plt.subplots(len(examples), 2, figsize=(12.5, max(7, 3.2*len(examples)+2)), squeeze=False)
    fig.subplots_adjust(left=.08, right=.96, top=.84, bottom=.14, hspace=.63, wspace=.3)
    for i, row in enumerate(examples.itertuples()):
        frame = clocks[clocks.window_id.eq(row.window_id)]
        start = pd.Timestamp(row.start_date)
        days = frame.timestamp_local.sub(start).dt.total_seconds().div(86400)
        for key, site, color in zip(("q_a_m3s", "q_b_m3s", "q_receiver_m3s"), (row.source_a, row.source_b, row.receiver), SIGNAL_COLORS):
            values = frame[key]
            axes[i, 0].plot(days, (values-values.min())/(values.max()-values.min()), color=color, linewidth=1.1, label=site)
            points = lab[lab.site.eq(site) & lab.timestamp_local.ge(start)
                         & lab.timestamp_local.lt(pd.Timestamp(row.end_date)+pd.Timedelta(days=1)) & lab.value.notna()]
            xx = points.timestamp_local.sub(start).dt.total_seconds().div(86400)
            axes[i, 1].scatter(xx, points.value, color=color, s=23, label=site, edgecolor="white", linewidth=.4)
        axes[i, 0].set_title(f"{row.source_a} + {row.source_b} → {row.receiver} · {row.year}", loc="left", fontsize=11)
        axes[i, 1].set_title("DOC 只保留真实采样点" if cn else "DOC retains actual laboratory samples", loc="left", fontsize=11)
        for ax in axes[i]:
            ax.set_xlim(0, 29)
            ax.set_xlabel("原窗口起点后的天数" if cn else "Days from original window start", fontsize=9)
        axes[i, 0].set_ylabel("流量（各站极差归一化）" if cn else "Flow (site range-normalized)", fontsize=9)
        axes[i, 1].set_ylabel("DOC (mg C/L)", fontsize=9)
        axes[i, 0].legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.suptitle("来水时序与碳观测：同一窗口，两种证据" if cn else "Incoming water timing and sampled carbon in the same windows", x=.07, y=.98, ha="left", fontsize=16)
    fig.text(.07, .921, "每个有合格流量窗口的汇流配置取最早两个年份；不按 DOC 响应方向挑选。" if cn else
             "Earliest two eligible years per configuration, selected without DOC-response direction.", fontsize=10)
    fig.text(.07, .035, "左图比较水波形状；右图不连线、不填补小时 DOC。两个支流及下游的采样时刻分别保留。" if cn else
             "Left: water-wave shapes. Right: no connected or filled hourly DOC curves; site-specific sample times are preserved.", fontsize=9, color="#53696E")
    save(fig, "incoming_water_doc_examples")

    sources = list(analysis.glob("*.csv"))+list(analysis.glob("*.parquet"))+[Path("scripts/plot_doc_river_confluence_response_v1.py")]
    (out / f"figure_sources{suffix}.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in sources}, "output_hashes": {str(p): sha256_file(p) for p in outputs},
        "example_windows": examples.window_id.tolist(), "selection": "earliest two eligible years per receiver, not DOC outcome",
        "language": "Chinese" if cn else "English"}, indent=2) + "\n")
    print(f"Saved {len(outputs)} figures")


if __name__ == "__main__":
    main()
