"""Export paired field-tracer figures with actual clocks and lab measurement gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_tracer_process_v1")
UP = "#257F88"
DOWN = "#C5814A"
REF = "#64749C"
INK = "#263B42"
PULSES = ["g1", "l", "g2"]


def observed_segments(ax, time, values, color, label, *, linestyle="-", markers=True):
    t, y = np.asarray(time), np.asarray(values)
    eligible = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (np.diff(t) <= 30)
    segments = np.stack([np.column_stack([t[:-1], y[:-1]]), np.column_stack([t[1:], y[1:]])], axis=1)
    ax.add_collection(LineCollection(segments[eligible], colors=color, linewidths=1.2,
                                    linestyles=linestyle))
    ax.plot(t, y, linestyle="none", marker="." if markers else "", markersize=2.5,
            color=color, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42})
    directory = ROOT / "analysis"
    points = pd.read_parquet(directory / "response_points.parquet")
    metrics = pd.read_csv(directory / "response_metrics.csv")
    primary = metrics[metrics.max_gap_min.eq(30)].set_index("code")
    pairs = pd.read_csv(directory / "paired_responses.csv").set_index("pulse")
    sensitivity = pd.read_csv(directory / "processing_sensitivity.csv")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    suffix = "_cn" if cn else ""
    names = {"g1": "8月8日 · 葡萄糖" if cn else "8 Aug · glucose",
             "l": "8月9日 · 叶片浸出液" if cn else "9 Aug · leaf leachate",
             "g2": "8月15日 · 葡萄糖" if cn else "15 Aug · glucose"}
    outputs = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out / f"{name}{suffix}.{extension}"
            fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.4))
    fig.subplots_adjust(left=.075, right=.975, top=.84, bottom=.14, hspace=.50, wspace=.34)
    for row, pulse in enumerate(PULSES):
        water_ax = axes[row, 0]
        for col, site in enumerate(("up", "down")):
            code = f"b_{pulse}_{site}"
            frame = points.loc[points.code.eq(code) & points.time_min.between(0, 300)].copy()
            color = UP if site == "up" else DOWN
            label = ("上游采样点" if cn else "Upstream location") if site == "up" else ("下游采样点" if cn else "Downstream location")
            observed_segments(water_ax, frame.time_min,
                              np.maximum(frame.reference_raw, 0) / primary.loc[code, "salt_area"], color, label)
            ax = axes[row, col + 1]
            observed_segments(ax, frame.time_min, np.maximum(frame.reference_raw, 0), REF,
                              "按盐示踪推得的保守响应" if cn else "Conservative reference from salt", linestyle="--")
            measured = frame.doc_label_raw.notna()
            ax.scatter(frame.loc[measured, "time_min"], frame.loc[measured, "doc_label_raw"],
                       color=color, s=15, edgecolor="white", linewidth=.3,
                       label="实验室标记 DOC" if cn else "Laboratory labelled DOC", zorder=4)
            missing = frame.n_lab_records.gt(0) & frame.doc_label_raw.isna()
            if missing.any():
                ax.scatter(frame.loc[missing, "time_min"], np.zeros(missing.sum()), marker="x",
                           color="#8B8E8F", s=25, linewidth=.9,
                           label="计划采样但 DOC 缺测" if cn else "Scheduled lab value missing")
            high = np.nanmax(np.r_[frame.reference_raw, frame.doc_label_raw])
            ax.set(xlim=(0, 300), ylim=(-.02 * high, 1.16 * high),
                   xlabel="自释放起的分钟数" if cn else "Minutes from release",
                   ylabel="标记碳 (mg C/L)" if cn else "Labelled carbon (mg C/L)")
            letter = "abcdefghi"[3*row + col + 1]
            ax.set_title(f"{letter}  {label}", loc="left", fontsize=11)
            fraction = primary.loc[code, "core_slope"]
            ax.text(.97, .92, (f"核心响应比 {fraction:.2f}" if cn else f"Core fraction {fraction:.2f}"),
                    transform=ax.transAxes, ha="right", color=INK, fontsize=10)
        water_ax.set(xlim=(0, 300), ylim=(0, .019),
                     xlabel="自释放起的分钟数" if cn else "Minutes from release",
                     ylabel="盐脉冲，窗内积分归一 (1/min)" if cn else "Salt response / window area (1/min)")
        water_ax.set_title(f"{'adg'[row]}  {names[pulse]}", loc="left", fontsize=11)
        water_ax.legend(frameon=False, fontsize=9, loc="upper right")
        if row == 0:
            axes[row, 1].legend(frameon=False, fontsize=8.4, loc="lower right")
    fig.suptitle("实测区分：河道输送与 DOC 途中处理" if cn else "Field separation of river transport and DOC processing",
                 x=.07, y=.975, ha="left", fontsize=17)
    fig.text(.07, .925, ("同一条河、三次同步盐 / 标记 DOC 释放；保留两个上下游采样点。点为原始测量，未填补 DOC 缺测。" if cn else
                        "One stream, three paired salt / labelled-DOC releases at both locations. Dots are original measurements; no lab imputation."), fontsize=10)
    fig.text(.07, .055, ("左列盐曲线采用相同的 0–300 min 观察窗归一；中、右列为各地点的实际浓度，纵轴范围不同。\n"
                        "保守参考沿用作者的盐–碳投入比例及校准；核心响应比为共测浓度的回归斜率，不是完整质量回收率。\n"
                        "所有时间以真实释放时刻为零；连接盐测量点仅展示观察形状。叶片浸出液下游尾部尚未完全返回背景。" if cn else
                        "Left: salt curves normalized over the same 0–300 min window. Middle/right: concentrations with panel-specific vertical ranges.\n"
                        "The conservative reference inherits the author salt/carbon calibration. Core fraction is a paired concentration slope, not full mass recovery.\n"
                        "All clocks start at actual release. Salt connections show the sampled shape; the downstream leachate salt tail remains incomplete."), fontsize=9, color="#53696E")
    save(fig, "conservative_and_doc_responses")

    fig, axes = plt.subplots(2, 2, figsize=(12.9, 9.3))
    fig.subplots_adjust(left=.095, right=.97, top=.83, bottom=.17, hspace=.44, wspace=.32)
    positions = np.arange(3)
    ax = axes[0, 0]
    for i, pulse in enumerate(PULSES):
        up, down = (primary.loc[f"b_{pulse}_{site}", "salt_centroid_min"] for site in ("up", "down"))
        ax.plot([up, down], [i, i], color="#BACACE", linewidth=2)
        ax.scatter([up, down], [i, i], c=[UP, DOWN], s=47, edgecolor="white", linewidth=.5)
        ax.text(down+4, i, f"+{down-up:.1f} min", va="center", fontsize=10)
    ax.set(xlim=(75, 211), yticks=positions, yticklabels=[names[p] for p in PULSES],
           xlabel="盐脉冲重心，自释放起 (min)" if cn else "Salt centroid, minutes from release")
    ax.set_title("a  下游晚到约一小时" if cn else "a  Downstream arrival is about one hour later", loc="left", fontsize=11)

    ax = axes[0, 1]
    width_changes = [(pairs.loc[p, "salt_duration80_ratio"]-1)*100 for p in PULSES]
    ax.bar(positions, width_changes, width=.55, color=DOWN, alpha=.87)
    for i, pulse in enumerate(PULSES):
        for gap, marker in ((20, "_"), (45, "x")):
            trial = metrics[metrics.pulse.eq(pulse) & metrics.max_gap_min.eq(gap)].set_index("site")
            value = (trial.loc["down", "salt_duration80_min"] / trial.loc["up", "salt_duration80_min"]-1)*100
            ax.scatter(i, value, marker=marker, color=INK, s=38, linewidth=.8, zorder=4)
        ax.text(i, width_changes[i]+2.7, f"+{width_changes[i]:.1f}%", ha="center", fontsize=10)
    ax.set(xticks=positions, xticklabels=[names[p] for p in PULSES], ylim=(0, 45),
           ylabel="盐脉冲中央 80% 持续时间变化 (%)" if cn else "Change in central 80% salt duration (%)")
    ax.set_title("b  晚到也伴随展宽" if cn else "b  The conservative response also broadens", loc="left", fontsize=11)
    ax.text(.02, .97, "横线 / 叉号：20 / 45 min 积分间隔" if cn else "Tick / cross: 20 / 45 min integration gap", transform=ax.transAxes, va="top", fontsize=8.5)

    ax = axes[1, 0]
    for i, pulse in enumerate(PULSES):
        for j, site in enumerate(("up", "down")):
            x = i+(-.16 if site == "up" else .16)
            color = UP if site == "up" else DOWN
            code = f"b_{pulse}_{site}"
            raw_values = sensitivity[(sensitivity.code == code) & sensitivity.policy.eq("original_points")].core_slope
            value = primary.loc[code, "core_slope"]
            ax.vlines(x, raw_values.min(), raw_values.max(), color=color, linewidth=2)
            ax.scatter(x, value, s=44, color=color, zorder=4,
                       label=("上游原始点" if cn else "Upstream original") if i == 0 and j == 0 else (("下游原始点" if cn else "Downstream original") if i == 0 else None))
            author_value = sensitivity[(sensitivity.code == code) & sensitivity.policy.eq("author_processed") & sensitivity.core_fraction.eq(.25)].core_slope.iloc[0]
            ax.scatter(x, author_value, marker="D", facecolors="white", edgecolors=INK, s=30, zorder=5,
                       label=("作者处理后" if cn else "Author processed") if i == 0 and j == 0 else None)
    ax.axhline(1, color="#9AAFB5", linewidth=1, linestyle="--")
    ax.set(xticks=positions, xticklabels=[names[p] for p in PULSES], ylim=(0, 1.15),
           ylabel="标记 DOC / 保守参考：核心浓度斜率" if cn else "Labelled DOC / conservative core slope")
    ax.set_title("c  碳响应低于盐示踪参考" if cn else "c  Carbon response is below its salt reference", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[1, 1]
    for i, pulse in enumerate(PULSES):
        value = pairs.loc[pulse, "down_up_core_fraction_ratio"]
        ax.scatter(i-.07, value, s=51, color=DOWN, zorder=4, label=("原始共测点" if cn else "Original paired points") if i == 0 else None)
        author_sub = sensitivity[sensitivity.code.str.startswith(f"b_{pulse}_")]
        author_sub = author_sub[author_sub.policy.eq("author_processed") & author_sub.core_fraction.eq(.25)].set_index("code")
        ratio = author_sub.loc[f"b_{pulse}_down", "core_slope"] / author_sub.loc[f"b_{pulse}_up", "core_slope"]
        ax.scatter(i+.07, ratio, marker="D", facecolors="white", edgecolors=INK, s=40,
                   label=("作者处理后" if cn else "Author processed") if i == 0 else None)
        ax.text(i, max(value, ratio)+.12, f"{value:.2f}", ha="center", fontsize=10)
    ax.axhline(1, color="#9AAFB5", linewidth=1, linestyle="--")
    ax.set(xticks=positions, xticklabels=[names[p] for p in PULSES], ylim=(0, 1.6),
           ylabel="下游 / 上游：盐归一 DOC 核心响应比" if cn else "Downstream / upstream DOC–salt core fraction")
    ax.set_title("d  相同河段，额外碳下降并不统一" if cn else "d  Extra carbon decline is not uniform in this reach", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("结构输送和碳处理，需要分开看" if cn else "Transport and carbon processing are separate operations",
                 x=.075, y=.975, ha="left", fontsize=17)
    fig.text(.075, .915, "绿色：上游采样点；橙色：下游采样点。三个实验日期均保留，来自同一河段。" if cn else
             "Teal: upstream sampling site. Orange: downstream sampling site. All three addition dates in the same reach are retained.", fontsize=10)
    fig.text(.075, .035, ("a/b 是 0–300 min 观察窗内的盐响应；c 竖线为背景 / 核心阈值敏感性范围，不是置信区间。\n"
                        "d < 1 表示 DOC 相对盐的核心响应进一步降低；> 1 不能单独证明 DOC 产生。\n"
                        "盐峰浓度下降同时包含稀释与展宽。这里只检验过程链条，未把同一段河道当作三种河网形态的比较。" if cn else
                        "a/b: salt response within the observed 0–300 min window. c: vertical ranges are background/core-threshold sensitivity, not confidence intervals.\n"
                        "d < 1 indicates an additional relative DOC response decline; > 1 alone does not demonstrate DOC production.\n"
                        "Salt concentration attenuation includes dilution and spreading. This reach tests the process link, not a ranking of three whole-network forms."), fontsize=9, color="#53696E")
    save(fig, "transport_and_carbon_decomposition")
    inputs = list(directory.iterdir()) + [Path(__file__)]
    receipt = {"input_hashes": {str(path): sha256_file(path) for path in inputs},
               "output_hashes": {str(path): sha256_file(path) for path in outputs},
               "original_lab_values_imputed": False, "whole_network_class_contrast": False}
    (out / f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Saved {len(outputs)} inspected-source figure files ({'Chinese' if cn else 'English'})")


if __name__ == "__main__":
    main()
