"""Plot current/history flow evidence and its geographic structure transfer."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager

ROOT = Path("experiments/phase4_transfer/doc_river_flow_memory_v1")
POPULATIONS = {"all_source_months": "全部源资料", "observed_temperature": "控制实测温度", "source_months_since_2009": "2009年以来"}
COLORS = ["#547681", "#B8864C", "#2B9486"]


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT/"figures"/f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#253F49", "text.color": "#253F49"})
    (ROOT/"figures").mkdir(exist_ok=True)
    temporal = pd.read_csv(ROOT/"analysis"/"chronological_gains.csv")
    structure = pd.read_csv(ROOT/"analysis"/"descriptor_gains.csv")
    counts = pd.read_csv(ROOT/"analysis"/"population_counts.csv").set_index("population")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for j, space in enumerate(("log1p", "native")):
        ax = axes[j]
        for i, (population, label) in enumerate(POPULATIONS.items()):
            r = temporal[(temporal.population.eq(population)) & temporal.space.eq(space)
                         & temporal.candidate.eq("flow_memory") & temporal.reference.eq("current_flow")].iloc[0]
            ax.plot([r.ci_low_pct, r.ci_high_pct], [i, i], color=COLORS[i], linewidth=2)
            ax.plot(r.gain_pct, i, "o", color=COLORS[i], markersize=7)
            ax.annotate(f"{r.gain_pct:+.1f}%", (r.gain_pct, i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10)
        ax.axvline(0, color="#9DAAAA", linestyle="--", linewidth=1)
        ax.set(yticks=range(3), yticklabels=[f"{label}\n{int(counts.loc[p, 'n_chronological_stations'])}站" for p, label in POPULATIONS.items()],
               xlabel="加入前月流量后的 MAE 降幅（%）", ylim=(-.55, 2.6))
        ax.invert_yaxis()
        ax.set_title(f"{'a' if j == 0 else 'b'}  后期预测：{'log1p DOC' if j == 0 else '原浓度尺度'}", loc="left", fontsize=12)
    ax = axes[2]
    names = {"footprint": "外轮廓", "branching": "支流组织", "paths": "路径组织", "all_form": "全部结构"}
    for i, (population, label) in enumerate(POPULATIONS.items()):
        g = structure[structure.population.eq(population) & structure.response.eq("previous_response")].set_index("candidate")
        for j, name in enumerate(names):
            r = g.loc[name]
            y = j+(i-1)*.17
            ax.plot([r.ci_low_pct, r.ci_high_pct], [y, y], color=COLORS[i], linewidth=1.4)
            ax.plot(r.gain_pct, y, "o", color=COLORS[i], markersize=4, label=label if j == 0 else None)
    ax.axvline(0, color="#9DAAAA", linestyle="--", linewidth=1)
    ax.set(yticks=range(4), yticklabels=list(names.values()), xlabel="跨区域响应描述预测 MAE 降幅（%）")
    ax.invert_yaxis()
    ax.set_title("c  结构能否解释前月响应差异？", loc="left", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="upper center", ncol=3,
               bbox_to_anchor=(.52, 1.01))
    fig.subplots_adjust(left=.09, right=.99, bottom=.23, top=.82, wspace=.65)
    fig.text(.5, .06, "a/b：同站前期拟合、后期评分；c：整片 HUC4 留出。正值表示误差下降，线段为 5,000 次区域 bootstrap 的 95% 区间。\n这些是水文响应诊断，不是当前神经 DOC 模型的性能提升；前月关联不代表一个月的实际传输时间。", ha="center", fontsize=10)
    save(fig, "river_flow_memory_evidence_cn")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    labels = ("前期当月流量响应", "前期前月流量响应")
    late_labels = ("后期当月流量响应", "后期前月流量响应")
    ledger = pd.read_csv(ROOT/"analysis"/"chronological_eligibility.csv")
    s = ledger[ledger.population.eq("all_source_months") & ledger.status.eq("included") & ledger.late_status.eq("included")]
    for j, term in enumerate(("current", "previous")):
        ax = axes[j]
        x, y = s[f"train_{term}_response"], s[f"late_{term}_response"]
        ax.scatter(x, y, s=30, color="#547681", alpha=.7, edgecolors="white", linewidths=.4)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#A2AEAC", linewidth=1)
        ax.axhline(0, color="#D9DFDF", linewidth=.8)
        ax.axvline(0, color="#D9DFDF", linewidth=.8)
        ax.set(xlabel=labels[j], ylabel=late_labels[j])
        fraction = np.mean((x > 0) == (y > 0))
        ax.set_title(f"{'a' if j == 0 else 'b'}  跨时段响应稳定性：{fraction:.0%} 同号", loc="left", fontsize=12)
    fig.text(.5, -.005, f"{len(s)} 个站点具有前后期可识别响应；系数为 log1p(DOC) 对 log(Q) 的偏响应，控制季节与趋势。", ha="center", fontsize=10)
    fig.tight_layout(w_pad=3)
    save(fig, "river_flow_response_stability_cn")


if __name__ == "__main__":
    main()
