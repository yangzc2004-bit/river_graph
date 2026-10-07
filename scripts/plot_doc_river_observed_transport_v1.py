"""Figures comparing real river-path arrival information in observed DOC."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_observed_transport_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")
ORDER = ("background", "same_month", "uniform_history", "mean_delay", "branch_arrival")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 220})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix, written = "_cn" if cn else "", []
    sources = [a/f"{n}.csv" for n in ("operator_summary", "paired_gains", "connection_metrics", "receiver_metrics", "receiver_coverage", "selection_trials")]
    summary = pd.read_csv(a/"operator_summary.csv", dtype={"group": str})
    gains = pd.read_csv(a/"paired_gains.csv", dtype={"group": str})
    connections = pd.read_csv(a/"connection_metrics.csv", dtype={"target": str, "huc4": str})
    receivers = pd.read_csv(a/"receiver_metrics.csv", dtype={"target": str, "huc4": str})
    coverage = pd.read_csv(a/"receiver_coverage.csv", dtype={"target": str, "huc4": str})
    labels = ("时间、水文与面积背景", "同月上游混合", "统一前月历史权重", "按平均路径统一滞后", "按两条路径分别滞后") if cn else (
        "Calendar / hydro / area", "Same-month upstream mix", "Uniform history weight", "Mean-path delay", "Branch-specific arrival")
    labels = dict(zip(ORDER, labels, strict=True))

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    def gain(candidate, reference):
        return gains[gains.candidate.eq(candidate) & gains.reference.eq(reference) & gains.group.eq("all") &
                     gains.metric.eq("mae") & gains.unit.eq("component")].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.5))
    ax = axes[0]
    s = summary[summary.group.eq("all") & summary.metric.eq("mae") & summary.unit.eq("component")]
    for i, op in enumerate(ORDER):
        r = s[s.operator.eq(op)].iloc[0]
        color = "#8A9CA1" if i == 0 else COLORS[(i-1) % 3]
        ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=2)
        ax.scatter(r.estimate, i, color=color, s=36, zorder=3)
        ax.annotate(f"{r.estimate:.3f}", (r.estimate, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set(yticks=range(5), yticklabels=[labels[x] for x in ORDER], ylim=(4.6, -.6),
           xlabel="下游 DOC MAE（mg/L）" if cn else "Downstream DOC MAE (mg/L)")
    ax.set_title("a  "+("已知上游 DOC 明显有帮助" if cn else "Known upstream DOC adds information"), loc="left", pad=17)
    ax = axes[1]
    candidates = ("uniform_history", "mean_delay", "branch_arrival")
    for i, op in enumerate(candidates):
        r = gain(op, "same_month")
        ax.plot([r.gain_ci_low_pct, r.gain_ci_high_pct], [i, i], color=COLORS[i], linewidth=2)
        ax.scatter(r.relative_reduction_pct, i, color=COLORS[i], s=36)
        ax.annotate(f"{r.relative_reduction_pct:+.2f}%", (r.relative_reduction_pct, i),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(0, color="#9CAAAF", linestyle="--", linewidth=.9)
    ax.set(yticks=range(3), yticklabels=[labels[x] for x in candidates], ylim=(2.6, -.6),
           xlabel="相对同月混合的 MAE 降低（%）" if cn else "MAE reduction vs same-month mix (%)")
    ax.set_title("b  "+("历史信息带来小幅改善" if cn else "History adds a small improvement"), loc="left", pad=17)
    ax = axes[2]
    for i, reference in enumerate(("mean_delay", "uniform_history")):
        r = gain("branch_arrival", reference)
        ax.plot([r.gain_ci_low_pct, r.gain_ci_high_pct], [i, i], color=COLORS[2], linewidth=2)
        ax.scatter(r.relative_reduction_pct, i, color=COLORS[2], s=36)
    ax.axvline(0, color="#9CAAAF", linestyle="--", linewidth=.9)
    ax.set(yticks=range(2), yticklabels=("对比平均路径滞后", "对比统一历史权重") if cn else (
           "vs mean-path delay", "vs uniform history"), ylim=(1.6, -.6),
           xlabel="两路径分别滞后的额外改善（%）" if cn else "Extra branch-specific MAE reduction (%)")
    ax.set_title("c  "+("路径差异尚未增加预测信息" if cn else "Added path-imbalance value unresolved"), loc="left", pad=17)
    fig.suptitle("实测 DOC 支持哪一种河网信息？" if cn else "Which river information is supported by observed DOC?",
                 x=.035, ha="left", y=.995, fontsize=17)
    fig.text(.035, .91, "相同评价月份、相同背景模型；按共享监测系统留出，滞后参数只在其余系统内选择。" if cn else
             "Identical evaluation months and shared background; each monitored system held out, delay selected only in the other systems.", fontsize=9)
    fig.text(.035, .025, ("59 个支流组合 / 22 个下游站 / 11 个共享监测系统；3,026 个组合–月对应 1,092 个不同下游站–月观测。\n"
              "点为下游站等权结果，线为配对系统 bootstrap 95% 区间（a 为单方法区间）。上游 DOC 已知；不是无观测站点预测。") if cn else
             ("59 tributary pairs / 22 receivers / 11 monitored systems; 3,026 pair-months represent 1,092 unique receiver-month truths.\n"
              "Receiver-equal points; paired-system bootstrap 95% (a: single-operator intervals). Upstream DOC is known; this is not unmonitored K0."), fontsize=8)
    fig.subplots_adjust(left=.155, right=.985, top=.78, bottom=.21, wspace=1.12)
    save(fig, "observed_doc_arrival_comparison")

    names = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated / tributary-rich", "Mainstem dominated / sparse", "Broad / tributary-rich")
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 5.7))
    p = receivers.pivot(index="target", columns="operator", values="log_mae").reset_index().merge(
        coverage, on="target", validate="one_to_one")
    ax = axes[0]
    for i, cluster in enumerate((1, 2, 3)):
        s = p[p.cluster.eq(cluster)].sort_values("component")
        ax.scatter(i+np.random.default_rng(42).uniform(-.17, .17, len(s)),
                   100*(1-s.branch_arrival/s.same_month), color=COLORS[i], s=35, edgecolor="white", linewidth=.5)
        ax.text(i, .98, f"n={len(s)} / systems={s.component.nunique()}" if not cn else f"{len(s)}站 / {s.component.nunique()}系统",
                ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.axhline(0, color="#A0AEB2", linestyle="--", linewidth=.9)
    ax.set(xticks=range(3), xticklabels=("细长", "少支流", "宽展") if cn else ("Elongated", "Sparse", "Broad"),
           ylabel="相对同月混合的 log1p MAE 降低（%）" if cn else "Log1p MAE reduction vs same-month mix (%)")
    ax.set_title("a  "+("不同形态：独立系统不均衡" if cn else "Forms have unequal system replication"), loc="left", pad=17)
    c = connections.pivot(index="pair_id", columns="operator", values="log_mae").reset_index().merge(
        connections[connections.operator.eq("same_month")][["pair_id", "target", "cluster", "path_difference_scaled", "branch_balance"]],
        on="pair_id", validate="one_to_one")
    for col, feature in enumerate(("path_difference_scaled", "branch_balance"), start=1):
        ax = axes[col]
        for i, cluster in enumerate((1, 2, 3)):
            s = c[c.cluster.eq(cluster)]
            ax.scatter(s[feature], 100*(1-s.branch_arrival/s.mean_delay), color=COLORS[i],
                       s=28, alpha=.72, edgecolor="white", linewidth=.3, label=names[i])
        ax.axhline(0, color="#A0AEB2", linestyle="--", linewidth=.9)
        ax.set(xlabel=("两条路径长度差 / √流域面积" if cn else "Path difference / √basin area") if col == 1 else (
            "两支流面积份额平衡度（4w(1−w)）" if cn else "Branch-area balance: 4w(1−w)"),
            ylabel="分别滞后 vs 平均滞后：log1p MAE 降低（%）" if cn else "Branch vs mean-delay log1p MAE reduction (%)")
        ax.set_title(f"{'abc'[col]}  "+(("路径不均衡的增量", "支流平衡的增量")[col-1] if cn else (
            "Information in unequal paths", "Information in branch balance")[col-1]), loc="left", pad=17)
    handles, legends = axes[2].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(.52, .89), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("河网形态与实测增益：哪些部分仍需更密的观测？" if cn else "River form and measured gains: where do we need denser observations?",
                 x=.035, ha="left", y=.995, fontsize=16)
    fig.text(.035, .025, ("a 每点一个下游站；b–c 每点一个支流组合，59 点不是 59 个独立系统。细长型 6 个站全部属于同一系统，少支流型仅 1 站。\n"
              "每条真实路径仅读取本月和前月；所有几何滞后均选到候选上界。图中权重不是实测流速，形态差异不能由点数直接推断。") if cn else
             ("a: one receiver/point; b–c: one tributary pair/point, not 59 independent systems. All six elongated receivers share one system; sparse form has one receiver.\n"
              "Current/previous monthly bins only; every geometry delay reached the candidate upper bound. These weights are not measured channel velocities."), fontsize=8)
    fig.subplots_adjust(left=.085, right=.985, top=.74, bottom=.23, wspace=.49)
    save(fig, "river_form_and_observed_gains")
    manifest = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in sources},
                "figure_hashes": {str(p): sha256_file(p) for p in written}, "generator_sha256": sha256_file(Path(__file__)),
                "interpretation": "Held-out observed DOC diagnostic; monthly path-guided weights, not physical velocities or unmonitored K0."}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
