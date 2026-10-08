"""Display observed participation changes separately from DOC mixing outcomes."""

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

ROOT = Path("experiments/phase4_transfer/doc_river_active_structure_v1")
BLUE, GOLD, DARK = "#297B8B", "#CA8748", "#344F5A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "text.color": DARK, "axes.labelcolor": DARK, "pdf.fonttype": 42})
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(parents=True, exist_ok=True)
    inputs = [folder/"network_hydro_changes.csv", folder/"network_doc_changes.csv", folder/"group_summary.csv",
              ROOT/"coverage_sensitivity.csv", ROOT/"leave_system_out.csv"]
    hydro, doc, summary, coverage, leave_out = [pd.read_csv(p, dtype={"target": str, "huc4": str}) for p in inputs]
    paths = []

    def label(en, zh):
        return zh if cn else en

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out/f"{name}{'_cn' if cn else ''}.{extension}"
            fig.savefig(path, dpi=220, facecolor="white")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.1))
    fig.subplots_adjust(left=.17, right=.98, top=.76, bottom=.29, wspace=.43)
    fig.suptitle(label("The same river arrangement can carry a different distribution of water",
                       "同一河网，参与混合的来水结构会随流量变化"), x=.07, y=.965, ha="left", fontsize=17)
    fig.text(.07, .88, label("17 fixed monitored frontiers · 7 elongated and 10 broad networks · 6 overlap systems",
        "17 个固定监测支流组合 · 7 个细长型、10 个宽阔多支流型 · 6 个相互重叠系统"), fontsize=10)
    ax = axes[0]
    order = ("all", "class_1", "class_3", "broad_minus_elongated")
    names = [label("All networks", "全部河网"), label("Elongated", "细长型"), label("Broad /\ntributary-rich", "宽阔多支流型"),
             label("Broad minus\nelongated", "宽阔型减细长型")]
    for metric, offset, color, legend in (("effective_fraction_raw_change", -.1, BLUE, label("Raw change", "原始变化")),
            ("effective_fraction_adjusted_change", .1, GOLD, label("Season/time adjusted", "季节与时间调整后"))):
        sub = summary.loc[summary.population.eq("hydro_only") & summary.metric.eq(metric)].set_index("group").reindex(order)
        y = np.arange(4)+offset
        ax.hlines(y, 100*sub.ci_low, 100*sub.ci_high, color=color, linewidth=1.2)
        ax.scatter(100*sub.estimate, y, color=color, s=35, label=legend, zorder=3)
    ax.axvline(0, color="#83939A", linestyle="--", linewidth=1)
    ax.set(yticks=range(4), yticklabels=names,
           xlabel=label("High − low effective-contributor fraction (pp)", "高流量减低流量：有效参与比例（百分点）"))
    ax.invert_yaxis()
    ax.set_title(label("a  How evenly incoming water is distributed", "a  监测支流之间的供水均衡程度"), loc="left", fontsize=11)
    ax.legend(frameon=False, bbox_to_anchor=(0, -.20), loc="upper left", fontsize=9)

    ax = axes[1]
    for cluster, color, marker, name in ((1, BLUE, "o", label("Elongated (n=7)", "细长型（7 个）")),
                                       (3, GOLD, "s", label("Broad (n=10)", "宽阔型（10 个）"))):
        sub = hydro.loc[hydro.cluster.eq(cluster)]
        ax.scatter(100*sub.effective_fraction_adjusted_change, sub.path_sd_reference_adjusted_change,
                   color=color, marker=marker, s=38, label=name)
    ax.axhline(0, color="#83939A", linewidth=.9)
    ax.axvline(0, color="#83939A", linewidth=.9)
    ax.set(xlabel=label("Effective-contributor fraction change (pp)", "有效参与比例变化（百分点）"),
           ylabel=label("Path-SD change / fixed reference path mean", "路径分散变化 / 固定参考平均路径"))
    ax.set_title(label("b  Water balance and path spread differ", "b  供水均衡与路径分散是不同维度"), loc="left", fontsize=11)
    ax.legend(frameon=False, bbox_to_anchor=(0, -.20), loc="upper left", fontsize=9)
    fig.text(.07, .045, label("Effective fraction = [1 / sum(w²)] / monitored-source count; paths and physical branch counts are unchanged.\n95% intervals resample complete overlap systems, conditional on the fitted network changes.",
        "有效参与比例 = [1 / 各支流水量占比平方之和] / 监测支流数；实际河道和支流数量未改变。\n区间按完整重叠系统抽样，以本次各河网的拟合变化为条件。"), fontsize=9)
    save(fig, "water_participation_and_shape")

    doc = doc.sort_values(["cluster", "target"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.3, 9.4))
    fig.subplots_adjust(left=.15, right=.93, bottom=.19, top=.84)
    fig.suptitle(label("More balanced water does not determine the observed DOC mixing response",
                       "供水更均衡，并不能单独决定 DOC 混合后的波动"), x=.08, y=.965, ha="left", fontsize=17)
    fig.text(.08, .905, label("All 17 eligible networks · fixed-state-weight variance summary · analytic contributions, not carbon removal",
        "全部 17 个可比较河网 · 固定组内来水比例的方差摘要 · 分项贡献不是 DOC 去除量"), fontsize=10)
    positive, negative = np.zeros(len(doc)), np.zeros(len(doc))
    for metric, color, name in (("weights_contribution_pp", BLUE, label("Water fractions", "来水比例")),
            ("correlation_contribution_pp", GOLD, label("Branch coordination", "支流同步性")),
            ("sd_contribution_pp", "#A1B3BB", label("Branch fluctuation amplitudes", "支流波动幅度"))):
        values = doc[metric].to_numpy()
        left = np.where(values >= 0, positive, negative)
        ax.barh(np.arange(len(doc)), values, left=left, height=.64, color=color, label=name)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    ax.scatter(doc.mixing_potential_change_pp, range(len(doc)), marker="D", s=26, color=DARK,
               label=label("Total change", "总变化"), zorder=4)
    ax.axvline(0, color="#71838B", linewidth=1)
    split = int(doc.cluster.eq(1).sum())
    ax.axhline(split-.5, color="#D0D9DC", linewidth=1)
    ax.set(yticks=range(len(doc)), yticklabels=doc.target,
           xlabel=label("High − low change in fixed-weight mixing potential (pp)", "高减低：固定比例混合平缓程度的变化（百分点）"))
    ax.invert_yaxis()
    ax.text(1.005, .985, label("Elongated", "细长型"), transform=ax.transAxes, fontsize=9, va="top")
    ax.text(1.005, 1-split/len(doc)-.04, label("Broad", "宽阔型"), transform=ax.transAxes, fontsize=9, va="top")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.095), ncol=2, fontsize=10)
    fig.text(.08, .035, label("Each bar uses the same network's low/high observations. Contributions sum exactly; sampling and unmonitored water remain explicit.",
        "每行比较同一河网的高低流量观测；各项之和等于总变化。月份数据不能识别真实传输时间。"), fontsize=9)
    save(fig, "doc_mixing_contributions")

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.1))
    fig.subplots_adjust(left=.11, right=.96, top=.76, bottom=.27, wspace=.40)
    fig.suptitle(label("Monitoring coverage changes the population-level participation result",
                       "监测覆盖范围改变了来水参与变化的总体结果"), x=.07, y=.965, ha="left", fontsize=17)
    fig.text(.07, .88, label("Supplementary checks after the primary analysis · fixed within-network estimates",
        "主分析之后补做的检验 · 每个河网的原拟合结果保持不变"), fontsize=10)
    sub = coverage.loc[coverage.metric.eq("effective_fraction_adjusted_change")]
    ax = axes[0]
    ax.scatter(np.arange(len(sub)), 100*sub.estimate, s=48, color=BLUE, zorder=3)
    ax.axhline(0, color="#83939A", linewidth=1)
    for position, row in enumerate(sub.itertuples()):
        ax.annotate(label(f"n={row.n_networks}", f"{row.n_networks} 个河网"),
                    (position, 100*row.estimate), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set(xticks=range(len(sub)), xticklabels=["0%", "50%", "80%", "90%"],
           xlabel=label("Minimum represented catchment area", "监测支流覆盖的流域面积下限"),
           ylabel=label("Adjusted high − low effective fraction (pp)", "调整后高减低有效参与比例（百分点）"),
           ylim=(-1.5, 5.6), xlim=(-.4, len(sub)-.6))
    ax.set_title(label("a  The ≥80% subset has only four networks", "a  覆盖至少 80% 的子集仅有 4 个河网"), loc="left", fontsize=11)
    sub = leave_out.loc[leave_out.group.eq("all") & leave_out.metric.eq("effective_fraction_adjusted_change")]
    ax = axes[1]
    ax.scatter(range(len(sub)), 100*sub.estimate, s=48, color=GOLD, zorder=3)
    overall = summary.loc[summary.population.eq("hydro_only") & summary.group.eq("all") &
                          summary.metric.eq("effective_fraction_adjusted_change"), "estimate"].iloc[0]
    ax.axhline(100*overall, color=DARK, linestyle="--", linewidth=1,
               label=label("All 17 networks", "全部 17 个河网"))
    ax.axhline(0, color="#83939A", linewidth=1)
    ax.set(xticks=range(len(sub)), xticklabels=[str(int(v)+1) for v in sub.omitted_system],
           xlabel=label("Omitted overlap system", "逐次移除的重叠系统"),
           ylabel=label("Adjusted high − low effective fraction (pp)", "调整后高减低有效参与比例（百分点）"),
           ylim=(-1.5, 5.6), xlim=(-.4, len(sub)-.6))
    ax.set_title(label("b  No single overlap system creates the sign", "b  逐个移除系统后，整体方向保持为正"), loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.19), fontsize=9)
    fig.text(.07, .07, label("Points are descriptive means, without new confidence intervals. Area thresholds retain different networks;\nthey do not represent a within-network change in monitoring coverage or activation of unmeasured tributaries.",
        "各点为描述性均值，没有新置信区间。面积阈值保留的是不同河网；\n不能将它理解为同一河网改变监测覆盖范围后的效果，也不能据此判断未监测支流的参与。"), fontsize=9)
    save(fig, "coverage_and_system_sensitivity")
    (ROOT/f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "code": {str(Path(__file__).relative_to(Path.cwd())): sha256_file(Path(__file__))},
        "figures": {str(p): sha256_file(p) for p in paths}}, indent=2)+"\n")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
