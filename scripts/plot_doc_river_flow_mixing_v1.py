"""Show branch mixing, measured outlet variation and incomplete flow coverage."""

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

ROOT = Path("experiments/phase4_transfer/doc_river_flow_mixing_v1")
COLORS = {"C7": "#257F88", "C9": "#C5814A"}


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
    analysis, output = ROOT / "analysis", ROOT / "figures"
    output.mkdir(exist_ok=True)
    comparison = pd.read_csv(analysis / "mixing_comparison.csv")
    samples = pd.read_parquet(analysis / "complete_mixing_campaigns.parquet")
    summary = json.loads((analysis / "summary.json").read_text())
    suffix, outputs = ("_cn" if cn else ""), []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = output / f"{name}{suffix}.{extension}"
            fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.1))
    fig.subplots_adjust(left=.075, right=.97, bottom=.14, top=.86, hspace=.53, wspace=.33)
    sub = comparison[comparison.receiver.eq("C7")].sort_values("year")
    stages = ["mean_upstream_cv", "dynamic_mix_cv", "receiver_cv"]
    ax = axes[0, 0]
    for row in sub.itertuples():
        ax.plot(range(3), [getattr(row, name) for name in stages], color="#BAC8CA", alpha=.8,
                linewidth=1, marker="o", markersize=3)
    ax.plot(range(3), sub[stages].median().to_numpy(), color=COLORS["C7"], linewidth=2,
            marker="o", markersize=6, label="七年中位数" if cn else "Median across seven years")
    ax.set(xticks=range(3), xticklabels=("支流平均", "流量加权混合", "实际出口") if cn else
           ("Branch average", "Flow-weighted mix", "Observed outlet"), ylabel="DOC 相对波动（SD / 均值）" if cn else "Relative DOC variation (SD / mean)",
           ylim=(0, max(sub[stages].max()) * 1.1))
    ax.set_title("a  混合能产生较小的 DOC 波动" if cn else "a  Mixing can produce lower DOC variation", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[0, 1]
    ordered = comparison.sort_values(["receiver", "year"])
    for i, row in enumerate(ordered.itertuples()):
        values = samples[(samples.receiver.eq(row.receiver)) & (samples.year.eq(row.year))].known_upstream_flow_share
        jitter = np.random.default_rng(42+i).uniform(-.11, .11, len(values))
        ax.scatter(values, i+jitter, color=COLORS[row.receiver], s=15, alpha=.6)
        ax.scatter(row.flow_share_median, i, marker="|", s=110, color="#263B42", zorder=5)
    ax.axvline(1, color="#80959C", linewidth=1, linestyle="--")
    ax.set(yticks=range(len(ordered)), yticklabels=[f"{r.receiver} · {r.year}" for r in ordered.itertuples()],
           xlim=(0, max(1.05, ordered.flow_share_max.max()+.05)),
           xlabel="已测上游流量 / 出口流量" if cn else "Measured upstream flow / outlet flow")
    ax.set_title("b  已测支流覆盖部分来水" if cn else "b  Monitored branches cover part of the flow", loc="left", fontsize=11)
    ax.invert_yaxis()

    ax = axes[1, 0]
    for receiver, group in comparison.groupby("receiver"):
        ax.scatter(group.dynamic_mix_sd, group.receiver_sd, color=COLORS[receiver], s=48,
                   label=f"{receiver} (n={len(group)})", edgecolor="white", linewidth=.6)
    maximum = max(comparison.dynamic_mix_sd.max(), comparison.receiver_sd.max()) * 1.1
    ax.plot([0, maximum], [0, maximum], color="#82969E", linestyle="--", linewidth=1)
    ax.set(xlim=(0, maximum), ylim=(0, maximum),
           xlabel="混合 DOC 标准差（mg C/L）" if cn else "Mixture DOC SD (mg C/L)",
           ylabel="出口 DOC 标准差（mg C/L）" if cn else "Outlet DOC SD (mg C/L)")
    ax.set_title("c  出口不总是比混合结果更平稳" if cn else "c  Outlet SD varies around the mixture", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1, 1]
    row = sub.sort_values("year").iloc[0]
    example = {"receiver": row.receiver, "year": int(row.year), "selection": "Earliest complete C7 window"}
    points = samples[(samples.receiver.eq(row.receiver)) & samples.year.eq(row.year)]
    peak = pd.Timestamp(row.peak_date)
    x = points.date_local.sub(peak).dt.total_seconds() / 86400
    for column, color, marker, label in (
        ("doc_a", "#A9B5B9", "o", row.source_a),
        ("doc_b", "#A9B5B9", "^", row.source_b),
        ("doc_dynamic_mix", "#C5814A", "D", "计算混合值" if cn else "Calculated mixture"),
        ("doc_receiver", COLORS["C7"], "s", "实测出口" if cn else "Observed outlet"),
    ):
        ax.scatter(x, points[column], color=color, marker=marker, s=36, label=label,
                   edgecolor="white", linewidth=.6)
    ax.axvline(0, color="#B5C3C6", linestyle=":", linewidth=1)
    ax.set(xlim=(-15, 15), xlabel="相对流量峰日（天）" if cn else "Days from flow maximum", ylabel="DOC (mg C/L)")
    ax.set_title(f"d  {row.source_a} + {row.source_b} → {row.receiver} · {int(row.year)}", loc="left", fontsize=11, pad=39)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), borderaxespad=0)
    fig.suptitle("支流混合与出口 DOC 波动" if cn else "Branch mixing and observed outlet DOC variation", x=.075, y=.985, ha="left", fontsize=17)
    fig.text(.075, .924, (f"{summary['n_windows_with_matched_flow']} 个窗口、{summary['n_campaigns_in_variation_comparison']} 轮 DOC 与流量匹配采样；其中七个窗口在同一汇流点 C7。" if cn else
             f"{summary['n_windows_with_matched_flow']} windows and {summary['n_campaigns_in_variation_comparison']} matched DOC/flow campaigns; seven windows share confluence C7."), fontsize=10)
    fig.text(.075, .06, ("a：灰线为各年份，彩线为中位数；b：每点是一轮采样，黑线为窗口中位数；c：虚线为出口与混合 SD 相等。\n"
              "每日流量用于实验室采样日的加权；这是已测支流的部分混合比较。d 为最早可用的 C7 窗口，DOC 点不插值。" if cn else
              "a: Thin lines show years, teal shows the median. b: Points show campaigns, black marks window medians. c: Dashed line denotes equal SD.\n"
              "Daily flow weights laboratory sample dates; this is a partial-branch comparison. d shows the earliest eligible C7 window, with no DOC interpolation."), fontsize=9, color="#53696E")
    save(fig, "branch_mixing_and_outlet")

    fig, ax = plt.subplots(figsize=(9.7, 5.8))
    fig.subplots_adjust(left=.16, right=.97, bottom=.21, top=.80)
    for i, row in enumerate(ordered.itertuples()):
        diag, cov = row.individual_variance_term, row.covariance_term
        ax.barh(i, diag, height=.38, color="#CBD5D8")
        start = diag if cov >= 0 else diag + cov
        ax.barh(i, abs(cov), left=start, height=.38, color="#C5814A" if cov >= 0 else COLORS["C7"])
        ax.scatter(row.fixed_mix_variance, i, color="#263B42", marker="|", s=160, zorder=4)
    ax.set(yticks=range(len(ordered)), yticklabels=[f"{r.receiver} · {r.year}" for r in ordered.itertuples()],
           xlabel="固定权重混合的方差贡献（mg C/L）²" if cn else "Fixed-weight mixture variance contributions (mg C/L)²")
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    fig.suptitle("支流同步性如何改变混合波动" if cn else "How branch synchrony changes mixture variation", x=.08, y=.985, ha="left", fontsize=17)
    fig.text(.08, .885, "灰色：支流各自方差；橙色：共同增减增加方差；青色：反向变化抵消方差；黑线：最终混合方差。" if cn else
             "Grey: individual variance terms. Orange: positive covariance. Teal: negative covariance. Black: resulting mixture variance.", fontsize=10)
    fig.text(.08, .045, ("固定权重取该窗口的平均流量比例；Var(mix) = w² Var(A) + (1−w)² Var(B) + 2w(1−w) Cov(A,B)。\n"
              "这是同轮实测浓度的方差分解，反映支流信号的同步性，不估计传播滞后。" if cn else
              "Weights are each window's mean flow shares. Var(mix) = w² Var(A) + (1−w)² Var(B) + 2w(1−w) Cov(A,B).\n"
              "This decomposes coeval observed concentrations and branch synchrony; it does not estimate propagation lag."), fontsize=9, color="#53696E")
    save(fig, "branch_covariance_decomposition")
    sources = [analysis / name for name in ("mixing_comparison.csv", "complete_mixing_campaigns.parquet", "summary.json")]
    sources += [Path("scripts/plot_doc_river_flow_mixing_v1.py")]
    receipt = {"input_hashes": {str(p): sha256_file(p) for p in sources},
               "output_hashes": {str(p): sha256_file(p) for p in outputs},
               "example": example,
               "chinese": cn, "doc_interpolation": False}
    (output / f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Saved {len(outputs)} figures")


if __name__ == "__main__":
    main()
