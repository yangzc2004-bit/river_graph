"""Standalone figures of river pathways, observed receiving variance and flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_pathway_context_v1 import ROOT, TYPES
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FuncFormatter, NullFormatter

from river_graph.experiments.provenance import sha256_file

COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chinese", action="store_true")
    p.add_argument("--shortest-routes", action="store_true")
    args = p.parse_args()
    cn, suffix = args.chinese, "_cn" if args.chinese else ""
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    data, out = root/"analysis", root/"figures"
    out.mkdir(exist_ok=True)
    f = pd.read_csv(data/"receiver_pathway_panel.csv", dtype=TYPES)
    coverage = pd.read_csv(data/"coverage_opportunities.csv")
    flow = pd.read_csv(data/"receiving_variance_budgets.csv", dtype=TYPES)
    names = {1: "细长多支流" if cn else "Elongated", 2: "主干主导" if cn else "Mainstem dominated",
             3: "宽阔多支流" if cn else "Broad"}
    written = []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.7, 10.2))
    for c in (1, 2, 3):
        g = coverage[coverage.cluster.eq(c)]
        axes[0, 0].plot(g.minimum_coverage*100, g.n_receivers, marker="o", color=COLORS[c], label=names[c])
    axes[0, 0].axvline(80, color="#859299", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="最低监测面积覆盖 (%)" if cn else "Minimum represented catchment area (%)",
        ylabel="可分析的出口河网数" if cn else "Eligible receiving networks", ylim=(0, 23),
        title="a  河网形态的可比观测" if cn else "a  Form-level sampling opportunity")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for index, (field, xlabel) in enumerate([
            ("monitored_common_fraction", "汇入后共有河段 / 平均路径" if cn else "Shared downstream corridor / mean path"),
            ("monitored_path_cv", "监测站至出口的路径长短差异 (CV)" if cn else "Gauge-to-receiver path dispersion (CV)"),
            ("monitored_storage_length_fraction", "路径中已标注湖泊/水库长度比例" if cn else "Mapped lake/reservoir fraction of paths")]):
        ax = axes.flat[index+1]
        for c in (1, 2, 3):
            for high, marker in ((False, "o"), (True, "s")):
                g = f[f.cluster.eq(c) & f.covered_area_fraction.ge(.8).eq(high)]
                ax.scatter(g[field], np.exp(g.outlet_mix_log_sd_ratio), s=40 if high else 32,
                    marker=marker, facecolor=COLORS[c] if high else "white", edgecolor=COLORS[c], linewidth=1, alpha=.9)
        ax.axhline(1, color="#859299", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="出口波动 / 支流混合波动 (SD)" if cn else "Receiving / upstream-mixture SD",
               yscale="log")
        title = ["b  汇入后的共同路径", "c  支流路径长短组合", "d  沿途已标注的蓄水水体"] if cn else [
            "b  Shared downstream path", "c  Arrangement of source-path lengths", "d  Tagged corridor storage"]
        ax.set_title(title[index], loc="left")
    fig.text(.09, .015, "实心方块：监测覆盖 ≥80%；空心圆：其余河网。无湖泊标签不等于没有湖泊。" if cn else
        "Filled squares: area coverage >=80%; open circles: remaining networks. No lake tag does not establish absence.", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    save(fig, "river_pathways_receiving_variability")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.9), gridspec_kw={"width_ratios": [1.3, 1.]})
    g = f[f.covered_area_fraction.ge(.8)].sort_values(["cluster", "target"])
    y = np.arange(len(g))
    ax = axes[0]
    ratios = np.exp(g.outlet_mix_log_sd_ratio.to_numpy())
    ax.barh(y, ratios, color=[COLORS[c] for c in g.cluster], height=.67)
    for i, value in enumerate(ratios):
        ax.text(value+.06, i, f"{value:.2f}", va="center", fontsize=9)
    ax.axvline(1, color="#273F48", linewidth=1, linestyle="--")
    ax.set(yticks=y, yticklabels=[f"{r.target} · {names[r.cluster]}" for r in g.itertuples()],
        xlim=(0, ratios.max()*1.2),
        xlabel="出口波动 / 支流混合波动（<1 更平缓，>1 更强）" if cn else "Receiving / mixture SD (<1 smoother; >1 larger)")
    ax.invert_yaxis()
    ax.set_title("a  高覆盖河网：出口更平缓还是更强？" if cn else "a  High-coverage networks: smoother or larger?", loc="left")
    ax = axes[1]
    flow = flow[flow.subset.eq("positive_common_flow")]
    area = flow[flow.weighting.eq("fixed_area")].set_index("target")
    q = flow[flow.weighting.eq("monthly_flow")].set_index("target").reindex(area.index)
    for c in (1, 2, 3):
        ids = area.index[area.cluster.eq(c)]
        ax.scatter(np.exp(area.loc[ids, "outlet_mix_log_sd_ratio"]),
            np.exp(q.loc[ids, "outlet_mix_log_sd_ratio"]), color=COLORS[c], s=35, label=names[c])
    maximum = max(np.exp(area.outlet_mix_log_sd_ratio).max(), np.exp(q.outlet_mix_log_sd_ratio).max())*1.15
    minimum = min(np.exp(area.outlet_mix_log_sd_ratio).min(), np.exp(q.outlet_mix_log_sd_ratio).min())/1.15
    ax.plot([minimum, maximum], [minimum, maximum], color="#859299", linestyle="--")
    ax.axvline(1, color="#B6C0C3", linewidth=.8)
    ax.axhline(1, color="#B6C0C3", linewidth=.8)
    ax.set(xscale="log", yscale="log", xlim=(minimum, maximum), ylim=(minimum, maximum),
        xlabel="面积权重：出口 / 混合波动 (SD)" if cn else "Area shares: receiving / mixture SD",
        ylabel="月流量权重：出口 / 混合波动 (SD)" if cn else "Monthly flow shares: receiving / mixture SD")
    ticks = [v for v in (.2, .3, .5, 1., 2., 4., 6.) if minimum <= v <= maximum]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("b  同一批月份更换混合权重" if cn else "b  Change shares on identical observed months", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.text(.08, .014, "差值包含未监测输入、变动流量及沿途过程；不能单独识别它们。月均流量不代表事件同步混合。" if cn else
        "Mismatch combines unmonitored inputs, variable shares and route processes. Monthly flow is not synchronized event mixing.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1), w_pad=2)
    save(fig, "receiving_variance_and_flow_sensitivity")
    contrasts = pd.read_csv(data/"form_contrasts.csv")
    contrasts = contrasts[contrasts.metric.eq("outlet_mix_log_sd_ratio")].sort_values("minimum_coverage")
    pairs = pd.read_csv(data/"same_region_form_pairs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.1), gridspec_kw={"width_ratios": [1.15, 1.]})
    ax = axes[0]
    for i, r in enumerate(contrasts.itertuples()):
        point, low, high = np.exp([r.estimate, r.ci_low, r.ci_high])
        ax.plot([low, high], [i, i], color="#66749F", linewidth=2)
        ax.scatter(point, i, color="#66749F", s=42)
    ax.axvline(1, color="#273F48", linestyle="--", linewidth=1)
    ax.set(yticks=np.arange(len(contrasts)), yticklabels=[
        f"{r.minimum_coverage*100:g}%  ·  {int(r.n_elongated)} / {int(r.n_broad)}" for r in contrasts.itertuples()],
        xlabel="宽阔型 / 细长型：出口相对上游的波动比例" if cn else "Broad / elongated: receiving-to-mixture SD ratio",
        ylabel="最低覆盖 · 细长 / 宽阔数量" if cn else "Minimum coverage · elongated / broad counts",
        title="a  观测中的形态差异线索" if cn else "a  Observed form contrast across coverage cuts",
        xscale="log", xlim=(.12, 1.65))
    ax.set_xticks([.2, .4, .7, 1., 1.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.invert_yaxis()
    counts = [len(pairs), int(pairs.area_comparable.sum()), int((pairs.area_comparable & pairs.both_observed).sum())]
    ax = axes[1]
    labels = ["同一区域的形态配对候选", "其中汇水面积相近", "双方均有共同支流—出口观测"] if cn else [
        "Same-HUC4 form-pair candidates", "With area ratio <=2", "Both have common source/outlet DOC"]
    ax.barh(np.arange(3), counts, color=["#D0DBDF", "#799BA4", "#66749F"], height=.6)
    for i, value in enumerate(counts):
        ax.text(value+7, i, str(value), va="center", fontsize=11)
    ax.set(yticks=np.arange(3), yticklabels=labels, xlim=(0, max(counts)*1.18),
        xlabel="河网配对候选数量（可重复使用站点）" if cn else "Candidate pairs (stations can recur)",
        title="b  相同地区的比较机会" if cn else "b  Comparison opportunity within the same region")
    ax.invert_yaxis()
    fig.text(.055, .018, "区间重抽整片河网系统 5,000 次；形态差异尚混有地域差异，不能单独归因于形态。" if cn else
        "Intervals resample whole catchment systems 5,000 times. Form and regional context remain confounded.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1), w_pad=2.8)
    save(fig, "whole_form_signal_and_comparison_gap")
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "input_hashes": {str(path): sha256_file(path) for path in (data/"receiver_pathway_panel.csv",
            data/"coverage_opportunities.csv", data/"receiving_variance_budgets.csv",
            data/"form_contrasts.csv", data/"same_region_form_pairs.csv")},
        "output_hashes": {str(path): sha256_file(path) for path in written},
        "description": "Actual observed receiver panels; cropped river paths, variance identity and same-date flow sensitivity"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
