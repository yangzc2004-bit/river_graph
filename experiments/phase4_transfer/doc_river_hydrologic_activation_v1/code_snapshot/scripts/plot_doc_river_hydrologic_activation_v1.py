"""Render monthly DOC-flow responses and source-position moderation."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_hydrologic_activation_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")
POPULATIONS = ("all_source_months", "observed_temperature", "source_months_since_2009")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "text.color": "#263B42", "axes.labelcolor": "#263B42",
                         "axes.spines.right": False, "axes.spines.top": False, "savefig.dpi": 230})
    out = ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix = "_cn" if cn else ""
    written = []
    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    classes = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated\ntributary-rich", "Mainstem dominated\nsparse", "Broad\ntributary-rich")
    population_labels = ("全部源月份", "控制温度", "2009 年以后") if cn else (
        "All source months", "Temperature adjusted", "Since 2009")
    fits = pd.read_csv(ROOT/"analysis/station_response_fits.csv", dtype={"station": str})
    metadata = pd.read_csv("experiments/phase4_transfer/doc_river_source_placement_v1/analysis/station_doc_source_panel.csv",
                           dtype={"station": str})
    fits = fits.merge(metadata[["station", "cluster"]], on="station", validate="many_to_one")
    descriptive = pd.read_csv(ROOT/"analysis/class_response_distributions.csv")
    adjusted = pd.read_csv(ROOT/"analysis/adjusted_class_responses.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    rng = np.random.default_rng(42)
    primary = fits[fits.population.eq(POPULATIONS[0]) & fits.response_status.eq("included")]
    for i, group in enumerate((1, 2, 3)):
        s = primary[primary.cluster.eq(group)]
        x = i+rng.uniform(-.21, .21, len(s))
        axes[0].scatter(x, s.interquartile_response, s=16, alpha=.5, color=COLORS[i])
        r = descriptive[descriptive.population.eq(POPULATIONS[0]) & descriptive.cluster.eq(group)
                        & descriptive.metric.eq("interquartile_response")].iloc[0]
        axes[0].plot([i, i], [r.ci_low, r.ci_high], color="#263B42", linewidth=2)
        axes[0].scatter(i, r["median"], s=45, facecolor="white", edgecolor="#263B42", zorder=4)
        axes[0].text(i, 1.01, f"n={len(s)}", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=9)
    axes[0].axhline(0, color="#879295", linewidth=.8, linestyle="--")
    axes[0].set(xticks=range(3), xticklabels=classes,
                ylabel="流量 Q25→Q75 时的 log1p DOC 变化" if cn else "Change in log1p DOC: flow Q25 to Q75")
    axes[0].set_title("站点内响应" if cn else "Within-station responses", loc="left", pad=27)
    for j, (population, title) in enumerate(zip(POPULATIONS, population_labels, strict=True)):
        for i, group in enumerate((1, 2, 3)):
            r = adjusted[adjusted.population.eq(population) & adjusted.cluster.eq(group)].iloc[0]
            x = i+(j-1)*.18
            axes[1].plot([x, x], [r.ci_low, r.ci_high], color=COLORS[j], linewidth=1.3)
            axes[1].scatter(x, r.adjusted_cq_linear, s=28, color=COLORS[j], marker=("o", "s", "^")[j], label=title if i == 0 else None)
    axes[1].axhline(0, color="#879295", linewidth=.8, linestyle="--")
    axes[1].set(xticks=range(3), xticklabels=classes,
                ylabel="调整后的局地 C–Q 斜率" if cn else "Adjusted local C–Q slope")
    axes[1].set_title("同一环境条件下的形态比较" if cn else "Classes at a common environment", loc="left", pad=27)
    axes[1].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("三类河网的 DOC 随流量如何变化？" if cn else "How does DOC respond to flow in three river planforms?",
                 x=.04, ha="left", fontsize=17)
    fig.text(.04, .025, "各站点去除季节与趋势。左：210 站点、13,890 月份配对及中位数；右：环境调整。区间为 HUC4 bootstrap 95%。" if cn else
             "Station-specific season/trend removed. Left: 210 stations, 13,890 pairs and medians. Right: adjusted classes. HUC4 bootstrap 95%.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.77, wspace=.35)
    save(fig, "planform_flow_responses")

    moderation = pd.read_csv(ROOT/"analysis/hydrologic_moderation.csv")
    terms = ("wetland", "forest", "wetland_distance_ratio", "forest_distance_ratio",
             "wetland_near_excess", "forest_near_excess", "wetland_riparian_enrichment", "forest_riparian_enrichment")
    labels = ("湿地总量", "森林总量", "湿地距离比", "森林距离比", "湿地近端集中", "森林近端集中", "湿地河岸富集", "森林河岸富集") if cn else (
        "Wetland amount", "Forest amount", "Wetland distance ratio", "Forest distance ratio",
        "Wetland near-outlet excess", "Forest near-outlet excess", "Wetland riparian enrichment", "Forest riparian enrichment")
    data = moderation[moderation.model.eq("source_shape_moderation") & moderation.term.isin(terms)]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.3), sharey=True)
    for j, (ax, population, title) in enumerate(zip(axes, POPULATIONS, population_labels, strict=True)):
        s = data[data.population.eq(population)].set_index("term")
        for i, term in enumerate(terms):
            r = s.loc[term]
            color = COLORS[0] if term.startswith("wetland") else COLORS[1]
            ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=1.5)
            ax.scatter(r.coefficient, i, s=32, color=color, zorder=3)
        ax.axvline(0, color="#879295", linewidth=.8, linestyle="--")
        ax.set(yticks=range(len(terms)), yticklabels=labels, ylim=(len(terms)-.5, -.5),
               xlabel="每个特征标准差的 C–Q 斜率变化" if cn else "C–Q slope change / feature SD")
        ax.set_title(f"{title}\nn={int(r.n_stations)}, HUC4={int(r.n_huc4)}", loc="left")
        bounds = data[~data.population.eq(POPULATIONS[2])] if j < 2 else s
        low, high = bounds.ci_low.min(), bounds.ci_high.max()
        ax.set_xlim(low-.07*(high-low), high+.07*(high-low))
    fig.suptitle("来源数量与位置如何改变流量响应？" if cn else "Source amount and position modify flow responses",
                 x=.04, ha="left", fontsize=17)
    fig.text(.04, .025, "站点等权；控制环境、形态、季节与趋势。HUC4 bootstrap 95%。近期图采用独立横轴范围；特征按各群体标准化。" if cn else
             "Equal station weight; environment, shape, season and trend adjusted. HUC4 bootstrap 95%. Recent panel uses a wider scale; SDs are population-specific.", fontsize=9)
    fig.subplots_adjust(left=.23, right=.98, bottom=.2, top=.78, wspace=.22)
    save(fig, "source_flow_moderation")

    gains = pd.read_csv(ROOT/"analysis/huc4_response_gains.csv")
    choices = (("environment_shape", "environment"), ("environment_placement", "environment"),
               ("environment_shape_placement", "environment_shape"),
               ("environment_shape_placement", "environment_placement"))
    labels = ("形态加入环境基线", "来源位置加入环境基线", "来源位置加入形态模型", "形态加入来源位置模型") if cn else (
        "Shape added to environment", "Placement added to environment", "Placement added to shape", "Shape added to placement")
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.6), sharey=True)
    for ax, population, title in zip(axes, POPULATIONS, population_labels, strict=True):
        for i, (candidate, reference) in enumerate(choices):
            r = gains[gains.population.eq(population) & gains.resampling_unit.eq("huc4")
                & gains.candidate.eq(candidate) & gains.reference.eq(reference)].iloc[0]
            ax.plot([r.ci_low_pct, r.ci_high_pct], [i, i], color=COLORS[i % 3], linewidth=1.5)
            ax.scatter(r.gain_pct, i, s=35, color=COLORS[i % 3], zorder=3)
        ax.axvline(0, color="#879295", linestyle="--", linewidth=.8)
        ax.set(yticks=range(4), yticklabels=labels, ylim=(3.5, -.5),
               xlabel="C–Q 描述量的 MAE 改善（%）" if cn else "C–Q descriptor MAE reduction (%)")
        ax.set_title(f"{title}\nn={int(r.n_stations)}", loc="left")
    low, high = gains.ci_low_pct.min(), gains.ci_high_pct.max()
    for ax in axes:
        ax.set_xlim(low-.05*(high-low), high+.05*(high-low))
    fig.suptitle("流量响应能否迁移到其他区域？" if cn else "Can flow-response descriptors transfer across regions?",
                 x=.04, ha="left", fontsize=17)
    fig.text(.04, .025, "五折 HUC4 留出，等折权重及 95% HUC4 bootstrap。预测对象为站点 C–Q 描述量，不是未监测站点 DOC。" if cn else
             "Five HUC4-blocked folds, equal fold weights and 95% HUC4 bootstrap. Response-descriptor diagnostic; not an unmonitored-site DOC benchmark.", fontsize=9)
    fig.subplots_adjust(left=.22, right=.98, bottom=.23, top=.76, wspace=.23)
    save(fig, "geographic_response_increment")
    source_paths = [ROOT/"analysis/station_response_fits.csv", ROOT/"analysis/class_response_distributions.csv",
                    ROOT/"analysis/adjusted_class_responses.csv", ROOT/"analysis/hydrologic_moderation.csv",
                    ROOT/"analysis/huc4_response_gains.csv", Path(__file__)]
    (ROOT/f"figure_sources{suffix}.json").write_text(json.dumps({
        "source_hashes": {str(p): sha256_file(p) for p in source_paths},
        "outputs": {str(p): sha256_file(p) for p in written},
        "language": "Chinese" if cn else "English", "AI_illustration": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
