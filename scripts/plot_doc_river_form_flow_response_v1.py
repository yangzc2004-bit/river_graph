"""Scientific figures for observed flow responses across real river forms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_form_flow_response_v1")
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
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 220})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    written = []
    suffix = "_cn" if cn else ""
    sources = [a/f"{n}.csv" for n in ("station_state_summaries", "class_flow_responses", "pair_response_contrasts",
               "pair_response_evidence", "joint_state_contrasts", "structure_response_associations")]
    classes = ("细长、多支流型", "主干主导、稀支流型", "宽展、多支流型") if cn else (
        "Elongated / tributary-rich", "Mainstem / sparse", "Broad / tributary-rich")
    summary = pd.read_csv(a/"class_flow_responses.csv")
    states = pd.read_csv(a/"station_state_summaries.csv")

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    def ci(ax, row, y, color, marker="o", scale=1.):
        ax.plot([scale*row.ci_low, scale*row.ci_high], [y, y], color=color, linewidth=1.8)
        value = row["mean"] if "mean" in row.index else row["coefficient"] if "coefficient" in row.index else row.difference_b_minus_a
        ax.scatter(scale*value, y, color=color, s=32, marker=marker, zorder=3)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.7))
    primary = states[states.population.eq("observed_flow")]
    grouped = primary.groupby(["cluster", "flow_state"]).doc_mean.mean()
    for i in (1, 2, 3):
        y = [grouped.loc[(i, "low")], grouped.loc[(i, "high")]]
        axes[0].plot([0, 1], y, marker="o", color=COLORS[i], label=classes[i-1], linewidth=1.6)
    axes[0].set(xticks=[0, 1], xticklabels=("低流量", "高流量") if cn else ("Low flow", "High flow"),
                ylabel="实测 DOC 均值 (mg/L)" if cn else "Observed mean DOC (mg/L)")
    axes[0].set_title("a  "+("每条河先分状态，再站点等权" if cn else "State means, stations weighted equally"), loc="left", pad=18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(.035, .925), ncol=3, fontsize=9)
    for i in (1, 2, 3):
        for offset, population, metric, marker in ((-.13, "observed_flow", "raw_log_contrast", "x"),
                (0, "observed_flow", "adjusted_log_contrast", "o"), (.13, "observed_temperature", "adjusted_log_contrast", "s")):
            r = summary[summary.cluster.eq(i) & summary.population.eq(population) & summary.metric.eq(metric)].iloc[0]
            ci(axes[1], r, i-1+offset, COLORS[i], marker)
        r = summary[summary.cluster.eq(i) & summary.population.eq("observed_flow") & summary.metric.eq("high_frequency_contrast")].iloc[0]
        ci(axes[2], r, i-1, COLORS[i], scale=100)
    for ax in axes[1:]:
        ax.axvline(0, color="#9FAFB4", linewidth=.9, linestyle="--")
        ax.set(yticks=range(3), yticklabels=classes, ylim=(2.5, -.5))
    axes[1].set_xlabel("高 − 低流量的 log1p DOC 差异" if cn else "High − low log1p DOC")
    axes[1].set_title("b  "+("季节、年份与温度调整" if cn else "Season, year and temperature adjustment"), loc="left", pad=18)
    axes[2].set_xlabel("高 DOC 频率变化 (百分点)" if cn else "High-DOC frequency change (percentage points)")
    axes[2].set_title("c  "+("高 DOC 阈值固定为 10 mg/L" if cn else "High DOC: fixed 10 mg/L threshold"), loc="left", pad=18)
    fig.suptitle("同一条河从低流量到高流量，DOC 如何变化？" if cn else "How does DOC change from low to high flow within the same river?",
                 x=.035, ha="left", y=.995, fontsize=16)
    fig.text(.035, .025, ("205 个源角色站点 / 49 个 HUC4；细长型 74、稀支流型 21、宽展型 110。流量状态由各站完整流量记录的三分位定义。\n"
        "b：× 未调整，● 调整季节与年份，■ 再调整温度；线为 HUC4 bootstrap 95% 区间。各类自身的响应不等于类间差异。") if cn else
        ("205 source-role stations / 49 HUC4; elongated 74, sparse 21, broad 110. Station flow terciles use measured flow, including non-DOC months.\n"
         "b: × unadjusted, ● season/year adjusted, ■ plus temperature. Lines: HUC4 bootstrap 95%. Within-class responses are not class differences."), fontsize=8)
    fig.subplots_adjust(left=.07, right=.985, top=.78, bottom=.22, wspace=.92)
    save(fig, "within_river_flow_responses")

    contrast = pd.read_csv(a/"pair_response_contrasts.csv")
    evidence = pd.read_csv(a/"pair_response_evidence.csv")
    paired = contrast[contrast.class_a.eq(1) & contrast.class_b.eq(3)]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.7))
    labels = ("实测差异", "调整季节＋年份", "再调整温度") if cn else ("Observed contrast", "Season + year adjusted", "+ temperature")
    for ax, kind in zip(axes[:2], ("log", "doc"), strict=True):
        for i, metric, population in ((0, f"raw_{kind}_contrast", "observed_flow"),
                (1, f"adjusted_{kind}_contrast", "observed_flow"), (2, f"adjusted_{kind}_contrast", "observed_temperature")):
            r = paired[paired.metric.eq(metric) & paired.population.eq(population)].iloc[0]
            ci(ax, r, i, COLORS[i+1])
            ax.annotate(f"{r.difference_b_minus_a:+.3f}" if kind == "log" else f"{r.difference_b_minus_a:+.2f}",
                        (r.difference_b_minus_a, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        ax.axvline(0, color="#9FAFB4", linestyle="--", linewidth=.9)
        ax.set(yticks=range(3), yticklabels=labels, ylim=(2.5, -.5),
               xlabel=("宽展型响应 − 细长型响应" if cn else "Broad response − elongated response")+f" ({'log1p DOC' if kind == 'log' else 'mg/L'})")
    axes[0].set_title("a  "+("主要响应：log1p DOC" if cn else "Primary response: log1p DOC"), loc="left", pad=18)
    axes[1].set_title("b  "+("原浓度单位的敏感性" if cn else "Native-concentration sensitivity"), loc="left", pad=18)
    p = evidence[evidence.class_a.eq(1) & evidence.class_b.eq(3) & evidence.population.eq("observed_flow") & evidence.metric.eq("adjusted_log_contrast")]
    ax = axes[2]
    lo = min(p.value_a.min(), p.value_b.min())-.1
    hi = max(p.value_a.max(), p.value_b.max())+.1
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#9FAFB4", linewidth=.9)
    ax.scatter(p.value_a, p.value_b, s=38, color=COLORS[3], edgecolor="white", linewidth=.5)
    ax.axhline(0, color="#D6E0E2", linewidth=.7)
    ax.axvline(0, color="#D6E0E2", linewidth=.7)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="细长型调整后响应" if cn else "Adjusted elongated response",
           ylabel="宽展型调整后响应" if cn else "Adjusted broad response")
    ax.set_title("c  "+("每点一对真实河网" if cn else "One fixed river pair per point"), loc="left", pad=18)
    fig.suptitle("面积和环境相近的河网，流量响应是否因形态而不同？" if cn else "Do environmentally matched river forms have different DOC-flow responses?",
                 x=.035, ha="left", y=.995, fontsize=16)
    fig.text(.035, .025, ("10 对固定细长／宽展河网 / 6 个 HUC4；两条河的拟合使用完全相同的 DOC＋流量观测月份，每状态至少 6 个观测。\n"
        "a–b 点为河网对等权均值，线为 HUC4 bootstrap 95% 区间。不同站点的低／高状态可能发生在不同月份；共同状态另表报告。") if cn else
        ("10 fixed elongated/broad river pairs / 6 HUC4; fits use identical observed DOC+flow months, with at least six observations per state.\n"
         "a/b: equal-pair means, HUC4 bootstrap 95%. Local low/high states can occur on different dates; joint-state comparisons are reported separately."), fontsize=8)
    fig.subplots_adjust(left=.14, right=.985, top=.78, bottom=.22, wspace=.65)
    save(fig, "matched_form_flow_contrasts")

    association = pd.read_csv(a/"structure_response_associations.csv")
    names = {"form_2": "稀支流 − 细长型", "form_3": "宽展 − 细长型", "log_basin_aspect": "流域长宽比",
             "log_network_axis_ratio": "河网伸长程度", "log_drainage_density": "河网密度", "mainstem_share": "主干河长占比",
             "mainstem_sinuosity": "主干弯曲度", "log_route_mean_scaled": "面积标准化路径长度", "route_distance_cv": "路径长短差异"} if cn else {
             "form_2": "Sparse − elongated", "form_3": "Broad − elongated", "log_basin_aspect": "Basin aspect ratio",
             "log_network_axis_ratio": "Network elongation", "log_drainage_density": "Drainage density", "mainstem_share": "Mainstem length share",
             "mainstem_sinuosity": "Mainstem sinuosity", "log_route_mean_scaled": "Area-scaled path length", "route_distance_cv": "Path-length dispersion"}
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6))
    for ax, features in zip(axes, (list(names)[:2], list(names)[2:]), strict=True):
        for i, feature in enumerate(features):
            for offset, population, marker in ((-.10, "observed_flow", "o"), (.10, "observed_temperature", "s")):
                r = association[association.feature.eq(feature) & association.population.eq(population)].iloc[0]
                ci(ax, r, i+offset, COLORS[3], marker)
        ax.axvline(0, color="#9FAFB4", linestyle="--", linewidth=.9)
        ax.set(yticks=range(len(features)), yticklabels=[names[f] for f in features], ylim=(len(features)-.5, -.5))
    axes[0].set_title("a  "+("三种形态的调整后差异" if cn else "Adjusted form-label associations"), loc="left", pad=18)
    axes[0].set_xlabel("响应差异 (log1p DOC)" if cn else "Response difference (log1p DOC)")
    axes[1].set_title("b  "+("内部结构与响应幅度" if cn else "Internal structure and response magnitude"), loc="left", pad=18)
    axes[1].set_xlabel("结构每增加 1 SD 的响应差异" if cn else "Response difference per 1 SD in structure")
    fig.suptitle("哪些河网结构与 DOC 的流量响应有关？" if cn else "Which river structures are associated with the DOC response to flow?",
                 x=.035, ha="left", y=.995, fontsize=16)
    fig.text(.035, .025, ("204 个协变量完整站点 / 49 个 HUC4；控制流域面积、生态、水文变化幅度和区域。● 季节／年份调整的响应；■ 再调整温度。\n"
        "外形、支流、路径分别建模，同一块内特征同时进入；5,000 次 HUC4 bootstrap，逐次重拟合。描述性关联，不是完整模型性能增益。") if cn else
        ("204 covariate-complete stations / 49 HUC4; controls include basin area, ecology, flow change and region. ● season/year responses; ■ plus temperature.\n"
         "Footprint, branches and paths fitted separately with features jointly within each block. 5,000 HUC4 bootstrap refits; descriptive associations."), fontsize=8)
    fig.subplots_adjust(left=.15, right=.985, top=.80, bottom=.19, wspace=.95)
    save(fig, "structure_flow_response_associations")
    manifest = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in sources},
                "generator_sha256": sha256_file(Path(__file__)), "figure_hashes": {str(p): sha256_file(p) for p in written}}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
