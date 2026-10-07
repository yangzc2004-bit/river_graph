"""Scientific figures for same-calendar river-form DOC comparisons."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_form_monthly_comparison_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")
POPULATIONS = ("common_doc", "complete_hydro", "at_least_24_months")


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
    sources = [a/f"{n}.csv" for n in ("paired_response_evidence", "paired_response_contrasts",
               "information_scores", "information_gains", "station_model_errors")]
    evidence = pd.read_csv(a/"paired_response_evidence.csv", dtype={"station_a": str, "station_b": str, "huc4": str})
    contrasts = pd.read_csv(a/"paired_response_contrasts.csv")
    primary = contrasts[contrasts.class_a.eq(1) & contrasts.class_b.eq(3)]
    pairs = evidence[evidence.class_a.eq(1) & evidence.class_b.eq(3) & evidence.population.eq("common_doc")]
    labels = ("共同 DOC 月份", "共同实测水文月份", "至少 24 个共同月份") if cn else (
        "Shared DOC months", "Shared measured-hydro months", "At least 24 shared months")

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.0))
    metrics = (("doc_mean", "log_doc_sd", "high_doc_fraction"),
               ("hydro_residual_mean", "hydro_residual_sd", "hydro_exceedance"))
    titles = (("DOC 平均水平", "DOC 波动幅度", "高 DOC 出现频率"),
              ("调整后的平均水平", "调整后的波动幅度", "调整后的高值频率")) if cn else (
        ("Mean DOC", "DOC variability", "High-DOC frequency"),
        ("Adjusted concentration level", "Adjusted variability", "Adjusted high-value frequency"))
    for row in range(2):
        for col in range(3):
            ax, metric = axes[row, col], metrics[row][col]
            scale = 100 if col == 2 else 1
            for i, population in enumerate(POPULATIONS):
                r = primary[primary.population.eq(population) & primary.metric.eq(metric)].iloc[0]
                ax.plot([scale*r.ci_low, scale*r.ci_high], [i, i], color=COLORS[i], linewidth=2)
                ax.scatter(scale*r.difference_b_minus_a, i, s=40, color=COLORS[i], zorder=3)
                ax.annotate(f"{scale*r.difference_b_minus_a:+.2f}" if col != 1 else f"{r.difference_b_minus_a:+.3f}",
                            (scale*r.difference_b_minus_a, i), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
            ax.axvline(0, color="#9FAFB4", linestyle="--", linewidth=.9)
            ax.set(ylim=(2.55, -.55), yticks=range(3), yticklabels=labels if col == 0 else [""]*3)
            if col == 2:
                unit = "百分点" if cn else "percentage points"
            elif row == 0 and col == 0:
                unit = "mg/L"
            else:
                unit = "log1p DOC"
            ax.set_xlabel(f"{'宽展型 − 细长型' if cn else 'Broad − elongated'} ({unit})")
            ax.set_title(f"{'abcdef'[row*3+col]}  {titles[row][col]}", loc="left", pad=18, fontsize=11)
    fig.suptitle("不同河网形态的 DOC，是否存在稳定差异？" if cn else "Do different river forms have stable differences in observed DOC?",
                 x=.035, ha="left", y=.995, fontsize=17)
    fig.text(.035, .935, ("同一 HUC4 内环境与面积相近的固定河网对，使用相同观测月份；下排扣除留区交叉拟合的环境、水文与时间背景。") if cn else
             "Fixed environment/area-matched rivers within HUC4, identical dates; bottom row removes a held-region environment/hydro/calendar background.", fontsize=9)
    fig.text(.035, .025, ("共同 DOC：20 对 / 11 个 HUC4；共同水文：17 对 / 10 个 HUC4；较长记录：13 对 / 7 个 HUC4。点为河网对等权均值，线为 HUC4 bootstrap 95% 区间。\n"
              "高 DOC 阈值固定为 10 mg/L。波动为 log1p 标准差。背景调整描述剩余关联；月度频率与标准差不代表实测事件峰值或物理缓冲率。") if cn else
             ("Shared DOC: 20 pairs / 11 HUC4; complete hydro: 17 / 10; longer record: 13 / 7. Equal-pair means with HUC4 bootstrap 95% intervals.\n"
              "High DOC: fixed 10 mg/L. Variability: log1p SD. Adjustments describe remaining associations; these are not event peaks or measured buffering rates."), fontsize=8)
    fig.subplots_adjust(left=.19, right=.985, top=.84, bottom=.14, wspace=.38, hspace=.60)
    save(fig, "same_month_form_contrasts")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.6))
    for i, (metric, scale) in enumerate((("doc_mean", 1), ("log_doc_sd", 1), ("high_doc_fraction", 100))):
        ax = axes[i]
        s = pairs[pairs.metric.eq(metric)]
        maximum = max(s.value_a.max(), s.value_b.max())*scale*1.12
        ax.plot([0, maximum], [0, maximum], color="#9FAFB4", linestyle="--", linewidth=.9)
        ax.scatter(s.value_a*scale, s.value_b*scale, color=COLORS[2], s=22+np.sqrt(s.n_months)*3,
                   alpha=.8, edgecolor="white", linewidth=.5)
        title = ("平均 DOC (mg/L)", "DOC 波动 (log1p SD)", "高 DOC 频率 (%)")[i] if cn else (
            "Mean DOC (mg/L)", "DOC variability (log1p SD)", "High-DOC frequency (%)")[i]
        ax.set(xlim=(-.02*maximum, maximum), ylim=(-.02*maximum, maximum), xlabel="细长、多支流型" if cn else "Elongated / tributary-rich",
               ylabel="宽展、多支流型" if cn else "Broad / tributary-rich")
        ax.set_title(f"{'abc'[i]}  {title}", loc="left", pad=18)
    fig.suptitle("相近环境中的 20 对真实河网：差异并非同一个方向" if cn else "20 real, environmentally matched river pairs: differences have mixed directions",
                 x=.035, ha="left", y=.995, fontsize=16)
    fig.text(.035, .025, ("每点一对固定河网，每对两站使用完全相同月份；点大小表示共同月份数。虚线表示两种形态数值相同。\n"
              "这些是有观测站点的同月描述性比较。站点对和类型未根据 DOC 结果重新选择。") if cn else
             ("One fixed pair per point, identical dates in both rivers; size indicates shared-month count. Dashed line: equal member values.\n"
              "Observed monthly descriptive comparisons; pairs and form classes were not reselected from DOC results."), fontsize=8)
    fig.subplots_adjust(left=.085, right=.985, top=.76, bottom=.22, wspace=.40)
    save(fig, "matched_river_doc_properties")

    scores = pd.read_csv(a/"information_scores.csv")
    gains = pd.read_csv(a/"information_gains.csv")
    station = pd.read_csv(a/"station_model_errors.csv", dtype={"station": str})
    arm_order = ("environment", "hydro", "form", "branching", "paths", "all_morphology")
    names = ("环境＋时间", "环境＋时间＋水文", "水文基线＋形态标签", "水文基线＋支流组织", "水文基线＋路径组织", "水文基线＋完整形态") if cn else (
        "Environment + calendar", "Environment + calendar + hydro", "Hydro baseline + form labels", "Hydro baseline + branches", "Hydro baseline + paths", "Hydro baseline + all morphology")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.9))
    ax = axes[0]
    for i, arm in enumerate(arm_order):
        r = scores[scores.arm.eq(arm) & scores.metric.eq("mae")].iloc[0]
        color = "#98A9AE" if i < 2 else COLORS[(i-2) % 3]
        ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=2)
        ax.scatter(r["mean"], i, s=34, color=color)
    ax.set(yticks=range(6), yticklabels=names, ylim=(5.6, -.6), xlabel="月度 DOC MAE (mg/L)" if cn else "Monthly DOC MAE (mg/L)")
    ax.set_title("a  "+("相同站点与月份的比较" if cn else "Same stations and months"), loc="left", pad=18)
    ax = axes[1]
    for i, arm in enumerate(arm_order[2:]):
        r = gains[gains.candidate.eq(arm) & gains.reference.eq("hydro") & gains.metric.eq("mae")].iloc[0]
        ax.plot([r.gain_ci_low_pct, r.gain_ci_high_pct], [i, i], color=COLORS[i % 3], linewidth=2)
        ax.scatter(r.relative_reduction_pct, i, s=34, color=COLORS[i % 3])
        ax.annotate(f"{r.relative_reduction_pct:+.2f}%", (r.relative_reduction_pct, i),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(0, color="#9FAFB4", linestyle="--", linewidth=.9)
    ax.set(yticks=range(4), yticklabels=names[2:], ylim=(3.6, -.6),
           xlabel="相对环境＋水文的 MAE 降低 (%)" if cn else "MAE reduction vs environment + hydro (%)")
    ax.set_title("b  "+("额外形态信息的总体收益" if cn else "Additional morphology information"), loc="left", pad=18)
    ax = axes[2]
    by_fold = station.groupby(["fold", "arm"]).log_mae.mean().unstack("arm")
    for i, arm in enumerate(arm_order[2:]):
        gain = 100*(1-by_fold[arm]/by_fold.hydro)
        x = i+np.linspace(-.12, .12, len(gain))
        ax.scatter(x, gain, color=COLORS[i % 3], s=30, alpha=.8)
    ax.axhline(0, color="#9FAFB4", linestyle="--", linewidth=.9)
    ax.set(xticks=range(4), xticklabels=("类型", "支流", "路径", "全部") if cn else ("Forms", "Branches", "Paths", "All"),
           ylabel="每个留区折的 log1p MAE 降低 (%)" if cn else "Held-region fold log1p MAE reduction (%)")
    ax.set_title("c  "+("五个地理折分别查看" if cn else "Five geographic folds"), loc="left", pad=18)
    fig.suptitle("河网形态为月度 DOC 增加了多少信息？" if cn else "How much monthly DOC information does river morphology add?",
                 x=.035, ha="left", y=.995, fontsize=17)
    fig.text(.035, .025, ("297 个源角色站点 / 62 个 HUC4 / 18,688 个观测月；五折按 HUC4 留出，训练与评分均先给予每站等权。\n"
              "形态信息分别加在环境＋水文基线上；a–b 为 HUC4 bootstrap 95% 区间；c 每点一个折。固定线性诊断，不是完整重建模型的提升。") if cn else
             ("297 source-role stations / 62 HUC4 / 18,688 observed months; HUC4-held folds, station-equal fitting and scoring.\n"
              "Morphology blocks separately augment the hydro baseline. a/b: HUC4 bootstrap 95%; c: one point/fold. Linear diagnostics, not released-model gains."), fontsize=8)
    fig.subplots_adjust(left=.16, right=.985, top=.78, bottom=.21, wspace=1.02)
    save(fig, "monthly_morphology_information")
    manifest = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in sources},
                "figure_hashes": {str(p): sha256_file(p) for p in written}, "generator_sha256": sha256_file(Path(__file__)),
                "interpretation": "Fixed matched forms, shared observed months, region-excluding backgrounds; exploratory associations."}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
