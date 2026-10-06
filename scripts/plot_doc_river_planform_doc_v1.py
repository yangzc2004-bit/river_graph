"""Render DOC dynamics, adjusted contrasts and matched river-message evidence."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_planform_doc_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    args = parser.parse_args()
    cn = args.chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#263B42", "text.color": "#263B42", "savefig.dpi": 230})
    labels = json.loads(Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/class_labels.json").read_text())
    short = ["细长多支流", "主干少支流", "宽展多支流"] if cn else ["Elongated\ntributary-rich", "Mainstem\ndominated", "Broad\ntributary-rich"]
    out = ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix = "_cn" if cn else ""

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{name}{suffix}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    panel = pd.read_csv(ROOT/"analysis/station_doc_response.csv", dtype={"station": str})
    panel = panel[panel.eligible]
    response = pd.read_csv(ROOT/"analysis/class_doc_response.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    rng = np.random.default_rng(42)
    for ax, metric, title, unit in zip(axes.flat[:3], ("doc_median", "harmonic_amplitude", "q90_fraction"),
         ("DOC 浓度水平", "DOC 季节振幅", "高 DOC 出现比例") if cn else
         ("DOC concentration", "Seasonal DOC amplitude", "High-DOC frequency"),
         ("站点中位数（mg/L）", "峰谷差（log1p DOC）", "观测样本中 DOC ≥ 10 mg/L（%）") if cn else
         ("Station median (mg/L)", "Peak-to-trough log1p DOC", "Observed DOC ≥ 10 mg/L (%)"), strict=True):
        scale = 100 if metric == "q90_fraction" else 1
        for c in (1, 2, 3):
            values = panel.loc[panel.cluster.eq(c), metric].dropna().to_numpy()*scale
            ax.scatter(c+rng.uniform(-.19, .19, len(values)), values, color=COLORS[c-1], s=10, alpha=.34, rasterized=True)
            row = response[response.cluster.eq(c) & response.metric.eq(metric)].iloc[0]
            median, low, high = scale*row["median"], scale*row.ci_low, scale*row.ci_high
            ax.errorbar(c, median, yerr=[[median-low], [high-median]], fmt="o", color="#203C43", capsize=4,
                        markersize=6, linewidth=1.7, zorder=4)
            ax.text(c, 1.02, f"n={len(values)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.set(xticks=[1, 2, 3], xticklabels=short, ylabel=unit)
        ax.set_title(title, pad=26, loc="left", fontweight="bold")
        if metric == "doc_median":
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 5, 10, 20, 50], labels=["1", "2", "5", "10", "20", "50"])
    seasonal = pd.read_csv(ROOT/"analysis/class_seasonal_response.csv")
    ax = axes[1, 1]
    for c in (1, 2, 3):
        s = seasonal[seasonal.cluster.eq(c)]
        ax.plot(s.month, s.doc_station_mean, color=COLORS[c-1], marker="o", markersize=3, label=short[c-1].replace("\n", " "))
    ax.set(xticks=[1, 3, 6, 9, 12], xlabel="月份" if cn else "Calendar month", ylabel="站点月中位数的均值（mg/L）" if cn else "Mean station monthly median (mg/L)")
    ax.set_title("观测日历曲线" if cn else "Observed calendar curves", loc="left", fontweight="bold", pad=26)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("真实河网形态与 DOC 动态" if cn else "Real river planform and DOC dynamics", fontsize=17, x=.06, ha="left")
    fig.text(.06, .018, "点：单个站点；黑点及区间：类型中位数和站点 bootstrap 95% 区间。日历曲线每月的站点组成可不同。" if cn else
             "Points: individual stations; dark marks: class median and station-bootstrap 95% interval. Calendar-month populations may differ.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.14, hspace=.65, wspace=.27)
    save(fig, "doc_dynamics_by_planform")

    association = pd.read_csv(ROOT/"analysis/adjusted_associations.csv")
    metrics = ("doc_median", "harmonic_amplitude", "cq_slope", "q90_fraction")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.9))
    for ax, metric, title in zip(axes, metrics,
         ("浓度", "季节振幅", "浓度—流量响应", "高值频率") if cn else
         ("Concentration", "Seasonal amplitude", "Concentration–flow", "High-value frequency"), strict=True):
        for c in (2, 3):
            for hydro, offset in ((True, -.12), (False, .12)):
                model = "class_hydro_adjusted" if hydro else "class_without_local_hydro"
                row = association[(association.metric == metric) & (association.model == model) & (association.term == f"class_{c}")].iloc[0]
                y = c+offset
                ax.plot([row.huc4_ci_low, row.huc4_ci_high], [y, y], color=COLORS[c-1], linewidth=1.6)
                ax.scatter(row.estimate, y, color=COLORS[c-1] if hydro else "white", edgecolor=COLORS[c-1], s=35, zorder=3)
        ax.axvline(0, color="#53676D", linestyle="--", linewidth=.8)
        ax.set(yticks=[2, 3], yticklabels=[short[1], short[2]], ylim=(3.55, 1.5), xlabel="调整后系数" if cn else "Adjusted coefficient")
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle("形态差异在环境与面积调整后是否保留？" if cn else "Do planform contrasts remain after environment and area adjustment?", fontsize=15, x=.05, ha="left")
    fig.text(.05, .04, "参考：细长多支流型。实心：含局地水文；空心：不含局地水文。横线：HUC4 block 95% 区间。各面板响应尺度不同。" if cn else
             "Reference: elongated tributary-rich. Filled: local hydro adjusted; open: no local hydro. Lines: HUC4-block 95% intervals. Outcome scales differ.", fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, top=.77, bottom=.23, wspace=.8)
    save(fig, "adjusted_planform_doc")

    gain = pd.read_csv(ROOT/"analysis/historical_message_gains.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4))
    for ax, family, title in zip(axes, ("e2a_strict", "e3_spatial_seed42"),
         ("严格时间外推", "空间站点留出") if cn else ("Strict temporal extrapolation", "Spatial station holdout"), strict=True):
        for arm, offset, marker in (("residual_upstream", -.1, "o"), ("residual_both", .1, "s")):
            for c in (1, 2, 3):
                s = gain[(gain.family == family) & (gain.candidate == arm) & (gain.group == f"class_{c}") & ~gain["tail"]]
                if s.empty:
                    continue
                r = s.iloc[0]
                y = c+offset
                ax.plot([r.gain_ci_low_pct, r.gain_ci_high_pct], [y, y], color=COLORS[c-1], linewidth=1.5)
                ax.scatter(r.relative_gain_pct, y, color=COLORS[c-1], marker=marker, s=40)
                ax.text(1.025, y, f"n={int(r.n_stations_unique)}"+("*" if r.few_stations else ""),
                        transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
        ax.axvline(0, color="#53676D", linestyle="--", linewidth=.8)
        ax.set(yticks=[1, 2, 3], yticklabels=short, ylim=(3.5, .5),
               xlabel="相对无消息对照的 MAE 改善（%）" if cn else "MAE reduction relative to no-message (%)")
        ax.set_title(title, loc="left", fontweight="bold", pad=15)
    fig.suptitle("河网消息在哪些形态中提供信息？" if cn else "Which planforms benefit from river messages?", fontsize=16, x=.05, ha="left")
    fig.text(.05, .035, "圆点：仅上游；方点：双向。区间：配对站点 bootstrap 95%。* 少于10站。历史 KGML 验证对照，不是当前模型的提升。" if cn else
             "Circles: upstream only; squares: both directions. Paired station-bootstrap 95% intervals. * Fewer than 10 stations. Historical KGML validation.", fontsize=9)
    fig.subplots_adjust(left=.16, right=.92, top=.79, bottom=.22, wspace=.65)
    save(fig, "river_message_gain_by_planform")

    gains = pd.read_csv(ROOT/"analysis/huc4_blocked_gains.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1))
    for ax, space, title in zip(axes, ("native", "log1p"),
         ("原始 DOC 尺度", "log1p DOC 尺度") if cn else ("Native DOC scale", "log1p DOC scale"), strict=True):
        for i, model in enumerate(("environment_area_classes", "environment_area_shape")):
            r = gains[(gains.model == model) & (gains.space == space) & (gains.resampling_unit == "huc4")].iloc[0]
            ax.plot([r.gain_ci_low_pct, r.gain_ci_high_pct], [i, i], color=COLORS[i], linewidth=2)
            ax.scatter(r.relative_gain_pct, i, color=COLORS[i], s=60)
            ax.text(r.relative_gain_pct, i-.13, f"{r.relative_gain_pct:.2f}%", ha="center", fontsize=11)
        ax.axvline(0, color="#53676D", linestyle="--", linewidth=.8)
        ax.set(yticks=[0, 1], yticklabels=["增加类型标签", "增加连续形态"] if cn else ["Add class labels", "Add continuous shape"],
               ylim=(1.5, -.6), xlim=(-2, 15), xlabel="相对环境＋面积的 MAE 改善（%）" if cn else "MAE reduction vs environment + area (%)")
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle("河网形态是否提供环境之外的信息？" if cn else "Does planform add information beyond environment?", fontsize=16, x=.05, ha="left")
    fig.text(.05, .035, "305站，63个 HUC4；五折按区域留出。区间：HUC4 block bootstrap 95%。目标为站点 DOC 中位数，不是逐月模型性能。" if cn else
             "305 stations, 63 HUC4s; five region-blocked folds. HUC4-block 95% intervals. Station median DOC diagnostic, not monthly model performance.", fontsize=9)
    fig.subplots_adjust(left=.16, right=.97, top=.79, bottom=.22, wspace=.52)
    save(fig, "planform_information_increment")
    (ROOT/f"figure_sources{suffix}.json").write_text(json.dumps({"analysis": [str(p) for p in (ROOT/"analysis").glob("*.csv")],
        "language": "Chinese" if cn else "English", "labels": labels,
        "error_bars": "95% paired station bootstrap or HUC4-block intervals, labelled per figure"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
