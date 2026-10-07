"""Plot actual DOC transmission by river form and conditional morphology terms."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_form_process_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")
TERMS = ("log_drainage_density", "mainstem_share", "log_route_mean_scaled", "route_distance_cv",
         "mainstem_sinuosity", "log_basin_aspect", "log_network_axis_ratio")
METRICS = ("log_anomaly_sd_ratio", "signal_rho", "mean_log_departure")


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
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix, written = "_cn" if cn else "", []
    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)
    desc = pd.read_csv(folder/"class_descriptors.csv", dtype={"group": str})
    desc = desc[desc.unit.eq("component")]
    names = ("细长\n多支流", "主干主导\n少支流", "宽展\n多支流") if cn else (
        "Elongated\ntributary-rich", "Mainstem\ndominated", "Broad\ntributary-rich")
    labels = ("下游 / 源波动幅度（log 比）", "季节调整后的信号秩相关", "平均 log1p(DOC) 浓度差") if cn else (
        "Log downstream / source anomaly SD", "Season-adjusted signal rank correlation", "Mean signed log1p(DOC) difference")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.5))
    for row, analysis in enumerate(("mixing_area", "path_monthly")):
        frame = pd.read_csv(folder/f"{analysis}_receivers.csv", dtype={"target": str})
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.axhline(0, color="#A1AEB1", linewidth=.8, linestyle="--")
            for i, group in enumerate((1, 2, 3)):
                s = frame[frame.cluster.eq(group)].dropna(subset=[metric])
                values = s[metric].to_numpy()
                rng = np.random.default_rng(42)
                ax.scatter(i+rng.uniform(-.19, .19, len(s)), values, color=COLORS[i],
                           s=18, alpha=.55, edgecolor="white", linewidth=.3)
                summary = desc[desc.analysis.eq(analysis) & desc.group.eq(str(group)) & desc.metric.eq(metric)]
                if len(summary):
                    r = summary.iloc[0]
                    if np.isfinite(r.ci_low):
                        ax.plot([i, i], [r.ci_low, r.ci_high], color="#263B42", linewidth=1.6)
                        ax.scatter(i, r.estimate, facecolor="white", edgecolor="#263B42", s=44, zorder=4)
                    else:
                        ax.scatter(i, r.estimate, marker="x", color="#263B42", s=44, zorder=4)
                    ax.text(i, 1.025, f"n={len(s)}, c={s.component.nunique()}", ha="center",
                            transform=ax.get_xaxis_transform(), fontsize=8)
                if row == 0 and col == 0:
                    expected = desc[desc.analysis.eq(analysis) & desc.group.eq(str(group))
                                    & desc.metric.eq("log_mixture_sd_ratio")]
                    if len(expected):
                        ax.scatter(i+.25, expected.estimate.iloc[0], marker="^", facecolor="#BDC6C8",
                                   edgecolor="#74868C", s=40, zorder=4)
            ax.set(xticks=range(3), xticklabels=names, ylabel=labels[col], xlim=(-.45, 2.55))
            letter = "abcdef"[row*3+col]
            title = ("支流汇合" if row == 0 else "沿河道变化") if cn else ("Tributary integration" if row == 0 else "Along-channel evolution")
            ax.set_title(f"{letter}  {title}", loc="left", pad=28)
            if col == 1:
                ax.set_ylim(-.3, 1.05)
    fig.suptitle("真实河网中，DOC 波动怎样传到下游？" if cn else "How do DOC fluctuations reach downstream stations in real river networks?",
                 x=.055, ha="left", fontsize=16, y=.98)
    fig.text(.055, .93, "a 图灰三角：面积加权的两支流混合信号" if cn else
             "Panel a grey triangles: area-weighted two-source mixture signal", fontsize=9, color="#74868C")
    fig.text(.055, .014, ("点为下游站；空心点和区间为共享站点系统 bootstrap 95%，× 表示区间不可估计。n=下游站，c=共享站点系统。\n"
                          "仅源角色月观测；波动为 log1p(DOC) 去除季节与年份趋势后的残差。负波动比表示幅度变小，不等于 DOC 被消耗。形态类比较未做环境匹配。") if cn else
             ("Dots: receiving stations. Open circles and intervals: connected-station bootstrap 95%; ×: interval not estimable. n=receivers, c=systems.\n"
              "Monthly source-role log1p(DOC) anomalies, calendar/year adjusted. Negative SD ratios describe buffering, not removal. Classes are unadjusted."), fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.865, wspace=.31, hspace=.65)
    save(fig, "observed_form_transmission")

    coefficients = pd.read_csv(folder/"morphology_associations.csv")
    term_names = ("支流密度", "主干占比", "归一化路径长度", "路径分散度", "主干弯曲度", "流域长宽比", "河网轴比") if cn else (
        "Drainage density", "Mainstem share", "Normalized path length", "Path dispersion", "Mainstem sinuosity", "Basin aspect", "Network axis ratio")
    fig, axes = plt.subplots(2, 3, figsize=(13.7, 8.1))
    for row, analysis in enumerate(("mixing_area", "path_monthly")):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.axvline(0, color="#9CAAAF", linestyle="--", linewidth=.9)
            sub = coefficients[coefficients.analysis.eq(analysis) & coefficients.adjustment.eq("connection_background")
                               & coefficients.outcome.eq(metric)]
            for j, term in enumerate(TERMS):
                primary = sub[sub.focal.eq(term) & sub.unit.eq("component")].iloc[0]
                region = sub[sub.focal.eq(term) & sub.unit.eq("huc4")].iloc[0]
                color = COLORS[0] if row == 0 else COLORS[2]
                ax.plot([region.ci_low, region.ci_high], [j+.08, j+.08], color="#B8C4C8", linewidth=1.2)
                ax.plot([primary.ci_low, primary.ci_high], [j, j], color=color, linewidth=2)
                ax.scatter(primary.estimate_per_receiver_sd, j, color=color, s=28, zorder=3)
            ax.set(yticks=range(7), yticklabels=term_names if col == 0 else [""]*7, ylim=(6.65, -.55),
                   xlabel="每 1 SD 形态变化的条件关联" if cn else "Association per 1 SD of morphology")
            title = ("波动幅度", "信号传递", "平均浓度变化")[col] if cn else ("Anomaly variability", "Signal transmission", "Mean concentration change")[col]
            ax.set_title(f"{'abcdef'[row*3+col]}  {title}", loc="left", pad=15)
            if col == 0:
                heading = ("支流汇合" if row == 0 else "沿河道变化") if cn else ("Tributary integration" if row == 0 else "Along-channel evolution")
                ax.text(0, 1.25, heading, transform=ax.transAxes, fontsize=11)
    fig.suptitle("是河网的外形，还是内部支流与路径排列？" if cn else "Is DOC transmission associated with footprint, branches or channel paths?",
                 x=.055, ha="left", fontsize=16, y=.985)
    fig.text(.055, .014, ("单形态变量＋固定连接背景控制；粗线：共享站点系统 95% 区间，灰线：HUC4 敏感性区间。\n"
                          "源角色探索分析，逐项区间；不同形态变量彼此相关。这些是观测关联，不是独立因果效应。") if cn else
             ("One morphology term + fixed connection controls. Thick: connected-system 95% CI; grey: HUC4 sensitivity CI.\n"
              "Exploratory source-role associations with pointwise intervals; morphology terms are correlated. These are not independent causal effects."), fontsize=8)
    fig.subplots_adjust(left=.145, right=.985, bottom=.14, top=.83, wspace=.25, hspace=.52)
    save(fig, "form_process_associations")
    source_paths = [folder/n for n in ("class_descriptors.csv", "morphology_associations.csv",
                                     "mixing_area_receivers.csv", "path_monthly_receivers.csv")]
    record = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in source_paths},
              "figure_hashes": {str(p): sha256_file(p) for p in written}, "generator_sha256": sha256_file(Path(__file__))}
    (out/f"manifest{suffix}.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"Rendered {len(written)} {record['language']} figures")


if __name__ == "__main__":
    main()
