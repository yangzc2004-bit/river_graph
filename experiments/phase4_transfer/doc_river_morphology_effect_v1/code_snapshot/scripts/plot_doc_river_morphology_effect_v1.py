"""Render morphology-centred matched DOC, routing and information blocks."""
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

ROOT = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")


def profile_interval(frame, draws=5000):
    table = frame.pivot(index=["station", "huc4"], columns="distance_fraction", values="area_mass")
    _, gi = np.unique(table.index.get_level_values("huc4"), return_inverse=True)
    n = gi.max()+1
    w = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
    boot = w@table.to_numpy()/w.sum(axis=1)[:, None]
    return table.columns.to_numpy(), table.mean().to_numpy(), np.quantile(boot, [.025, .975], axis=0)


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
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix, written = "_cn" if cn else "", []
    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)
    names = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated / tributary-rich", "Mainstem dominated / sparse", "Broad / tributary-rich")
    evidence = pd.read_csv(folder/"paired_doc_evidence.csv")
    pairs = evidence[evidence.class_a.eq(1) & evidence.class_b.eq(3) & evidence.metric.eq("doc_median")]
    contrasts = pd.read_csv(folder/"paired_doc_contrasts.csv")
    r = contrasts[contrasts.class_a.eq(1) & contrasts.class_b.eq(3) & contrasts.metric.eq("doc_median")].iloc[0]
    balance = pd.read_csv(folder/"matching_balance.csv")
    balance = balance[balance.class_a.eq(1) & balance.class_b.eq(3)].set_index("covariate")
    labels = ("流域面积", "湿地比例", "森林比例", "农业比例", "城镇比例", "降水", "气候温度", "纬度", "经度", "观测年份") if cn else (
        "Basin area", "Wetland cover", "Forest cover", "Agriculture", "Urban cover", "Precipitation", "Climate temperature", "Latitude", "Longitude", "Observation year")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5))
    ax = axes[0]
    upper = max(pairs.value_a.max(), pairs.value_b.max())*1.12
    ax.plot([0, upper], [0, upper], color="#899598", linestyle="--", linewidth=1)
    ax.scatter(pairs.value_a, pairs.value_b, color=COLORS[2], edgecolor="white", linewidth=.5, s=42, alpha=.8)
    ax.set(xlim=(0, upper), ylim=(0, upper),
           xlabel="细长多支流型 DOC（mg/L）" if cn else "Elongated tributary-rich DOC (mg/L)",
           ylabel="宽展多支流型 DOC（mg/L）" if cn else "Broad tributary-rich DOC (mg/L)")
    ax.set_aspect("equal")
    ax.set_title("a  同一区域、相近环境的形态比较" if cn else "a  Different forms, comparable environments", loc="left", pad=14)
    statement = f"{int(r.n_pairs)} pairs / {int(r.n_huc4)} HUC4\nB − E: {r.difference_b_minus_a:+.2f} mg/L\n95% CI [{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
    ax.text(.04, .95, statement, transform=ax.transAxes, va="top", fontsize=10)
    ax = axes[1]
    for i, name in enumerate(balance.index):
        row = balance.loc[name]
        ax.plot([row.unmatched_standardized_difference, row.matched_standardized_difference], [i, i], color="#B4BEC0", linewidth=1)
    ax.scatter(balance.unmatched_standardized_difference, range(len(balance)), facecolor="white", edgecolor="#8E9B9F", s=33, label="匹配前" if cn else "Before matching")
    ax.scatter(balance.matched_standardized_difference, range(len(balance)), color=COLORS[0], s=27, label="匹配后" if cn else "After matching")
    ax.axvline(0, color="#9CA7AA", linewidth=.8, linestyle="--")
    ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels)-.5, -.5),
           xlabel="环境协变量的标准化差异" if cn else "Standardized environmental difference")
    ax.set_title("b  匹配质量" if cn else "b  Covariate balance", loc="left", pad=14)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("河网外形不同，DOC 浓度是否也不同？" if cn else "Do contrasting river forms have different DOC concentrations?", x=.04, ha="left", fontsize=17)
    fig.text(.04, .02, "不嵌套站点配对；不按 DOC 选择流域。浓度为源观测的站点中位数，区间为配对 HUC4 bootstrap 95%。" if cn else
             "Non-nested pairs selected without DOC. Concentration: source-record station median. Paired HUC4 bootstrap 95%.", fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, bottom=.17, top=.79, wspace=.5)
    save(fig, "matched_forms_doc")

    profiles = pd.read_csv(folder/"station_path_profiles.csv", dtype={"station": str, "huc4": str})
    pulse = pd.read_parquet(folder/"identical_input_routing.parquet")
    descriptors = pd.read_csv(folder/"class_routing_descriptors.csv")
    panel = pd.read_csv(folder/"station_morphology_doc_panel.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 5.4))
    for i, group in enumerate((1, 2, 3)):
        sub = profiles[profiles.cluster.eq(group)]
        x, mean, ci = profile_interval(sub)
        axes[0].plot(x, mean, color=COLORS[i], linewidth=1.7, label=names[i])
        axes[0].fill_between(x, ci[0], ci[1], color=COLORS[i], alpha=.12)
        response = pulse[pulse.cluster.eq(group)].groupby("normalized_time").routed_anomaly.mean()
        axes[1].plot(response.index, response, color=COLORS[i], linewidth=1.8)
        s = panel[panel.cluster.eq(group)]
        rng = np.random.default_rng(42)
        axes[2].scatter(i+rng.uniform(-.2, .2, len(s)), s.routing_pulse_peak, color=COLORS[i], s=10, alpha=.4)
        r = descriptors[descriptors.cluster.eq(group) & descriptors.metric.eq("routing_pulse_peak")].iloc[0]
        axes[2].plot([i, i], [r.ci_low, r.ci_high], color="#263B42", linewidth=2)
        axes[2].scatter(i, r["mean"], facecolor="white", edgecolor="#263B42", s=35, zorder=4)
        axes[2].text(i, 1.01, f"n={len(s)}", transform=axes[2].get_xaxis_transform(), ha="center", fontsize=9)
    base = pulse[pulse.station.eq(pulse.station.iloc[0])]
    axes[1].plot(base.normalized_time, base.no_delay_anomaly, color="#889598", linestyle="--", linewidth=1.2)
    axes[0].set(xlabel="到出口的相对河道距离" if cn else "Relative channel distance to outlet",
                ylabel="各距离段的汇水面积份额" if cn else "Catchment-area share per distance bin", xlim=(0, 1))
    axes[0].set_title("a  实际河网的路径排列" if cn else "a  Actual channel-path organization", loc="left", pad=18)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set(xlabel="归一化时间（最大路径延迟 = 1）" if cn else "Normalized time (maximum route delay = 1)",
                ylabel="出口脉冲 / 输入脉冲峰值" if cn else "Outlet anomaly / input pulse peak", xlim=(-.25, 1.25), ylim=(0, 1.05))
    axes[1].set_title("b  各形态的平均出口响应" if cn else "b  Class-mean routed responses", loc="left", pad=18)
    axes[1].text(.03, .86, "虚线：无延迟输入" if cn else "Dashed: no-delay input", transform=axes[1].transAxes, fontsize=9, color="#667478")
    axes[2].set(xticks=range(3), xticklabels=("细长多支流", "主干少支流", "宽展多支流") if cn else ("Elongated", "Sparse", "Broad"),
                ylabel="出口 / 输入峰值" if cn else "Outlet / input peak", ylim=(0, 1.05))
    axes[2].set_title("c  每个河网的峰值传递" if cn else "c  Peak transmission per network", loc="left", pad=18)
    fig.suptitle("只改变河网路径，出口 DOC 信号怎样变化？" if cn else "How does channel organization reshape an identical DOC input signal?", x=.04, ha="left", fontsize=17)
    fig.text(.04, .02, "全部汇水区、无植被加权。恒定流量与相同输入；归一化恒速路由情景，非实测 DOC。面积权重基于河段中点，少支流型受离散分辨率影响。" if cn else
             "All catchments; no vegetation weights. Constant-flow normalized routing scenario, not field DOC. Midpoint discretization matters for sparse networks.", fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.99, bottom=.19, top=.77, wspace=.37)
    save(fig, "river_form_routing")

    gains = pd.read_csv(folder/"morphology_block_gains.csv")
    gains = gains[gains.space.eq("native")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    collections = ([('footprint', 'environment'), ('branching', 'environment'), ('paths', 'environment'), ('all_morphology', 'environment')],
                   [('all_morphology', 'without_footprint'), ('all_morphology', 'without_branching'), ('all_morphology', 'without_paths')])
    for k, (ax, choices) in enumerate(zip(axes, collections, strict=True)):
        labels = (["外轮廓", "分支组织", "路径组织", "全部形态"] if k == 0 else ["外轮廓的独立信息", "分支组织的独立信息", "路径组织的独立信息"]) if cn else (
            ["Footprint", "Branch organization", "Path organization", "All morphology"] if k == 0 else ["Unique footprint information", "Unique branch information", "Unique path information"])
        for i, (candidate, reference) in enumerate(choices):
            r = gains[gains.candidate.eq(candidate) & gains.reference.eq(reference)].iloc[0]
            color = COLORS[i % 3]
            ax.plot([r.ci_low_pct, r.ci_high_pct], [i, i], color=color, linewidth=1.8)
            ax.scatter(r.gain_pct, i, color=color, s=36)
            ax.text(17.5, i, f"{r.gain_pct:+.2f}%", ha="right", va="center", fontsize=9)
        ax.axvline(0, color="#879295", linewidth=.8, linestyle="--")
        ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels)-.5, -.5), xlim=(-2, 18), xticks=[0, 5, 10, 15],
               xlabel="站点中位 DOC 的 MAE 改善（%）" if cn else "Station-median DOC MAE reduction (%)")
        ax.set_title(("a  形态信息加入环境基线" if k == 0 else "b  控制其他形态信息") if cn else (
            "a  Added to environment + area" if k == 0 else "b  Other morphology blocks retained"), loc="left", pad=16)
    fig.suptitle("河网的哪一部分提供 DOC 信息？" if cn else "Which aspect of river morphology adds DOC information?", x=.04, ha="left", fontsize=17)
    fig.text(.04, .02, "297 站点、62 HUC4；五折区域留出，95% HUC4 bootstrap。站点中位浓度诊断，不是现有月度重建模型的性能增益。" if cn else
             "297 stations, 62 HUC4s; five blocked folds and 95% HUC4 bootstrap. Station-median diagnostic; not monthly reconstruction-model gains.", fontsize=9)
    fig.subplots_adjust(left=.16, right=.98, bottom=.19, top=.75, wspace=.8)
    save(fig, "morphology_information_blocks")
    (ROOT/f"figure_sources{suffix}.json").write_text(json.dumps({
        "figures": {str(p): sha256_file(p) for p in written}, "script_sha256": sha256_file(Path(__file__)),
        "tables": {str(p): sha256_file(p) for p in folder.glob("*") if p.is_file()},
        "synthetic_panel": "river_form_routing b/c: identical-input conservative normalized-delay scenario"}, indent=2)+'\n')


if __name__ == "__main__":
    main()
