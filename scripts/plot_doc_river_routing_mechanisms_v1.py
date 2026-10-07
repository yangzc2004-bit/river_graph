"""Plot measured river geometry and controlled DOC routing experiments."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager
from plot_doc_river_planform_v1 import draw_network

from river_graph.analysis.river_routing_mechanisms import two_branch_process
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1")
PLANFORM = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis")
CACHE = Path("data/raw/river_planform_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")
PROCESS_ORDER = ("conservative_uniform", "conservative_flow_speed", "uniform_processing", "common_processing")


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
    names = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated / tributary-rich", "Mainstem dominated / sparse", "Broad / tributary-rich")
    process_names = ("恒速、无损耗", "支流速度随流量份额变化", "全路径相同处理率", "汇合后处理率较高") if cn else (
        "Uniform speed / conservative", "Flow-sensitive branch speed", "Uniform processing", "Higher common-segment processing")
    suffix, written = "_cn" if cn else "", []
    sources = [a/n for n in ("network_representatives.csv", "whole_network_routing.csv", "whole_class_summary.csv",
        "reach_resolution_class_summary.csv",
        "whole_matched_contrasts.csv", "whole_matched_evidence.csv", "whole_representative_traces.parquet",
        "tributary_routing.csv", "tributary_contrasts.csv", "tributary_representative_traces.parquet", "arrival_alignment.csv")]
    sources += [PLANFORM/"classes.csv", Path("scripts/plot_doc_river_planform_v1.py"),
                Path("src/river_graph/topology/river_planform.py"), CACHE/"flowlines.sqlite"]

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    reps = pd.read_csv(a/"network_representatives.csv", dtype={"station": str})
    classes = pd.read_csv(PLANFORM/"classes.csv", dtype={"station": str})
    reps = reps.merge(classes, on=["station", "comid", "cluster"], validate="one_to_one")
    whole = pd.read_csv(a/"whole_network_routing.csv", dtype={"station": str})
    actual = whole[whole.scenario.eq("actual_paths")]
    summary = pd.read_csv(a/"whole_class_summary.csv")
    resolution = pd.read_csv(a/"reach_resolution_class_summary.csv")
    traces = pd.read_parquet(a/"whole_representative_traces.parquet")
    paired = pd.read_csv(a/"whole_matched_evidence.csv")
    paired = paired[paired.scenario.eq("actual_paths") & paired.class_a.eq(1) & paired.class_b.eq(3)]
    contrast = pd.read_csv(a/"whole_matched_contrasts.csv")
    contrast = contrast[contrast.scenario.eq("actual_paths") & contrast.class_a.eq(1) & contrast.class_b.eq(3)]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.6))
    connection = sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True)
    for i, row in enumerate(reps.sort_values("cluster").itertuples()):
        draw_network(axes[0, i], connection, row, COLORS[i])
        axes[0, i].set_title(f"{'abc'[i]}  {names[i]}\n{row.station}", loc="left", pad=9, fontsize=11)
        for directory, ext in (("members_full", "npz"), ("basins", "json")):
            sources.append(CACHE/directory/f"comid_{int(row.comid)}.{ext}")
        ax = axes[1, i]
        mean = actual[actual.station.eq(row.station)].pulse_centroid.iloc[0]
        for scenario, color, style, label in (
            ("actual_paths", COLORS[i], "-", "真实路径" if cn else "Actual paths"),
            ("half_spread", "#A0AEB2", "--", "路径分散度减半" if cn else "Half path spread"),
            ("zero_spread", "#33494F", ":", "相同到达时刻" if cn else "Equal arrival times")):
            t = traces[traces.station.eq(row.station) & traces.scenario.eq(scenario)]
            ax.plot(t.time-mean, t.outlet_anomaly, color=color, linestyle=style, linewidth=1.7, label=label)
        ax.set(xlim=(-3, 3), ylim=(0, 1.04), xlabel="相对平均到达时刻（情景单位）" if cn else "Time relative to mean arrival (scenario units)",
               ylabel="出口脉冲 / 输入峰值" if cn else "Outlet pulse / input peak")
        ax.set_title(f"{'def'[i]}  "+("同一河网的路径对照" if cn else "Within-network path controls"), loc="left", pad=12)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    connection.close()
    ax = axes[2, 0]
    for i, group in enumerate((1, 2, 3)):
        s = actual[actual.cluster.eq(group)]
        r = summary[summary.scenario.eq("actual_paths") & summary.cluster.eq(group) & summary.metric.eq("pulse_peak")].iloc[0]
        ax.scatter(i+np.random.default_rng(42).uniform(-.2, .2, len(s)), s.pulse_peak, color=COLORS[i], s=11, alpha=.45)
        ax.plot([i, i], [r.ci_low, r.ci_high], color="#263B42", linewidth=2)
        ax.scatter(i, r["mean"], facecolor="white", edgecolor="#263B42", s=42, zorder=4)
        refined = resolution[resolution.cluster.eq(group) & resolution.metric.eq("pulse_peak")].iloc[0]
        ax.plot([i+.22, i+.22], [refined.ci_low, refined.ci_high], color="#8D9FA4", linewidth=1.3)
        ax.scatter(i+.22, refined["mean"], color="#8D9FA4", s=27, marker="s", zorder=4)
        ax.text(i, .98, f"n={len(s)}", ha="center", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set(xticks=range(3), xticklabels=("细长多支流", "主干少支流", "宽展多支流") if cn else ("Elongated", "Sparse", "Broad"),
           ylabel="出口 / 输入峰值" if cn else "Outlet / input peak", ylim=(0, 1.04))
    ax.set_title("g  "+("全部河网：○ 中点，■ 沿河段" if cn else "All networks: ○ midpoint, ■ within reach"), loc="left", pad=12)
    ax = axes[2, 1]
    p = paired[paired.metric.eq("pulse_peak")]
    r = contrast[contrast.metric.eq("pulse_peak")].iloc[0]
    ax.plot([0, .5], [0, .5], color="#9CAAAF", linestyle="--", linewidth=.8)
    ax.scatter(p.value_a, p.value_b, color=COLORS[2], s=34, edgecolor="white", linewidth=.4)
    ax.set(xlim=(0, .5), ylim=(0, .5), xlabel="细长多支流型峰值" if cn else "Elongated-network peak",
           ylabel="宽展多支流型峰值" if cn else "Broad-network peak")
    ax.set_title("h  "+("22 对相近环境河网" if cn else "22 environmentally matched pairs"), loc="left", pad=12)
    ax.text(.035, .96, f"B − E: {r.difference_b_minus_a:+.3f}\n95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", transform=ax.transAxes, va="top", fontsize=9)
    ax = axes[2, 2]
    for i, group in enumerate((1, 2, 3)):
        rs = summary[summary.scenario.eq("actual_paths") & summary.cluster.eq(group)]
        for j, period in enumerate((1, 4, 12)):
            r = rs[rs.metric.eq(f"period_{period}_gain")].iloc[0]
            x = j+(i-1)*.16
            ax.plot([x, x], [r.ci_low, r.ci_high], color=COLORS[i], linewidth=1.3)
            ax.scatter(x, r["mean"], color=COLORS[i], s=25)
    ax.set(xticks=range(3), xticklabels=("1", "4", "12"), xlabel="输入周期（情景单位）" if cn else "Input period (scenario units)",
           ylabel="信号幅度传递比" if cn else "Signal amplitude gain", ylim=(0, 1.04))
    ax.set_title("i  "+("不同时间尺度的传递" if cn else "Transmission depends on timescale"), loc="left", pad=12)
    fig.suptitle("河网形态怎样重新排列 DOC 的到达时间？" if cn else "How does river form reorganize the arrival of a DOC signal?",
                 x=.045, ha="left", fontsize=17, y=.99)
    fig.text(.045, .955, "真实河道与固定代表站点；相同输入浓度、总流量和归一化速度。" if cn else
             "Measured channels and fixed representatives; identical concentration forcing, total flow and normalized velocity.", fontsize=10)
    fig.text(.045, .018, ("距离 / √流域面积；时间不是实测天数或月份。每幅地图保留真实形状，使用独立比例尺。d–f 对照仅收缩路径分散度，平均时距不变。\n"
              "g、i 为 HUC4 bootstrap 95% 区间；h 为固定配对 HUC4 区间。沿段输入为五点均匀分布敏感性。无损耗情景保留总输入负荷与恒定浓度。") if cn else
             ("Distance / √basin area; time is not measured days/months. Maps retain native form with independent physical extents. Controls preserve mean delay.\n"
              "g/i: HUC4 bootstrap 95%; h: paired-HUC4 95%. Five inputs/reach check spatial resolution. Conservative routing preserves load and steady concentration."), fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.90, bottom=.115, wspace=.36, hspace=.43)
    save(fig, "real_forms_and_routed_signals")

    branch = pd.read_csv(a/"tributary_routing.csv", dtype={"target": str})
    btraces = pd.read_parquet(a/"tributary_representative_traces.parquet")
    bc = pd.read_csv(a/"tributary_contrasts.csv")
    junction_names = ("真实汇合位置", "共同河段较短", "共同河段较长") if cn else ("Observed split", "Short common segment", "Long common segment")
    junction_colors = ("#A7B4B6", COLORS[0], COLORS[1])
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0))
    for i, process in enumerate(("conservative_uniform", "conservative_flow_speed", "common_processing")):
        ax = axes[0, i]
        for j, junction in enumerate(("observed", "short_common", "long_common")):
            s = btraces[btraces.process.eq(process) & btraces.weighting.eq("area_proxy") & btraces.junction.eq(junction)]
            ax.plot(s.time, s.outlet_anomaly, color=junction_colors[j], linestyle=(":", "-", "--")[j], linewidth=1.8, label=junction_names[j])
        ax.set(xlabel="出口时间（情景单位）" if cn else "Outlet time (scenario units)",
               ylabel="出口脉冲 / 输入峰值" if cn else "Outlet pulse / input peak", ylim=(0, 1.02))
        ax.set_title(f"{'abc'[i]}  {process_names[PROCESS_ORDER.index(process)]}", loc="left", pad=14)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for col, metric in enumerate(("pulse_peak", "pulse_centroid", "anomaly_mass_fraction")):
        ax = axes[1, col]
        ax.axvline(0, color="#9CAAAF", linestyle="--", linewidth=.8)
        for j, process in enumerate(PROCESS_ORDER):
            s = bc[bc.comparison.eq("long_minus_short_common") & bc.process.eq(process) & bc.weighting.eq("area_proxy") &
                   bc.forcing.eq("synchronous") & bc.metric.eq(metric)]
            r, h = s[s.unit.eq("component")].iloc[0], s[s.unit.eq("huc4")].iloc[0]
            ax.plot([h.ci_low, h.ci_high], [j+.08, j+.08], color="#BEC8CB", linewidth=1.3)
            ax.plot([r.ci_low, r.ci_high], [j, j], color=COLORS[col], linewidth=2)
            ax.scatter(r.estimate, j, color=COLORS[col], s=32, zorder=3)
        ax.set(yticks=range(4), yticklabels=process_names if col == 0 else [""]*4, ylim=(3.5, -.5))
        title = ("峰值变化", "平均到达时间变化", "保留负荷比例变化")[col] if cn else ("Peak change", "Arrival-centroid change", "Retained-load change")[col]
        ax.set_title(f"{'def'[col]}  {title}", loc="left", pad=14)
        ax.set_xlabel("共同河段较长 − 较短" if cn else "Long − short common segment")
    fig.suptitle("汇合位置何时会改变下游 DOC 信号？" if cn else "When does junction position change the downstream DOC signal?",
                 x=.045, ha="left", fontsize=17, y=.99)
    fig.text(.045, .93, ("固定两条源站到出口的总距离，仅重新分配支流与共同河段。上排：固定代表支流组合；下排：38 个下游站。") if cn else
             "Two fixed source-to-outlet distances; only branch/common partition changes. Top: fixed representative pair. Bottom: 38 receivers.", fontsize=9)
    fig.text(.045, .018, ("121 个真实独立支流组合，38 个下游站，17 个共享站点系统；先在下游站内平均。粗线：系统 bootstrap 95%；灰线：HUC4 敏感性。\n"
              "流速随流量和处理率均为预设情景，未用 DOC 拟合；两条监测支流不是完整流域预算。虚拟汇合位置不代表真实河道改造。") if cn else
             ("121 independent tributary pairs / 38 receivers / 17 connected systems; receiver-first averages. Thick: system 95%; grey: HUC4 sensitivity.\n"
              "Speeds and processing are imposed, not DOC-fitted; two monitored branches are not a full basin budget. Virtual splits are not real channel alterations."), fontsize=8)
    fig.subplots_adjust(left=.215, right=.985, top=.86, bottom=.15, wspace=.31, hspace=.50)
    save(fig, "junction_process_experiment")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.1))
    selected = branch[branch.pair_id.eq(btraces.pair_id.iloc[0]) & branch.process.eq("conservative_uniform") &
                      branch.weighting.eq("area_proxy") & branch.junction.eq("observed")]
    forcing_names = ("同时输入", "支流 B 提前", "支流 B 延后") if cn else ("Synchronous input", "Branch B leads", "Branch B lags")
    ax = axes[0]
    for i, forcing in enumerate(("synchronous", "branch_b_leads", "branch_b_lags")):
        r = selected[selected.forcing.eq(forcing)].iloc[0]
        paths = np.array([r.branch_a_path, r.branch_b_path])+r.common_path
        _, time, pulse = two_branch_process(paths, r.common_path, r.weight_a, offset_b=r.offset_b)
        ax.plot(time, pulse, color=COLORS[i], linewidth=1.7, label=forcing_names[i])
    r = selected.iloc[0]
    paths = np.array([r.branch_a_path, r.branch_b_path])+r.common_path
    _, time, pulse = two_branch_process(paths, r.common_path, r.weight_a, offset_b=paths[0]-paths[1])
    ax.plot(time, pulse, color="#263B42", linestyle="--", linewidth=1.3, label="出口同时到达" if cn else "Arrival aligned")
    ax.set(xlabel="出口时间（情景单位）" if cn else "Outlet time (scenario units)", ylabel="出口 / 输入峰值" if cn else "Outlet / input peak", ylim=(0, 1.03))
    ax.set_title("a  "+("同一河网，不同到达相位" if cn else "Same network, different arrival phases"), loc="left", pad=16)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    s = branch[branch.process.eq("conservative_uniform") & branch.forcing.eq("synchronous") & branch.junction.eq("observed")]
    values = s.groupby(["target", "weighting"]).pulse_peak.mean().unstack("weighting")
    ax.plot([0, 1], [0, 1], color="#A0AEB2", linestyle="--", linewidth=.8)
    ax.scatter(values.area_proxy, values.balanced, s=30, color=COLORS[0], alpha=.8, edgecolor="white", linewidth=.4)
    r = bc[bc.comparison.eq("balanced_minus_area") & bc.process.eq("conservative_uniform") & bc.forcing.eq("synchronous") &
           bc.unit.eq("component") & bc.metric.eq("pulse_peak")].iloc[0]
    ax.text(.03, .97, f"Δ: {r.estimate:+.3f}\n95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", va="top", transform=ax.transAxes, fontsize=9)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="面积代理流量的峰值" if cn else "Peak with area-proxy flow", ylabel="平衡两支流的峰值" if cn else "Peak with balanced branch flow")
    ax.set_title("b  "+("支流更平衡，峰值通常更低" if cn else "Balanced branches usually reduce peaks"), loc="left", pad=16)
    ax = axes[2]
    labels = ("提前 / 面积权重", "延后 / 面积权重", "提前 / 平衡支流", "延后 / 平衡支流") if cn else (
        "Lead / area proxy", "Lag / area proxy", "Lead / balanced", "Lag / balanced")
    choices = [(w, c) for w in ("area_proxy", "balanced") for c in ("leading_minus_synchronous", "lagging_minus_synchronous")]
    for j, (weighting, comparison) in enumerate(choices):
        r = bc[bc.comparison.eq(comparison) & bc.weighting.eq(weighting) & bc.process.eq("conservative_uniform") &
               bc.metric.eq("pulse_peak") & bc.unit.eq("component")].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [j, j], color=COLORS[j % 2], linewidth=2)
        ax.scatter(r.estimate, j, color=COLORS[j % 2], s=32)
    ax.axvline(0, color="#9CAAAF", linestyle="--", linewidth=.8)
    ax.set(yticks=range(4), yticklabels=labels, ylim=(3.5, -.5), xlabel="峰值变化：不同步 − 同步输入" if cn else "Peak change: shifted − synchronous input")
    ax.set_title("c  "+("到达相位可以加强或抵消缓冲" if cn else "Arrival phase can offset buffering"), loc="left", pad=16)
    fig.suptitle("支流平衡与到达时间共同塑造 DOC 脉冲" if cn else "Branch balance and arrival timing jointly shape the DOC pulse",
                 x=.045, ha="left", fontsize=17, y=.99)
    fig.text(.045, .90, "相同源脉冲积分与总流量；仅改变两支流的流量份额或脉冲时刻。" if cn else
             "Identical integrated source pulses and total flow; only branch flow shares or pulse timing change.", fontsize=9)
    fig.text(.045, .018, ("恒速、无损耗路由。b、c：38 个下游站；c 使用共享站点系统 95% 区间。所有情景总异常负荷 = 1。\n"
              "a 为按路径差中位数选出的固定组合；不是按效果选例。出口到达对齐可恢复输入峰值，说明缓冲依赖路径与脉冲相位共同作用。") if cn else
             ("Uniform-speed conservative routing. b/c: 38 receivers; c: connected-system 95%. Integrated anomaly load remains one in every case.\n"
              "a: fixed median path-imbalance pair, not effect-selected. Aligning arrivals restores input peak, demonstrating the joint role of paths and phase."), fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.78, bottom=.21, wspace=.61)
    save(fig, "branch_balance_and_arrival_phase")
    manifest = {"language": "Chinese" if cn else "English", "source_hashes": {str(p): sha256_file(p) for p in sources},
                "figure_hashes": {str(p): sha256_file(p) for p in written}, "generator_sha256": sha256_file(Path(__file__)),
                "data_vs_scenario": "Mapped river geometry; pulse/rate/velocity experiments are controlled scenarios, not measured DOC."}
    (out/f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Rendered {len(written)} {manifest['language']} figures")


if __name__ == "__main__":
    main()
