"""Scientific figures of controlled fluctuation scales and arrival alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from analyze_doc_river_signal_timescale_v1 import ROOT, TYPES
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FuncFormatter, NullFormatter

from river_graph.experiments.provenance import sha256_file

PALETTE = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chinese", action="store_true")
    p.add_argument("--shortest-routes", action="store_true")
    args = p.parse_args()
    cn, suffix = args.chinese, "_cn" if args.chinese else ""
    if cn:
        path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    data, out = root/"analysis", root/"figures"
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(data/"scenario_summary.csv")
    context = pd.read_csv(data/"observed_and_controlled_context.csv", dtype=TYPES)
    source = pd.read_csv(data/"source_paths.csv", dtype=TYPES)
    metrics = pd.read_csv(data/"illustration_metrics.csv", dtype=TYPES)
    waves = pd.read_parquet(data/"illustration_waves.parquet")
    written = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out/f"{name}{suffix}.{extension}"
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    def curve(ax, *, cut=0., coherence=.5, memory=0., group="all", color="#257F88", label="", style="-", band=True):
        f = summary[summary.metric.eq("sd_ratio") & summary.minimum_coverage.eq(cut)
            & summary.coherence.eq(coherence) & summary.memory_allocation.eq(memory) & summary.group.eq(group)].sort_values("correlation_time")
        ax.plot(f.correlation_time, f.estimate, color=color, linestyle=style, label=label, linewidth=2)
        if band and f.ci_low.notna().any():
            ax.fill_between(f.correlation_time, f.ci_low, f.ci_high, color=color, alpha=.10, linewidth=0)

    def time_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks([.025, .1, .5, 2., 8.])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(1, color="#98A6AC", linestyle=":", linewidth=1)
        ax.set(xlabel="输入起伏持续尺度 / 平均名义传输时间" if cn else "Input correlation time / mean nominal travel time",
            ylabel="出口波动 / 同时上游混合波动 (SD)" if cn else "Outlet / instantaneous-mixture SD", ylim=(.28, 1.045))

    fig, axes = plt.subplots(2, 2, figsize=(12.9, 10.0))
    ax = axes[0, 0]
    curve(ax, label="只有路径长短差异" if cn else "Actual arrival paths", color="#257F88")
    curve(ax, memory=.5, label="再加入共同河段弥散" if cn else "+ Shared-corridor dispersion", color="#C5814A")
    time_axis(ax)
    ax.set_title("a  短促与缓慢变化的响应不同（32个河网）" if cn else "a  Fast and slow inputs differ (32 networks)", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax = axes[0, 1]
    for coherence, color, name in ((0., "#8D9FA6", "独立起伏" if cn else "Independent"),
            (.5, "#66749F", "部分共同起伏" if cn else "Partly common"),
            (1., "#257F88", "完全共同起伏" if cn else "Fully common")):
        curve(ax, coherence=coherence, color=color, label=name)
    time_axis(ax)
    ax.set_title("b  上游是否一起变化，决定错开的作用" if cn else "b  Source coherence governs delay smoothing", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax = axes[1, 0]
    for c, name, n in ((1, "细长型" if cn else "Elongated", 4), (3, "宽阔型" if cn else "Broad", 8)):
        for memory, style in ((0., "--"), (.5, "-")):
            addition = "路径" if cn and not memory else "路径＋共同弥散" if cn else "paths" if not memory else "paths + shared dispersion"
            curve(ax, cut=.8, memory=memory, group=f"class_{c}", color=PALETTE[c],
                label=f"{name} (n={n}) · {addition}", style=style, band=False)
    time_axis(ax)
    ax.set_title("c  同输入下的形态比较（覆盖≥80%）" if cn else "c  Same-input form comparison (coverage >=80%)", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax = axes[1, 1]
    names = {1: "细长型" if cn else "Elongated", 2: "主干主导" if cn else "Mainstem dominated", 3: "宽阔型" if cn else "Broad"}
    for c in (1, 2, 3):
        for high, marker in ((False, "o"), (True, "s")):
            g = context[context.cluster.eq(c) & context.covered_area_fraction.ge(.8).eq(high)]
            ax.scatter(g.observed_sd_ratio, g.sd_ratio, facecolor=PALETTE[c] if high else "white",
                edgecolor=PALETTE[c], marker=marker, s=44 if high else 32, linewidth=1,
                label=names[c] if high else None)
    ax.axvline(1, color="#859299", linestyle=":", linewidth=1)
    ax.axhline(1, color="#859299", linestyle=":", linewidth=1)
    ax.set(xscale="log", xlim=(.15, 5.2), ylim=(.28, 1.045),
        xlabel="实测月 DOC：出口 / 混合波动 (SD)" if cn else "Observed monthly DOC: receiving / mixture SD",
        ylabel="控制实验：出口 / 混合波动 (SD)" if cn else "Controlled experiment: outlet / mixture SD")
    ax.set_xticks([.2, .5, 1., 2., 4.])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title("d  共同起伏的被动模型不包含全部实测变化" if cn else "d  The positive-coherence control does not span field variation", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(.055, .018,
        "控制实验：相同输入统计、均匀速度代理、单位增益；不是实测停留时间或 DOC 去除。阴影：整片系统重抽的95%区间。\n"
        "c 为情景均值；d 参数固定且未拟合实测值。实心方块表示监测覆盖≥80%，不同形态仍混有地域差异。" if cn else
        "Controlled scenarios: identical input statistics, uniform-speed proxy, unit gain; no measured residence time or DOC removal.\n"
        "Bands: 95% whole-system bootstrap intervals. Panel c shows scenario means; d is not fitted to DOC. Filled squares: coverage >=80%.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1), w_pad=2.8, h_pad=2.8)
    save(fig, "river_form_timescale_response")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), gridspec_kw={"width_ratios": [1., 1.3, 1.3]})
    f = source[source.target.eq("401733105392404")].sort_values("normalized_delay")
    independent = f.independent_path_km.to_numpy()
    common = f.path_km.to_numpy()-independent
    ax = axes[0]
    ax.barh([0, 1], independent, color="#257F88", height=.46, label="各自路径" if cn else "Separate route")
    ax.barh([0, 1], common, left=independent, color="#C5814A", height=.46, label="共同河段" if cn else "Shared corridor")
    ax.set(yticks=[0, 1], yticklabels=f.source_station.tolist(), xlabel="实际裁剪路径长度 (km)" if cn else "Actual cropped path length (km)")
    for i, r in enumerate(f.itertuples()):
        ax.text(r.path_km+.04, i, f"{r.path_km:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, f.path_km.max()*1.25)
    ax.invert_yaxis()
    ax.set_title("a  Loch Vale 的两条真实路径" if cn else "a  Two real Loch Vale paths", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.23))
    for ax, forcing, title in ((axes[1], "synchronous", "b  输入一起起伏：路径使它们错开" if cn else "b  Common input: paths separate arrival"),
            (axes[2], "path_compensated", "c  输入原本错开：路径也能让它们重叠" if cn else "c  Offset input: paths can align arrival")):
        group = waves[waves.forcing.eq(forcing) & waves.cycle.le(2.)]
        reference = group[group.scenario.eq("aligned_arrivals")]
        ax.plot(reference.cycle, reference.instantaneous_mixture, color="#8D9FA6", linestyle=":", linewidth=1.5,
            label="上游即时混合" if cn else "Instantaneous input mixture")
        for scenario, color, label in (("actual_delays", "#257F88", "经真实路径" if cn else "Actual arrival delays"),
                ("actual_plus_shared_memory", "#C5814A", "再经共同弥散" if cn else "+ Shared dispersion")):
            g = group[group.scenario.eq(scenario)]
            r = metrics[metrics.forcing.eq(forcing) & metrics.scenario.eq(scenario)].iloc[0]
            ax.plot(g.cycle, g.outlet_anomaly, color=color, linewidth=2,
                label=f"{label} · {r.amplitude_ratio:.2f}×")
        ax.axhline(0, color="#B6C0C3", linewidth=.7)
        ax.set(xlabel="构造输入的周期数" if cn else "Constructed input cycles",
            ylabel="构造的浓度异常（任意单位）" if cn else "Constructed concentration anomaly", ylim=(-1.1, 1.1), xlim=(0, 2))
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -.23))
    fig.text(.065, .018,
        "示意输入按路径差构造，未使用实测 DOC 相位；数值为出口幅度 / 即时混合幅度。所有情景保持单位增益及平均路径时间。" if cn else
        "Constructed phases, not measured DOC. Ratios compare outlet and instantaneous-mixture amplitudes; all scenarios preserve unit gain and path means.", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1), w_pad=2.8)
    save(fig, "source_timing_and_arrival_alignment")
    (out/f"figure_sources{suffix}.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in [data/"scenario_summary.csv", data/"observed_and_controlled_context.csv",
            data/"source_paths.csv", data/"illustration_metrics.csv", data/"illustration_waves.parquet"]},
        "output_hashes": {str(p): sha256_file(p) for p in written},
        "description": "Actual mapped lengths, analytical controlled scenarios and explicitly constructed periodic input phases"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
