"""Plot real network geometry, laboratory DOC campaigns and a separate optical case."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from shapely.ops import substring

from river_graph.analysis.river_event_observations import mapped_network
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
RAW = Path("data/raw/river_event_observations_v1")
RECEIVER_COLORS = {"C7": "#257F88", "C9": "#C5814A", "C16": "#66749F"}
SERIES_COLORS = ("#257F88", "#C5814A", "#344B6B")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    out, analysis = ROOT / "figures", ROOT / "analysis"
    out.mkdir(exist_ok=True)
    cadence = pd.read_csv(analysis / "station_cadence.csv")
    stations = pd.read_csv(analysis / "stations.csv")
    connections = pd.read_csv(analysis / "monitored_confluences.csv")
    coverage = pd.read_csv(analysis / "confluence_event_coverage.csv")
    variation = pd.read_csv(analysis / "campaign_variation.csv")
    summary = json.loads((analysis / "summary.json").read_text())
    optical_summary = json.loads((analysis / "optical_summary.json").read_text())
    doc = pd.read_parquet(analysis / "laboratory_doc.parquet")
    flow = pd.read_parquet(analysis / "daily_discharge.parquet")
    features = json.loads((RAW / "streams.geojson").read_text())["features"]
    graph, snaps, lines = mapped_network(features, stations)
    suffix, outputs = ("_cn" if cn else ""), []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out / f"{name}{suffix}.{extension}"
            fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.3, 9.7))
    fig.subplots_adjust(left=.065, right=.96, bottom=.15, top=.86, wspace=.34, hspace=.43)
    ax = axes[0, 0]
    ax.add_collection(LineCollection([np.asarray(x.coords) / 1000 for x in lines], colors="#C5D2D4", linewidths=.55))
    arc_lines = {f["properties"]["ARCID"]: line for f, line in zip(features, lines)}
    for row in connections.itertuples():
        color = RECEIVER_COLORS.get(row.receiver, "#72878C")
        for site in (row.source_a, row.source_b):
            route = nx.shortest_path(graph, f"site:{site}", f"site:{row.receiver}", weight="length_m")
            routed_segments = []
            for a, b in pairwise(route):
                edge = graph[a][b]
                cropped = substring(arc_lines[edge["arcid"]], edge["along_start_m"], edge["along_end_m"])
                if cropped.geom_type == "LineString":
                    routed_segments.append(np.asarray(cropped.coords) / 1000)
            ax.add_collection(LineCollection(routed_segments, colors=color, linewidths=1.4))
    points = snaps[snaps.mapped]
    ax.scatter(points.x_m / 1000, points.y_m / 1000, s=15, color="white", edgecolor="#435B65", zorder=5)
    for site, color in RECEIVER_COLORS.items():
        row = snaps[snaps.site.eq(site)].iloc[0]
        ax.scatter(row.x_m / 1000, row.y_m / 1000, s=30, color=color, zorder=6)
        offset = {"C7": (7, -12), "C9": (14, -4), "C16": (7, 6)}[site]
        ax.annotate(site, (row.x_m / 1000, row.y_m / 1000), xytext=offset,
                    textcoords="offset points", fontsize=9, color=color)
    ax.autoscale()
    ax.set(aspect="equal", xlabel="SWEREF 99 TM easting (km)", ylabel="Northing (km)")
    ax.set_title("a  真实河网与监测位置" if cn else "a  Real network and monitoring positions", loc="left", fontsize=11)

    ax = axes[0, 1]
    values = [cadence.median_gap_days, cadence.spring_median_gap_days]
    bp = ax.boxplot(values, tick_labels=("全年", "春季") if cn else ("All year", "Spring"),
                    patch_artist=True, widths=.4, showfliers=False)
    for patch, color in zip(bp["boxes"], SERIES_COLORS):
        patch.set(facecolor=color, alpha=.2, edgecolor=color)
    for i, v in enumerate(values, 1):
        jitter = np.random.default_rng(42+i).uniform(-.09, .09, len(v))
        ax.scatter(i+jitter, v, s=14, color=SERIES_COLORS[i-1], alpha=.65)
    ax.axhline(28, color="#93A2A7", linestyle="--", linewidth=1)
    ax.text(1.02, 28.4, "既有采样审计：28 天" if cn else "Original sampling audit: 28 days", fontsize=8, color="#6B7E83")
    ax.set(ylim=(0, 32), ylabel="DOC 采样间隔（天）" if cn else "DOC sampling interval (days)")
    ax.set_title("b  实验室 DOC 采样更密集" if cn else "b  Laboratory DOC sampling cadence", loc="left", fontsize=11)

    ax = axes[0, 2]
    groups = coverage.groupby("receiver").agg(total=("year", "size"), usable=("all_three_span_response", "sum"))
    for i, (site, row) in enumerate(groups.iterrows()):
        ax.barh(i, row.total, color="#E2E8E9", height=.5)
        ax.barh(i, row.usable, color=RECEIVER_COLORS.get(site, "#72878C"), height=.5)
        ax.text(row.total+.3, i, f"{int(row.usable)} / {int(row.total)}", va="center", fontsize=9)
    ax.set(yticks=range(len(groups)), yticklabels=groups.index, xlim=(0, groups.total.max()+3),
           xlabel="可用 / 候选春季窗口" if cn else "Spanning / candidate spring windows")
    ax.set_title("c  三个位置均覆盖上升和回落" if cn else "c  All three sites span rise and recession", loc="left", fontsize=11)
    ax.invert_yaxis()

    ax = axes[1, 0]
    for site, sub in variation.groupby("receiver"):
        ax.scatter(sub.mean_upstream_cv, sub.receiver_cv, s=42, color=RECEIVER_COLORS[site],
                   label=f"{site} (n={len(sub)})", edgecolor="white", linewidth=.6)
    maximum = max(variation.mean_upstream_cv.max(), variation.receiver_cv.max()) * 1.18
    ax.plot([0, maximum], [0, maximum], color="#8E9FA4", linestyle="--", linewidth=1)
    ax.text(.025, .31, "线下：出口波动较小" if cn else "Below line: smaller outlet variation", fontsize=8, color="#5B737A")
    ax.set(xlim=(0, maximum), ylim=(0, maximum),
           xlabel="两条上游平均 DOC SD / 均值" if cn else "Mean upstream DOC SD / mean",
           ylabel="出口 DOC SD / 均值" if cn else "Outlet DOC SD / mean")
    ax.set_title("d  同轮采样的实际 DOC 波动" if cn else "d  DOC variation at matched campaigns", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    examples = []
    for ax, receiver, letter in zip(axes[1, 1:], ("C7", "C16"), ("e", "f")):
        # Earliest usable comparison at each site, selected without DOC outcomes.
        row = variation[variation.receiver.eq(receiver)].sort_values("year").iloc[0]
        peak = pd.Timestamp(row.peak_date)
        start, end = peak - pd.Timedelta(days=14), peak + pd.Timedelta(days=14)
        q = flow[(flow.site.eq(receiver)) & flow.date_local.between(start, end) & flow.value.notna()]
        right = ax.twinx()
        right.spines["top"].set_visible(False)
        right.plot(q.date_local.sub(peak).dt.days, q.value / q.value.max(), color="#C6D0D2", linewidth=1.4, zorder=0)
        right.set(ylim=(0, 1.05), ylabel="相对流量" if cn else "Relative flow")
        right.tick_params(colors="#7F939A", labelsize=8)
        ax.set_zorder(right.get_zorder()+1)
        ax.patch.set_visible(False)
        for site, color, marker in zip((row.source_a, row.source_b, receiver), SERIES_COLORS, ("o", "^", "s")):
            sub = doc[doc.site.eq(site) & doc.date_local.between(start, end) & doc.value.notna()]
            ax.scatter(sub.date_local.sub(peak).dt.days, sub.value, s=30, color=color, marker=marker, label=site, edgecolor="white", linewidth=.5)
        ax.axvline(0, color="#B5C3C6", linestyle=":", linewidth=1)
        ax.set(xlim=(-15, 15), xlabel="相对流量峰日（天）" if cn else "Days from flow maximum", ylabel="DOC (mg C/L)")
        ax.set_title(f"{letter}  {row.source_a} + {row.source_b} → {receiver} · {int(row.year)}", loc="left", fontsize=11, pad=28)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.01), borderaxespad=0)
        examples.append({"receiver": receiver, "year": int(row.year), "selection": "Earliest eligible observed window"})
    fig.suptitle("真实河网中的 DOC 汇流观测" if cn else "Observed DOC across real river confluences", x=.065, y=.99, ha="left", fontsize=17)
    fig.text(.065, .93, (f"SITES：{summary['n_doc_stations']} 个站点、{summary['n_valid_laboratory_doc']:,} 次实验室测量；地图显示 Krycklan 内的真实汇流配置。" if cn else
             f"SITES collection: {summary['n_doc_stations']} sites, {summary['n_valid_laboratory_doc']:,} laboratory values; the map shows real Krycklan confluences."), fontsize=10)
    fig.text(.065, .065, ("d：11 个同轮采样窗口中，9 个出口相对波动较小（8 个在 C7）；重复年份按汇流点区分。\n"
              "e/f：实测 DOC 点与每日流量；每轮三站采样跨度 ≤12 小时。地图颜色标记汇流配置，尚未分配原来的三类形态。" if cn else
              "d: 9/11 matched windows show smaller outlet variation (8 at C7); repeated years are grouped by confluence.\n"
              "e/f: Laboratory points against daily flow; campaign spans ≤12 h. Map colors identify confluences; original morphology classes are not assigned."),
             fontsize=9, color="#53696E")
    save(fig, "observed_confluences_and_doc")

    paired = pd.read_parquet(analysis / "optical_paired_hours.parquet")
    blocks = pd.read_csv(analysis / "optical_paired_blocks.csv")
    selected = blocks[blocks.duration_hours.ge(24)].sort_values("start").head(3)
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 9.4), sharex=False)
    fig.subplots_adjust(left=.08, right=.975, top=.83, bottom=.13, hspace=.5, wspace=.24)
    for i, row in enumerate(selected.itertuples()):
        sub = paired[paired.block.eq(row.block)].copy()
        elapsed = sub.date_time_source_clock.sub(sub.date_time_source_clock.min()).dt.total_seconds() / 3600
        for ax, field, normalize, label in ((axes[i, 0], "Discharge (m3 s-1)", True, "各站相对流量" if cn else "Site-normalized flow"),
                                             (axes[i, 1], "DOC (mg l-1)", False, "光学 DOC 估计 (mg/L)" if cn else "Optical DOC estimate (mg/L)")):
            for suffix_name, color, name in (("upstream", SERIES_COLORS[0], "上游 SN" if cn else "Upstream SN"),
                                             ("receiver", SERIES_COLORS[1], "下游 FITT" if cn else "Receiver FITT")):
                values = sub[f"{field}_{suffix_name}"]
                values = values / values.max() if normalize else values
                ax.plot(elapsed, values, color=color, linewidth=1.4, marker="o", markersize=2.2, label=name)
            ax.set(xlabel="相对记录开始（小时）" if cn else "Hours from record start", ylabel=label)
            ax.set_title(f"{str(row.start)[:10]} · n={row.n_paired_hours}", loc="left", fontsize=10)
            if i == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("小时级上下游记录：独立的光学案例" if cn else "Paired hourly records: a separate optical case", x=.08, y=.99, ha="left", fontsize=17)
    n_records, n_blocks = optical_summary["n_paired_hourly_records"], optical_summary["n_paired_blocks_ge24hours"]
    fig.text(.08, .91, (f"Turbolo：{n_records} 对小时记录、{n_blocks} 段跨度 ≥24 小时连续记录；按时间选最早三段。" if cn else
             f"Turbolo: {n_records} paired hours and {n_blocks} continuous blocks spanning ≥24 h; the earliest three are shown."), fontsize=10)
    fig.text(.08, .045, ("DOC 来自校正荧光估计，部分高浊度峰值在原论文中经过统计补算；发布表没有逐点标记。\n"
              "原始时钟时区未注明；保留发布时钟。此图用于准备事件诊断，不作为实验室 DOC 或三类形态效应的验证。" if cn else
              "DOC is estimated from corrected fluorescence; the paper reports statistical retrieval of some high-turbidity peaks without row flags.\n"
              "Source timezone is unspecified; the source clock is retained. This prepares event diagnostics, not laboratory DOC or three-form effect validation."),
             fontsize=9, color="#53696E")
    save(fig, "paired_hourly_optical_case")
    inputs = [*analysis.glob("*.csv"), *analysis.glob("*.parquet"), analysis / "summary.json",
              analysis / "optical_summary.json", RAW / "streams.geojson",
              Path("scripts/plot_doc_river_event_observations_v1.py"), Path("src/river_graph/analysis/river_event_observations.py")]
    receipt = {"input_hashes": {str(p): sha256_file(p) for p in inputs},
               "output_hashes": {str(p): sha256_file(p) for p in outputs},
               "laboratory_examples": examples, "chinese": cn,
               "doc_points_not_interpolated": True, "optical_case_separate": True}
    (out / f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Saved {len(outputs)} figures")


if __name__ == "__main__":
    main()
