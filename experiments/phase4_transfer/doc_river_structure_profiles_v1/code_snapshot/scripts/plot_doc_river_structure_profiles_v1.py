"""Plot real river forms, continuous mechanism profiles and monthly DOC evidence."""
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
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from plot_doc_river_planform_v1 import draw_network, segments

from river_graph.analysis.river_structure_profiles import POSITION_ORDER
from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import project_geometry, read_cached_lines

ROOT = Path("experiments/phase4_transfer/doc_river_structure_profiles_v1")
CACHE = Path("data/raw/river_planform_v1")
COLORS = {1: "#257F88", 2: "#C5814A", 3: "#66749F"}
STORAGE_COLORS = ("#E5E9E8", "#8BC3C6", "#7585B4", "#354F6A")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "savefig.dpi": 210, "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(a/"station_mechanism_profiles.csv", dtype={"station": str, "huc4": str})
    summary = pd.read_csv(a/"form_profile_summary.csv")
    counts = pd.read_csv(a/"storage_position_counts.csv")
    paired = pd.read_csv(a/"matched_chain_summary.csv")
    evidence = pd.read_csv(a/"matched_chain_evidence.csv")
    flow = pd.read_csv(a/"field_flow_context.csv")
    reps = pd.read_csv(a/"representatives.csv", dtype={"station": str})
    forms = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else (
        "Elongated /\ntributary-rich", "Mainstem dominated /\nsparse", "Broad /\ntributary-rich")
    short = ("细长", "主干型", "宽展") if cn else ("Elongated", "Sparse", "Broad")
    inputs = list(a.glob("*.csv"))+[Path(__file__), Path("scripts/plot_doc_river_planform_v1.py"),
        Path("src/river_graph/topology/river_planform.py"), Path("cache/nldplus_vaa.parquet"), CACHE/"flowlines.sqlite"]
    outputs = []
    suffix = "_cn" if cn else ""

    def save(fig, name):
        for extension in ("png", "pdf"):
            p = out/f"{name}{suffix}.{extension}"
            fig.savefig(p, bbox_inches="tight", facecolor="white")
            outputs.append(p)
        plt.close(fig)

    def distribution(ax, metric, ylabel, title, factor=1):
        values = [frame.loc[frame.cluster.eq(c), metric].dropna().to_numpy()*factor for c in (1, 2, 3)]
        box = ax.boxplot(values, positions=[1, 2, 3], widths=.5, patch_artist=True, showfliers=False,
                         medianprops={"color": "#243B43", "linewidth": 1.2},
                         whiskerprops={"color": "#758486"}, capprops={"color": "#758486"})
        for c, patch in enumerate(box["boxes"], 1):
            patch.set(facecolor=COLORS[c], alpha=.22, edgecolor=COLORS[c])
        for c, v in enumerate(values, 1):
            jitter = np.random.default_rng(100+c).uniform(-.14, .14, len(v))
            ax.scatter(c+jitter, v, color=COLORS[c], s=8, alpha=.35, edgecolor="none", zorder=2)
            row = summary[(summary.cluster.eq(c)) & summary.metric.eq(metric)].iloc[0]
            ax.plot([c+.24]*2, [factor*row.ci_low, factor*row.ci_high], color="#243B43", linewidth=1.3)
            ax.scatter(c+.24, factor*row["mean"], s=25, facecolor="white", edgecolor="#243B43", zorder=5)
        ax.set(xticks=[1, 2, 3], xticklabels=[f"{short[c-1]}\nn={len(values[c-1])}" for c in (1, 2, 3)],
               ylabel=ylabel, xlim=(.5, 3.5))
        ax.set_title(title, loc="left", fontsize=11, pad=10)
        ax.grid(axis="y", color="#EBEEEE", linewidth=.7, zorder=0)

    # Distributions and means share the same station-instance denominators.
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.6))
    fig.subplots_adjust(left=.07, right=.985, top=.89, bottom=.14, wspace=.34, hspace=.48)
    distribution(axes[0, 0], "arrival_dispersion", "路径 SD / 均值" if cn else "Path SD / mean",
                 "a  全河网到达分散" if cn else "a  Whole-network arrival dispersion")
    distribution(axes[0, 1], "major_common_share", "共同主干 / 加权总路径 (%)" if cn else "Common / weighted total path (%)",
                 "b  主要汇流后的共同路径" if cn else "b  Shared path after the major confluence", 100)
    ax = axes[0, 2]
    bottom = np.zeros(3)
    position_names = ("选定路径无湖库", "仅独立支流", "仅共同主干", "两处都有") if cn else (
        "None on selected paths", "Independent paths only", "Shared trunk only", "Both segments")
    for pos, color, label in zip(POSITION_ORDER, STORAGE_COLORS, position_names):
        values = counts[counts.storage_position.eq(pos)].sort_values("cluster").fraction.to_numpy()*100
        ax.bar([1, 2, 3], values, bottom=bottom, color=color, width=.54, label=label, edgecolor="white", linewidth=.7)
        for x, v, b in zip((1, 2, 3), values, bottom):
            if v >= 10:
                ax.text(x, b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white" if pos == "branches_and_common" else "#263B42")
        bottom += values
    numbers = counts.groupby("cluster").n.sum()
    ax.set(xticks=[1, 2, 3], xticklabels=[f"{short[c-1]}\nn={int(numbers.loc[c])}" for c in (1, 2, 3)],
           ylim=(0, 103), ylabel="河网比例 (%)" if cn else "Networks (%)")
    ax.set_title("c  真实湖库位于哪段路径？" if cn else "c  Where is mapped storage?", loc="left", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.19), ncol=2)
    distribution(axes[1, 0], "relative_time_pulse_peak", "出口峰值 / 输入峰值" if cn else "Outlet peak / input peak",
                 "d  相同输入下的结构响应" if cn else "d  Identical-input structural response")
    axes[1, 0].axhline(1, color="#65787E", linestyle=":", linewidth=1)
    distribution(axes[1, 1], "doc_cv", "实测 DOC SD / 均值" if cn else "Observed DOC SD / mean",
                 "e  实测月度 DOC 波动" if cn else "e  Measured monthly DOC variation")
    ax = axes[1, 2]
    for c, row in enumerate(flow.sort_values("cluster").itertuples(), 1):
        ax.plot([c]*2, [row.ci_low, row.ci_high], color=COLORS[c], linewidth=2)
        ax.scatter(c, row.mean, s=42, color=COLORS[c], edgecolor="white", linewidth=.6, zorder=3)
    ax.axhline(0, color="#B8C4C5", linewidth=1)
    ax.set(xticks=[1, 2, 3], xticklabels=[f"{short[int(r.cluster)-1]}\nn={r.n_stations}" for r in flow.sort_values("cluster").itertuples()],
           ylabel="高流量 − 低流量：log1p DOC" if cn else "High-flow − low-flow log1p DOC")
    ax.set_title("f  季节 / 年份调整后的响应" if cn else "f  Season/year-adjusted flow response", loc="left", fontsize=11, pad=10)
    fig.suptitle("河网形态 → 结构机制 → DOC 证据" if cn else "River form → structural mechanisms → DOC evidence", x=.07, y=.995, ha="left", fontsize=17)
    fig.text(.07, .943, "原来的三类形态不变；连续结构刻画类内差异" if cn else "Three original forms retained; continuous structure resolves within-form differences", fontsize=10)
    fig.text(.07, .06, ("a/b/d/e：点为河网，盒为四分位范围；空心点及线为均值与 HUC4 95% 区间。c：选定代表路径，不代表全河网无湖库。\n"
              "d 为河段中点输入的受控情景（平均到达=1），e/f 为实测月度 DOC；后两者不能当作事件峰值。区间为 5,000 次区域重采样。" if cn else
              "a/b/d/e: dots are station instances; boxes show quartiles; open points/lines are means and HUC4 95% intervals. c: selected representative paths only.\n"
              "d uses controlled reach-midpoint forcing (mean arrival = 1); e/f use monthly observations, not event peaks. Intervals use 5,000 regional resamples."), fontsize=9, color="#53696E")
    save(fig, "forms_mechanisms_and_doc")

    # Same 22 matched pairs, with honest units and signed comparison intervals.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.7))
    fig.subplots_adjust(left=.07, right=.985, top=.76, bottom=.23, wspace=.36)
    terms = [("arrival_dispersion", 1, "到达分散" if cn else "Arrival dispersion", "Δ SD / 均值" if cn else "Δ SD / mean"),
             ("major_common_share", 100, "共同路径" if cn else "Shared path", "差值（百分点）" if cn else "Difference (percentage points)"),
             ("mapped_storage_path_share", 100, "湖库路径份额" if cn else "Mapped storage path share", "差值（百分点）" if cn else "Difference (percentage points)"),
             ("doc_cv", 1, "实测 DOC 波动" if cn else "Measured DOC variation", "Δ DOC SD / 均值" if cn else "Δ DOC SD / mean")]
    for index, (ax, (metric, factor, title, label)) in enumerate(zip(axes, terms)):
        sub = evidence[evidence.metric.eq(metric)]
        row = paired[paired.metric.eq(metric)].iloc[0]
        x = np.random.default_rng(42).uniform(-.13, .13, len(sub))
        ax.scatter(x, factor*sub.difference_b_minus_a, color="#738C93", alpha=.55, s=21)
        ax.plot([.4]*2, factor*np.array([row.ci_low, row.ci_high]), color="#293F4B", linewidth=2)
        ax.scatter(.4, factor*row.difference_b_minus_a, s=42, color="#66749F", edgecolor="white", zorder=3)
        ax.axhline(0, color="#A6B7BD", linewidth=1, linestyle="--")
        ax.set(xlim=(-.35, .65), xticks=[0, .4], xticklabels=("配对河流", "总体均值") if cn else ("River pairs", "Mean"), ylabel=label)
        ax.set_title(f"{'abcd'[index]}  {title}\nn={int(row.n_pairs)} · HUC4={int(row.n_huc4)}", loc="left", fontsize=11)
    fig.suptitle("相似环境中：宽展河网 − 细长河网" if cn else "Within matched environments: broad − elongated", x=.07, ha="left", fontsize=17)
    fig.text(.07, .07, "原来的 22 对河流保持固定；点是每对差值，线是配对 HUC4 95% 区间。几何差异与实测波动差异分别评价。" if cn else
             "The original 22 pairs stay fixed; dots show pair differences and lines show paired HUC4 95% intervals. Geometry and observed variation are separate outcomes.", fontsize=9, color="#53696E")
    save(fig, "matched_structure_and_monthly_doc")

    # Real projected NHD geometry and basin boundaries, not an illustration.
    connection = sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True)
    vaa = pd.read_parquet("cache/nldplus_vaa.parquet", columns=["comid", "wbareatype"])
    stored_ids = set(vaa.loc[vaa.wbareatype.isin(["LakePond", "Reservoir"]), "comid"])
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.4))
    fig.subplots_adjust(left=.025, right=.985, top=.83, bottom=.25, wspace=.13)
    for index, (ax, row) in enumerate(zip(axes, reps.sort_values("example_group").itertuples())):
        draw_network(ax, connection, row, COLORS[int(row.cluster)])
        member = np.load(CACHE/"members_full"/f"comid_{int(row.comid)}.npz")
        selected_ids = [int(cid) for cid in member["comids"] if int(cid) in stored_ids]
        lines = {cid: project_geometry(g) for cid, g in read_cached_lines(connection, selected_ids).items()}
        if len(lines) != len(selected_ids):
            raise ValueError("every mapped storage reach in representatives needs cached geometry")
        if lines:
            ax.add_collection(LineCollection(segments(lines.values()), colors="#325FAD", linewidths=2., zorder=5))
        ax.set_title(f"{'abc'[index]}  {forms[int(row.cluster)-1].replace(chr(10), ' ')}\n{row.station}", loc="left", fontsize=11, pad=12)
        position = position_names[POSITION_ORDER.index(row.storage_position)]
        text = (f"全网路径分散  {row.arrival_dispersion:.3f}\n主要共同路径  {row.major_common_share:.1%}\n全网湖库路径份额  {row.mapped_storage_path_share:.1%}\n选定路径湖库位置  {position}" if cn else
                f"Whole-network path CV  {row.arrival_dispersion:.3f}\nMajor shared path  {row.major_common_share:.1%}\nWhole-network storage path share  {row.mapped_storage_path_share:.1%}\nSelected-path storage  {position}")
        ax.text(.01, -.12, text, transform=ax.transAxes, va="top", fontsize=9, linespacing=1.65)
        inputs.extend([CACHE/"members_full"/f"comid_{int(row.comid)}.npz", CACHE/"basins"/f"comid_{int(row.comid)}.json"])
    connection.close()
    fig.suptitle("三类真实河网的结构画像" if cn else "Structural portraits of three real river forms", x=.03, ha="left", fontsize=17)
    fig.legend(handles=[Line2D([], [], color="#325FAD", lw=2, label="实际湖库河段" if cn else "Mapped LakePond / Reservoir reaches"),
                        Line2D([], [], color="#203C43", lw=1.3, label="主干河道" if cn else "Mainstem")],
               loc="upper right", bbox_to_anchor=(.98, .995), frameon=False, ncol=2, fontsize=9)
    fig.text(.03, .025, "保留原来按几何选择的三个实例；地图尺度各自独立。路径份额是河段长度指标，不是实测停留时间。" if cn else
             "Original geometry-selected examples retained; independent physical map extents. Path shares are channel-length measures, not measured residence times.", fontsize=9, color="#53696E")
    save(fig, "real_river_structure_portraits")
    receipt = {"input_hashes": {str(p): sha256_file(p) for p in inputs},
               "output_hashes": {str(p): sha256_file(p) for p in outputs}, "chinese": cn,
               "real_geometry": True, "n_form_station_instances": len(frame),
               "n_selected_corridors": int(frame.major_confluence_available.sum()),
               "n_matched_pairs": 22, "bootstrap_draws": 5000}
    (out/f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("Saved bilingual-ready structural evidence figures", flush=True)


if __name__ == "__main__":
    main()
