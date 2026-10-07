"""Actual river footprints and controlled branch/storage DOC responses."""
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
from analyze_doc_river_storage_transport_v1 import DTYPES, FOOTPRINT, ROOT
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from plot_doc_monitored_river_footprint_v1 import CACHE, draw_corridor, mapped_corridor

from river_graph.experiments.provenance import sha256_file

TEAL, ORANGE, NAVY, GRAY = "#267F88", "#C78169", "#304B58", "#A4AEB1"
DURATION_COLORS = {.075: ORANGE, .15: TEAL, .30: NAVY}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 210, "pdf.fonttype": 42})
    a, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(exist_ok=True)
    geometry = pd.read_csv(FOOTPRINT, dtype=DTYPES)
    reps = pd.read_csv(a/"representatives.csv", dtype=DTYPES)
    metrics = pd.read_csv(a/"scenario_metrics.csv", dtype=DTYPES)
    contrasts = pd.read_csv(a/"structural_contrasts.csv", dtype=DTYPES)
    summary = pd.read_csv(a/"cohort_summary.csv")
    reaches_path = FOOTPRINT.parent/"corridor_reaches.csv"
    reaches = pd.read_csv(reaches_path, dtype=DTYPES)
    traces = pd.read_parquet(a/"representative_responses.parquet")
    selected = reps.merge(geometry, on="pair_id", validate="one_to_one").sort_values("example_group")
    written, suffix = [], "_cn" if cn else ""

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out/f"{name}{suffix}.{ext}"
            fig.savefig(path, facecolor="white", bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    labels = (("等长支路，纯平移", "实际支路，纯平移", "实际支路，50% 主干时间用于滞留", "实际支路，100% 主干时间用于滞留") if cn else
              ("Equal branches, translation", "Actual branches, translation", "Actual branches, 50% common-time storage", "Actual branches, 100% common-time storage"))
    lines = (("equal", 0., GRAY, ":"), ("actual", 0., TEAL, "-"),
             ("actual", .5, ORANGE, "--"), ("actual", 1., NAVY, "-"))
    titles = ({1: "支路差异小 · 主干占比较低", 2: "支路差异大 · 主干占比较低",
               3: "支路差异小 · 主干占比较高", 4: "支路差异大 · 主干占比较高", 5: "共同主干湖库路径占比最高的实测连接"} if cn else
              {1: "Lower dispersion / lower common share", 2: "Higher dispersion / lower common share",
               3: "Lower dispersion / higher common share", 4: "Higher dispersion / higher common share",
               5: "Highest common-trunk waterbody share"})
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), gridspec_kw={"width_ratios": [1, 1.25]})
    example_metrics = metrics[metrics.pair_id.isin(selected.pair_id) & metrics.input_sd.eq(.15)]
    end = max(3., example_metrics.t90.max()+.75)
    geometry_hashes, gaps = {}, {}
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for k, row in enumerate(selected.itertuples()):
            ax = axes[k, 0]
            r = reaches[reaches.pair_id.eq(row.pair_id)]
            parts, gap, gh = mapped_corridor(connection, r)
            geometry_hashes[row.pair_id], gaps[row.pair_id] = gh, gap
            for reach in r[r.storage].itertuples():
                ax.plot(*parts[int(reach.comid)].xy, color="#D2AC69", linewidth=5,
                        alpha=.45, solid_capstyle="round", zorder=1)
            draw_corridor(ax, r, parts, row, cn)
            ax.set_title(f"{chr(97+k)}  {titles[row.example_group]}", loc="left", fontsize=11.5, pad=16)
            storage_label = (f"共同主干湖库路径 {row.common_storage_fraction:.1%}" if cn else
                             f"Mapped common-trunk waterbody path {row.common_storage_fraction:.1%}")
            ax.text(.02, 1.025, storage_label, transform=ax.transAxes, fontsize=9, color=ORANGE)
            ax = axes[k, 1]
            trace = traces[traces.pair_id.eq(row.pair_id)]
            for (branch, fraction, color, style), label in zip(lines, labels, strict=True):
                f = trace[trace.branch_condition.eq(branch) & trace.storage_fraction.eq(fraction)]
                ax.plot(f.relative_time, f.outlet_anomaly, color=color, linestyle=style, linewidth=1.8, label=label)
            ax.axvline(1., color=GRAY, linewidth=.8, linestyle="--", zorder=0)
            c = contrasts[contrasts.pair_id.eq(row.pair_id) & contrasts.input_sd.eq(.15)]
            c0, c5 = [c[c.storage_fraction.eq(f)].iloc[0] for f in (0., .5)]
            detail = ((f"支路错峰削峰 {c0.branch_peak_reduction_pct:.1f}%\n"
                       f"50% 滞留额外削峰 {c5.storage_peak_reduction_pct:.1f}%") if cn else
                      (f"Branch peak reduction {c0.branch_peak_reduction_pct:.1f}%\n"
                       f"Additional storage reduction (50%) {c5.storage_peak_reduction_pct:.1f}%"))
            ax.text(.98, .92, detail, transform=ax.transAxes, fontsize=9.5, ha="right", va="top")
            ax.set(xlim=(-.4, end), ylim=(0, 1.065), xlabel="相对时间（所有质心 = 1）" if cn else "Relative time (all centroids = 1)",
                   ylabel="DOC 异常 / 相同输入的峰高" if cn else "DOC anomaly / identical input peak")
            ax.set_title("相同平均到达时间，比较响应形状" if cn else "Same mean arrival, different response shapes", loc="left", pad=16, fontsize=11.5)
    handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=1.8, label=l)
               for (_, _, c, s), l in zip(lines, labels, strict=True)]
    fig.legend(handles=handles, ncol=2, loc="upper left", bbox_to_anchor=(.03, .964), frameon=False, fontsize=10)
    fig.suptitle("河网怎样把同一 DOC 峰错开、摊宽？" if cn else "How river structure disperses and broadens the same DOC pulse",
                 x=.03, y=.998, ha="left", fontsize=18)
    fig.text(.03, .017, ("左：真实 NHDPlus 河道；青色独立 A，橙色独立 B，深蓝共同主干，金色底线为湖库河段；比例尺独立。\n"
        "右：相同输入、混合比例与平均到达时间；输入 SD = 0.15，滞留比例分配主干时间预算，不由湖库路径占比换算。\n"
        "相对时间不是天数，情景峰高不是实测 DOC 削减率。") if cn else
        ("Left: real NHDPlus channels; teal branch A, orange B, navy common trunk; gold underlay marks waterbody reaches. Independent scales.\n"
         "Right: identical forcing, shares and mean arrival; input SD = 0.15. Storage allocates common time, not mapped waterbody length.\n"
         "Relative time is not days; these are simulated conservative responses, not measured DOC reductions."), fontsize=9)
    fig.subplots_adjust(left=.07, right=.97, top=.90, bottom=.08, wspace=.32, hspace=.68)
    save(fig, "matched_branch_storage_responses")

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10))
    ax = axes[0, 0]
    for fraction, color, marker, label in ((0., TEAL, "o", "纯平移" if cn else "Translation"),
        (.5, ORANGE, "s", "50% 主干时间滞留" if cn else "50% common-time storage"),
        (1., NAVY, "^", "100% 主干时间滞留" if cn else "100% common-time storage")):
        f = contrasts[contrasts.input_sd.eq(.15) & contrasts.storage_fraction.eq(fraction)]
        ax.scatter(f.total_path_cv/.15, f.branch_peak_reduction_pct, color=color, marker=marker,
                   s=26, alpha=.75, edgecolor="white", linewidth=.4, label=label)
    ax.set(xlabel="支路到达 SD / 输入峰 SD" if cn else "Branch arrival SD / input pulse SD",
           ylabel="支路长短差带来的削峰 (%)" if cn else "Peak reduction from branch differences (%)", ylim=(-2, 55))
    ax.set_title("a  "+("支路错峰作用随主干响应改变" if cn else "Branch effects depend on the shared response"), loc="left", pad=15)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[0, 1]
    for sigma, label in zip(DURATION_COLORS, ("短峰", "中峰", "长峰") if cn else ("Short pulse", "Middle pulse", "Long pulse"), strict=True):
        f = contrasts[contrasts.input_sd.eq(sigma) & contrasts.storage_fraction.eq(.5)]
        ax.scatter(.5*f.common_fraction/sigma, f.storage_peak_reduction_pct, color=DURATION_COLORS[sigma],
                   s=25, alpha=.75, edgecolor="white", linewidth=.4, label=f"{label} (SD={sigma:g})")
    ax.set(xlabel="滞留平均时间 / 输入峰 SD" if cn else "Storage mean time / input pulse SD",
           ylabel="相对实际支路纯平移的额外削峰 (%)" if cn else "Additional peak reduction (%)\nvs actual-branch translation",
           ylim=(-2, 78))
    ax.set_title("b  "+("固定 50% 滞留：短峰更容易削峰" if cn else "50% storage: shorter pulses attenuate more"), loc="left", pad=15)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1, 0]
    for sigma, label in zip(DURATION_COLORS, ("短峰", "中峰", "长峰") if cn else ("Short pulse", "Middle pulse", "Long pulse"), strict=True):
        f = summary[summary.input_sd.eq(sigma) & summary.metric.eq("storage_peak_reduction_pct")]
        ax.plot(f.storage_fraction*100, f.receiver_equal_mean, color=DURATION_COLORS[sigma], marker="o",
                linewidth=1.8, label=f"{label} (SD={sigma:g})")
    ax.set(xlabel="共同主干时间预算用于滞留的比例 (%)" if cn else "Common mean-time budget allocated to storage (%)",
           ylabel="22 个接收站等权平均的额外削峰 (%)" if cn else "Receiver-equal additional peak reduction (%)",
           ylim=(-2, 80), xticks=[0, 25, 50, 100])
    ax.set_title("c  "+("滞留强度 × 输入持续时间" if cn else "Storage strength × input duration"), loc="left", pad=15)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 1]
    f = example_metrics[example_metrics.branch_condition.eq("actual") & example_metrics.storage_fraction.eq(.5)]
    f = selected[["pair_id", "example_group"]].merge(f, on="pair_id", validate="one_to_one").sort_values("example_group")
    variance = np.column_stack([f.input_sd**2, f.branch_variance, f.storage_variance])
    share = variance/variance.sum(axis=1, keepdims=True)*100
    bottom = np.zeros(len(f))
    for k, (color, label) in enumerate(zip((GRAY, TEAL, ORANGE), ("输入峰宽", "独立支路差", "共同主干滞留") if cn else
                                            ("Input duration", "Branch differences", "Common storage"), strict=True)):
        ax.bar(f.example_group, share[:, k], bottom=bottom, color=color, width=.64, label=label,
               edgecolor="white", linewidth=.5)
        bottom += share[:, k]
    ax.set(xticks=range(1, 6), xticklabels=["a", "b", "c", "d", "e"], ylim=(0, 100),
           xlabel="上图五个真实河网例（50% 滞留情景）" if cn else "Five mapped examples (50% storage scenario)",
           ylabel="输出到达时间方差的组成 (%)" if cn else "Composition of output timing variance (%)")
    ax.set_title("d  "+("输入、支路与主干贡献可分开" if cn else "Separate forcing, branch and common-trunk contributions"), loc="left", pad=15)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.23), ncol=3)
    fig.suptitle("控制输入后，结构作用取决于时间尺度" if cn else "With forcing controlled, structural effects depend on time scale",
                 x=.03, y=.99, ha="left", fontsize=17)
    fig.text(.03, .015, ("a、b：59 个真实连接的几何情景，不是实测事件；a、d 的输入 SD = 0.15，b 固定 50% 滞留。c：22 个接收站等权。\n"
        "d：时间方差 = 输入方差 + 支路到达方差 + 滞留方差；峰高百分比不能直接相加。固定流量情景中总异常量保持不变。") if cn else
        ("a, b: 59 geometry scenarios, not field events; a, d: input SD = 0.15; b: 50% storage. c: pairs within receiver, then 22 receivers equal.\n"
         "d: timing variance = input + branch + storage variance; peak reductions do not add. Integrated anomaly is preserved at constant flow."), fontsize=9)
    fig.subplots_adjust(left=.085, right=.98, top=.91, bottom=.17, wspace=.33, hspace=.48)
    save(fig, "structural_time_scale_effects")
    inputs = [FOOTPRINT, reaches_path, ROOT/"config.json", Path(__file__),
        Path("scripts/plot_doc_monitored_river_footprint_v1.py"), Path("src/river_graph/analysis/river_monitored_footprint.py")]
    inputs.extend(a/p for p in ("representatives.csv", "scenario_metrics.csv", "structural_contrasts.csv",
                               "cohort_summary.csv", "representative_responses.parquet"))
    receipt = {"inputs": {str(p): sha256_file(p) for p in inputs},
        "outputs": {str(p): sha256_file(p) for p in written}, "geometry_hashes": geometry_hashes,
        "maximum_reach_endpoint_gap_m": gaps, "crs": "EPSG:5070",
        "curves": "controlled simulation, not observed DOC series"}
    (out/f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("Saved:", *written, sep="\n")


if __name__ == "__main__":
    main()
