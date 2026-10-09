"""Scientific figures for measured DOC mixing and channel geometry."""

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

ROOT = Path("experiments/phase4_transfer/doc_river_measured_confluences_v1")
BLUE, ORANGE, DARK = "#297B8B", "#CA8748", "#344F5A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chinese", action="store_true")
    args = parser.parse_args()
    cn = args.chinese
    if cn:
        path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#7B898E"})
    campaigns = pd.read_csv(ROOT / "analysis/campaigns.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.8))
    fig.subplots_adjust(left=.085, right=.98, bottom=.14, top=.87, wspace=.28, hspace=.48)
    for ax in axes.flat:
        ax.tick_params(colors=DARK)
        ax.grid(axis="y", color="#E7EBED", linewidth=.7)
        ax.set_axisbelow(True)
    seasons = {"summer": (BLUE, "o", "夏季" if cn else "Summer"),
               "fall": (ORANGE, "s", "秋季" if cn else "Fall")}
    for season, (color, marker, label) in seasons.items():
        group = campaigns[campaigns.season == season].sort_values("location")
        x = np.arange(5) + (-.13 if season == "summer" else .13)
        ax = axes[0, 0]
        y = group.doc_departure_pct.to_numpy()
        ax.errorbar(x, y, yerr=np.array([y - group.doc_departure_min_pct,
                                       group.doc_departure_max_pct - y]),
                    fmt=marker, color=color, capsize=4, ms=7, label=label, linewidth=1)
        ax = axes[0, 1]
        ax.plot(x, group.water_closure_pct, marker, color=color, ms=7, label=label + ("：水" if cn else ": water"))
        ax.plot(x, group.doc_flux_discrepancy_pct, marker, color=color,
                markerfacecolor="white", ms=7, label=label + ("：碳" if cn else ": carbon"))
        ax = axes[1, 0]
        ax.scatter(group.wrt_downstream_main_ratio, group.doc_departure_pct,
                   color=color, marker=marker, s=48, label=label)
        for row in group.itertuples():
            offset = (5, -11) if season == "summer" and row.location == "Con-2" else (5, 4)
            ax.annotate(row.location.replace("Con-", ""),
                        (row.wrt_downstream_main_ratio, row.doc_departure_pct),
                        xytext=offset, textcoords="offset points", color=color, fontsize=9)
    for ax in axes[0]:
        ax.axhline(0, color="#8B999E", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(5), [f"汇流点 {i}" if cn else f"Confluence {i}" for i in range(1, 6)])
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    axes[0, 0].set_ylabel("相对流量加权混合值的差异（%）" if cn else "DOC departure from input mixture (%)")
    axes[0, 1].set_ylabel("下游相对两端输入总量的差异（%）" if cn else "Downstream minus incoming total (%)")
    axes[1, 0].axhline(0, color="#8B999E", linestyle="--", linewidth=1)
    axes[1, 0].axvline(1, color="#CBD3D6", linestyle=":", linewidth=1)
    axes[1, 0].set_xlabel("下游 / 主干上游：每 100 米停留时间" if cn else "Downstream / mainstem residence time per 100 m")
    axes[1, 0].set_ylabel("DOC 混合差异（%）" if cn else "DOC mixing departure (%)")
    ax = axes[1, 1]
    fall = campaigns[campaigns.season == "fall"].sort_values("location")
    for row in fall.itertuples():
        x = int(row.location.split("-")[1]) - 1
        ax.plot([x-.12, x+.12], [row.width_downstream_upstream_ratio, row.depth_downstream_upstream_ratio],
                color="#BCC8CD", linewidth=1.5)
        ax.plot(x-.12, row.width_downstream_upstream_ratio, "o", color=BLUE, ms=7)
        ax.plot(x+.12, row.depth_downstream_upstream_ratio, "s", color=ORANGE, ms=7)
    ax.plot([], [], "o", color=BLUE, label="宽度" if cn else "Width")
    ax.plot([], [], "s", color=ORANGE, label="深度" if cn else "Depth")
    ax.legend(frameon=False, ncol=2)
    ax.axhline(1, color="#8B999E", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(fall.width_downstream_upstream_ratio.max(), fall.depth_downstream_upstream_ratio.max()) * 1.2)
    ax.set_xticks(np.arange(5), [f"汇流点 {i}" if cn else f"Confluence {i}" for i in range(1, 6)])
    ax.set_ylabel("下游 / 上游：河道尺寸比" if cn else "Downstream / upstream channel dimension")
    titles = (["a  混合后的 DOC：平均值与三个位置范围", "b  水量收支与碳通量差额", "c  停留时间与 DOC 混合差异", "d  实测河道宽深变化（秋季）"]
              if cn else ["a  DOC mixing: mean and three-position range", "b  Water closure and carbon-flux discrepancy",
                          "c  Residence time and DOC departure", "d  Measured channel dimensions (fall)"])
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=11, pad=12)
    fig.suptitle("真实汇流点如何改变 DOC：混合、水文与河道形态" if cn else
                 "Measured confluences: DOC mixing, hydrology and channel geometry",
                 x=.085, ha="left", fontsize=15, y=.97, color=DARK)
    fig.text(.085, .915, "Tom’s Creek：五处汇流点，夏秋两次基流采样；同一河网内的局地结构对照。" if cn else
             "Tom’s Creek: five confluences, two baseflow campaigns; local structures within one river network.", color=DARK)
    fig.text(.085, .045, ("a：误差线为左、中、右位置的实测范围，非置信区间。b：DOC × 流量为瞬时碳通量，非事件总量。\n"
                        "流量在水质采样后三天内以示踪法测得；宽度按断面等权，深度为断面均值。数据：Plont（2022）。"
                        if cn else
                        "a: bars show the range across left/middle/right positions, not confidence intervals. b: DOC × Q is snapshot flux.\n"
                        "Tracer flow measured within three days of chemistry. Widths and mean depths use equal transects. Data: Plont (2022)."),
             fontsize=9, color=DARK, linespacing=1.65)
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    suffix = "_cn" if cn else ""
    files = []
    for ext in ["png", "svg"]:
        path = folder / f"measured_doc_mixing_geometry{suffix}.{ext}"
        fig.savefig(path, dpi=200)
        files.append(path)
    plt.close(fig)
    (ROOT / f"figure_sources{suffix}.json").write_text(json.dumps({
        "analysis_sources_sha256": sha256_file(ROOT / "analysis_sources.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "images": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        "range": "Observed three-position range, not inferential uncertainty",
    }, indent=2) + "\n")
    print("Saved scientific figure", suffix or "English")


if __name__ == "__main__":
    main()
