"""Paper figures separating imposed geometry responses from field observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_morphology_closure_v1 import ROOT, TYPES
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FuncFormatter

from river_graph.experiments.provenance import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chinese", action="store_true")
    args = p.parse_args()
    cn = args.chinese
    suffix = "_cn" if cn else ""
    if cn:
        path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    data, out = ROOT / "analysis", ROOT / "figures"
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(data / "mechanism_summary.csv")
    waves = pd.read_parquet(data / "mechanism_example.parquet")
    field = pd.read_csv(data / "observed_leave_system_out.csv")
    recovered = pd.read_csv(data / "recovered_receivers.csv", dtype=TYPES)
    written = []
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.3))
    ax = axes[0, 0]
    for comparison, color, label in (
            ("path_difference", "#257F88", "路径长短差异" if cn else "Unequal arrival paths"),
            ("shared_spreading", "#C5814A", "共同河段弥散的额外作用" if cn else "Additional shared-corridor spreading")):
        f = summary[summary.minimum_coverage.eq(0) & summary.metric.eq("pulse_peak_relative")
            & summary.comparison.eq(comparison)].sort_values("sigma")
        ax.plot(f.sigma, f.estimate*100, "o-", color=color, label=label, linewidth=2)
        ax.fill_between(f.sigma, f.ci_low*100, f.ci_high*100, color=color, alpha=.13)
    ax.axhline(0, color="#9AA7AB", linewidth=1, linestyle=":")
    ax.set(xscale="log", xlabel="输入峰宽 / 平均名义传输时间" if cn else "Input pulse SD / mean nominal travel time",
        ylabel="峰值变化（%）" if cn else "Peak change (%)")
    ax.set_xticks([.025, .1, .5, 2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("a  短促高峰更受路径与弥散影响" if cn else "a  Geometry affects short pulses more", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax = axes[0, 1]
    for name, color, label, style in (
            ("aligned_arrivals", "#8D9FA6", "同时到达" if cn else "Aligned arrival", "--"),
            ("actual_paths", "#257F88", "真实路径使到达错开" if cn else "Actual unequal paths", "-"),
            ("actual_shared_0.5", "#C5814A", "再经共同河段弥散" if cn else "+ Shared-corridor spreading", "-")):
        f = waves[waves.scenario.eq(name)]
        ax.plot(f.time, f.response, color=color, linestyle=style, linewidth=2, label=label)
    ax.set(xlim=(.1, 3.1), ylim=(0, 1.08),
        xlabel="名义时间（平均路径时间=1）" if cn else "Nominal time (mean path time = 1)",
        ylabel="单位输入的出口响应" if cn else "Outlet response to unit input")
    ax.set_title("b  Loch Vale 真实路径上的受控峰值" if cn else "b  Controlled pulses on real Loch Vale paths", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[1, 0]
    f = field[field.minimum_coverage.eq(0)].sort_values("omitted_system")
    for j, r in enumerate(f.itertuples()):
        color = "#C5814A" if r.omitted_system == -1 else "#257F88"
        ax.errorbar(r.estimate*100, j, xerr=[[100*(r.estimate-r.ci_low)], [100*(r.ci_high-r.estimate)]],
            fmt="o", color=color, capsize=3, markersize=6)
    labels = [("完整样本" if cn else "All systems") if r.omitted_system == -1 else
        (f"去掉系统 {r.omitted_system}" if cn else f"Omit system {r.omitted_system}") for r in f.itertuples()]
    ax.set(yticks=np.arange(len(f)), yticklabels=labels,
        xlabel="多支流一起偏高时，出口偏高概率增加（百分点）" if cn else "Receiving excursion probability difference (pp)")
    ax.axvline(0, color="#9AA7AB", linewidth=1, linestyle=":")
    ax.invert_yaxis()
    ax.set_title("c  实测关联：逐个去掉河系核对" if cn else "c  Observed association: omit one system at a time", loc="left")
    ax = axes[1, 1]
    categories = ["same_day_eligible", "within_month_eligible"]
    for j, (form, color, label) in enumerate(((1, "#257F88", "细长型" if cn else "Elongated"),
            (3, "#66749F", "宽分支型" if cn else "Broad"))):
        g = recovered[recovered.cluster.eq(form)]
        values = [int(g[c].sum()) for c in categories]
        ax.bar(np.arange(2)+(j-.5)*.32, values, width=.3, color=color, label=f"{label} (n={len(g)})")
        for x, v in zip(np.arange(2)+(j-.5)*.32, values, strict=True):
            ax.text(x, v+.10, str(v), ha="center", fontsize=10)
    ax.set(xticks=[0, 1], xticklabels=["共同日期足够" if cn else "Adequate common dates",
        "月内密集日期足够" if cn else "Adequate within-month dates"],
        ylabel="可用出口数" if cn else "Eligible receiving networks", ylim=(0, max(3, recovered.same_day_eligible.sum()+1)))
    ax.set_title("d  15组形态对补查后的观测覆盖" if cn else "d  Observation recovery for 15 form pairs", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(.06, .017,
        "a–b：同输入、实际路径、假定速度与弥散的机制实验；不是实测 DOC 降幅。c：月观测关联，不代表统计因果。\n"
        "阴影和误差线为整片重叠河系重抽的95%区间。d：同区域、面积相近的优先清单；出口在多个形态对中重复。" if cn else
        "a–b: prescribed inputs, real paths, assumed speed/spreading; not measured DOC reductions. c: observed monthly association.\n"
        "Bands/whiskers: 95% overlapping-system bootstrap intervals. d: metadata-selected, similar-area form opportunities with reused receivers.", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1), h_pad=3, w_pad=3)
    for extension in ("png", "pdf"):
        path = out / f"geometry_mechanisms_and_field_evidence{suffix}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(fig)
    (out / f"figure_sources{suffix}.json").write_text(json.dumps({
        "source_hashes": {str(data / name): sha256_file(data / name) for name in (
            "mechanism_summary.csv", "mechanism_example.parquet", "observed_leave_system_out.csv", "recovered_receivers.csv")},
        "figure_hashes": {str(p): sha256_file(p) for p in written},
        "evidence_labels": "controlled pulse experiment versus observed concentrations, separately labelled"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
