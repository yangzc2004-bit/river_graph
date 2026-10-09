"""Scientific figures connecting geometry, concentration and carbon budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_geometry_budget_v1")
BLUE, ORANGE, GREY, DARK = "#297B8B", "#CA8748", "#BAC3C7", "#344F5A"
COLORS = {"Kervidy": BLUE, "Rappbode": ORANGE, "Bouleau": "#738453"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": DARK, "axes.labelcolor": DARK, "svg.fonttype": "none"})
    out, source = ROOT / "figures", ROOT / "analysis"
    out.mkdir(exist_ok=True)
    paths = []

    def text(en, zh):
        return zh if cn else en

    def save(fig, name):
        for ext in ("png", "svg"):
            path = out / f"{name}{'_cn' if cn else ''}.{ext}"
            fig.savefig(path, dpi=190, facecolor="white")
            paths.append(path)
        plt.close(fig)

    trace = pd.read_csv(source / "geometry_examples.csv")
    fig, axes = plt.subplots(4, 2, figsize=(12.8, 13.4))
    fig.subplots_adjust(left=.09, right=.97, top=.86, bottom=.11, hspace=.54, wspace=.28)
    fig.suptitle(text("River geometry changes when carbon arrives; redistribution and loss differ",
                     "河网改变碳的到达过程：展宽与总量减少可以区分"), x=.06, y=.97, ha="left", fontsize=17)
    fig.text(.06, .925, text("One fixed real tributary pair · imposed identical input pulses · water and carbon routed separately",
        "固定一组真实支流路径；相同输入脉冲；分别传输水量与碳量，再计算浓度"), fontsize=10)
    contrasts = [
        ("equal_mean_paths", "actual_paths", text("a  Differing paths", "a  路径长短不同"),
         text("Equal arrivals", "相同到达时间"), text("Actual path differences", "真实路径差异")),
        ("actual_paths", "arrival_compensated", text("b  Arrival alignment", "b  到达时间同步"),
         text("Synchronous source release", "源头同时释放"), text("Compensated arrival times", "补偿路径后同步到达")),
        ("short_shared_distributed", "long_shared_distributed", text("c  Shared downstream spreading", "c  共同下游河段展宽"),
         text("Short shared segment", "共同河段短"), text("Long shared segment", "共同河段长")),
        ("long_shared_distributed", "long_shared_reactive", text("d  Additional DOC processing", "d  额外 DOC 处理过程"),
         text("Conservative routing", "保留全部新增碳"), text("Imposed pulse loss", "设定新增碳处理")),
    ]
    for row, (reference, candidate, title, reference_name, candidate_name) in enumerate(contrasts):
        a, b = trace.loc[trace.scenario.eq(reference)], trace.loc[trace.scenario.eq(candidate)]
        for frame, label, color, style in ((a, reference_name, BLUE, "--"), (b, candidate_name, ORANGE, "-")):
            axes[row, 0].plot(frame.time, frame.concentration - 5, color=color, ls=style, lw=1.7, label=label)
            # All scenario source inputs have identical extra carbon budgets.
            dt = float(frame.time.diff().dropna().median())
            denominator = float(a.carbon_excess_flux.sum() * dt)
            axes[row, 1].plot(frame.time, frame.carbon_excess_flux.cumsum() * dt / denominator,
                              color=color, ls=style, lw=1.7)
        axes[row, 0].set_title(title, loc="left", fontsize=12)
        axes[row, 0].set_ylabel(text("DOC concentration increment", "DOC 浓度增量"))
        axes[row, 1].set_ylabel(text("Cumulative extra-carbon fraction", "累计到达的新增碳比例"))
        axes[row, 0].set_ylim(-.03, 1.05)
        axes[row, 1].set_ylim(-.03, 1.05)
        axes[row, 1].axhline(1, color=GREY, ls=":", lw=.8)
        axes[row, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
        for ax in axes[row]:
            ax.set_xlabel(text("Normalized scenario time", "归一化情景时间"))
    fig.text(.06, .035, text(
        "Geometry is measured; velocities, gamma spreading and processing are imposed scenarios. Time is not hours.\nConservative cases finish at the same carbon total; only the explicit processing case removes additional DOC.",
        "路径来自真实河网；速度、共同河段展宽和处理强度为设定情景，时间不代表实测小时。\n前三行改变到达过程，最终新增碳总量保持相同；第四行加入处理过程才出现总量减少。"), fontsize=9)
    save(fig, "geometry_peak_timing_export")

    budgets = pd.read_csv(source / "observed_budgets.csv")
    budgets = budgets.loc[budgets.window.eq("response_window") & budgets.budget_complete & budgets.lag_eligible]
    summary = pd.read_csv(source / "observed_summary.csv")
    summary = summary.loc[summary.window.eq("response_window") & summary.cohort.eq("positive_response")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.8))
    fig.subplots_adjust(left=.065, right=.97, top=.73, bottom=.27, wspace=.31)
    fig.suptitle(text("Observed outlet events: peak, spreading and export need joint interpretation",
                     "真实出口事件：峰高、展宽与输送量需要一起看"), x=.045, y=.96, ha="left", fontsize=17)
    fig.text(.045, .865, text("Complete hourly DOC–flow response windows only · each point is one event, not a river form",
        "仅用 DOC 与流量均完整的整点响应窗口；每个点是一次事件，并非一个独立河网形态"), fontsize=10)
    for case, group in budgets.groupby("case"):
        color = COLORS[case]
        width = group.loc[group.width_pair_eligible]
        axes[0].scatter(width.doc_width_hours, width.doc_peak_excess_mg_l, s=24, color=color,
                        alpha=.7, label=f"{case} (n={len(width)})")
        axes[1].scatter(group.doc_peak_excess_mg_l, group.carbon_yield_kg_km2, s=24,
                        color=color, alpha=.7, label=f"{case} (n={len(group)})")
    axes[0].set_xlabel(text("DOC half-excess width (h)", "DOC 半峰宽（小时）"))
    axes[0].set_ylabel(text("DOC peak above baseline (mg/L)", "DOC 峰高于事前浓度（mg/L）"))
    axes[0].set_title(text("a  Peak versus duration", "a  峰高与持续时间"), loc="left", fontsize=12)
    axes[1].set_xlabel(text("DOC peak above baseline (mg/L)", "DOC 峰高于事前浓度（mg/L）"))
    axes[1].set_ylabel(text("Bounded-window carbon yield (kg/km²)", "有界窗口碳输出（kg/km²）"))
    axes[1].set_title(text("b  Peak versus total export", "b  峰高与碳输送量"), loc="left", fontsize=12)
    for ax in axes[:2]:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8.5)
    field = summary.loc[summary.metric.eq("carbon_minus_water_after_peak_share")].set_index("case")
    for i, case in enumerate(("Kervidy", "Rappbode", "Bouleau")):
        row = field.loc[case]
        axes[2].errorbar(row["median"] * 100, i,
            xerr=[[100 * (row["median"] - row.ci_low)], [100 * (row.ci_high - row["median"])]],
            fmt="o", color=COLORS[case], capsize=3, ms=6)
        axes[2].text(.02, i - .21, f"n={int(row.n_events)}", transform=axes[2].get_yaxis_transform(), fontsize=8.5)
    axes[2].set_yticks(range(3), ["Kervidy", "Rappbode", "Bouleau"])
    axes[2].set_ylim(2.5, -.5)
    axes[2].axvline(0, color=GREY, ls="--", lw=.9)
    axes[2].set_xlabel(text("Carbon − water share after flow peak (pp)", "流量峰后：碳比例减水比例（百分点）"))
    axes[2].set_title(text("c  Does carbon arrive later than water?", "c  碳是否比水更偏向后半程？"), loc="left", fontsize=11)
    fig.text(.045, .105, text(
        "Outlet budgets measure concentration × flow over observed bounded windows, not net DOC loss within a river.\nIntervals: 5,000 month-block draws. Rappbode is provider-smoothed; Bouleau has few events. Field associations do not isolate geometry.",
        "出口碳量按浓度×流量在有界窗口内积分，不能据此推断河道净去除量。区间来自 5,000 次月份分块抽样。\n德国信号已由原作者平滑；加拿大事件少。实测关系反映出口响应；河网形态作用另由受控情景检验。"), fontsize=9)
    save(fig, "observed_peak_width_carbon_export")
    (ROOT / f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "code_sha256": sha256_file(Path(__file__)),
        "analysis_sources_sha256": sha256_file(ROOT / "analysis_sources.json"),
        "inputs": {str(p): sha256_file(p) for p in (source / "geometry_examples.csv", source / "observed_budgets.csv", source / "observed_summary.csv")},
        "outputs": {str(p): sha256_file(p) for p in paths}}, indent=2) + "\n")
    print(f"Saved {len(paths)} figure files")


if __name__ == "__main__":
    main()
