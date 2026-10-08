"""Plot measured signal changes, branch mixing summaries, and partial water budgets."""

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

ROOT = Path("experiments/phase4_transfer/doc_river_flow_state_mechanisms_v1")
ORDER = ("C12", "C16", "C7", "C9")
BLUE, GOLD, DARK = "#297B8B", "#CA8748", "#344F5A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": DARK, "axes.labelcolor": DARK, "pdf.fonttype": 42})
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(parents=True, exist_ok=True)
    inputs = [folder/"state_summary.csv", folder/"high_minus_low.csv", folder/"water_budget_ledger.parquet"]
    states = pd.read_csv(inputs[0])
    states = states.loc[states.population.eq("joint_calendar")].set_index(["receiver", "flow_state"])
    changes = pd.read_csv(inputs[1])
    changes = changes.loc[changes.population.eq("joint_calendar")].set_index("receiver").reindex(ORDER)
    ledger = pd.read_parquet(inputs[2])
    ledger = ledger.loc[ledger.population.eq("joint_calendar")]
    paths = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out/f"{name}{'_cn' if cn else ''}.{extension}"
            fig.savefig(path, dpi=220, facecolor="white")
            paths.append(path)
        plt.close(fig)

    def label(en, zh):
        return zh if cn else en

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.07, right=.97, bottom=.29, top=.77, wspace=.32)
    fig.suptitle(label("Separate outlet variation from changing incoming variation",
                       "把出口波动和支流混合波动分开看"), x=.07, ha="left", y=.965, fontsize=17)
    fig.text(.07, .87, label("Four actual confluences · 47 common dates (2014–2017) · 95% whole-year bootstrap intervals",
        "四处真实汇合点 · 47 个共同日期（2014–2017）· 95% 整年抽样区间"), fontsize=10)
    ax = axes[0]
    for name, offset, color, legend in (("mixture_sd", -.16, BLUE, label("Partial branch mixture", "已测支流混合")),
            ("outlet_sd", .16, GOLD, label("Measured outlet", "实测出口"))):
        for i, site in enumerate(ORDER):
            x = i+offset+np.array([-.055, .055])
            sub = states.loc[site].reindex(["low", "high"])
            values = sub[name].to_numpy()
            ax.plot(x, values, color=color, linewidth=1.3)
            ax.vlines(x, sub[name+"_lo"], sub[name+"_hi"], color=color, linewidth=1)
            ax.scatter(x[0], values[0], marker="o", facecolors="white", edgecolors=color, s=38, zorder=4)
            ax.scatter(x[1], values[1], marker="o", color=color, s=38, zorder=4, label=legend if i == 0 else None)
    ax.set(xticks=range(4), xticklabels=ORDER, ylabel=label("Season/time-adjusted DOC SD (mg C/L)", "去除季节与时间趋势后的 DOC 波动（mg C/L）"),
           ylim=(0, None), xlim=(-.5, 3.5))
    ax.set_title(label("a  Absolute fluctuation magnitude", "a  两边的绝对波动"), loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(.0, -.25, label("Open: low flow    Filled: high flow", "空心：低流量    实心：高流量"), transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    for name, offset, color, legend in (
        ("log_outlet_sd_contribution", -.17, GOLD, label("Outlet SD change", "出口波动变化")),
        ("log_mixture_sd_contribution", 0, BLUE, label("Minus mixture SD change", "负的支流混合波动变化")),
        ("log_ratio_change", .17, DARK, label("Ratio change (sum)", "波动比变化（两项之和）"))):
        y = np.arange(4)+offset
        ax.hlines(y, changes[name+"_lo"], changes[name+"_hi"], color=color, linewidth=1.2)
        ax.scatter(changes[name], y, color=color, s=30, label=legend, zorder=3)
    ax.axvline(0, color="#83939A", linewidth=1, linestyle="--")
    ax.set(yticks=range(4), yticklabels=ORDER, xlabel=label("High − low change in natural-log SD", "高流量相对低流量的对数波动变化"))
    ax.invert_yaxis()
    ax.set_title(label("b  Exact numerator / denominator decomposition", "b  波动比的分子与分母贡献"), loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.19), fontsize=9)
    fig.text(.07, .045, label("Same population-wide seasonal projection in every flow state; sparse samples do not measure continuous event peaks.",
        "各流量组使用同一季节投影；这是稀疏采样的波动，不是连续观测到的事件峰值。"), fontsize=9)
    save(fig, "flow_signal_decomposition")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.7))
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.855, wspace=.35, hspace=.58)
    fig.suptitle(label("Branch coordination, water contribution, and the unresolved outlet budget",
                       "支流同步性、来水比例与出口水量缺口"), x=.09, ha="left", y=.97, fontsize=17)
    fig.text(.09, .92, label("Same four confluences and joint calendar · summary substitutions are algebraic, not causal interventions",
                           "相同四处汇合点和共同采样日期 · 分项替换是数学分解"), fontsize=10)
    for ax, name, title, xlabel, lim in (
        (axes[0, 0], "rho", label("a  Coordination of branch DOC", "a  两条支流的 DOC 同步性"),
         label("Correlation of adjusted branch DOC", "去除季节后的支流 DOC 相关系数"), (-1.05, 1.05)),
        (axes[0, 1], "w", label("b  Water contribution of branch A", "b  支流 A 在两支流中的来水比例"),
         label(r"Mean $Q_A / (Q_A + Q_B)$", r"平均 $Q_A / (Q_A + Q_B)$"), (-.03, 1.03))):
        for i, site in enumerate(ORDER):
            low, high = (states.loc[(site, s)] for s in ("low", "high"))
            ax.plot([low[name], high[name]], [i, i], color="#B4C1C4", linewidth=1.5)
            for offset, row, color, state in ((-.08, low, BLUE, "low"), (.08, high, GOLD, "high")):
                ax.hlines(i+offset, row[name+"_lo"], row[name+"_hi"], color=color, linewidth=1.2)
                ax.scatter(row[name], i+offset, color=color, s=34,
                           label=label("Low flow", "低流量") if i == 0 and state == "low" else (
                               label("High flow", "高流量") if i == 0 else None), zorder=3)
        ax.set(yticks=range(4), yticklabels=ORDER, xlabel=xlabel, xlim=lim)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.20), ncol=2, fontsize=9)

    ax = axes[1, 0]
    for name, offset, color, legend in (("correlation_pp", -.17, BLUE, label("Branch correlation", "同步性")),
            ("amplitude_balance_pp", 0, DARK, label("Branch SDs", "支流波动幅度")),
            ("mean_flow_share_pp", .17, GOLD, label("Mean water balance", "平均来水比例"))):
        y = np.arange(4)+offset
        ax.hlines(y, changes[name+"_lo"], changes[name+"_hi"], color=color, linewidth=1.2)
        ax.scatter(changes[name], y, color=color, s=30, label=legend, zorder=3)
    ax.axvline(0, color="#83939A", linestyle="--", linewidth=1)
    ax.set(yticks=range(4), yticklabels=ORDER, xlabel=label("Contribution to fixed-weight mixing potential (pp)", "对固定比例混合平缓程度的贡献（百分点）"))
    ax.invert_yaxis()
    ax.set_title(label("c  High − low all-order summary decomposition", "c  高低流量差异的分项贡献"), loc="left", fontsize=11)
    ax.legend(frameon=False, bbox_to_anchor=(0, -.24), loc="upper left", fontsize=9)

    ax = axes[1, 1]
    conditions = (
        (ledger.feasibility_budget_usable & ledger.nonnegative_unmonitored_solution,
         BLUE, label("Nonnegative solution", "非负解")),
        (ledger.feasibility_budget_usable & ~ledger.nonnegative_unmonitored_solution,
         "#C6909C", label("Negative solution", "负值解")),
        (ledger.budget_exclusion.eq("near_complete_share"), GOLD, label("Share near 1", "水量比接近 1")),
        (ledger.budget_exclusion.eq("measured_share_exceeds_one"), "#B2BCC1", label("Share > 1", "水量比 > 1")))
    left = np.zeros(4)
    for condition, color, legend in conditions:
        counts = ledger.loc[condition].groupby("receiver").size().reindex(ORDER, fill_value=0).to_numpy()
        ax.barh(range(4), counts, left=left, height=.55, color=color, label=legend)
        left += counts
    if not np.all(left == 47):
        raise ValueError("Budget category plot does not exhaust the common dates")
    ax.set(yticks=range(4), yticklabels=ORDER, xlim=(0, 50), xlabel=label("Common sampling dates (47 per configuration)", "共同采样日期数量（每处 47 个）"))
    ax.invert_yaxis()
    ax.set_title(label("d  No-processing partial-water feasibility", "d  未测来水能否在无净处理假设下解释出口"), loc="left", fontsize=11)
    ax.legend(frameon=False, bbox_to_anchor=(0, -.24), loc="upper left", fontsize=9, ncol=2)
    fig.text(.09, .035, label("Positive solutions are required unmonitored DOC, not observations. Water shares use daily Q; no complete instantaneous load is available.",
        "非负解是需要的未测来水浓度，并非实测值。水量比使用日流量，不能代替完整的瞬时 DOC 负荷。"), fontsize=9)
    save(fig, "branch_mixing_and_water_budget")
    metadata = {"inputs": {str(p): sha256_file(p) for p in inputs},
        "code": {str(Path(__file__).relative_to(Path.cwd())): sha256_file(Path(__file__))},
        "figures": {str(p): sha256_file(p) for p in paths}, "language": "Chinese" if cn else "English"}
    (ROOT/f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
