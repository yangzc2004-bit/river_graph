"""Draw independently observed pulse timing, examples and actual map evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_doc_river_multicatchment_pulses_v1 import load_cases
from matplotlib.font_manager import FontProperties, fontManager
from shapely.geometry import shape

from river_graph.analysis.river_multicatchment_pulses import hourly_observations
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_multicatchment_pulses_v1")
RAW = Path("data/raw/river_multicatchment_pulses_v1")
CASES = ["Kervidy", "Rappbode", "Bouleau"]
COLORS = {"Kervidy": "#297B8B", "Rappbode": "#CA8748", "Bouleau": "#738453"}
DARK, GREY = "#344F5A", "#BAC3C7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(path)
        plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": DARK, "axes.labelcolor": DARK, "svg.fonttype": "none"})
    out, source = ROOT / "figures", ROOT / "analysis"
    out.mkdir(exist_ok=True)
    saved = []

    def label(en, zh):
        return zh if cn else en

    def name(case):
        return {"Kervidy": "法国 Kervidy", "Rappbode": "德国 Rappbode", "Bouleau": "加拿大 Bouleau"}[case] if cn else case

    def save(fig, stem):
        for ext in ("png", "svg"):
            path = out / f"{stem}{'_cn' if cn else ''}.{ext}"
            fig.savefig(path, facecolor="white", dpi=190)
            saved.append(path)
        plt.close(fig)

    summary = pd.read_csv(source / "pulse_comparison.csv")
    main_rows = summary.loc[summary.resolution.eq("hourly") & summary.relative_prominence.eq(.2)].set_index("case").loc[CASES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7))
    fig.subplots_adjust(left=.13, right=.97, top=.73, bottom=.26, wspace=.34)
    fig.suptitle(label("DOC pulses are delayed and broadened across independent rivers", "不同真实河流的 DOC 峰：延迟时间不同，展宽较一致"),
        ha="left", x=.08, y=.96, fontsize=17)
    fig.text(.08, .865, label("Common hourly observations · medians and 95% calendar-month block intervals · no new gap filling",
        "统一到原有整点观测；中位数及 95% 月份分块区间；未新增插值或补值"), fontsize=10)
    for k, (field, title, xlabel, ref) in enumerate([
            ("doc_peak_lag_hours", label("a  DOC peak minus flow peak", "a  DOC 峰比流量峰晚多久"), label("Hours (negative = earlier)", "小时（负值表示 DOC 更早）"), 0),
            ("doc_flow_width_ratio", label("b  DOC / flow half-excess width", "b  DOC 峰宽是流量峰的多少倍"), label("Width ratio (1 = same width)", "半峰宽比（1 表示同样宽）"), 1)]):
        ax = axes[k]
        for i, case in enumerate(CASES):
            row = main_rows.loc[case]
            point, lo, hi = (row[f"{field}_{end}"] for end in ("median", "ci_low", "ci_high"))
            ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", ms=8,
                        lw=1.9, capsize=4, color=COLORS[case])
            n = int(row.n_positive_doc if k == 0 else row.n_width_pairs)
            ax.text(.02, i-.24, f"n = {n}; {point:.2f}"+label(" h" if k == 0 else "×", " 小时" if k == 0 else " 倍"),
                    transform=ax.get_yaxis_transform(), fontsize=9, color=COLORS[case])
        ax.axvline(ref, ls="--", color=GREY, lw=1)
        ax.set_yticks(range(3), [name(c) for c in CASES]); ax.set_ylim(2.55, -.55)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel(xlabel)
    axes[0].set_xlim(-3, 38); axes[1].set_xlim(.9, 1.95)
    fig.text(.08, .11, label("Rappbode signals were already smoothed over 2.5 h and had short gaps filled by the data provider.\nBouleau has only 8 resolved positive responses (6 complete widths); its lag interval reaches zero. These are outlet responses, not estimates of a causal form effect.",
        "德国信号已由原作者作 2.5 小时平滑和短缺口补值。加拿大只有 8 次可辨认正响应、6 对完整峰宽，延迟区间触及零。\n这里测量的是出口响应；三处流域的气候、水文和观测处理也不同，不能把全部差异归给河网形态。"), fontsize=9)
    save(fig, "independent_pulse_comparison")

    loaded = {audit["case"]: (flow, doc) for flow, doc, audit, _ in load_cases()}
    examples = pd.read_csv(source / "chronological_examples.csv")
    examples = examples.loc[examples.resolution.eq("hourly")].set_index("case")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    fig.subplots_adjust(left=.06, right=.98, top=.69, bottom=.27, wspace=.27)
    fig.suptitle(label("Recorded pulses: one chronological example per river", "真实曲线：每条河最早能测完整峰宽的一次事件"),
        x=.06, y=.96, ha="left", fontsize=17)
    fig.text(.06, .87, label("Hourly observations · each curve scaled by its own baseline and peak excess · panel time ranges differ",
        "原有整点观测；按各自事前水平和峰值增量归一化；三个面板时间范围不同"), fontsize=10)
    for i, (ax, case) in enumerate(zip(axes, CASES)):
        row = examples.loc[case]
        clock_args = {"utc": True} if case == "Kervidy" else {}
        start, end, peak = (pd.to_datetime(row[key], **clock_args) for key in
            ("antecedent_start_clock", "response_end_clock", "flow_peak_clock"))
        flow, doc = hourly_observations(*loaded[case])
        for frame, clock, field, base, maximum, color, title in [
            (flow.loc[flow.flow_valid], "flow_timestamp_utc", "q_m3_s", row.q_baseline_m3_s, row.q_peak_m3_s, "#CA8748", label("Flow", "流量")),
            (doc, "timestamp_utc", "doc_mg_l", row.doc_baseline_mg_l, row.doc_peak_mg_l, "#297B8B", "DOC")]:
            part = frame.loc[frame[clock].between(start, end)]
            groups = part[clock].diff().dt.total_seconds().gt(3600).cumsum()
            for j, (_, block) in enumerate(part.groupby(groups)):
                ax.plot((block[clock]-peak).dt.total_seconds()/3600,
                    (block[field]-base)/(maximum-base), lw=1.7, color=color, label=title if j == 0 else None)
        ax.axvline(0, lw=.9, color=GREY, ls="--"); ax.axhline(.5, lw=.7, color=GREY)
        ax.set_title(f"{'abc'[i]}  {name(case)}\n{peak:%Y-%m-%d}", loc="left", fontsize=11)
        ax.set_xlabel(label("Hours from flow peak", "距流量峰的小时数"))
        ax.text(.96, .93, label(f"Lag {row.doc_peak_lag_hours:+.0f} h\nWidth {row.doc_flow_width_ratio:.2f}×",
            f"延迟 {row.doc_peak_lag_hours:+.0f} 小时\n峰宽比 {row.doc_flow_width_ratio:.2f} 倍"),
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    axes[0].set_ylabel(label("Normalized pulse excess", "归一化峰值增量"))
    axes[0].legend(loc="upper left", frameon=False, ncols=2, fontsize=9)
    fig.text(.06, .10, label("Examples are chosen by chronology, not by DOC peak size or lag. Lines never cross missing hourly records.\nA single example is not the catchment median; negative normalized values mean below the antecedent baseline.",
        "按时间顺序取例，不按 DOC 高低或延迟筛选；缺失整点处断线。\n单次事件不代表流域中位数；负值表示低于事前水平。"), fontsize=9)
    save(fig, "independent_pulse_examples")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.98, top=.68, bottom=.27, wspace=.20)
    fig.suptitle(label("Same mapped river; varying event responses", "同一河网形态下，事件响应仍会变化"),
        x=.06, y=.96, ha="left", fontsize=17)
    fig.text(.06, .865, label("All resolved positive responses · common hourly observations · the same pulse measurement rules",
        "全部可辨认的正向响应；相同整点口径与事件测量规则"), fontsize=10)
    for i, (case, ax) in enumerate(zip(CASES, axes)):
        events = pd.read_csv(source / f"{case.lower()}_hourly_p20_events.csv")
        events = events.loc[events.lag_eligible].copy()
        if case == "Bouleau":
            events["year"] = pd.to_datetime(events.flow_peak_clock).dt.year
            for year, marker in ((2018, "o"), (2019, "^")):
                data = events.loc[events.year.eq(year)]
                ax.scatter(data.flow_span_hours, data.doc_peak_lag_hours, s=55, color=COLORS[case],
                    marker=marker, facecolors=COLORS[case] if year == 2018 else "none", label=str(year), lw=1.3)
            ax.legend(frameon=False, fontsize=9, loc="upper left")
        else:
            ax.scatter(events.flow_span_hours, events.doc_peak_lag_hours, s=28,
                color=COLORS[case], alpha=.7, linewidths=0)
        ax.axhline(0, color=GREY, ls="--", lw=.9)
        ax.set_title(f"{'abc'[i]}  {name(case)} (n={len(events)})", loc="left", fontsize=11)
        ax.set_xlabel(label("Bounded flow pulse duration (h)", "有界流量脉冲持续时间（小时）"))
        ax.set_xlim(0, 100); ax.set_ylim(-28, 95)
    axes[0].set_ylabel(label("DOC peak minus flow peak (h)", "DOC 峰相对流量峰延迟（小时）"))
    fig.text(.06, .10, label("Bouleau median lag: 34 h in 2018 (n=5), 0 h in 2019 (n=3), with the same mapped river form.\nDuration strata are exploratory diagnostics. Climate, event forcing, DOC mobilisation and source processing remain distinct from geometric routing.",
        "加拿大同一河网：2018 年延迟中位数 34 小时（5 次），2019 年 0 小时（3 次）。\n持续时间分层为探索性诊断。要研究形态作用，需把河道汇流展宽与降雨事件、DOC 动员过程分开。"), fontsize=9)
    save(fig, "within_form_event_variation")

    fig, axes = plt.subplots(1, 3, figsize=(15, 8.8))
    fig.subplots_adjust(left=.08, right=.97, top=.77, bottom=.25, wspace=.17)
    fig.suptitle(label("Real river maps accompanying the response measurements", "用于比较的是真实河网形态"), x=.04, y=.97,
        ha="left", fontsize=18)
    fig.text(.04, .90, label("Kervidy: measured provider vectors · Rappbode and Bouleau: original published maps",
        "法国：官方矢量河网；德国与加拿大：原论文的真实河网图"), fontsize=11)
    geometry = json.loads(Path("experiments/phase4_transfer/doc_river_kervidy_pulses_v1/analysis/topage_path_geometry.json").read_text())
    boundary = shape(geometry["catchment"]).exterior
    x, y = boundary.xy
    axes[0].plot((np.array(x)-264000)/1000, (np.array(y)-6783000)/1000, color=GREY, lw=.9, zorder=0)
    for reach in geometry["reaches"].values():
        line = shape(reach); x, y = line.xy
        axes[0].plot((np.array(x)-264000)/1000, (np.array(y)-6783000)/1000, color=COLORS["Kervidy"], lw=2)
    outlet = shape(geometry["mapped_terminal"])
    axes[0].scatter((outlet.x-264000)/1000, (outlet.y-6783000)/1000, marker="D", s=32, color=DARK)
    axes[0].set_aspect("equal"); axes[0].set_xlabel(label("Projected easting (km offset)", "投影东向距离（公里偏移）"))
    axes[0].set_ylabel(label("Projected northing (km offset)", "投影北向距离（公里偏移）"))
    for ax, site in zip(axes[1:], ["rappbode", "bouleau"]):
        ax.imshow(plt.imread(RAW / site / "published_river_map_web.png")); ax.axis("off")
    titles = [label("a  Kervidy: merging mapped headwaters", "a  法国：多条源头河段汇合"),
        label("b  Rappbode: tributaries along a trunk", "b  德国：支流沿主河道逐级汇入"),
        label("c  Bouleau: a winding dominant channel", "c  加拿大：弯曲主河道占主导")]
    for ax, title in zip(axes, titles):
        fig.text(ax.get_position().x0, .81, title, ha="left", fontsize=11)
    fig.text(.04, .13, label("Kervidy BD Topage: 4 mapped headwater paths, length CV 4.2%, shared terminal 569 m.\nThe other published maps support visual descriptions, not comparable measured path lengths or complete tributary inventories.\nSources: BD Topage; Werner et al. (2019), Fig. 1 (base map © Google, GeoBasis-DE-BKG); Prijac et al. (2023), Fig. 1. Original figure colours retained.",
        "法国矢量：4 条源头到出口路径；长度变异系数 4.2%；共同末端 569 米。\n另两张原图支持目视形态描述；尚不能给出可直接比较的路径长度或完整支流数量。\n来源：BD Topage；Werner 等（2019）图 1（底图 © Google、GeoBasis-DE-BKG）；Prijac 等（2023）图 1。保留原图配色。"), fontsize=9)
    save(fig, "real_river_map_evidence")
    (ROOT / f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "code_sha256": sha256_file(Path(__file__)), "analysis_sources_sha256": sha256_file(ROOT / "analysis_sources.json"),
        "outputs": {str(p): sha256_file(p) for p in saved},
        "map_inputs": {str(p): sha256_file(p) for p in [
            Path("experiments/phase4_transfer/doc_river_kervidy_pulses_v1/analysis/topage_path_geometry.json"),
            RAW / "rappbode/published_river_map_web.png", RAW / "bouleau/published_river_map_web.png"]},
        "map_sources": "Official BD Topage vectors and original attributed published maps; no AI geometry"}, indent=2)+"\n")
    print(f"Saved {len(saved)} {'Chinese' if cn else 'English'} figure files")


if __name__ == "__main__":
    main()
