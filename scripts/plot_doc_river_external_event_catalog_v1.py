"""Show inspected event records and an exact repeated sequence, not form effects."""

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

ROOT = Path("experiments/phase4_transfer/doc_river_external_event_catalog_v1")
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
                        "text.color": DARK, "axes.labelcolor": DARK, "svg.fonttype": "none"})
    folder, out = ROOT/"analysis", ROOT/"figures"
    out.mkdir(parents=True, exist_ok=True)
    series = pd.read_parquet(folder/"arctic_doc_record_audit.parquet")
    events = pd.read_csv(folder/"author_storm_audit.csv", parse_dates=["start_source_clock", "end_source_clock"])
    replay = pd.read_csv(folder/"cross_year_sequences.csv")
    paths = []

    def label(en, zh):
        return zh if cn else en

    def save(fig, name):
        for extension in ("png", "svg"):
            path = out/f"{name}{'_cn' if cn else ''}.{extension}"
            fig.savefig(path, dpi=210, facecolor="white")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), sharey=True)
    fig.subplots_adjust(left=.09, right=.96, top=.81, bottom=.15, hspace=.55, wspace=.25)
    fig.suptitle(label("Real storm records resolve DOC fluctuations within days",
                       "真实洪水记录可以分辨 DOC 在几天内的涨落"), x=.08, y=.965, ha="left", fontsize=17)
    fig.text(.08, .89, label("Two earliest record candidates per catchment; unchanged published storm intervals",
        "每个流域按日期展示最早两个记录候选；保留作者划定的洪水起止时间"), fontsize=10)
    for column, (site, color) in enumerate((("Kuparuk River", BLUE), ("Oksrukuyik Creek", GOLD))):
        selected = events.loc[events.watershed.eq(site) & events.doc_record_candidate].sort_values("start_source_clock").head(2)
        if len(selected) != 2:
            raise ValueError("Need two chronological candidates per catchment")
        for row_index, event in enumerate(selected.itertuples()):
            ax = axes[row_index, column]
            frame = series.loc[series.watershed.eq(site) & series.timestamp.between(
                event.start_source_clock, event.end_source_clock) & np.isfinite(series.doc_mg_l)].sort_values("timestamp")
            block = frame.timestamp.diff().dt.total_seconds().gt(3600).cumsum()
            for _, g in frame.groupby(block):
                x = (g.timestamp-event.start_source_clock).dt.total_seconds()/3600
                ax.plot(x, g.doc_mg_l, color=color, linewidth=1.1)
                ax.scatter(x, g.doc_mg_l, color=color, s=3, alpha=.45, linewidths=0)
            ax.set_title(f"{chr(97+row_index*2+column)}  {site}\n{event.start_source_clock:%Y-%m-%d}", loc="left", fontsize=10)
            ax.set(xlim=(0, event.elapsed_hours), ylim=(0, 13.5))
            ax.set_xlabel(label("Hours since published storm start", "距作者记录的洪水起点（小时）"))
            if column == 0:
                ax.set_ylabel(label("Optical estimate of DOC (mg/L)", "光学估算 DOC（mg/L）"))
            ax.text(.99, .04, label(f"DOC-bin coverage {event.nominal_doc_coverage:.1%}",
                f"DOC 时段覆盖 {event.nominal_doc_coverage:.1%}"), transform=ax.transAxes, ha="right", fontsize=9)
    fig.text(.08, .045, label("Published source clocks retained; dots are existing measurements, not interpolated values.\nThese are outlet traces from different storms. Continuous flow and independent form replication are still missing.",
        "保留原档案时钟；点为已有记录，没有补值。两流域展示的是不同洪水的出口曲线。\n目前尚缺连续流量和独立河网形态重复，不能据此比较细长型与多支流型。"), fontsize=9)
    save(fig, "observed_storm_record_examples")

    match = replay.loc[replay.requires_source_resolution].sort_values("longest_exact_sequence_values", ascending=False).iloc[0]
    source = series.loc[series.watershed.eq(match.watershed)]
    a = source.loc[source.year.eq(match.earlier_year) & np.isfinite(source.doc_mg_l)].sort_values("timestamp").reset_index(drop=True)
    b = source.loc[source.year.eq(match.later_year) & np.isfinite(source.doc_mg_l)].sort_values("timestamp").reset_index(drop=True)
    n, i, j = int(match.longest_exact_sequence_values), int(match.earlier_index), int(match.later_index)
    a, b = a.iloc[i:i+n], b.iloc[j:j+n]
    np.testing.assert_array_equal(a.doc_mg_l, b.doc_mg_l)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    fig.subplots_adjust(left=.09, right=.97, top=.76, bottom=.27, wspace=.35)
    fig.suptitle(label("An archive issue can create an apparent timing difference",
                       "档案中的重复片段可能制造虚假的时间差异"), x=.08, y=.965, ha="left", fontsize=17)
    fig.text(.08, .87, label(f"Oksrukuyik Creek: {n:,} ordered finite DOC values are exactly identical in 2017 and 2022",
        f"Oksrukuyik Creek：2017 与 2022 年有 {n:,} 个按顺序排列的有效 DOC 值完全相同"), fontsize=10)
    ax = axes[0]
    ax.plot(np.arange(n), a.doc_mg_l, color=BLUE, linewidth=1.2, label=str(int(match.earlier_year)))
    ax.plot(np.arange(n), b.doc_mg_l, color=GOLD, linewidth=.9, linestyle="--", label=str(int(match.later_year)))
    ax.set(xlabel=label("Ordered finite-record index", "有效记录的顺序编号"),
           ylabel=label("Archived optical DOC (mg/L)", "档案中的光学 DOC（mg/L）"))
    ax.set_title(label("a  Identical values at every matched position", "a  每个对应位置的数值相同"), loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax = axes[1]
    for frame, color, year in ((a, BLUE, match.earlier_year), (b, GOLD, match.later_year)):
        x = (frame.timestamp-frame.timestamp.iloc[0]).dt.total_seconds()/86400
        ax.plot(x, frame.doc_mg_l, color=color, linewidth=1, label=str(int(year)))
    ax.set(xlabel=label("Elapsed days on each archive clock", "各自档案时钟上的经过天数"),
           ylabel=label("Archived optical DOC (mg/L)", "档案中的光学 DOC（mg/L）"))
    ax.set_title(label("b  Same sequence, different elapsed duration", "b  同一数值序列，对应不同的时间跨度"), loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.text(.08, .075, label("A record-reuse diagnostic, not an event-shape or river-form result. The source assembly remains unexplained.\nLater affected site/years are set aside; original values and all storm intervals are retained.",
        "这是记录重复诊断，不是洪水形状或河网形态的结论；档案组装原因尚未解释。\n暂不采用后续受影响的站点年份，原始数值和全部洪水时间段均保留。"), fontsize=9)
    save(fig, "cross_year_record_sequence_diagnostic")
    inputs = [folder/"arctic_doc_record_audit.parquet", folder/"author_storm_audit.csv", folder/"cross_year_sequences.csv"]
    (ROOT/f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "code_hash": sha256_file(Path(__file__)),
        "figure_hashes": {str(p): sha256_file(p) for p in paths}}, indent=2)+"\n")
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
