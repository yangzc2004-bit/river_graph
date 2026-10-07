"""Plot sampled wave clocks and DOC quality without masking missing peaks."""

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
from matplotlib.lines import Line2D

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_hourly_response_v1")
COLORS = {"upstream": "#277F89", "receiver": "#C78349"}


def plot_segment(ax_q, ax_c, frame, cn, compact=False):
    start = frame.date_time_source_clock.iloc[0]
    x = frame.date_time_source_clock.sub(start).dt.total_seconds().div(3600).to_numpy()
    for site, color in COLORS.items():
        q = frame[f"Discharge (m3 s-1)_{site}"].to_numpy()
        c = frame[f"DOC (mg l-1)_{site}"].to_numpy()
        risk = frame[f"doc_reported_retrieval_regime_{site}"].to_numpy(dtype=bool)
        label = ("上游 SN" if site == "upstream" else "下游 FITT") if cn else ("Upstream SN" if site == "upstream" else "Downstream FITT")
        ax_q.plot(x, (q-q.min())/(q.max()-q.min()), color=color, linewidth=1.6, label=label)
        ax_q.scatter(x[q == q.max()], np.ones(int(np.sum(q == q.max()))), s=20, color=color)
        ax_c.plot(x, c, color=color, linewidth=1.05, alpha=.8)
        ax_c.scatter(x[~risk], c[~risk], color=color, s=10 if compact else 15, edgecolor="white", linewidth=.35)
        ax_c.scatter(x[risk], c[risk], color=color, marker="x", s=20 if compact else 27, linewidths=1)
    for ax in (ax_q, ax_c):
        ax.set_xlim(-.5, x.max()+.5)
        ax.set_xlabel("记录起点后的小时" if cn else "Hours from segment start", fontsize=9)
        ax.tick_params(labelsize=8)
    ax_q.set_ylabel("流量（范围归一化）" if cn else "Flow (range-normalized)", fontsize=9)
    ax_c.set_ylabel("光学 DOC (mg C/L)" if cn else "Optical DOC (mg C/L)", fontsize=9)
    return start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263B42", "axes.labelcolor": "#263B42", "pdf.fonttype": 42})
    analysis, output = ROOT / "analysis", ROOT / "figures"
    output.mkdir(exist_ok=True)
    segments = pd.read_csv(analysis / "segment_comparison.csv")
    sites = pd.read_csv(analysis / "site_waveforms.csv")
    paired = pd.read_parquet(analysis / "paired_hourly_quality.parquet")
    suffix, outputs = ("_cn" if cn else ""), []

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = output / f"{name}{suffix}.{ext}"
            fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=.15, right=.97, bottom=.14, top=.86, hspace=.47, wspace=.5)
    main = segments[segments.main_coverage_subset]
    ax = axes[0, 0]
    y = np.arange(len(main))
    lo, hi = main.flow_peak_difference_lower_hours.to_numpy(), main.flow_peak_difference_upper_hours.to_numpy()
    middle = (lo+hi)/2
    ax.errorbar(middle, y, xerr=np.vstack([middle-lo, hi-middle]), fmt="o", color=COLORS["upstream"], capsize=3, markersize=5)
    ax.axvline(0, color="#B7C4C6", linestyle=":", linewidth=1)
    ax.set(yticks=y, yticklabels=[f"{str(r.start)[:10]} · #{r.block}" for r in main.itertuples()],
           xlabel="下游－上游流量峰值时差（小时）" if cn else "Downstream − upstream flow peak clock (h)", xlim=(-.2, 2.5))
    ax.invert_yaxis()
    ax.set_title("a  七段长记录：流量峰值晚到" if cn else "a  Seven long segments: later flow maxima", loc="left", fontsize=11)

    ax = axes[0, 1]
    width = main[main.flow_both_widths_available]
    clean = width.flow_single_half_height_lobes.to_numpy(dtype=bool)
    for selected, marker, color, label in (
        (clean, "o", COLORS["upstream"], "单一半高波峰" if cn else "Single half-height lobe"),
        (~clean, "s", "#A9B8BD", "多波峰记录" if cn else "Multiple half-height lobes"),
    ):
        ax.scatter(width.loc[selected, "flow_upstream_width_hours"], width.loc[selected, "flow_receiver_width_hours"],
                   marker=marker, s=45, color=color, label=label, edgecolor="white", linewidth=.5)
    for row in width.itertuples():
        offset = {3: (5, 10), 10: (5, -14), 4: (5, -10)}.get(row.block, (4, 3))
        ax.annotate(f"#{row.block}", (row.flow_upstream_width_hours, row.flow_receiver_width_hours),
                    xytext=offset, textcoords="offset points", fontsize=8)
    limit = float(max(width.flow_upstream_width_hours.max(), width.flow_receiver_width_hours.max())*1.16)
    ax.plot([0, limit], [0, limit], color="#A3B2B7", linestyle="--", linewidth=1)
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="上游半高宽度（小时）" if cn else "Upstream half-height width (h)",
           ylabel="下游半高宽度（小时）" if cn else "Downstream half-height width (h)")
    ax.set_title("b  晚到不等于曲线变宽" if cn else "b  Later arrival is not wider duration", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, 0]
    for i, row in enumerate(segments.itertuples()):
        marker = "s" if row.multiple_author_events else "o"
        color = COLORS["upstream"] if row.doc_peak_pair_outside_retrieval_regime else "#A7B6BA"
        low, high = row.optical_doc_peak_difference_lower_hours, row.optical_doc_peak_difference_upper_hours
        mid = (low+high)/2
        ax.errorbar(mid, i, xerr=[[mid-low], [high-mid]], marker=marker, color=color, markersize=4, capsize=2, linestyle="none")
    ax.axvline(0, color="#B7C4C6", linestyle=":", linewidth=1)
    ax.set(yticks=np.arange(len(segments)), yticklabels=[f"#{b}" for b in segments.block],
           xlabel="下游－上游光学 DOC 峰值时差（小时）" if cn else "Downstream − upstream optical-DOC peak clock (h)")
    ax.invert_yaxis()
    ax.set_title("c  DOC 峰值并非固定平移" if cn else "c  DOC peak clocks do not share one offset", loc="left", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[1, 1]
    totals = [15, 7]
    peak = [int(segments.doc_peak_pair_outside_retrieval_regime.sum()), int(main.doc_peak_pair_outside_retrieval_regime.sum())]
    widths = [int(segments.doc_optical_only_width_pair.sum()), int(main.doc_optical_only_width_pair.sum())]
    x = np.arange(2)
    for dx, values, color, label in (
        (-.25, totals, "#D8E0E1", "记录数" if cn else "Segments"),
        (0, peak, COLORS["upstream"], "两端峰值 ≤600 FNU" if cn else "Both peak samples ≤600 FNU"),
        (.25, widths, COLORS["receiver"], "半高区间全部 ≤600 FNU" if cn else "All width-support samples ≤600 FNU"),
    ):
        ax.bar(x+dx, values, width=.23, color=color, label=label)
        for xi, value in zip(x+dx, values):
            ax.text(xi, value+.25, str(value), ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=("全部记录", "至少 24 小时") if cn else ("All segments", "≥24 elapsed hours"),
           ylabel="上下游配对段数" if cn else "Paired segments", ylim=(0, 20))
    ax.set_title("d  DOC 波形宽度受到估算时段影响" if cn else "d  Retrieval affects DOC width evidence", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("流量峰值晚到，与 DOC 缓冲是两件事" if cn else "Later flow peaks and DOC buffering are distinct questions", x=.07, y=.98, ha="left", fontsize=16)
    fig.text(.07, .921, "一个上下游嵌套河段；422 个配对小时。所有段保留；不把每个小时或记录段当作独立流域。" if cn else
             "One nested reach, 422 paired hours. All segments retained; hours and windows are not independent catchments.", fontsize=10)
    fig.text(.07, .035, ("峰值时差以 1 小时采样给出；宽度仅在线性连接相邻样本时估算。c：方形为多个作者事件；灰色为峰值进入 >600 FNU 的估算区。\n"
              ">600 FNU 的 DOC 使用流量与降雨回归估算；≤600 FNU 仍是校正后的光学估计。宽度支持包含峰体及两侧交叉样本。\n"
              "所有 DOC 峰值均超过原文实验室校准上限 3.88 mg C/L；低于浊度阈值不等于峰值已被实验室验证。" if cn else
              "Peak clocks have 1-h resolution; widths interpolate adjacent samples only. c: Squares mark multiple author events; grey marks >600-FNU peaks.\n"
              "DOC above 600 FNU uses flow/rainfall regressions; lower-turbidity DOC is still a corrected optical estimate. Width support includes crossing samples.\n"
              "All DOC maxima exceed the source laboratory calibration maximum of 3.88 mg C/L; the turbidity screen does not validate their accuracy."),
             fontsize=9, color="#53696E")
    save(fig, "hourly_response_and_quality")

    # Chronological coverage selection, not DOC-response selection.
    examples = main.head(3).block.tolist()
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.6))
    fig.subplots_adjust(left=.07, right=.97, bottom=.14, top=.87, hspace=.65, wspace=.25)
    for i, block in enumerate(examples):
        frame = paired[paired.block.eq(block)].reset_index(drop=True)
        start = plot_segment(*axes[i], frame, cn)
        axes[i, 0].set_title(f"{'记录段' if cn else 'Segment'} #{block} · {start:%Y-%m-%d} · {len(frame)-1} h", loc="left", fontsize=11)
        axes[i, 1].set_title("实心：校正光学值；×：高浊度估算区" if cn else "Dots: corrected optical; ×: retrieval regime", loc="left", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("上下游流量与 DOC 的事件响应" if cn else "Paired flow and optical-DOC responses", x=.07, y=.982, ha="left", fontsize=16)
    fig.text(.07, .928, "按时间先后展示最早三个 ≥24 小时记录段，没有按 DOC 响应挑选。" if cn else
             "The earliest three ≥24-h segments, selected chronologically rather than by DOC outcome.", fontsize=10)
    fig.text(.07, .045, "流量按各段、各站的最小值和极差归一化，仅比较形状。DOC 保留 mg C/L 原值，× 不是独立光学峰值。\n"
             "源时钟时区未说明；图中记录段可能包含多次波峰。" if cn else
             "Flow is normalized by each site-segment minimum and range to compare shape. DOC retains mg C/L; × is not an independent optical peak.\n"
             "Source-clock timezone is unspecified; an observation segment can contain several waves.", fontsize=9, color="#53696E")
    save(fig, "hourly_waveform_examples")

    # Diagnostic atlas: every segment, in time order, five per page.
    for page in range(3):
        blocks = segments.block.iloc[page*5:(page+1)*5].tolist()
        fig, axes = plt.subplots(5, 2, figsize=(12.4, 15))
        fig.subplots_adjust(left=.07, right=.97, bottom=.08, top=.92, hspace=.8, wspace=.25)
        for i, block in enumerate(blocks):
            frame = paired[paired.block.eq(block)].reset_index(drop=True)
            start = plot_segment(*axes[i], frame, cn, compact=True)
            axes[i, 0].set_title(f"#{block} · {start:%Y-%m-%d} · {len(frame)-1} h", loc="left", fontsize=10)
            metric = segments[segments.block.eq(block)].iloc[0]
            axes[i, 1].set_title("多作者事件" if cn and metric.multiple_author_events else
                                "Multiple author events" if metric.multiple_author_events else
                                "单一作者事件" if cn else "Single author event", loc="left", fontsize=9)
        handles = [Line2D([0], [0], color=COLORS[site], label=("上游 SN" if site == "upstream" else "下游 FITT") if cn else site) for site in COLORS]
        handles.append(Line2D([0], [0], color="#53696E", marker="x", linestyle="none", label=">600 FNU"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.97, .966), frameon=False, ncol=3, fontsize=9)
        fig.suptitle(f"{'全部记录段检查' if cn else 'All-segment diagnostic atlas'} · {page+1}/3", x=.07, y=.983, ha="left", fontsize=15)
        fig.text(.07, .022, "每段不插值缺测小时。× 为原论文报告的流量/降雨 DOC 估算区；保留全部记录，不选择低浊度替代峰。" if cn else
                 "No interpolation across missing hours. × marks the reported flow/rainfall DOC retrieval regime; no replacement low-turbidity maxima.", fontsize=9, color="#53696E")
        save(fig, f"hourly_diagnostic_atlas_{page+1}")

    # Preserve the complete, shorter low-retrieval example rather than hiding it.
    clean = segments[segments.doc_optical_only_width_pair]
    selection = {"main_examples": examples, "atlas": segments.block.tolist(),
                 "optical_only_width_pairs": clean.block.tolist(),
                 "selected_without_doc_direction": True}
    sources = [analysis / name for name in ("segment_comparison.csv", "site_waveforms.csv", "paired_hourly_quality.parquet")]
    sources += [Path("scripts/plot_doc_river_hourly_response_v1.py")]
    receipt = {"input_hashes": {str(p): sha256_file(p) for p in sources},
               "output_hashes": {str(p): sha256_file(p) for p in outputs},
               "selection": selection, "chinese": cn, "n_site_waveforms": len(sites)}
    (output / f"figure_sources{suffix}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Saved {len(outputs)} figures; all 15 segments retained")


if __name__ == "__main__":
    main()
