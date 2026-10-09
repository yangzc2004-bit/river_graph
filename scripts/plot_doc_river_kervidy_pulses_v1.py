"""Publish actual single-pulse responses, distributions, and rooted river paths."""

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
from shapely.geometry import shape

from river_graph.analysis.river_kervidy_observations import prepare_flow
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_kervidy_pulses_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1")
FLOW = Path("data/raw/river_kervidy_geometry_v1/discharge_quarter_hour.parquet")
BLUE, GOLD, DARK, GREY = "#297B8B", "#CA8748", "#344F5A", "#BAC3C7"


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
    saved = []

    def label(en, zh):
        return zh if cn else en

    def save(fig, name):
        for ext in ("png", "svg"):
            path = out / f"{name}{'_cn' if cn else ''}.{ext}"
            fig.savefig(path, dpi=200, facecolor="white")
            saved.append(path)
        plt.close(fig)

    flow = prepare_flow(pd.read_parquet(FLOW))
    doc = pd.read_parquet(PREVIOUS / "analysis/corrected_doc_flow_matches.parquet")
    examples = pd.read_csv(source / "chronological_examples.csv")
    for col in examples.columns:
        if col.endswith("_utc"):
            examples[col] = pd.to_datetime(examples[col], utc=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharex=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.17, top=.80, hspace=.37, wspace=.20)
    fig.suptitle(label("DOC and water do not form the same pulse", "DOC 高峰与水量高峰不是同一条曲线"),
                 x=.08, y=.966, ha="left", fontsize=18)
    fig.text(.08, .89, label("First width-resolved pulse in each observation year · Kervidy–Naizin · original UTC clocks",
        "每个观测年份中，最早能测完整宽度的脉冲；法国 Kervidy–Naizin；沿用原始 UTC 时间"), fontsize=10)
    xmin, xmax = 0., 0.

    def trace(ax, frame, clock, field, peak, baseline, maximum, color, name):
        groups = frame[clock].diff().dt.total_seconds().gt(3600).cumsum()
        for k, (_, part) in enumerate(frame.groupby(groups)):
            hours = (part[clock]-peak).dt.total_seconds()/3600
            ax.plot(hours, (part[field]-baseline)/(maximum-baseline), color=color, lw=1.8,
                    label=name if k == 0 else None)

    for i, (ax, w) in enumerate(zip(axes.flat, examples.itertuples())):
        q = flow.loc[flow.flow_timestamp_utc.between(w.antecedent_start_utc, w.response_end_utc)]
        d = doc.loc[doc.timestamp_utc.between(w.antecedent_start_utc, w.response_end_utc)]
        trace(ax, q, "flow_timestamp_utc", "q_m3_s", w.flow_peak_utc, w.q_baseline_m3_s,
              w.q_peak_m3_s, GOLD, label("Flow", "流量"))
        trace(ax, d, "timestamp_utc", "doc_mg_l", w.flow_peak_utc, w.doc_baseline_mg_l,
              w.doc_peak_mg_l, BLUE, "DOC")
        ax.axvline(0, color=GOLD, ls="--", lw=.9)
        ax.axvline(w.doc_peak_lag_hours, color=BLUE, ls=":", lw=1.0)
        ax.axhline(.5, color=GREY, lw=.7, zorder=0)
        ax.axhline(0, color=GREY, lw=.7, zorder=0)
        for field, y, color in (("flow", -.17, GOLD), ("doc", -.28, BLUE)):
            left = (getattr(w, f"{field}_left_crossing_utc")-w.flow_peak_utc).total_seconds()/3600
            right = (getattr(w, f"{field}_right_crossing_utc")-w.flow_peak_utc).total_seconds()/3600
            ax.plot([left, right], [y, y], color=color, lw=2.3, marker="|", ms=8)
        ax.set_title(f"{'abcd'[i]}  {w.flow_peak_utc:%Y-%m-%d}\n"+label(
            f"DOC lag {w.doc_peak_lag_hours:+.2f} h · width ratio {w.doc_flow_width_ratio:.2f}",
            f"DOC 峰相对延迟 {w.doc_peak_lag_hours:+.2f} 小时；宽度比 {w.doc_flow_width_ratio:.2f}"),
            loc="left", fontsize=10)
        ax.text(.98, .94, label(f"DOC peak {w.doc_peak_mg_l:.1f} mg/L\nFlow peak {w.q_peak_m3_s:.3f} m³/s",
            f"DOC 峰 {w.doc_peak_mg_l:.1f} mg/L\n流量峰 {w.q_peak_m3_s:.3f} m³/s"),
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
        normalized = np.r_[
            (q.q_m3_s-w.q_baseline_m3_s)/(w.q_peak_m3_s-w.q_baseline_m3_s),
            (d.doc_mg_l-w.doc_baseline_mg_l)/(w.doc_peak_mg_l-w.doc_baseline_mg_l)]
        ax.set_ylim(min(-.38, normalized.min()-.08), max(1.15, normalized.max()+.08))
        xmin = min(xmin, (w.antecedent_start_utc-w.flow_peak_utc).total_seconds()/3600)
        xmax = max(xmax, (w.response_end_utc-w.flow_peak_utc).total_seconds()/3600)
    for ax in axes.flat:
        ax.set_xlim(xmin-1, xmax+1)
    for ax in axes[:, 0]:
        ax.set_ylabel(label("Excess / own pulse peak excess", "较事前水平的增量／各自峰值增量"))
    for ax in axes[-1, :]:
        ax.set_xlabel(label("Hours from flow peak", "距流量峰的小时数"))
    axes[0, 0].legend(loc="upper left", frameon=False, ncols=2, fontsize=9)
    fig.text(.08, .075, label("Curves use their own antecedent baseline and peak excess; panel y-ranges differ. Bars below zero mark half-excess widths.\nNegative values mean below antecedent baseline. Examples are chronological, not selected by DOC size; gaps > 1 h are not joined.",
        "按各自事前水平和峰值增量归一化；各面板纵轴范围不同；下方横线为半峰宽。\n负值表示低于事前水平。按时间顺序选例，不按 DOC 高低选例；超过一小时的缺口不连线。"), fontsize=9)
    save(fig, "single_pulse_examples")

    events = pd.read_csv(source / "resolved_doc_pulses.csv")
    widths = events.loc[events.width_pair_eligible]
    sensitivity = pd.read_csv(source / "threshold_sensitivity.csv")
    main_row = sensitivity.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7))
    fig.subplots_adjust(left=.07, right=.96, top=.72, bottom=.25, wspace=.35)
    fig.suptitle(label("Observed DOC peaks are usually later and broader", "真实观测中，DOC 峰通常更晚，也更宽"),
                 x=.07, y=.96, ha="left", fontsize=18)
    fig.text(.07, .87, label(f"{len(events)} positive DOC responses · {len(widths)} complete width pairs · 2020–2023 · one mapped catchment",
        f"{len(events)} 次可辨认的正向 DOC 响应；{len(widths)} 对完整宽度；2020–2023 年；同一真实流域"), fontsize=10)
    lag = np.sort(events.doc_peak_lag_hours)
    axes[0].step(lag, np.arange(1, len(lag)+1)/len(lag), where="post", color=BLUE, lw=2)
    axes[0].axvline(0, color=GREY, ls="--", lw=1)
    axes[0].set(xlabel=label("DOC peak − flow peak (hours)", "DOC 峰减去流量峰（小时）"),
                ylabel=label("Cumulative event fraction", "累计事件比例"), ylim=(0, 1.04))
    axes[0].set_title(label(f"a  Peak delay (all {len(events)} events)", f"a  峰值延迟（全部 {len(events)} 次）"), loc="left", fontsize=10)
    lag_mid, lag_lo, lag_hi = [main_row[f"doc_peak_lag_hours_{key}"] for key in ("median", "ci_low", "ci_high")]
    n_later = int(events.doc_peak_lag_hours.gt(0).sum())
    axes[0].text(.98, .13, label(f"Median {lag_mid:.2f} h\n95% month-block CI: {lag_lo:.2f}–{lag_hi:.2f} h\n{n_later}/{len(events)} DOC peaks follow flow",
        f"中位延迟 {lag_mid:.2f} 小时\n月分组 95% 区间：{lag_lo:.2f}–{lag_hi:.2f} 小时\n{n_later}/{len(events)} 次 DOC 峰晚于流量峰"),
        transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9)
    axes[1].scatter(widths.flow_width_hours, widths.doc_width_hours, s=30, color=BLUE, alpha=.7,
                    edgecolors="white", linewidths=.4)
    maximum = float(max(widths.flow_width_hours.max(), widths.doc_width_hours.max()))*1.07
    axes[1].plot([0, maximum], [0, maximum], color=DARK, ls="--", lw=1)
    axes[1].set(xlabel=label("Flow half-excess width (hours)", "流量半峰宽（小时）"),
                ylabel=label("DOC half-excess width (hours)", "DOC 半峰宽（小时）"),
                xlim=(0, maximum), ylim=(0, maximum))
    axes[1].set_title(label(f"b  Pulse spreading (all {len(widths)} pairs)", f"b  脉冲展宽（全部 {len(widths)} 对）"), loc="left", fontsize=10)
    ratio = main_row.doc_flow_width_ratio_median
    n_wider = int(widths.doc_flow_width_ratio.gt(1).sum())
    axes[1].text(.97, .08, label(f"Median DOC/flow ratio {ratio:.2f}\n{n_wider}/{len(widths)} DOC pulses are wider\nDashed: equal width",
        f"DOC／流量宽度比中位数 {ratio:.2f}\n{n_wider}/{len(widths)} 次 DOC 脉冲更宽\n虚线：宽度相等"),
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=9)
    for j, percent in enumerate((15, 20, 25)):
        row = sensitivity.loc[sensitivity.setting.str.contains(f"{percent}pct")].iloc[0]
        lo, mid, hi = [row[f"doc_flow_width_ratio_{key}"] for key in ("ci_low", "median", "ci_high")]
        axes[2].plot([lo, hi], [j, j], color=BLUE, lw=1.6)
        axes[2].scatter(mid, j, color=BLUE, s=55 if percent == 20 else 30, zorder=3)
        axes[2].text(hi+.02, j, f"n={int(row.n_width_pairs)}", va="center", fontsize=8)
    axes[2].axvline(1, color=DARK, ls="--", lw=1)
    axes[2].set(yticks=[0, 1, 2], yticklabels=["15%", "20%", "25%"], ylim=(2.6, -.6), xlim=(.95, 2.13),
                xlabel=label("DOC / flow half-width ratio", "DOC／流量半峰宽比"),
                ylabel=label("Relative flow prominence", "流量相对突出度阈值"))
    axes[2].set_title(label("c  Threshold sensitivity", "c  换阈值后的结果"), loc="left", fontsize=10)
    fig.text(.07, .135, label(f"Primary flow threshold: 20% relative prominence. Width ratio 95% month-block CI: {main_row.doc_flow_width_ratio_ci_low:.2f}–{main_row.doc_flow_width_ratio_ci_high:.2f}.\nNo time shift or interpolation is fitted. This measures outlet DOC response; it is not a river travel-speed estimate or a cross-form effect.",
        f"主分析按流量相对突出度 20% 选事件；宽度比的月分组 95% 区间：{main_row.doc_flow_width_ratio_ci_low:.2f}–{main_row.doc_flow_width_ratio_ci_high:.2f}。\n没有拟合时间平移或插值。这里测的是出口 DOC 响应，不把延迟直接当流速，也不据此判定不同形态的效应。"), fontsize=9)
    fig.text(.07, .035, "Source: Faucheux et al. (2024), doi:10.57745/OFOUWE; AgrHyS observed level/rating-curve discharge", fontsize=8)
    save(fig, "observed_pulse_delay_and_spreading")

    geometry = json.loads((source / "topage_path_geometry.json").read_text())
    inventory = json.loads((source / "topage_path_inventory.json").read_text())
    basin, gauge = shape(geometry["catchment"]), shape(geometry["gauge"])
    origin = basin.bounds[:2]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 8))
    fig.subplots_adjust(left=.08, right=.96, top=.80, bottom=.19, wspace=.35)
    fig.suptitle(label("Real tributary paths meet before the monitored outlet", "真实支流如何汇合到 DOC 监测出口"),
                 x=.08, y=.966, ha="left", fontsize=18)
    fig.text(.08, .89, label("BD Topage mapped river network · official Kervidy catchment · unique geometric paths to the observed outlet",
        "BD Topage 真实河网；官方 Kervidy 流域边界；沿真实河道量取源头到监测出口的路径"), fontsize=10)
    ax = axes[0]
    polygons = list(basin.geoms) if hasattr(basin, "geoms") else [basin]
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.fill((np.asarray(x)-origin[0])/1000, (np.asarray(y)-origin[1])/1000,
                facecolor="#EEF2E8", edgecolor="#899A82", lw=1)
    for edge_id, geo in geometry["reaches"].items():
        line = shape(geo)
        x, y = line.xy
        shared = edge_id in geometry["common_terminal_edge_ids"]
        ax.plot((np.asarray(x)-origin[0])/1000, (np.asarray(y)-origin[1])/1000,
                color=GOLD if shared else BLUE, lw=3 if shared else 1.8)
    for path in geometry["paths"]:
        x, y = path["headwater_xy"]
        ax.scatter((x-origin[0])/1000, (y-origin[1])/1000, color=BLUE, s=25, zorder=5)
        ax.annotate(path["path_id"], ((x-origin[0])/1000, (y-origin[1])/1000),
                    xytext=(7, 4), textcoords="offset points", fontsize=10)
    ax.scatter((gauge.x-origin[0])/1000, (gauge.y-origin[1])/1000, color=GOLD,
               marker="*", s=160, edgecolors="white", lw=.6, zorder=5)
    ax.set(aspect="equal", xlabel=label("Eastward distance (km)", "向东距离（公里）"),
           ylabel=label("Northward distance (km)", "向北距离（公里）"))
    ax.set_title(label("a  Four mapped headwater paths", "a  四条已绘源头路径"), loc="left", fontsize=10)
    ax.margins(.12)
    ax.annotate("N", xy=(.91, .95), xytext=(.91, .82), xycoords="axes fraction", ha="center",
                arrowprops={"arrowstyle": "-|>", "color": DARK})
    lengths = [p["path_length_m"]/1000 for p in geometry["paths"]]
    paths = [p["path_id"] for p in geometry["paths"]]
    shared = inventory["shared_terminal_length_m"]/1000
    axes[1].barh(paths, np.asarray(lengths)-shared, color=BLUE, height=.53,
                 label=label("Before final common route", "最终汇合之前的路径"))
    axes[1].barh(paths, shared, left=np.asarray(lengths)-shared, color=GOLD, height=.53,
                 label=label("Final route shared by all four", "四条路径最后共用的河段"))
    for j, length in enumerate(lengths):
        axes[1].text(length+.03, j, f"{length:.2f} km", va="center", fontsize=10)
    axes[1].set(xlim=(0, max(lengths)+.43), xlabel=label("Mapped headwater-to-outlet length (km)", "源头到出口的河道路径长度（公里）"))
    axes[1].invert_yaxis()
    axes[1].set_title(label("b  Similar lengths, a common downstream route", "b  路径长度相近，共用最后约 570 米"), loc="left", fontsize=10)
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, loc="lower left", bbox_to_anchor=(.075, .835), ncols=2, fontsize=9)
    fig.text(.08, .085, label("Mapped river total 5.05 km; 3 confluences; headwater paths 1.96–2.21 km. The mapped terminal is 1.01 m from the gauge.\nLengths are geometric, not measured travel times. Digitization agrees with the known outlet; no missing connections are bridged.",
        "边界内已绘河道共 5.05 公里；3 处汇合；源头到出口 1.96–2.21 公里；河道末端距监测位置 1.01 米。\n这些是几何长度，不是实测传输时间；河道方向与已知出口一致，没有人为补接断口。"), fontsize=9)
    fig.text(.08, .025, "Source: GeoSAS / AgrHyS; BD Topage IGN–OFB; UMR 1069 SAS INRA – Agrocampus Ouest", fontsize=8)
    save(fig, "actual_tributary_outlet_paths")
    inputs = [source / "chronological_examples.csv", source / "resolved_doc_pulses.csv",
        source / "threshold_sensitivity.csv", source / "topage_path_geometry.json", source / "topage_path_inventory.json",
        FLOW, PREVIOUS / "analysis/corrected_doc_flow_matches.parquet"]
    (ROOT / f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs}, "script_hash": sha256_file(Path(__file__)),
        "figures": {str(p): sha256_file(p) for p in saved}}, indent=2)+"\n")
    print("\n".join(map(str, saved)))


if __name__ == "__main__":
    main()
