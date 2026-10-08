"""Draw the real Kervidy river map and flow-selected DOC response windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager
from shapely.geometry import shape

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1")
RAW = Path("data/raw/river_kervidy_geometry_v1")
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
    paths = []

    def label(en, zh):
        return zh if cn else en

    def save(fig, name):
        for extension in ("png", "svg"):
            p = out / f"{name}{'_cn' if cn else ''}.{extension}"
            fig.savefig(p, dpi=210, facecolor="white")
            paths.append(p)
        plt.close(fig)

    geometry = json.loads((source / "mapped_geometry.json").read_text())
    inventory = json.loads((source / "geometry_inventory.json").read_text())
    river, inside, basin, gauge = [shape(geometry[n]) for n in (
        "river_network", "clipped_river_network", "catchment", "gauge")]
    origin = basin.bounds[0], basin.bounds[1]

    def line(ax, geom, **kwargs):
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for g in parts:
            x, y = g.xy
            ax.plot([(v-origin[0])/1000 for v in x], [(v-origin[1])/1000 for v in y], **kwargs)

    def basin_patch(ax):
        polygons = list(basin.geoms) if hasattr(basin, "geoms") else [basin]
        for p in polygons:
            x, y = p.exterior.xy
            ax.fill([(v-origin[0])/1000 for v in x], [(v-origin[1])/1000 for v in y],
                    facecolor="#EEF2E8", edgecolor="#899A82", linewidth=1.1, zorder=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 8.1))
    fig.subplots_adjust(left=.08, right=.96, top=.81, bottom=.20, wspace=.30)
    fig.suptitle(label("The observed DOC outlet belongs to a real mapped river network",
                       "把 DOC 监测出口放回真实河网"), x=.08, y=.965, ha="left", fontsize=17)
    fig.text(.08, .89, label("Kervidy–Naizin, France · public field river map + official catchment boundary + measured outlet location",
        "法国 Kervidy–Naizin · 真实河道底图、官方流域边界与监测出口位置"), fontsize=10)
    for ax in axes:
        basin_patch(ax)
        line(ax, river, color=GREY, linewidth=1.4, zorder=1)
        line(ax, inside, color=BLUE, linewidth=2.1, zorder=2)
        gx, gy = (gauge.x-origin[0])/1000, (gauge.y-origin[1])/1000
        ax.scatter(gx, gy, color=GOLD, marker="*", s=170, edgecolors="white", linewidths=.7, zorder=4)
        ax.annotate(label("DOC + flow outlet", "DOC 与流量监测出口"), (gx, gy), xytext=(12, -20),
                    textcoords="offset points", fontsize=9, color=DARK,
                    arrowprops={"arrowstyle": "-", "color": DARK, "lw": .7})
        ax.set_aspect("equal")
        ax.set_xlabel(label("Eastward distance (km; local origin)", "向东距离（公里；局地坐标）"))
        ax.set_ylabel(label("Northward distance (km; local origin)", "向北距离（公里；局地坐标）"))
        ax.annotate("N", xy=(.94, .96), xytext=(.94, .86), xycoords="axes fraction", ha="center",
                    arrowprops={"arrowstyle": "-|>", "color": DARK}, fontsize=10)
    axes[0].set_title(label("a  Full mapped context includes downstream reaches", "a  整份河道图包含出口下游河段"),
                      loc="left", fontsize=10)
    axes[0].margins(.10)
    axes[1].set_title(label("b  Isolate the monitored catchment", "b  单独识别监测出口上游流域"), loc="left", fontsize=10)
    xmin, ymin, xmax, ymax = basin.bounds
    axes[1].set_xlim(-.18, (xmax-xmin)/1000+.22)
    axes[1].set_ylim(-.32, (ymax-ymin)/1000+.20)
    axes[1].text(.03, .97, label(f"Area {inventory['catchment_area_km2']:.2f} km²\nMapped river length inside {inventory['within_catchment_river_length_km']:.2f} km",
        f"流域面积 {inventory['catchment_area_km2']:.2f} 平方公里\n边界内已绘河道 {inventory['within_catchment_river_length_km']:.2f} 公里"),
        transform=axes[1].transAxes, va="top", fontsize=9)
    for ax in axes:
        ax.plot([.04, .04+.5/(ax.get_xlim()[1]-ax.get_xlim()[0])], [.05, .05], transform=ax.transAxes,
                color=DARK, linewidth=2)
        ax.text(.04, .075, "500 m", transform=ax.transAxes, fontsize=8)
    fig.text(.08, .115, label("Blue: mapped rivers inside the official basin · Grey: mapped context outside · Star: observed gauge\nThe exported river layer has no flow-direction fields; the map does not assign a morphology effect to DOC.",
        "蓝色：官方流域边界内的已绘河道；灰色：边界外河道；星号：真实监测位置。\n这份河道导出没有流向字段；这里建立几何对应，不从一条曲线判定形态效应。"), fontsize=9)
    fig.text(.08, .038, "Source : UMR 1069 SAS INRA - Agrocampus Ouest", fontsize=9)
    save(fig, "actual_river_network_and_outlet")

    windows = pd.read_csv(source / "flow_selected_windows.csv", parse_dates=[
        "selected_flow_peak_utc", "window_start_utc", "window_end_utc"])
    joined = pd.read_parquet(source / "corrected_doc_flow_matches.parquet")
    flow = pd.read_parquet(RAW / "discharge_quarter_hour.parquet")
    fig, axes = plt.subplots(2, len(windows), figsize=(15.5, 8.6), sharex="col", sharey="row")
    fig.subplots_adjust(left=.07, right=.96, top=.78, bottom=.20, hspace=.33, wspace=.20)
    fig.suptitle(label("Observed DOC responses around flow-selected maxima",
                       "按流量选择窗口，查看 DOC 的真实涨落"), x=.07, y=.965, ha="left", fontsize=17)
    fig.text(.07, .89, label("Same catchment, four observation years · 3 days before / 4 days after each reported flow maximum",
        "同一真实流域，四个观测年份；每次展示记录流量最大值之前三天与之后四天"), fontsize=10)

    def trace(ax, data, clock, value, peak, color, gap_hours):
        frame = data.loc[data[value].notna()].sort_values(clock)
        block = frame[clock].diff().dt.total_seconds().gt(gap_hours*3600).cumsum()
        for _, g in frame.groupby(block):
            x = (g[clock]-peak).dt.total_seconds()/3600
            ax.plot(x, g[value], color=color, linewidth=1)
            ax.scatter(x, g[value], color=color, s=3, alpha=.40, linewidths=0)

    for column, w in enumerate(windows.itertuples()):
        q = flow.loc[flow.timestamp_utc.between(w.window_start_utc, w.window_end_utc)].copy()
        q["q_m3_s"] = q.q_dm3_s*.001
        d = joined.loc[joined.timestamp_utc.between(w.window_start_utc, w.window_end_utc)]
        peak = w.selected_flow_peak_utc
        trace(axes[0, column], q, "timestamp_utc", "q_m3_s", peak, GREY, 1)
        trace(axes[1, column], d, "timestamp_utc", "doc_mg_l", peak, BLUE, 1)
        suffix = label("partial-year span", "部分年度跨度") if w.partial_observation_year else label("full-year span", "全年观测跨度")
        axes[0, column].set_title(f"{w.year} · {suffix}\n{peak:%Y-%m-%d %H:%M} UTC", loc="left", fontsize=10)
        status = label("Dense joint window", "联合记录密集") if w.dense_joint_window else label("DOC gap > 1 h", "DOC 有超过一小时的缺口")
        axes[1, column].text(.02, .94, status, transform=axes[1, column].transAxes, va="top", fontsize=8,
                             color=DARK if w.dense_joint_window else GOLD)
        if w.reported_flow_max_censored:
            axes[0, column].text(.02, .93, label("Reported flow ceiling", "流量峰触及记录上限"),
                transform=axes[0, column].transAxes, va="top", fontsize=8, color=GOLD)
        for ax in axes[:, column]:
            ax.axvline(0, color=GOLD, linestyle="--", linewidth=.85)
            ax.set_xlim(-72, 96)
            ax.set_xticks([-72, 0, 72])
        axes[1, column].set_xlabel(label("Hours from reported flow maximum", "距记录流量最大值（小时）"), fontsize=9)
    axes[0, 0].set_ylabel(label("Discharge (m³/s)", "流量（立方米／秒）"))
    axes[1, 0].set_ylabel(label("Corrected optical DOC (mg/L)", "校正光学 DOC（mg/L）"))
    axes[0, 0].set_ylim(0, 1.38)
    axes[1, 0].set_ylim(0, 36)
    fig.text(.07, .095, label("Dots are archived observations; lines stop at gaps over one hour. DOC values do not select these windows.\n2022 has a 4.75-hour DOC gap; 2023 reaches the reported flow cap. Repeated events here are not independent river forms.",
        "点为档案观测；超过一小时的缺口不连线；窗口选择没有查看 DOC 值。\n2022 年有 4.75 小时 DOC 缺口，2023 年流量触及记录上限。这些是同一河网的重复事件，不是不同形态。"), fontsize=9)
    fig.text(.07, .03, "DOC: Faucheux et al. (2024), doi:10.57745/OFOUWE · AgrHyS flow: Fovet et al. (2018), doi:10.2136/vzj2018.04.0066", fontsize=8)
    save(fig, "flow_selected_doc_response_windows")
    inputs = [source / "mapped_geometry.json", source / "geometry_inventory.json", source / "flow_selected_windows.csv",
              source / "corrected_doc_flow_matches.parquet", RAW / "discharge_quarter_hour.parquet"]
    (ROOT / f"figure_sources{'_cn' if cn else ''}.json").write_text(json.dumps({
        "inputs": {str(p): sha256_file(p) for p in inputs}, "code_hash": sha256_file(Path(__file__)),
        "figures": {str(p): sha256_file(p) for p in paths}}, indent=2) + "\n")
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
