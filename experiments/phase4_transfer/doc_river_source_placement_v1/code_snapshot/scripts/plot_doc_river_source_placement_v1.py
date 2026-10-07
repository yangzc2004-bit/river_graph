"""Map real wetland/forest source placement and render its DOC diagnostics."""
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, fontManager
from shapely.geometry import shape

from river_graph.analysis.river_source_placement import PLACEMENT
from river_graph.topology.river_planform import (
    project_geometry,
    project_lines,
    read_cached_lines,
)

ROOT = Path("experiments/phase4_transfer/doc_river_source_placement_v1")
CACHE = Path("data/raw/river_source_placement_v1")
COLORS = ("#257F88", "#C5814A", "#66749F")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", action="store_true")
    cn = parser.parse_args().chinese
    if cn:
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        fontManager.addfont(font)
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#263B42", "axes.labelcolor": "#263B42", "savefig.dpi": 230})
    out = ROOT/"figures"
    out.mkdir(exist_ok=True)
    suffix = "_cn" if cn else ""
    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{name}{suffix}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    panel = pd.read_csv(ROOT/"analysis/station_doc_source_panel.csv", dtype={"station": str})
    s = panel[panel.included].copy()
    classes = ("细长、多支流", "主干主导、少支流", "宽展、多支流") if cn else ("Elongated tributary-rich", "Mainstem dominated sparse", "Broad tributary-rich")
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.8))
    for ax, name, label in zip(axes[:2], ("wetland", "forest"), ("湿地", "森林") if cn else ("Wetland", "Forest"), strict=True):
        for group, title in zip((1, 2, 3), classes, strict=True):
            v = s[s.cluster.eq(group)]
            ax.scatter(v[f"{name}_pct"], v[f"{name}_distance_ratio"], s=22, alpha=.6, color=COLORS[group-1], label=f"{title} (n={len(v)})")
        ax.axhline(1, color="#879295", linestyle="--", linewidth=.8)
        ax.set(xlabel=f"{label}面积比例（%）" if cn else f"{label} area (%)",
               ylabel="来源平均距离／流域平均距离" if cn else "Source mean distance / drainage mean distance")
        ax.set_title(label, loc="left", fontweight="bold")
    for group in (1, 2, 3):
        v = s[s.cluster.eq(group)]
        axes[2].scatter(v.wetland_riparian_enrichment, v.forest_riparian_enrichment, s=22, alpha=.6, color=COLORS[group-1])
    axes[2].axhline(0, color="#879295", linewidth=.8)
    axes[2].axvline(0, color="#879295", linewidth=.8)
    axes[2].set(xlabel="湿地河岸富集（百分点）" if cn else "Wetland riparian enrichment (percentage points)",
                ylabel="森林河岸富集（百分点）" if cn else "Forest riparian enrichment (percentage points)")
    axes[2].set_title("河岸连接" if cn else "Riparian concentration", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .89), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("河网中的潜在 DOC 来源位置" if cn else "Potential DOC source placement within real networks", fontsize=17, x=.06, ha="left")
    fig.text(.06, .015, "每点为一个源站点；无该来源的距离点省略。距离比 <1 表示更靠近出口；河岸比较 100 m 范围与集水区。NLCD 2019。" if cn else
        "One point per station; zero-source positions omitted. Ratio <1: nearer outlet. Riparian contrast: 100-m buffers versus catchments. NLCD 2019.", fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, bottom=.2, top=.72, wspace=.43)
    save(fig, "source_placement_distribution")

    landscape = pd.read_parquet(CACHE/"reach_landscape.parquet").set_index("comid")
    chosen = []
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.1))
    with sqlite3.connect("data/raw/river_planform_v1/flowlines.sqlite") as connection:
        for col, group in enumerate((1, 2, 3)):
            candidates = s[s.cluster.eq(group) & s.n_upstream_reaches.between(25, 5000)].copy()
            features = ["log_basin_aspect", "log_network_axis_ratio", "mainstem_share", "mainstem_sinuosity"]
            scaled = (candidates[features]-s[s.cluster.eq(group)][features].median()).div(s[s.cluster.eq(group)][features].std().replace(0, 1))
            candidates["shape_centrality"] = (scaled**2).mean(axis=1)
            # Representatives are selected on geometry/cache coverage only.
            representative = None
            for r in candidates.sort_values(["shape_centrality", "station"]).itertuples():
                with np.load(CACHE/"routing"/f"comid_{r.comid}.npz") as z:
                    members = z["comids"]
                lines = read_cached_lines(connection, members)
                if len(lines) == len(members):
                    representative = r
                    break
            if representative is None:
                raise ValueError(f"no complete real geometry representative for class {group}")
            r = representative
            chosen.append({"cluster": group, "station": r.station, "comid": r.comid, "reaches": len(members)})
            lines = project_lines(lines)
            basin_path = Path("data/raw/river_planform_v1/basins")/f"comid_{r.comid}.json"
            basin = project_geometry(shape(json.loads(basin_path.read_text())["features"][0]["geometry"]))
            f = landscape.reindex(members)
            for row, name, label, cmap, vmax in ((0, "wetland", "湿地" if cn else "Wetland", "YlGnBu", 50),
                                               (1, "forest", "森林" if cn else "Forest", "YlGn", 100)):
                ax = axes[row, col]
                for polygon in (basin.geoms if basin.geom_type == "MultiPolygon" else [basin]):
                    coords = np.asarray(polygon.exterior.coords)/1000
                    ax.plot(coords[:, 0], coords[:, 1], color="#B7C2C4", linewidth=.7)
                cover = (f.pctwdwet2019cat+f.pcthbwet2019cat) if name == "wetland" else (f.pctdecid2019cat+f.pctconif2019cat+f.pctmxfst2019cat)
                pieces, values = [], []
                for cid, value in zip(members, cover, strict=True):
                    line = lines[int(cid)]
                    for part in (line.geoms if line.geom_type == "MultiLineString" else [line]):
                        pieces.append(np.asarray(part.coords)/1000)
                        values.append(value)
                collection = LineCollection(pieces, cmap=cmap, norm=Normalize(0, vmax), linewidths=1.1)
                collection.set_array(np.asarray(values))
                ax.add_collection(collection)
                ax.scatter(r.outlet_x/1000, r.outlet_y/1000, marker="v", s=27,
                           color="#263B42", edgecolor="white", linewidth=.5, zorder=4)
                ax.autoscale()
                ax.set_aspect("equal")
                ax.set_axis_off()
                xmin, ymin, xmax, ymax = basin.bounds
                bar = max(1., round((xmax-xmin)/1000/5))
                x, y = xmin/1000+.03*(xmax-xmin)/1000, ymin/1000-.03*(ymax-ymin)/1000
                ax.plot([x, x+bar], [y, y], color="#263B42", linewidth=1.5)
                ax.text(x+bar/2, y, f"{bar:g} km", va="top", ha="center", fontsize=8)
                ax.set_title(f"{classes[col]}\n{r.station} · {label}", fontsize=10, pad=6)
                if col == 2:
                    fig.colorbar(collection, ax=axes[row, :], fraction=.017, pad=.025,
                                 label=f"{label}集水区比例（%）" if cn else f"{label} in catchment (%)", extend="max" if row == 0 else "neither")
    fig.suptitle("真实河网与来源景观" if cn else "Actual river networks and source landscapes", fontsize=17, x=.055, ha="left")
    fig.text(.055, .025, "河段颜色：对应集水区土地覆盖；三角：出口。代表按形态中心性与几何完整性选取。" if cn else
        "Reach colour: local catchment land cover; triangle: outlet. Representatives selected on shape centrality and complete geometry.", fontsize=9)
    fig.subplots_adjust(left=.04, right=.87, top=.89, bottom=.08, wspace=.17, hspace=.25)
    save(fig, "real_network_source_maps")

    gains = pd.read_csv(ROOT/"analysis/huc4_blocked_gains.csv")
    choices = [("environment_area_shape", "environment_area"),
               ("environment_area_placement", "environment_area"),
               ("environment_area_shape_placement", "environment_area_shape"),
               ("environment_area_shape_placement", "environment_area_placement")]
    labels = ["形态加入环境基线", "来源位置加入环境基线", "来源位置加入形态模型", "形态加入来源位置模型"] if cn else [
        "Shape added to environment", "Placement added to environment", "Placement added to shape model", "Shape added to placement model"]
    fig, axes = plt.subplots(1, 2, figsize=(12., 5.1))
    for ax, population, title in zip(axes, ("all_source_months", "source_months_since_2009"),
                                    ("全部源月份", "2009 年以后的源月份") if cn else ("All permitted source months", "Source months since 2009"), strict=True):
        for i, (candidate, reference) in enumerate(choices):
            v = gains[gains.population.eq(population) & gains.space.eq("native")
                & gains.candidate.eq(candidate) & gains.reference.eq(reference)]
            for unit, offset, filled in (("station", -.1, True), ("huc4", .1, False)):
                r = v[v.resampling_unit.eq(unit)].iloc[0]
                ax.plot([r.ci_low_pct, r.ci_high_pct], [i+offset]*2, color=COLORS[i % 3], linewidth=1.5)
                ax.scatter(r.gain_pct, i+offset, s=35, facecolor=COLORS[i % 3] if filled else "white", edgecolor=COLORS[i % 3], zorder=3)
        ax.axvline(0, color="#879295", linestyle="--", linewidth=.8)
        ax.set(yticks=range(4), yticklabels=labels, ylim=(3.5, -.5),
               xlabel="站点 DOC 中位值的 MAE 改善（%）" if cn else "MAE reduction for station median DOC (%)")
        ax.set_title(f"{title}\nn={int(r.n_stations)}, HUC4={int(r.n_huc4)}", loc="left", fontweight="bold")
    fig.suptitle("来源位置能增加多少可迁移信息？" if cn else "What transferable information does source placement add?", fontsize=16, x=.04, ha="left")
    fig.text(.04, .025, "五折 HUC4 留出，等折权重。实心／空心：站点／HUC4 bootstrap 95%。这是站点汇总诊断。" if cn else
        "Five HUC4-blocked folds, equal fold weights. Filled/open: station/HUC4 bootstrap 95%. Station-summary diagnostic.", fontsize=9)
    fig.subplots_adjust(left=.23, right=.97, top=.76, bottom=.2, wspace=1.0)
    save(fig, "source_placement_prediction_increment")
    association = pd.read_csv(ROOT/"analysis/source_placement_associations.csv")
    names = ["湿地·距离比", "森林·距离比", "湿地·近端集中", "森林·近端集中", "湿地·河岸富集", "森林·河岸富集"] if cn else [
        "Wetland distance ratio", "Forest distance ratio", "Wetland near-outlet excess", "Forest near-outlet excess",
        "Wetland riparian enrichment", "Forest riparian enrichment"]
    outcomes = ("doc_median", "doc_cv", "harmonic_amplitude")
    outcome_titles = ("DOC 中位值", "DOC 波动（CV）", "DOC 季节幅度") if cn else ("DOC median", "DOC variation (CV)", "DOC seasonal amplitude")
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.1), sharey=True)
    for row, population in enumerate(("all_source_months", "source_months_since_2009")):
        for col, (outcome, label) in enumerate(zip(outcomes, outcome_titles, strict=True)):
            ax = axes[row, col]
            for i, term in enumerate(PLACEMENT):
                r = association[association.population.eq(population) & association.outcome.eq(outcome) & association.term.eq(term)].iloc[0]
                color = COLORS[0] if term.startswith("wetland") else COLORS[1]
                ax.plot([r.huc4_ci_low, r.huc4_ci_high], [i, i], color=color, linewidth=1.5)
                ax.scatter(r.coefficient_per_station_sd, i, s=30, color=color, zorder=3)
            ax.axvline(0, color="#879295", linewidth=.8, linestyle="--")
            ax.set(yticks=range(6), yticklabels=names, ylim=(5.5, -.5))
            if row == 0:
                ax.set_title(label, loc="left", fontweight="bold")
            if col == 0:
                ax.text(0, 1.14, ("全部源月份" if row == 0 else "2009 年以后") if cn else
                        ("All source months" if row == 0 else "Since 2009"), transform=ax.transAxes, fontsize=10)
            if row == 1:
                ax.set_xlabel("变换后响应／特征标准差" if cn else "Transformed response / feature SD")
            bounds = association[association.outcome.eq(outcome) & association.term.isin(PLACEMENT)]
            low, high = bounds.huc4_ci_low.min(), bounds.huc4_ci_high.max()
            ax.set_xlim(low-.06*(high-low), high+.06*(high-low))
    fig.suptitle("来源位置与 DOC 动态的调整关联" if cn else "Adjusted source-placement associations with DOC dynamics", fontsize=16, x=.04, ha="left")
    fig.text(.04, .025, "控制来源比例、形态、水文、气候及观测时间；HUC4 bootstrap 95%。上／下：297／92 站点。探索性关联。" if cn else
        "Adjusted for source amount, shape, hydro, climate and sampling time; HUC4 bootstrap 95%. Top/bottom: 297/92 stations. Exploratory associations.", fontsize=9)
    fig.subplots_adjust(left=.23, right=.97, top=.85, bottom=.13, wspace=.25, hspace=.35)
    save(fig, "source_placement_dynamic_associations")
    (ROOT/f"figure_sources{suffix}.json").write_text(json.dumps({"representatives": chosen,
        "selection": "class shape centrality, 25-5000 reaches, complete cached geometry; no DOC-fit selection",
        "map_colour": "catchment NLCD 2019 percentages, not DOC", "language": "Chinese" if cn else "English"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
