"""Plot actual basin boundaries and every real flowline of class representatives."""
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
from matplotlib.font_manager import FontProperties, fontManager
from shapely.geometry import shape
from shapely.ops import unary_union

from river_graph.topology.river_planform import project_geometry, read_cached_lines

ROOT = Path("experiments/phase4_transfer/doc_river_planform_typology_v1")
CACHE = Path("data/raw/river_planform_v1")
COLORS = ("#257F88", "#C5814A", "#66749F", "#968C45", "#AB6D84", "#426276", "#719E72", "#8D745B")


def segments(geometries):
    parts = []
    for geom in geometries:
        pieces = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        parts.extend(np.asarray(piece.coords)[:, :2]/1000 for piece in pieces)
    return parts


def boundary(ax, geometry):
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for poly in polygons:
        xy = np.asarray(poly.exterior.coords)/1000
        ax.fill(xy[:, 0], xy[:, 1], color="#EFF3F1", zorder=0)
        ax.plot(xy[:, 0], xy[:, 1], color="#BAC6C4", linewidth=.6, zorder=1)
        for hole in poly.interiors:
            h = np.asarray(hole.coords)/1000
            ax.fill(h[:, 0], h[:, 1], color="white", zorder=1)


def draw_network(ax, connection, row, color):
    member_dir = "members_full" if hasattr(row, "network_definition") else "members"
    member = np.load(CACHE / member_dir / f"comid_{int(row.comid)}.npz")
    lines = {cid: project_geometry(line) for cid, line in read_cached_lines(connection, member["comids"]).items()}
    basin_json = json.loads((CACHE / "basins" / f"comid_{int(row.comid)}.json").read_text())
    basin = project_geometry(unary_union([shape(f["geometry"]) for f in basin_json["features"]]))
    boundary(ax, basin)
    width = .32 if len(lines) < 5000 else .15
    ax.add_collection(LineCollection(segments(lines.values()), colors=color, linewidths=width,
                                     alpha=.8, zorder=2, rasterized=len(lines) > 20000))
    ax.add_collection(LineCollection(segments(lines[int(cid)] for cid in member["mainstem"]),
                                     colors="#203C43", linewidths=1.05, zorder=3))
    ax.scatter(row.outlet_x/1000, row.outlet_y/1000, color="#CC7155", s=26, edgecolors="white", linewidths=.6, zorder=4)
    x0, y0, x1, y1 = np.asarray(basin.bounds)/1000
    span = max(x1-x0, y1-y0)*1.14
    xm, ym = (x0+x1)/2, (y0+y1)/2
    ax.set(xlim=(xm-span/2, xm+span/2), ylim=(ym-span/2, ym+span/2), aspect="equal")
    ax.set_axis_off()
    # Physical units retained; independent extent explicitly disclosed below.
    target = span*.22
    power = 10**np.floor(np.log10(target))
    scale = max(v*power for v in (1, 2, 5) if v*power <= target) if target >= power else power/2
    startx, starty = xm-.42*span, ym-.49*span
    ax.plot([startx, startx+scale], [starty, starty], color="#34464C", linewidth=1.5)
    ax.text(startx+scale/2, starty+.013*span, f"{scale:g} km", ha="center", fontsize=8, color="#34464C")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--chinese", action="store_true")
    args = parser.parse_args()
    analysis = args.analysis_dir or ROOT / ("provisional_analysis" if args.provisional else "analysis")
    summary = json.loads((analysis / "classification_summary.json").read_text())
    labels_path = analysis / "class_labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    if args.chinese:
        font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        if not font_path.exists():
            raise SystemExit("Chinese export needs a configured Chinese font")
        fontManager.addfont(font_path)
        plt.rcParams["font.family"] = FontProperties(fname=font_path).get_name()
    suffix = "_cn" if args.chinese else ""
    classes = pd.read_csv(analysis / "classes.csv", dtype={"station": str})
    reps = pd.read_csv(analysis / "representatives.csv", dtype={"station": str})
    selected = reps.loc[reps["rank"].eq(1)].merge(classes, on=["station", "comid", "cluster"], validate="one_to_one")
    provisional = summary["status"].startswith("provisional")
    out = analysis / "figures" if args.analysis_dir else ROOT / ("provisional_figures" if args.provisional else "figures")
    out.mkdir(exist_ok=True)
    connection = sqlite3.connect(f"file:{CACHE / 'flowlines.sqlite'}?mode=ro", uri=True)
    n = len(selected)
    columns = min(3, n)
    rows = int(np.ceil(n/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.4*columns, 4.8*rows+1.0), squeeze=False)
    for ax, row in zip(axes.ravel(), selected.itertuples()):
        color = COLORS[int(row.cluster)-1]
        draw_network(ax, connection, row, color)
        count = summary["counts"][str(int(row.cluster))]
        label = labels.get(str(int(row.cluster)), {})
        if args.chinese:
            title = f"{int(row.cluster)}  {label.get('zh', '河网形态组')}\n{count} 个河网  ·  代表站点 {row.station}"
            detail = f"流域长宽比 {row.basin_aspect:.2f}  ·  主干占比 {100*row.mainstem_share:.1f}%\n{row.basin_area_km2:,.0f} km²  ·  {int(row.n_reaches):,} 条实际河段"
        else:
            title = f"Type {int(row.cluster)}  ·  n = {count}\nStation {row.station}"
            if label:
                title = label["en"]+"\n"+title
            detail = f"Basin aspect {row.basin_aspect:.2f}  ·  Mainstem share {100*row.mainstem_share:.1f}%\n{row.basin_area_km2:,.0f} km²  ·  {int(row.n_reaches):,} mapped reaches"
        ax.set_title(title, fontsize=11, loc="left", color="#273D43", pad=8)
        ax.text(.02, -.06, detail,
                transform=ax.transAxes, fontsize=9, color="#46565D", va="top")
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    prefix = "Provisional " if provisional else ""
    if args.chinese:
        headline = ("初步" if provisional else "")+"河网整体形态分类：来自真实河道与流域边界"
        note = f"数据来源：NHDPlus / USGS NLDI。完整几何覆盖 {summary['measured_stations']}/{summary.get('requested_stations', 357)} 个站点；{summary['classified_networks']} 个不同上游河网进入分类。\n"
        note += "每类展示最接近该组中心的真实河网。保持真实形状，按各图比例尺读取距离。\n灰色：流域边界；彩色：所有连接的支流；深色：主干；橙色点：出口。分类只使用河网形态指标。"
    else:
        headline = f"{prefix}river planform classes from real upstream networks"
        note = f"NHDPlus / USGS NLDI. Complete geometry for {summary['measured_stations']}/{summary.get('requested_stations', 357)} stations; {summary['classified_networks']} unique networks classified.\n"
        note += "Actual centroid-nearest networks; native projected orientation; panels have independent extents and physical scale bars.\nDark line: mainstem; colored lines: all connected tributaries; orange point: receiving reach outlet. No DOC used in clustering."
    fig.suptitle(headline, x=.025, ha="left", fontsize=17, color="#263A40")
    fig.text(.025, .035, note, fontsize=9, color="#46565D")
    fig.subplots_adjust(top=.86, bottom=.18, wspace=.13, hspace=.39)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"real_river_planform_representatives{suffix}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    # Three real examples for each class prevent a single case defining a type.
    alternatives = reps.merge(classes, on=["station", "comid", "cluster"], validate="one_to_one")
    for c, group in alternatives.groupby("cluster"):
        fig, axes = plt.subplots(1, len(group), figsize=(4.3*len(group), 5.2), squeeze=False)
        for ax, row in zip(axes.ravel(), group.itertuples()):
            draw_network(ax, connection, row, COLORS[int(c)-1])
            ax.set_title(f"Type {int(c)} · example {int(row.rank)}\nStation {row.station}", fontsize=11, loc="left")
        fig.text(.03, .04, "Actual mapped channels; independent physical extents; darkest line is mainstem; pale outline is upstream basin.", fontsize=9)
        fig.subplots_adjust(top=.86, bottom=.12, wspace=.15)
        fig.savefig(out / f"type_{int(c)}_real_examples{suffix}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    connection.close()
    if not args.chinese:
        profiles(analysis, out, summary)
    print(out / f"real_river_planform_representatives{suffix}.png")


def profiles(analysis, out, summary):
    centroids = pd.read_csv(analysis / "centroids.csv").set_index("cluster")
    candidates = pd.read_csv(analysis / "candidates.csv")
    stability = pd.read_csv(analysis / "stability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    ax = axes[0]
    vmax = max(1., abs(centroids.to_numpy()).max())
    im = ax.imshow(centroids.to_numpy(), cmap="BrBG", vmin=-vmax, vmax=vmax, aspect="auto")
    labels = [v.replace("log_", "").replace("_", " ") for v in centroids.columns]
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels, yticks=np.arange(len(centroids)),
           yticklabels=[f"Type {v}" for v in centroids.index], title="a   Geometric class profiles")
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    fig.colorbar(im, ax=ax, shrink=.6, label="Block-weighted standardized feature")
    ax = axes[1]
    ax.plot(candidates.k, candidates.silhouette, "o-", color=COLORS[0])
    selected = candidates.loc[candidates.k.eq(summary["selected_k"])].iloc[0]
    ax.scatter([selected.k], [selected.silhouette], s=100, facecolors="none", edgecolors=COLORS[1], linewidths=2)
    for row in candidates.itertuples():
        ax.annotate(f"n≥{row.minimum_class}", (row.k, row.silhouette), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)
    spread = candidates.silhouette.max()-candidates.silhouette.min()
    ax.set(xlabel="Number of classes", ylabel="Silhouette", xticks=candidates.k,
           ylim=(candidates.silhouette.min()-.2*spread-.01, candidates.silhouette.max()+.35*spread+.01),
           title="b   Data-derived class count")
    ax = axes[2]
    box = ax.boxplot([stability.loc[stability.k.eq(k), "ari"] for k in candidates.k],
                     tick_labels=candidates.k, patch_artist=True, widths=.55)
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(.6)
    ax.set(xlabel="Number of classes", ylabel="Adjusted Rand index", ylim=(-.02, 1.05),
           title="c   80% network-subset stability")
    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Whole-network shape classification", x=.03, ha="left", fontsize=16)
    fig.text(.03, .025, f"{summary['classified_networks']} unique full networks; shape, branching and organization blocks equally weighted.\n"
             "Choose maximum silhouette with minimum class n≥10. Nested upstream basins are dependent geometric units.", fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.33, left=.06, right=.97, wspace=.45)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"planform_classification_evidence.{ext}", dpi=230, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
