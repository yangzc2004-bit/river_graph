"""Actual river maps and replicated DOC comparisons; no illustrative geometry."""
from __future__ import annotations

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

from river_graph.topology.river_planform import (
    project_geometry,
    project_lines,
    read_cached_lines,
)

ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")
RAW = Path("data/raw/river_neon_form_validation_v1")
OLD = Path("data/raw/river_planform_v1")
COLORS = {1: "#247F89", 2: "#BE8246", 3: "#6B73A7"}
NAMES = {1: "细长多支流", 2: "主干主导少支流", 3: "宽展多支流"}


def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"{name}.{extension}", bbox_inches="tight", facecolor="white", dpi=210)
    plt.close(fig)


def main():
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    fontManager.addfont(font)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(), "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#243D43", "axes.labelcolor": "#243D43"})
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    p = pd.read_csv(ROOT / "analysis" / "site_panel_primary.csv")
    old = sqlite3.connect(f"file:{OLD / 'flowlines.sqlite'}?mode=ro", uri=True)
    new = sqlite3.connect(f"file:{RAW / 'flowlines.sqlite'}?mode=ro", uri=True)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10.8))
    ordered = p.sort_values(["cluster", "basin_area_km2", "site"])
    for ax, row in zip(axes.flat, ordered.itertuples(), strict=False):
        with np.load(RAW / "members" / f"{row.site}.npz") as z:
            members, mainstem = z["comids"], set(z["mainstem"].tolist())
        lines = read_cached_lines(old, members)
        lines.update(read_cached_lines(new, members))
        if set(lines) != set(members):
            raise ValueError(f"incomplete mapped network: {row.site}")
        projected = project_lines(lines)
        basin_path = OLD / "basins" / f"comid_{row.comid}.json"
        if not basin_path.exists():
            basin_path = RAW / "basins" / f"{row.comid}.json"
        basin = project_geometry(unary_union([shape(f["geometry"]) for f in json.loads(basin_path.read_text())["features"]]))
        polygons = list(basin.geoms) if basin.geom_type == "MultiPolygon" else [basin]
        for polygon in polygons:
            coords = np.asarray(polygon.exterior.coords) / 1000
            ax.fill(coords[:, 0], coords[:, 1], color="#F1F2F3", edgecolor="#D7DCDE", linewidth=.5)
        for main, width, color in ((False, .35, COLORS[row.cluster]), (True, .9, "#213E49")):
            pieces = []
            for cid, line in projected.items():
                if (cid in mainstem) != main:
                    continue
                for piece in (list(line.geoms) if line.geom_type == "MultiLineString" else [line]):
                    pieces.append(np.asarray(piece.coords)[:, :2] / 1000)
            ax.add_collection(LineCollection(pieces, colors=color, linewidths=width, rasterized=True))
        ax.plot(row.outlet_x/1000, row.outlet_y/1000, "o", color="#C87840", markersize=4)
        ax.autoscale_view()
        ax.set_aspect("equal")
        xmin, ymin, xmax, ymax = np.asarray(basin.bounds)/1000
        extent = max(xmax-xmin, ymax-ymin)
        scale = 10**np.floor(np.log10(extent*.2))
        scale *= max(v for v in (1, 2, 5, 10) if v*scale <= extent*.25)
        x, y = xmin+.05*extent, ymin-.10*extent
        ax.plot([x, x+scale], [y, y], color="#253D44", linewidth=2)
        ax.text(x+scale/2, y-.04*extent, f"{scale:g} km", ha="center", va="top", fontsize=9)
        ax.set_title(f"{row.site} · {NAMES[row.cluster]}\n{row.basin_area_km2:,.0f} km² · DOC {row.doc_level_mgl:.2f} mg/L", fontsize=10, loc="left")
        ax.axis("off")
    for ax in list(axes.flat)[len(ordered):]:
        ax.axis("off")
    fig.suptitle("13 个实测 DOC 站点的完整上游河网", fontsize=16, y=1.02)
    fig.text(.5, -.005, "真实 NHDPlus 河道与流域边界；各面板保持真实纵横比、独立比例尺。形态沿用原分类中心，未用 DOC 重新聚类。", ha="center", fontsize=11)
    fig.tight_layout(h_pad=2, w_pad=1)
    save(fig, "actual_neon_networks_cn")
    old.close()
    new.close()

    gains = pd.read_csv(ROOT / "analysis" / "heldout_form_gains.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.3))
    ax = axes[0, 0]
    label_offsets = {"BLUE": (-28, -14), "LEWI": (5, 10), "FLNT": (5, 11), "BLWA": (5, -14)}
    for cluster, g in p.groupby("cluster"):
        ax.scatter(g.basin_aspect, g.doc_level_mgl, color=COLORS[cluster], label=NAMES[cluster], s=50)
        for r in g.itertuples():
            offset = (4, 5) if r.site == "BLWA" else label_offsets.get(r.site, (4, 5))
            ax.annotate(r.site, (r.basin_aspect, r.doc_level_mgl), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set(xlabel="流域长宽比（越大越细长）", ylabel="季节均衡 DOC 水平（mg/L）")
    ax.set_ylim(0, 6)
    ax.set_title("a  形态与 DOC：原始观测分布", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    for cluster, g in p.groupby("cluster"):
        ax.scatter(g.mainstem_share, g.doc_level_mgl, color=COLORS[cluster], s=50)
        for r in g.itertuples():
            ax.annotate(r.site, (r.mainstem_share, r.doc_level_mgl), xytext=label_offsets.get(r.site, (4, 5)), textcoords="offset points", fontsize=8)
    ax.set(xlabel="主干长度 / 总河道长度", ylabel="季节均衡 DOC 水平（mg/L）")
    ax.set_ylim(0, 6)
    ax.set_title("b  支流组织与 DOC", loc="left")
    labels = {"footprint": "外轮廓", "branching": "支流组织", "paths": "路径组织", "all_form": "全部结构"}
    for ax, outcome, title in ((axes[1, 0], "doc_level_mgl", "c  对 DOC 水平的额外预测信息"),
                                (axes[1, 1], "doc_relative_iqr", "d  对 DOC 波动的额外预测信息")):
        for j, population in enumerate(("primary", "no_st357_upstream_overlap")):
            g = gains[gains.outcome.eq(outcome) & gains.population.eq(population)].set_index("candidate").loc[list(labels)]
            y = np.arange(4) + (.12 if j else -.12)
            color = "#247F89" if j else "#68737A"
            ax.errorbar(g.gain_pct, y, xerr=np.vstack([g.gain_pct-g.ci_low_pct, g.ci_high_pct-g.gain_pct]),
                        fmt="o", color=color, capsize=3, markersize=5,
                        label="无 ST357 重叠：10 站 / 8 组" if j else "全部：13 站 / 11 组")
        ax.axvline(0, color="#A0A9AA", linestyle="--", linewidth=1)
        ax.set(yticks=np.arange(4), yticklabels=list(labels.values()), xlabel="相对背景模型的 MAE 降幅（%）")
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.text(.5, -.008, "正值表示结构改善预测；横线为 5,000 次整组 bootstrap 的 95% 区间。站点水平诊断，不是 DOC 月度重建模型提升。", ha="center", fontsize=10)
    fig.tight_layout(h_pad=3, w_pad=2)
    save(fig, "neon_form_doc_evidence_cn")


if __name__ == "__main__":
    main()
