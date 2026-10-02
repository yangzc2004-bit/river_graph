"""Embed verified numbers and vector plots into the spatial draft.

Run after analyze_spatial_manuscript_evidence.py. This edits only marked
generated blocks, preserving the manuscript prose, and requires no TeX engine
to generate the numeric figures. The architecture illustration is a separate
image asset; compile locally with access to the adjacent figures directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/phase4_transfer/spatial_adaptation/manuscript_evidence_v2"
TEX = ROOT / "docs/paper/latex/spatial_transfer_draft_v1.tex"


def coordinates(x, y):
    return " ".join(f"({a:.6f},{b:.6f})" for a, b in zip(x, y, strict=True))


def project(lon, lat):
    lon, lat = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    lon0, lat0 = np.deg2rad([-96.0, 39.0])
    k = np.sqrt(2 / (1 + np.sin(lat0) * np.sin(lat)
                     + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)))
    return (6371 * k * np.cos(lat) * np.sin(lon - lon0),
            6371 * k * (np.cos(lat0) * np.sin(lat)
                        - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)))


def block(source, name, body):
    pattern = rf"(% BEGIN GENERATED {name}\n).*?(% END GENERATED {name})"
    source, count = re.subn(pattern, lambda m: m[1] + body + "\n" + m[2],
                            source, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected exactly one generated {name} block")
    return source


def main():
    data = pd.read_csv(EVIDENCE / "main_results.csv").sort_values("k")
    station = pd.read_csv(EVIDENCE / "station_gain.csv", dtype={"site_no": str})
    stats = json.loads((EVIDENCE / "station_summary.json").read_text())
    shuffle = pd.read_csv(EVIDENCE / "support_shuffle_summary.csv")
    k5 = data[data.k == 5].iloc[0]
    shuffled = shuffle[(shuffle.k == 5) & (shuffle.condition == "shuffled_support")].iloc[0]
    source = TEX.read_text()
    numbers = {
        "MainDelta": f"${k5.delta_mae:.3f}$",
        "MainDeltaCI": f"$[{k5.delta_lo:.3f}, {k5.delta_hi:.3f}]$",
        "MacroDelta": f"${k5.station_macro_mae - k5.reference_station_macro_mae:.3f}$",
        "ImprovedStations": str(stats["improved_stations"]),
        "MedianStationGain": f"{stats['median_relative_reduction_pct']:.2f}",
        "BiasGainCorrelation": f"{stats['bias_gain_pearson']:.2f}",
        "ShuffleMAE": f"{shuffled['mean']:.3f}",
    }
    source = block(source, "NUMBERS", "\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in numbers.items()))
    table = [r"\begin{tabular}{rrrrr}\toprule",
             r"$K$ & MAE (mg/L) & Seed SD & RMSE (mg/L) & Reduction (\%)\\\midrule"]
    for row in data.itertuples():
        table.append(f"{row.k} & {row.mae:.3f} & {row.seed_mae_sd:.3f} & "
                     f"{row.rmse:.3f} & {row.reduction_pct:.2f}" + r"\\")
    table.append(r"\bottomrule\end{tabular}")
    source = block(source, "MAIN TABLE", "\n".join(table))

    curve = [r"\begin{tikzpicture}\begin{axis}[width=.85\linewidth,height=5.8cm,",
             r"xlabel={Support observations per station ($K$)},ylabel={DOC MAE (mg/L)},",
             r"xmin=-.2,xmax=5.2,ymin=1.95,ymax=2.55,xtick={0,1,3,5},",
             r"axis lines=left,ymajorgrids,grid style={gray!20},tick label style={font=\small}]",
             r"\addplot+[riverblue,thick,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {"]
    for row in data.itertuples():
        curve.append(f"({row.k},{row.mae:.8f}) +- (0,{row.seed_mae_sd:.8f})")
    curve += [r"};", r"\end{axis}\end{tikzpicture}"]
    source = block(source, "K FIGURE", "\n".join(curve))

    nodes = pd.read_csv(ROOT / "data/processed/graph_nodes_graphfix_st357.csv",
                        dtype={"site_no": str})
    nodes["site_no"] = nodes.site_no.str.zfill(8)
    edges = pd.read_csv(ROOT / "data/processed/graph_edges_graphfix_st357.csv", dtype=str)
    node_x, node_y = project(nodes.dec_long_va, nodes.dec_lat_va)
    xy = dict(zip(nodes.site_no, zip(node_x, node_y, strict=True), strict=True))
    mp = [r"\begin{tikzpicture}\begin{axis}[width=.98\linewidth,height=8.3cm,",
          r"xlabel={East--west distance from center (km)},",
          r"ylabel={North--south distance from center (km)},",
          r"xmin=-1600,xmax=1800,ymin=-1150,ymax=1350,axis equal image,",
          r"axis lines=left,tick label style={font=\scriptsize},",
          r"legend style={font=\scriptsize,draw=none,at={(.99,.02)},anchor=south east}]" ]
    for edge in edges.itertuples():
        a, b = xy[edge.source.zfill(8)], xy[edge.target.zfill(8)]
        mp.append(r"\addplot[gray!25,thin,no marks,forget plot] coordinates {"
                  + coordinates([a[0], b[0]], [a[1], b[1]]) + "};")
    mp.append(r"\addplot[only marks,mark=*,gray!35,mark size=.65pt,forget plot] coordinates {"
              + coordinates(node_x, node_y) + "};")
    for row in station.itertuples():
        x, y = project(row.longitude, row.latitude)
        color, mark = ("riverblue", "*") if row.relative_reduction_pct > 0 else (
            "supportorange", "triangle*")
        size = 1.5 + 4 * np.sqrt(abs(row.relative_reduction_pct) / 100)
        mp.append(r"\addplot[only marks,forget plot," + color + f",mark={mark},"
                  f"mark size={size:.3f}pt] coordinates {{({x:.6f},{y:.6f})}};")
    mp += [r"\addlegendimage{only marks,mark=*,riverblue}\addlegendentry{Improvement}",
           r"\addlegendimage{only marks,mark=triangle*,supportorange}\addlegendentry{Deterioration}",
           (r"\node[anchor=north west,font=\scriptsize,align=left] at (rel axis cs:0,1)"
            r" {Lambert equal-area projection\\center: $96^{\circ}$W, $39^{\circ}$N};"),
           r"\end{axis}\end{tikzpicture}"]
    source = block(source, "MAP FIGURE", "\n".join(mp))

    bias = [r"\begin{tikzpicture}\begin{axis}[width=.85\linewidth,height=6.1cm,",
            r"xlabel={Absolute mean baseline bias (mg/L)},ylabel={MAE reduction (mg/L)},",
            r"xmin=0,ymin=-1,axis lines=left,ymajorgrids,grid style={gray!20},",
            r"tick label style={font=\small}]",
            r"\addplot[gray,dashed,no marks] coordinates {(0,0)(30,0)};",
            r"\addplot[only marks,mark=*,riverblue,mark size=2pt] coordinates {",
            coordinates(np.abs(station.bias_k0), station.mae_reduction),
            r"};\end{axis}\end{tikzpicture}"]
    source = block(source, "BIAS FIGURE", "\n".join(bias))
    if "IfFileExists" in source:
        raise ValueError("Draft must not silently substitute missing figures")
    for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
        if not (TEX.parent / asset).is_file():
            raise FileNotFoundError(f"Missing manuscript figure: {asset}")
    TEX.write_text(source)
    print(f"Updated numbers and three data figures in {TEX}")


if __name__ == "__main__":
    main()
