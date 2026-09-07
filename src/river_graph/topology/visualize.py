"""Visual sanity check for the station graph.

Deliberately dependency-light (matplotlib only, no basemap): the point is to
verify edge *directions and connectivity* against the known geography of the
basin, not to make a pretty map.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: str | Path,
    title: str = "Mississippi Basin DOC station graph",
) -> None:
    """Scatter stations and draw directed edges as arrows."""
    xy = nodes.set_index("site_no")[["dec_long_va", "dec_lat_va"]].astype(float)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(xy["dec_long_va"], xy["dec_lat_va"], s=8, c="tab:blue", zorder=2)
    n_drawn = 0
    for _, e in edges.iterrows():
        if e["source"] not in xy.index or e["target"] not in xy.index:
            continue
        x0, y0 = xy.loc[e["source"]]
        x1, y1 = xy.loc[e["target"]]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 0.5, "alpha": 0.6},
            zorder=1,
        )
        n_drawn += 1
    ax.set_title(f"{title}\n{len(nodes)} nodes, {n_drawn} edges")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
