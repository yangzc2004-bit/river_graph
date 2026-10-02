"""Plot station-level few-shot transfer gains on the ST357 river network.

The map is a descriptive view of the frozen station metrics in
``adapter_comparison_v1``.  It does not refit a model or select stations.  The
colour at each held-out station is the relative MAE reduction from K=0 to K=5
for the residual adapter, while the background lines show the ST357 directed
station graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "experiments/phase4_transfer/spatial_adaptation/adapter_comparison_v1/station_metrics.csv"
NODES = ROOT / "data/processed/graph_nodes_graphfix_st357.csv"
EDGES = ROOT / "data/processed/graph_edges_graphfix_st357.csv"
OUT = ROOT / "experiments/phase4_transfer/spatial_adaptation/station_transfer_map_v1"


def _site_key(series: pd.Series) -> pd.Series:
    """Normalize USGS site numbers without losing leading zeros."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(8)


def load_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load station metrics and join ST357 coordinates and directed edges."""
    metrics = pd.read_csv(METRICS, dtype={"site_no": str})
    metrics["site_no"] = _site_key(metrics["site_no"])
    residual = metrics[metrics["model"].eq("residual") & metrics["k"].isin([0, 5])]
    wide = residual.pivot(index=["station", "site_no"], columns="k", values="mae").reset_index()
    wide = wide.rename(columns={0: "mae_k0", 5: "mae_k5"})
    wide["mae_reduction"] = wide["mae_k0"] - wide["mae_k5"]
    wide["relative_reduction_pct"] = 100.0 * wide["mae_reduction"] / wide["mae_k0"]
    n_by_site = residual[residual["k"].eq(0)][["site_no", "n_cells"]].drop_duplicates("site_no")
    wide = wide.merge(n_by_site, on="site_no", how="left", validate="one_to_one")

    nodes = pd.read_csv(NODES, dtype={"site_no": str})
    nodes["site_no"] = _site_key(nodes["site_no"])
    coords = nodes[["site_no", "dec_lat_va", "dec_long_va"]].rename(
        columns={"dec_lat_va": "latitude", "dec_long_va": "longitude"},
    )
    wide = wide.merge(coords, on="site_no", how="left", validate="one_to_one")
    if wide[["latitude", "longitude"]].isna().any().any() or len(wide) != 43:
        raise ValueError("Expected 43 E3 target stations with complete ST357 coordinates")

    edges = pd.read_csv(EDGES, dtype=str)
    edges["source"] = _site_key(edges["source"])
    edges["target"] = _site_key(edges["target"])
    edge_xy = edges.merge(
        coords.rename(columns={"site_no": "source", "latitude": "lat_source", "longitude": "lon_source"}),
        on="source", how="inner",
    ).merge(
        coords.rename(columns={"site_no": "target", "latitude": "lat_target", "longitude": "lon_target"}),
        on="target", how="inner",
    )
    return wide.sort_values("site_no").reset_index(drop=True), nodes, edge_xy


def plot_map(table: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    """Render a clean network map with a diverging K=5 gain scale."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)

    # The full graph gives geographic context; target stations are overplotted below.
    for row in edges.itertuples(index=False):
        ax.plot(
            [row.lon_source, row.lon_target],
            [row.lat_source, row.lat_target],
            color="#aeb7c2", linewidth=0.38, alpha=0.24, zorder=1,
        )
    ax.scatter(
        nodes["dec_long_va"], nodes["dec_lat_va"], s=8, color="#c8ced6",
        linewidth=0, alpha=0.65, zorder=2,
    )

    max_cells = float(table["n_cells"].max())
    marker_size = 34 + 95 * np.sqrt(table["n_cells"] / max_cells)
    norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=80)
    points = ax.scatter(
        table["longitude"], table["latitude"],
        c=table["relative_reduction_pct"], cmap="RdBu_r", norm=norm,
        s=marker_size, edgecolor="white", linewidth=0.65, alpha=0.98, zorder=4,
    )

    # Label the largest gains and the clearest negative responses.
    labels = pd.concat([
        table.nlargest(3, "relative_reduction_pct"),
        table.nsmallest(2, "relative_reduction_pct"),
    ]).drop_duplicates("site_no")
    for row in labels.itertuples(index=False):
        site_label = row.site_no
        dx = 0.18 if row.relative_reduction_pct >= 0 else -0.18
        dy = 0.10 if row.latitude < table["latitude"].median() else -0.14
        ax.annotate(
            site_label, (row.longitude, row.latitude), xytext=(dx, dy),
            textcoords="offset points", fontsize=7.2, color="#26323d",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
            zorder=5,
        )

    improved = int((table["relative_reduction_pct"] > 0).sum())
    median_gain = float(table["relative_reduction_pct"].median())
    ax.text(
        0.015, 0.03,
        f"43 held-out stations  |  {improved}/43 improved  |  median gain {median_gain:.1f}%",
        transform=ax.transAxes, fontsize=8, color="#26323d",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#d5dbe2", "alpha": 0.92},
        zorder=6,
    )
    ax.set_title("A  Station-level benefit of five target observations", loc="left", weight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(-112.4, -77.0)
    ax.set_ylim(29.8, 49.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e1e5e9", linewidth=0.55, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#c8ced6", markersize=4.5, label="ST357 stations"),
            Line2D([0], [0], color="#aeb7c2", linewidth=1.0, label="directed river edge"),
        ], loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#d5dbe2",
    )
    colorbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("K=5 MAE reduction vs K=0 (%)")
    colorbar.ax.axhline(0, color="#38424d", linewidth=0.7)
    colorbar.outline.set_linewidth(0.6)
    fig.savefig(path.with_suffix(".png"), dpi=350, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table, nodes, edges = load_table()
    table.to_csv(OUT / "station_map_data.csv", index=False)
    plot_map(table, nodes, edges, OUT / "station_transfer_map")
    spec = {
        "description": "E3 target-station spatial map of residual-adapter K=5 relative MAE reduction",
        "metric_input": str(METRICS.relative_to(ROOT)),
        "node_input": str(NODES.relative_to(ROOT)),
        "edge_input": str(EDGES.relative_to(ROOT)),
        "n_target_stations": len(table),
        "improved_stations": int((table["relative_reduction_pct"] > 0).sum()),
        "median_relative_reduction_pct": float(table["relative_reduction_pct"].median()),
        "color_definition": "100 * (station MAE at K=0 - station MAE at K=5) / station MAE at K=0",
        "network_context": "all 357 ST357 stations and 324 directed edges; target stations overlaid",
    }
    (OUT / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(table[["site_no", "relative_reduction_pct", "n_cells"]].to_string(index=False))


if __name__ == "__main__":
    main()
