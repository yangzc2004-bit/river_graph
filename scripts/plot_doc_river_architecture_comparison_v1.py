"""Standalone scientific figure from audited architecture comparison tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")
ARMS = ["local", "directed_gnn", "hierarchical_gnn", "graph_transformer"]
LABELS = ["Local only", "Directed GNN", "Hierarchical GNN", "Graph Transformer"]
COLORS = ["#737d8c", "#2977b8", "#278565", "#bf7633"]


def main():
    summary = pd.read_csv(ROOT / "analysis/summary.csv").set_index("model_name").loc[ARMS]
    basin = pd.read_csv(ROOT / "analysis/per_basin_metrics.csv")
    wide = basin.pivot(index="target_huc4", columns="model_name", values="mae")
    audit = json.loads((ROOT / "analysis/audit.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1, 1.15]})
    axis = axes[0]
    bars = axis.bar(np.arange(4), summary.mae, color=COLORS, width=.65)
    axis.set_xticks(np.arange(4), ["Local\nonly", "Directed\nGNN", "Hierarchical\nGNN", "Graph\nTransformer"])
    axis.set_ylabel("DOC MAE (mg/L); lower is better")
    axis.set_ylim(0, summary.mae.max() * 1.2)
    axis.set_title("A  Pooled error across five held-out regions", loc="left", fontsize=11)
    for bar, value in zip(bars, summary.mae):
        axis.text(bar.get_x() + bar.get_width()/2, value + summary.mae.max() * .025,
                  f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
    axis = axes[1]
    positions = np.arange(len(wide))
    for i, arm in enumerate(ARMS[1:]):
        gain = 100 * (wide.local - wide[arm]) / wide.local
        axis.scatter(positions + (i-1)*.16, gain, color=COLORS[i+1], s=52,
                     marker=["o", "s", "^"][i], label=LABELS[i+1])
    axis.axhline(0, color="#555555", linewidth=.9)
    axis.set_xticks(positions, [str(h).zfill(4) for h in wide.index])
    axis.set_xlabel("Held-out HUC4 region")
    axis.set_ylabel("MAE reduction vs local (%); positive is better")
    axis.set_title("B  Region-specific changes", loc="left", fontsize=11)
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", alpha=.15)
    fig.suptitle("DOC reconstruction: matched spatial architecture comparison", x=.06,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.06, .015, f"{audit['query_stations']} held-out stations; {audit['unique_query_cells']:,} unique DOC observations; "
             "3 seeds per region. Mean seed errors, not ensemble errors. Internal development comparison.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(.01, .055, 1, .91))
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"architecture_comparison.{suffix}", dpi=180)
    plt.close(fig)
    record = {"script_sha256": sha256_file(__file__), "sources": {name: sha256_file(ROOT / "analysis" / name)
        for name in ("summary.csv", "per_basin_metrics.csv", "audit.json")},
        "figures": {path.name: sha256_file(path) for path in destination.glob("architecture_comparison.*")}}
    (destination / "sources.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
