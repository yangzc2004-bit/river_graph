"""Plot station-level heterogeneity of the spatial support adapter."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path("experiments/phase4_transfer/spatial_adaptation/fewshot_residual_paired_v1")
OUT = Path("experiments/phase4_transfer/spatial_adaptation/adapter_comparison_v1")
ALPHA = {0: 0.0, 1: 0.25, 3: 0.5, 5: 0.75}


def station_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, alpha in ALPHA.items():
        current = frame[(frame.k == k) & (frame.alpha == alpha)].copy()
        seed_mean = current.groupby(["cell", "station"], as_index=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        )
        grouped = seed_mean.groupby("station")
        rows.append(pd.DataFrame({
            "station": grouped.station.first().index,
            "k": k,
            "mae": grouped.apply(
                lambda g: np.mean(np.abs(g.y_true - g.y_pred)),
                include_groups=False,
            ).to_numpy(),
            "bias": grouped.apply(
                lambda g: np.mean(g.y_pred - g.y_true),
                include_groups=False,
            ).to_numpy(),
        }))
    table = pd.concat(rows, ignore_index=True)
    base = table[table.k == 0][["station", "mae", "bias"]].rename(
        columns={"mae": "mae_k0", "bias": "bias_k0"},
    )
    final = table[table.k == 5][["station", "mae"]].rename(
        columns={"mae": "mae_k5"},
    )
    out = base.merge(final, on="station", validate="one_to_one")
    out["mae_reduction"] = out.mae_k0 - out.mae_k5
    out["relative_reduction_pct"] = 100 * out.mae_reduction / out.mae_k0
    return out.sort_values("station").reset_index(drop=True)


def plot(table: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": "#d9dde2",
        "grid.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.scatter(
        np.abs(table.bias_k0), table.mae_reduction,
        c=table.mae_k0, cmap="viridis", s=36, alpha=0.9,
        edgecolor="white", linewidth=0.35,
    )
    rho = np.corrcoef(np.abs(table.bias_k0), table.mae_reduction)[0, 1]
    ax.axhline(0, color="#555b63", lw=0.8)
    ax.set_xlabel("Absolute K=0 station bias (mg/L)")
    ax.set_ylabel("K=5 MAE reduction (mg/L)")
    ax.set_title(f"A  Support helps baseline mismatch (r={rho:.2f})", loc="left", weight="bold")

    ax = axes[1]
    order = table.sort_values("mae_reduction")
    colors = np.where(order.mae_reduction >= 0, "#c45a11", "#6b7280")
    ax.bar(np.arange(len(order)), order.mae_reduction, color=colors, width=0.85)
    ax.axhline(0, color="#555b63", lw=0.8)
    ax.set_xlabel("Target station (sorted)")
    ax.set_ylabel("K=5 MAE reduction (mg/L)")
    ax.set_title("B  Heterogeneous station response", loc="left", weight="bold")
    ax.set_xticks([])
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    frame = pd.read_parquet(RUN / "test_predictions.parquet")
    table = station_summary(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "station_mechanism_summary.csv", index=False)
    plot(table, OUT / "station_mechanism")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

