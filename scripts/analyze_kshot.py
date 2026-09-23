#!/usr/bin/env python3
"""Summarize K-shot runs: learning curves, support-distance effects, maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cells(pred_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(pred_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        # run_kshot names files f"{region}_s{seed}.parquet"
        stem = p.stem
        if "_s" in stem and "seed" not in df.columns:
            region, seed = stem.rsplit("_s", 1)
            df = df.copy()
            df["region"] = region
            df["seed"] = int(seed)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no prediction parquet under {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ae"] = (df.y_true - df.y_pred).abs()
    df["se"] = (df.y_true - df.y_pred) ** 2
    df["log_ae"] = (np.log1p(df.y_true) - np.log1p(df.y_pred)).abs()
    return df


def curve_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["region", "seed", "k"]
    for key, sub in cells.groupby(group_cols, sort=True):
        region, seed, k = key
        yt, yp = sub.y_true.to_numpy(), sub.y_pred.to_numpy()
        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        lt, lp = np.log1p(yt), np.log1p(yp)
        lss_tot = ((lt - lt.mean()) ** 2).sum()
        lss_res = ((lt - lp) ** 2).sum()
        log_r2 = 1 - lss_res / lss_tot if lss_tot > 0 else np.nan
        rows.append(
            {
                "region": region,
                "seed": int(seed),
                "k": int(k),
                "n_query": len(sub),
                "mae": sub.ae.mean(),
                "rmse": np.sqrt(sub.se.mean()),
                "r2": r2,
                "log_mae": sub.log_ae.mean(),
                "log_r2": log_r2,
            }
        )
    return pd.DataFrame(rows)


def plot_curves(curve: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for region, sub in curve.groupby("region"):
        mean = sub.groupby("k", as_index=False).agg(mae=("mae", "mean"), r2=("r2", "mean"))
        axes[0].plot(mean.k, mean.mae, marker="o", label=region)
        axes[1].plot(mean.k, mean.r2, marker="o", label=region)
    axes[0].set_xlabel("K (same-month support)")
    axes[0].set_ylabel("query MAE (mg/L)")
    axes[0].set_title("K-shot learning curve")
    axes[1].set_xlabel("K (same-month support)")
    axes[1].set_ylabel("query R²")
    axes[1].set_title("K-shot R²")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(sorted(curve.k.unique()))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_maps(cells: pd.DataFrame, out_dir: Path) -> None:
    """Scatter lon/lat is not in cells; map mean |error| by station index per K."""
    # Station-level MAE vs K (compact, no basemap required).
    stations = (
        cells.groupby(["region", "k", "row"], as_index=False)
        .agg(mae=("ae", "mean"), n=("ae", "size"), y_true=("y_true", "mean"))
    )
    regions = stations.region.unique()
    fig, axes = plt.subplots(1, len(regions), figsize=(4.5 * len(regions), 4),
                             constrained_layout=True, squeeze=False)
    for ax, region in zip(axes[0], regions):
        sub = stations[stations.region == region]
        pivot = sub.pivot(index="row", columns="k", values="mae")
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma_r", interpolation="nearest")
        ax.set_title(f"{region}: station MAE by K")
        ax.set_xlabel("K")
        ax.set_ylabel("station index")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "station_mae_by_k.png", dpi=150)
    plt.close(fig)

    # Prediction vs truth for K=0 and max K
    ks = sorted(cells.k.unique())
    pick = [ks[0], ks[-1]] if len(ks) > 1 else ks
    fig, axes = plt.subplots(1, len(pick), figsize=(4 * len(pick), 4),
                             constrained_layout=True, squeeze=False)
    for ax, k in zip(axes[0], pick):
        sub = cells[cells.k == k]
        ax.scatter(sub.y_true, sub.y_pred, s=8, alpha=0.35)
        lim = max(sub.y_true.max(), sub.y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("true DOC")
        ax.set_ylabel("predicted DOC")
        ax.set_title(f"K={int(k)} query predictions")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.savefig(out_dir / "pred_vs_true.png", dpi=150)
    plt.close(fig)


def plot_support_distance(cells: pd.DataFrame, out: Path) -> None:
    has = cells.dropna(subset=["min_support_hops"]) if "min_support_hops" in cells else None
    if has is None or has.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for (region, k), sub in has.groupby(["region", "k"]):
        if sub.min_support_hops.nunique() < 2:
            continue
        binned = sub.groupby(sub.min_support_hops.round()).ae.mean()
        ax.plot(binned.index, binned.values, marker="o", label=f"{region} K={int(k)}")
    ax.set_xlabel("min support→query river hops")
    ax.set_ylabel("query MAE (mg/L)")
    ax.set_title("Error vs support river distance")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kshot-dir", default="experiments/kshot_st357")
    args = ap.parse_args()
    root = Path(args.kshot_dir)
    cells = error_columns(load_cells(root / "predictions"))
    curve = curve_table(cells)
    curve.to_csv(root / "kshot_curve.csv", index=False)
    cells.to_csv(root / "kshot_cells.csv", index=False)

    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Nested-query subset per region: same tasks across that region's K ladder.
    task_ids = ["region", "month", "month_index", "seed"]
    nested_parts = []
    for region, reg in cells.groupby("region"):
        k_nested = tuple(sorted(int(k) for k in reg.k.unique() if k <= 5))
        if len(k_nested) < 2:
            continue
        sub_k = reg[reg.k.isin(k_nested)]
        counts = sub_k.groupby(task_ids)["k"].nunique()
        full_tasks = counts[counts == len(k_nested)].index
        nested = sub_k[sub_k.set_index(task_ids).index.isin(full_tasks)].copy()
        if nested.empty:
            continue
        nested_parts.append(nested)
        nested_curve = curve_table(nested)
        print(f"\n[{region}] nested-query curve across K={k_nested}")
        print(nested_curve.to_string(index=False))
        print("mean by K")
        print(
            nested_curve.groupby("k")[["mae", "r2", "log_r2", "n_query"]].mean().to_string()
        )
    if nested_parts:
        nested_all = pd.concat(nested_parts, ignore_index=True)
        nested_curve = curve_table(nested_all)
        nested_curve.to_csv(root / "kshot_curve_nested.csv", index=False)
        plot_curves(nested_curve, fig_dir / "kshot_curves_nested.png")

    print("\npooled curve mean by K (query sets differ across K)")
    print(curve.groupby("k")[["mae", "r2", "log_r2", "n_query"]].mean().to_string())

    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_curves(curve, fig_dir / "kshot_curves.png")
    plot_maps(cells, fig_dir)
    plot_support_distance(cells, fig_dir / "mae_vs_support_hops.png")
    print(f"wrote {root}/kshot_curve.csv and figures under {fig_dir}")


if __name__ == "__main__":
    main()
