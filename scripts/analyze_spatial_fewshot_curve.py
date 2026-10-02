"""Summarize and plot the fixed-query spatial few-shot curve."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import load_split

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.spatial_fewshot import (
    support_view,
)
from river_graph.experiments.transfer import DATASETS

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
SELECTED_ALPHA = {0: 0.0, 1: 0.25, 3: 0.5, 5: 0.5}
COLORS = {"raw": "#1f4e79", "corrected": "#d9772a", "shuffle": "#8a8f98"}


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["seed", "k", "alpha", "station"], as_index=False)
        .apply(lambda group: pd.Series({
            "mae": np.mean(np.abs(group.y_true - group.y_pred)),
            "n": len(group),
        }), include_groups=False)
        .reset_index(drop=True)
    )


def bootstrap_delta(station: pd.DataFrame, k: int, *, seed: int = 1729,
                    draws: int = 5000) -> tuple[float, float, float]:
    zero = station[station.k == 0][["seed", "station", "mae"]].rename(columns={"mae": "mae0"})
    current = station[station.k == k][["seed", "station", "mae"]].rename(columns={"mae": "maek"})
    paired = current.merge(zero, on=["seed", "station"], validate="one_to_one")
    paired["delta"] = paired.maek - paired.mae0
    station_delta = paired.groupby("station", as_index=False).delta.mean()
    point = float(station_delta.delta.mean())
    rng = np.random.default_rng(seed + k)
    values = station_delta.delta.to_numpy()
    samples = np.empty(draws)
    for i in range(draws):
        samples[i] = rng.choice(values, size=len(values), replace=True).mean()
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def shuffled_scores(predictions: pd.DataFrame, data: dict, split: dict) -> pd.DataFrame:
    y = np.asarray(data["y"], dtype=float).reshape(-1)
    rows = []
    for k, alpha in SELECTED_ALPHA.items():
        if k == 0:
            continue
        view, support, query = support_view(split, target_role="test", k=k, n_months=T)
        del view
        support_station = support // T
        query_station = query // T
        station_ids = np.unique(query_station)
        support_means = {
            int(station): float(np.log1p(y[support[support_station == station]]).mean())
            for station in station_ids
        }
        for seed in sorted(predictions.seed.unique()):
            base = predictions[
                (predictions.seed == seed) & (predictions.k == k) & (predictions.alpha == 0.0)
            ].sort_values("cell")
            if len(base) != len(query) or not np.array_equal(base.cell.to_numpy(), query):
                raise ValueError("prediction rows are not aligned with fixed query")
            true = predictions[
                (predictions.seed == seed) & (predictions.k == k)
                & (predictions.alpha == alpha)
            ].sort_values("cell")
            true_score = metrics(true.y_true.to_numpy(), true.y_pred.to_numpy())["mae"]
            rng = np.random.default_rng(9100 + 17 * seed + k)
            shuffled_values = rng.permutation(np.array([support_means[int(s)] for s in station_ids]))
            shuffled = dict(zip(station_ids, shuffled_values, strict=True))
            z = np.log1p(np.maximum(base.y_pred.to_numpy(), 0.0))
            for station in station_ids:
                take = base.station.to_numpy() == station
                z[take] += alpha * (shuffled[int(station)] - z[take].mean())
            shuffle_score = metrics(base.y_true.to_numpy(), np.expm1(z))["mae"]
            rows.extend([
                {"seed": seed, "k": k, "condition": "true_support", "mae": true_score},
                {"seed": seed, "k": k, "condition": "shuffled_support", "mae": shuffle_score},
            ])
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, improvement: pd.DataFrame, shuffle: pd.DataFrame,
         out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#d9dde2",
        "grid.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    ax = axes[0]
    for variant, group in summary.groupby("variant"):
        ax.errorbar(group["k"], group["mean"], yerr=group["std"], marker="o", lw=2,
                    capsize=3, color=COLORS[variant], label=variant)
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("MAE (mg/L)")
    ax.set_title("A  Fixed-query reconstruction curve", loc="left", weight="bold")
    ax.legend(frameon=False, title=None)

    ax = axes[1]
    x = improvement.k.to_numpy()
    y = improvement.reduction.to_numpy()
    lo = y - improvement.reduction_lo.to_numpy()
    hi = improvement.reduction_hi.to_numpy() - y
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o-", lw=2, capsize=3,
                color=COLORS["corrected"])
    ax.axhline(0, color="#555b63", lw=0.8)
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("MAE reduction vs K=0 (%)")
    ax.set_title("B  Paired station bootstrap", loc="left", weight="bold")

    ax = axes[2]
    if len(shuffle):
        sns = shuffle.groupby(["k", "condition"]).mae.mean().reset_index()
        for condition, group in sns.groupby("condition"):
            ax.plot(group.k, group.mae, "o-", lw=2, label=condition.replace("_", " "))
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("MAE (mg/L)")
    ax.set_title("C  Support-label diagnostic", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.run_dir / "test_predictions.parquet")
    selected_alpha = {
        int(row.k): float(row.alpha)
        for row in pd.read_csv(args.run_dir / "selected_alpha.csv").itertuples()
    }
    all_results = pd.read_csv(args.run_dir / "test_all_alphas.csv")
    raw_summary = (
        all_results[all_results.variant == "raw"]
        .groupby(["k", "variant"], as_index=False).mae.agg(
            mean="mean", std="std", n="count",
        )
    )
    corrected_summary = (
        all_results[
            (all_results.variant == "corrected")
            & all_results.apply(
                lambda row: row.alpha == selected_alpha[int(row.k)], axis=1,
            )
        ]
        .groupby(["k", "variant"], as_index=False).mae.agg(
            mean="mean", std="std", n="count",
        )
    )
    summary = pd.concat([raw_summary, corrected_summary], ignore_index=True)
    selected_predictions = predictions[
        predictions.apply(
            lambda row: row.alpha == selected_alpha[int(row.k)], axis=1,
        )
    ]
    station = station_metrics(selected_predictions)
    base = float(raw_summary[raw_summary.k == 0].iloc[0]["mean"])
    improvement_rows = []
    for k in (0, 1, 3, 5):
        if k == 0:
            delta = (0.0, 0.0, 0.0)
        else:
            delta = bootstrap_delta(station, k)
        improvement_rows.append({
            "k": k,
            "delta_mae": delta[0],
            "delta_lo": delta[1],
            "delta_hi": delta[2],
            "reduction": -100 * delta[0] / base,
            "reduction_lo": -100 * delta[2] / base,
            "reduction_hi": -100 * delta[1] / base,
        })
    improvement = pd.DataFrame(improvement_rows)
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    split = load_split(MASK)
    shuffle = shuffled_scores(predictions, data, split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "curve_summary.csv", index=False)
    improvement.to_csv(args.out_dir / "paired_improvement.csv", index=False)
    shuffle.to_csv(args.out_dir / "support_shuffle.csv", index=False)
    plot(summary, improvement, shuffle, args.out_dir / "spatial_fewshot_curve")
    print(summary.to_string(index=False))
    print(improvement.to_string(index=False))
    print(shuffle.groupby(["k", "condition"]).mae.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
