"""Analyze and plot the paired support-residual spatial transfer experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
COLORS = {
    "raw": "#1f4e79",
    "residual_corrected": "#c45a11",
    "true_support": "#c45a11",
    "shuffled_support": "#8a8f98",
}


def station_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, k, alpha, station), group in frame.groupby(
        ["seed", "k", "alpha", "station"],
    ):
        rows.append({
            "seed": int(seed),
            "k": int(k),
            "alpha": float(alpha),
            "station": int(station),
            "mae": float(np.mean(np.abs(group.y_true - group.y_pred))),
            "n": len(group),
        })
    return pd.DataFrame(rows)


def bootstrap_delta(
    station: pd.DataFrame,
    k: int,
    *,
    seed: int = 1729,
    draws: int = 5000,
) -> tuple[float, float, float]:
    zero = station[station.k == 0][["seed", "station", "mae"]].rename(
        columns={"mae": "mae0"},
    )
    current = station[station.k == k][["seed", "station", "mae"]].rename(
        columns={"mae": "maek"},
    )
    paired = current.merge(zero, on=["seed", "station"], validate="one_to_one")
    station_delta = (
        paired.assign(delta=paired.maek - paired.mae0)
        .groupby("station", as_index=False).delta.mean()
    )
    point = float(station_delta.delta.mean())
    rng = np.random.default_rng(seed + k)
    values = station_delta.delta.to_numpy()
    samples = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(draws)
    ])
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def shuffled_scores(
    predictions: pd.DataFrame,
    support_predictions: pd.DataFrame,
    selected_alpha: dict[int, float],
) -> pd.DataFrame:
    rows = []
    for k, alpha in selected_alpha.items():
        if k == 0:
            continue
        for seed in sorted(predictions.seed.unique()):
            base = predictions[
                (predictions.seed == seed)
                & (predictions.k == k)
                & (predictions.alpha == 0.0)
            ].sort_values("cell")
            support = support_predictions[
                (support_predictions.seed == seed)
                & (support_predictions.k == k)
            ]
            true = predictions[
                (predictions.seed == seed)
                & (predictions.k == k)
                & (predictions.alpha == alpha)
            ].sort_values("cell")
            if len(support) != 43 * k or len(true) != len(base):
                raise ValueError("support/query rows are incomplete")
            true_score = metrics(true.y_true.to_numpy(), true.y_pred.to_numpy())["mae"]
            station_ids = np.sort(support.station.unique())
            residual_by_station = (
                support.groupby("station").support_residual.mean()
                .reindex(station_ids)
            )
            rng = np.random.default_rng(9100 + 17 * seed + k)
            shuffled_values = rng.permutation(residual_by_station.to_numpy())
            shuffled = dict(zip(station_ids, shuffled_values, strict=True))
            z = np.log1p(np.maximum(base.base_pred.to_numpy(), 0.0))
            for station in station_ids:
                take = base.station.to_numpy() == station
                z[take] += alpha * shuffled[int(station)]
            shuffle_score = metrics(
                base.y_true.to_numpy(), np.expm1(z),
            )["mae"]
            rows.extend([
                {
                    "seed": seed, "k": k,
                    "condition": "true_support", "mae": true_score,
                },
                {
                    "seed": seed, "k": k,
                    "condition": "shuffled_support", "mae": shuffle_score,
                },
            ])
    return pd.DataFrame(rows)


def plot(
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    shuffle: pd.DataFrame,
    out: Path,
) -> None:
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
    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.25), constrained_layout=True,
    )
    ax = axes[0]
    for variant, group in summary.groupby("variant"):
        ax.errorbar(
            group["k"], group["mean"], yerr=group["std"],
            marker="o", lw=2, capsize=3, color=COLORS[variant],
            label="residual calibration" if variant == "residual_corrected" else "raw",
        )
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("MAE (mg/L)")
    ax.set_title("A  Fixed-query reconstruction curve", loc="left", weight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    x = improvement.k.to_numpy()
    y = improvement.reduction.to_numpy()
    lo = y - improvement.reduction_lo.to_numpy()
    hi = improvement.reduction_hi.to_numpy() - y
    ax.errorbar(
        x, y, yerr=np.vstack([lo, hi]), fmt="o-", lw=2, capsize=3,
        color=COLORS["residual_corrected"],
    )
    ax.axhline(0, color="#555b63", lw=0.8)
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("MAE reduction vs K=0 (%)")
    ax.set_title("B  Paired station bootstrap", loc="left", weight="bold")

    ax = axes[2]
    if len(shuffle):
        grouped = shuffle.groupby(["k", "condition"]).mae.mean().reset_index()
        for condition, group in grouped.groupby("condition"):
            ax.plot(
                group.k, group.mae, "o-", lw=2,
                color=COLORS[condition],
                label=condition.replace("_", " "),
            )
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("MAE (mg/L)")
    ax.set_title("C  Support-residual diagnostic", loc="left", weight="bold")
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
    support_predictions = pd.read_parquet(
        args.run_dir / "support_predictions.parquet",
    )
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
            (all_results.variant == "residual_corrected")
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
        delta = (0.0, 0.0, 0.0) if k == 0 else bootstrap_delta(station, k)
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
    shuffle = shuffled_scores(
        predictions, support_predictions, selected_alpha,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "curve_summary.csv", index=False)
    improvement.to_csv(args.out_dir / "paired_improvement.csv", index=False)
    shuffle.to_csv(args.out_dir / "support_residual_shuffle.csv", index=False)
    plot(summary, improvement, shuffle, args.out_dir / "spatial_fewshot_residual")
    print(summary.to_string(index=False))
    print(improvement.to_string(index=False))
    print(
        shuffle.groupby(["k", "condition"]).mae.agg(["mean", "std"]).to_string(),
    )


if __name__ == "__main__":
    main()
