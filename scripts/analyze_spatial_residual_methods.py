"""Select and score a small set of support-residual calibrators.

Method and regularization selection uses internal E3 validation only. The
outer test is evaluated only for the selected candidate and the previously
frozen mean-residual control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_spatial_fewshot_curve import T

from river_graph.experiments.spatial_fewshot import K_VALUES
from river_graph.experiments.spatial_residual_methods import (
    candidate_specs,
    spec_label,
    station_residual_adjustment,
)
from river_graph.experiments.transfer import DATASETS

ALPHA_CONTROL = {0: 0.0, 1: 0.25, 3: 0.5, 5: 0.75}
K_EVAL = (1, 3, 5)


def _rows(
    cache: pd.DataFrame,
    values: np.ndarray,
    stage: str,
    k: int,
    seed: int,
    spec: dict[str, float | str | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = cache[
        (cache.stage == stage) & (cache.k == k)
        & (cache.seed == seed) & (cache.role == "query")
    ].sort_values("cell")
    support = cache[
        (cache.stage == stage) & (cache.k == k)
        & (cache.seed == seed) & (cache.role == "support")
    ].sort_values("cell")
    if k == 0:
        return (
            query.base_pred.to_numpy(dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int64),
        )
    if len(support) != 43 * k if stage == "test" else len(support) != 31 * k:
        raise ValueError(f"unexpected support count for {stage}, K={k}")
    q_cells = query.cell.to_numpy(dtype=np.int64)
    s_cells = support.cell.to_numpy(dtype=np.int64)
    q_z = np.log1p(np.maximum(query.base_pred.to_numpy(dtype=float), 0.0))
    s_z = np.log1p(np.maximum(support.base_pred.to_numpy(dtype=float), 0.0))
    support_truth = np.log1p(values[s_cells])
    support_residual = support_truth - s_z
    correction = np.zeros_like(q_z)
    regularization = (
        float(spec["regularization"])
        if spec["regularization"] is not None else 10.0
    )
    for station in query.station.unique():
        q_take = query.station.to_numpy() == station
        s_take = support.station.to_numpy() == station
        if not s_take.any():
            continue
        correction[q_take] = station_residual_adjustment(
            str(spec["method"]),
            s_z[s_take],
            support_residual[s_take],
            q_z[q_take],
            support.month.to_numpy(dtype=np.int64)[s_take],
            query.month.to_numpy(dtype=np.int64)[q_take],
            alpha=float(spec["alpha"]), regularization=regularization,
        )
    return np.expm1(q_z + correction), q_cells, s_cells


def _score_spec(
    cache: pd.DataFrame,
    values: np.ndarray,
    stage: str,
    k: int,
    seed: int,
    spec: dict[str, float | str | None],
) -> float:
    prediction, query_cells, _support_cells = _rows(
        cache, values, stage, k, seed, spec,
    )
    if k == 0:
        query = cache[
            (cache.stage == stage) & (cache.k == 0)
            & (cache.seed == seed) & (cache.role == "query")
        ].sort_values("cell")
        query_cells = query.cell.to_numpy(dtype=np.int64)
    return float(np.mean(np.abs(values[query_cells] - prediction)))


def select_validation(
    cache: pd.DataFrame,
    values: np.ndarray,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict[int, dict[str, float | str | None]]]:
    rows = []
    for k in K_EVAL:
        for spec in candidate_specs():
            label = spec_label(spec)
            for seed in seeds:
                rows.append({
                    "stage": "val",
                    "k": k,
                    "seed": seed,
                    "method_label": label,
                    "method": spec["method"],
                    "alpha": spec["alpha"],
                    "regularization": spec["regularization"],
                    "mae": _score_spec(cache, values, "val", k, seed, spec),
                })
    frame = pd.DataFrame(rows)
    mean_scores = (
        frame.groupby(
            ["k", "method_label", "method", "alpha", "regularization"],
            as_index=False,
            dropna=False,
        ).mae.mean()
        .sort_values(["k", "mae", "method_label"])
    )
    selected_rows = mean_scores.groupby("k", sort=False).head(1)
    selected = {}
    for row in selected_rows.itertuples():
        selected[int(row.k)] = {
            "method": row.method,
            "alpha": float(row.alpha),
            "regularization": (
                None if pd.isna(row.regularization)
                else float(row.regularization)
            ),
            "method_label": row.method_label,
            "validation_mae": float(row.mae),
        }
    return frame, selected


def score_outer(
    cache: pd.DataFrame,
    values: np.ndarray,
    seeds: list[int],
    selected: dict[int, dict[str, float | str | None]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    for k in K_VALUES:
        specs = []
        if k == 0:
            specs.append({
                "method": "mean", "alpha": 0.0,
                "regularization": None, "method_label": "raw",
            })
        else:
            specs.append(selected[k])
            specs.append({
                "method": "mean",
                "alpha": ALPHA_CONTROL[k],
                "regularization": None,
                "method_label": f"mean_control_a{ALPHA_CONTROL[k]:g}",
            })
        for spec in specs:
            for seed in seeds:
                query = cache[
                    (cache.stage == "test") & (cache.k == k)
                    & (cache.seed == seed) & (cache.role == "query")
                ].sort_values("cell")
                prediction, query_cells, _support_cells = _rows(
                    cache, values, "test", k, seed, spec,
                )
                if k == 0:
                    query_cells = query.cell.to_numpy(dtype=np.int64)
                y_true = values[query_cells]
                mae = float(np.mean(np.abs(y_true - prediction)))
                label = str(spec["method_label"])
                rows.append({
                    "stage": "test",
                    "k": k,
                    "seed": seed,
                    "method_label": label,
                    "method": spec["method"],
                    "alpha": spec["alpha"],
                    "regularization": spec["regularization"],
                    "mae": mae,
                    "query_cells": len(query_cells),
                })
                predictions.append(pd.DataFrame({
                    "cell": query_cells,
                    "station": query_cells // T,
                    "month": query_cells % T,
                    "y_true": y_true,
                    "y_pred": prediction,
                    "seed": seed,
                    "k": k,
                    "method_label": label,
                }))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def plot(summary: pd.DataFrame, out: Path) -> None:
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
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    for label, group in summary.groupby("method_label"):
        group = group.sort_values("k")
        ax.errorbar(
            group["k"], group["mean"], yerr=group["std"],
            marker="o", lw=2, capsize=3, label=label,
        )
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Target-station DOC support (K)")
    ax.set_ylabel("Outer E3 MAE (mg/L)")
    ax.set_title("Selected residual calibrator vs mean control", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = pd.read_parquet(args.run_dir / "base_predictions.parquet")
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    values = np.asarray(data["y"], dtype=float).reshape(-1)
    seeds = sorted(cache.seed.unique().tolist())
    if set(cache.columns) != {
        "stage", "seed", "k", "role", "cell", "station", "month", "base_pred",
    }:
        raise ValueError("base cache contains unexpected columns")
    validation, selected = select_validation(cache, values, seeds)
    outer, predictions = score_outer(cache, values, seeds, selected)
    selected_json = {
        str(k): {key: value for key, value in spec.items()}
        for k, spec in selected.items()
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out / "validation_candidates.csv", index=False)
    outer.to_csv(out / "test_selected_and_control.csv", index=False)
    predictions.to_parquet(out / "test_selected_and_control_predictions.parquet", index=False)
    (out / "selected_methods.json").write_text(
        json.dumps(selected_json, indent=2) + "\n",
    )
    summary = (
        outer.groupby(["k", "method_label"], as_index=False).mae.agg(
            mean="mean", std="std", n="count",
        )
    )
    summary.to_csv(out / "test_summary.csv", index=False)
    plot(summary, out / "residual_method_curve")
    manifest = {
        "experiment": "fewshot_residual_methods_v1",
        "selection": "internal validation only",
        "outer_scored": "selected method per K plus pre-existing mean residual control",
        "candidate_set": [
            spec_label(spec) for spec in candidate_specs()
        ],
        "development_extension": (
            "candidate methods were specified after the prior outer mean "
            "result was observed; this is an extension for method development"
        ),
        "selected_methods": selected_json,
        "test_candidate_enumeration": "disabled",
    }
    (out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print("selected methods")
    print(json.dumps(selected_json, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
