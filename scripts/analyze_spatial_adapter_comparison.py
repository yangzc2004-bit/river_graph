"""Compare fixed-query spatial adapters with a consistent paired estimand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_comparison import paired_station_comparison
from river_graph.experiments.transfer import DATASETS

ROOT = Path("experiments/phase4_transfer/spatial_adaptation")
LABELS = {"base": "Source expert", "level": "Previous level correction",
          "residual": "Support residual", "adapter": "Validation-selected adapter"}
COLORS = {"base": "#6c757d", "level": "#1f4e79",
          "residual": "#d9772a", "adapter": "#6b752a"}


def selected_predictions(folder: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(folder / "test_predictions.parquet")
    chosen = pd.read_csv(folder / "selected_alpha.csv")[["k", "alpha"]]
    selected = frame.merge(chosen, on=["k", "alpha"], validate="many_to_one")
    selected["model"] = name
    return selected[["seed", "cell", "station", "month", "k", "y_true", "y_pred", "model"]]


def build_report(out: Path, adapter: Path | None = None) -> None:
    paths = {
        "level": ROOT / "fewshot_curve_paired_v1",
        "residual": ROOT / "fewshot_residual_paired_v1",
    }
    frames = [selected_predictions(path, name) for name, path in paths.items()]
    reference = frames[1][frames[1].k == 0].copy()
    base = pd.concat([reference.assign(k=k, model="base") for k in (0, 1, 3, 5)])
    frames.append(base)
    inputs = [path / name for path in paths.values()
              for name in ("test_predictions.parquet", "selected_alpha.csv")]
    if adapter is not None:
        extra = pd.read_parquet(adapter)
        extra["model"] = "adapter"
        frames.append(extra[frames[0].columns])
        inputs.append(adapter)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["model", "seed", "cell", "k"]).any():
        raise ValueError("duplicate prediction rows")
    dataset = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    mask_path = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
    with np.load(mask_path) as m:
        q90 = float(np.quantile(np.asarray(dataset["y"]).ravel()[m["train"]], .9))
    data["error"] = np.abs(data.y_pred - data.y_true)
    data["squared_error"] = np.square(data.y_pred - data.y_true)
    data["log_error"] = np.abs(np.log1p(data.y_pred) - np.log1p(data.y_true))
    rows = []
    comparisons = []
    for (model, k), frame in data.groupby(["model", "k"]):
        comparison = paired_station_comparison(frame, reference, seed=1729 + k)
        comparisons.append({"model": model, "k": k, "reference": "base", **comparison})
        if model not in {"base", "level"}:
            old = data[(data.model == "level") & (data.k == k)]
            comparisons.append({"model": model, "k": k, "reference": "level",
                                **paired_station_comparison(frame, old, seed=1729 + k)})
        tail = frame[frame.y_true >= q90]
        seed_mae = frame.groupby("seed").error.mean()
        rows.append({"model": model, "k": k, "mae": frame.error.mean(),
                     "seed_mae_sd": seed_mae.std(),
                     "rmse": np.sqrt(frame.groupby("seed").squared_error.mean()).mean(),
                     "log_mae": frame.log_error.mean(),
                     "q90_mae": tail.error.mean(), "q90_threshold": q90,
                     "q90_cells": tail.cell.nunique(),
                     "q90_stations": tail.station.nunique(),
                     "n_cells": frame.cell.nunique(), "n_seeds": frame.seed.nunique()})
    summary = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    # Assert the reported main MAE and bootstrap point are exactly the same estimand.
    checked = summary.merge(comparison[comparison.reference == "base"], on=["model", "k"])
    np.testing.assert_allclose(checked.mae_x, checked.mae_y, rtol=1e-12)
    station = data.groupby(["model", "k", "station"]).agg(
        mae=("error", "mean"), mean_prediction=("y_pred", "mean"),
        mean_truth=("y_true", "mean"), n_cells=("cell", "nunique"),
    ).reset_index()
    station["site_no"] = station.station.map(dict(enumerate(dataset["site_no"])))
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    comparison.to_csv(out / "paired_comparisons.csv", index=False)
    station.to_csv(out / "station_metrics.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), layout="constrained")
    for model, label in LABELS.items():
        group = summary[summary.model == model].sort_values("k")
        if group.empty:
            continue
        axes[0].errorbar(group.k, group.mae, yerr=group.seed_mae_sd, label=label,
                        color=COLORS[model], marker="o", capsize=3, lw=1.6)
    axes[0].set(title="a  Matched-query DOC reconstruction", xlabel="Support months per station (K)",
                ylabel="MAE (mg/L)", xticks=[0, 1, 3, 5])
    axes[0].legend(frameon=False, fontsize=8)
    comp = comparison[(comparison.reference == "base") & (comparison.k == 5)
                      & (comparison.model != "base")].set_index("model")
    names = [name for name in LABELS if name in comp.index]
    for i, name in enumerate(names):
        row = comp.loc[name]
        axes[1].errorbar(row.reduction_pct, i,
                        xerr=[[row.reduction_pct-row.reduction_lo],
                              [row.reduction_hi-row.reduction_pct]],
                        fmt="o", capsize=4, color=COLORS[name])
    axes[1].axvline(0, color="#777777", lw=.8)
    axes[1].set(title="b  K=5 improvement over no support",
                yticks=range(len(names)), yticklabels=[LABELS[n] for n in names],
                xlabel="MAE reduction (%) · station bootstrap 95% CI")
    for ax in axes:
        ax.grid(axis="x" if ax is axes[1] else "y", color="#dddddd", lw=.5)
    fig.savefig(out / "adapter_comparison.png", dpi=300)
    fig.savefig(out / "adapter_comparison.pdf")
    plt.close(fig)
    definition = {
        "estimand": "cell-weighted MAE, mean of losses over five seeds",
        "bootstrap": "5000 whole-station paired draws; draw-specific reference denominator",
        "seeds_are_not_independent_ecological_samples": True,
        "experiment_status": "development extension after outer E3 results were seen",
        "repair": "Earlier curves divided a station-equal delta by a cell-weighted MAE; this report supersedes those confidence intervals without changing predictions.",
        "data": {str(p): sha256_file(p) for p in [Path(DATASETS["doc"]), mask_path]},
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "code": {str(p): sha256_file(p) for p in [Path(__file__), Path(
            "src/river_graph/experiments/spatial_comparison.py")]},
    }
    definition["outputs"] = {p.name: sha256_file(p) for p in out.iterdir()
                             if p.suffix in {".csv", ".png", ".pdf"}}
    (out / "analysis_manifest.json").write_text(json.dumps(definition, indent=2)+"\n")
    print(summary.to_string(index=False))
    print(comparison[comparison.k == 5].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "adapter_comparison_v1")
    parser.add_argument("--adapter-predictions", type=Path)
    args = parser.parse_args()
    build_report(args.out_dir, args.adapter_predictions)


if __name__ == "__main__":
    main()
