"""Build the reproducible Source-regional + K-shot residual product.

This is a post-processing product builder.  It does not fit ExtraTrees or
alter the historical experiments.  The five frozen source-regional E3
prediction files are read, the preselected station-mean log1p residual
correction is applied for K={0,1,3,5}, and paired-query/full-E3 tables,
metrics, bootstrap summaries and a publication-ready curve are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.spatial_fewshot import support_schedule

T = 654
SEEDS = (42, 43, 44, 45, 46)
K_VALUES = (0, 1, 3, 5)
ALPHA = {0: 0.0, 1: 0.25, 3: 0.5, 5: 0.75}
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
SOURCE_ROOT = Path(
    "experiments/phase4_transfer/spatial_adaptation/"
    "source_selection_v1_5seed/product"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def apply_adapter(
    y: np.ndarray,
    base: np.ndarray,
    cells: np.ndarray,
    support_by_station: dict[int, np.ndarray],
    query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[int, float]]:
    by_cell = {int(c): i for i, c in enumerate(cells)}
    correction: dict[int, float] = {}
    for station, ordered in support_by_station.items():
        opened = ordered[:k]
        if len(opened):
            idx = np.asarray([by_cell[int(c)] for c in opened], dtype=int)
            correction[station] = float(np.mean(
                np.log1p(np.maximum(y[idx], 0.0))
                - np.log1p(np.maximum(base[idx], 0.0)),
            ))
        else:
            correction[station] = 0.0
    qidx = np.asarray([by_cell[int(c)] for c in query], dtype=int)
    z = np.log1p(np.maximum(base[qidx], 0.0))
    z += ALPHA[k] * np.asarray([
        correction.get(int(c) // T, 0.0) for c in query
    ])
    return np.maximum(np.expm1(z), 0.0), correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(MASK, allow_pickle=False) as saved:
        test_cells = np.sort(np.asarray(saved["test"], dtype=np.int64))
    support_by_station, query_cells = support_schedule(test_cells, T)
    query_cells = np.sort(query_cells)
    support_all = np.sort(np.concatenate(list(support_by_station.values())))
    if len(test_cells) != 2531 or len(query_cells) != 2316:
        raise ValueError("unexpected E3/query population")

    full_rows: list[pd.DataFrame] = []
    query_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = SOURCE_ROOT / f"predictions_seed{seed}.parquet"
        source_hashes[str(path)] = sha256(path)
        source = pd.read_parquet(path).sort_values("cell").reset_index(drop=True)
        cells = source.cell.to_numpy(dtype=np.int64)
        if not np.array_equal(cells, test_cells):
            raise ValueError(f"seed {seed} cells do not match frozen E3 test")
        y = source.y_true.to_numpy(float)
        base = source.y_pred.to_numpy(float)
        by_cell = {int(c): i for i, c in enumerate(cells)}
        for k in K_VALUES:
            pred_query, correction = apply_adapter(
                y, base, cells, support_by_station, query_cells, k,
            )
            qidx = np.asarray([by_cell[int(c)] for c in query_cells], dtype=int)
            metric_rows.append({
                "seed": seed, "k": k, "alpha": ALPHA[k],
                "query_cells": len(query_cells), "support_cells": 43 * k,
                **metrics(y[qidx], pred_query),
            })
            # Full E3 outer product: all 2,531 test cells are retained, while
            # support rows are explicitly marked and excluded from evaluation.
            zfull = np.log1p(np.maximum(base, 0.0))
            zfull += ALPHA[k] * np.asarray([
                correction.get(int(c) // T, 0.0) for c in cells
            ])
            full_pred = np.maximum(np.expm1(zfull), 0.0)
            full_rows.append(pd.DataFrame({
                "cell": cells,
                "station": cells // T,
                "month_index": cells % T,
                "y_true": y,
                "base_pred": base,
                "y_pred": full_pred,
                "support_correction": [correction.get(int(c) // T, 0.0) for c in cells],
                "seed": seed, "k": k, "alpha": ALPHA[k],
                "is_support": np.isin(cells, support_all),
                "is_paired_query": np.isin(cells, query_cells),
                "visibility_role": "outer_e3_test",
                "model_name": "source_regional_et_mean_residual",
            }))
            query_rows.append(pd.DataFrame({
                "cell": query_cells,
                "station": query_cells // T,
                "month_index": query_cells % T,
                "y_true": y[qidx],
                "base_pred": base[qidx],
                "y_pred": pred_query,
                "support_correction": [correction.get(int(c) // T, 0.0) for c in query_cells],
                "seed": seed, "k": k, "alpha": ALPHA[k],
                "visibility_role": "outer_e3_paired_query",
                "model_name": "source_regional_et_mean_residual",
            }))

    full = pd.concat(full_rows, ignore_index=True)
    query = pd.concat(query_rows, ignore_index=True)
    full.to_parquet(args.out_dir / "e3_full_grid.parquet", index=False)
    query.to_parquet(args.out_dir / "paired_query_predictions.parquet", index=False)
    per_seed = pd.DataFrame(metric_rows)
    per_seed.to_csv(args.out_dir / "metrics_by_seed.csv", index=False)

    summary = (per_seed.groupby("k", as_index=False)
               .agg(alpha=("alpha", "first"), mae=("mae", "mean"),
                    mae_sd=("mae", "std"), rmse=("rmse", "mean"),
                    log_mae=("log_mae", "mean"), n_seeds=("seed", "count"),
                    query_cells=("query_cells", "first"),
                    support_cells=("support_cells", "first")))
    k0 = float(summary.loc[summary.k == 0, "mae"].iloc[0])
    summary["relative_reduction_pct"] = 100.0 * (k0 - summary.mae) / k0
    summary.to_csv(args.out_dir / "k_curve_summary.csv", index=False)

    # Station-clustered paired bootstrap versus each seed's K=0 prediction.
    station = (query.groupby(["seed", "k", "station"], as_index=False)
               .apply(lambda x: pd.Series({
                   "mae": float(np.mean(np.abs(x.y_true - x.y_pred))),
               }), include_groups=False).reset_index(drop=True))
    boot_rows: list[dict] = []
    rng = np.random.default_rng(20261002)
    for k in K_VALUES:
        cur = station[station.k == k].groupby("station").mae.mean()
        zero = station[station.k == 0].groupby("station").mae.mean()
        delta = (cur - zero).dropna().to_numpy()
        draws = np.asarray([
            rng.choice(delta, len(delta), replace=True).mean()
            for _ in range(args.bootstrap_draws)
        ])
        boot_rows.append({
            "k": k, "alpha": ALPHA[k], "delta_mae_vs_k0": float(delta.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "n_stations": len(delta),
        })
    pd.DataFrame(boot_rows).to_csv(args.out_dir / "station_bootstrap.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    })
    fig, ax = plt.subplots(figsize=(5.7, 3.8), constrained_layout=True)
    x = summary.k.to_numpy(float)
    ax.errorbar(x, summary.mae, yerr=summary.mae_sd, fmt="o-", lw=2,
                capsize=3, color="#1f5a8a")
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Target-station support (K)")
    ax.set_ylabel("MAE (mg/L)")
    ax.set_title("Spatial transfer with mean residual adaptation", loc="left", weight="bold")
    ax.text(0.02, 0.04, "five source-pool seeds; fixed paired query", transform=ax.transAxes,
            fontsize=8, color="#4d5964")
    fig.savefig(args.out_dir / "k_curve.png", dpi=300)
    fig.savefig(args.out_dir / "k_curve.pdf")
    plt.close(fig)

    spec = {
        "model": "Source regional ExtraTrees (pool=40) + station mean log1p residual",
        "seeds": list(SEEDS), "K": list(K_VALUES), "alpha_by_k": ALPHA,
        "mask": str(MASK), "test_cells": len(test_cells),
        "paired_query_cells": len(query_cells), "support_cells_by_k": {str(k): 43 * k for k in K_VALUES},
        "source_product": str(SOURCE_ROOT),
        "training": "none; post-processing of frozen five-seed predictions",
        "mask_sha256": sha256(MASK),
        "builder_script_sha256": sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "query_definition": "fixed five candidate months are reserved for all K",
    }
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "artifact_type": "postprocessed_spatial_transfer_product",
        "source_prediction_files": source_hashes,
        "mask": str(MASK), "mask_sha256": sha256(MASK),
        "builder": str(Path(__file__)),
        "builder_script_sha256": sha256(Path(__file__)),
        "n_seeds": len(SEEDS), "seeds": list(SEEDS),
        "no_retraining": True,
    }, indent=2) + "\n")
    (args.out_dir / "verdict.md").write_text(
        "# Regional residual product\n\n"
        "This directory is a post-processing product of the frozen five-seed "
        "source-pool-40 ExtraTrees spatial transfer runs. It provides an E3 "
        "outer-test table with all 2,531 cells, a fixed 2,316-cell paired "
        "query table, the K curve, station-clustered bootstrap intervals and "
        "a publication-ready figure. Support labels are used only to compute "
        "the station mean residual; support rows are marked and excluded from "
        "paired-query metrics. No model is retrained.\n"
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
