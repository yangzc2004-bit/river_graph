"""Reconcile manuscript evidence from one frozen regional-residual product.

No fitting or prediction changes occur. All metrics and figures use the
120-tree, five-seed product and its fixed 2,316-cell query. The primary loss is
cell-weighted and averaged over seeds; whole stations are bootstrap clusters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_comparison import paired_station_comparison
from river_graph.experiments.spatial_fewshot import support_schedule

ROOT = Path("experiments/phase4_transfer/spatial_adaptation")
PRODUCT = ROOT / "regional_residual_product_v1"
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
EDGES = Path("data/processed/graph_edges_graphfix_st357.csv")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
BLUE = "#265d82"
ORANGE = "#c87932"
GREY = "#7b8288"


def verify_support_corrections(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute each correction from the product's frozen support predictions."""
    full = pd.read_parquet(PRODUCT / "e3_full_grid.parquet")
    base = full[full.k == 0]
    rows = []
    for seed, group in base.groupby("seed"):
        lookup = group.set_index("cell")
        schedule, query = support_schedule(lookup.index.to_numpy(), 654)
        for k in (0, 1, 3, 5):
            selected = frame[(frame.seed == seed) & (frame.k == k)].sort_values("cell")
            np.testing.assert_array_equal(selected.cell, query)
            correction = {}
            for station, cells in schedule.items():
                support = lookup.loc[cells[:k]]
                delta = (float(np.mean(np.log1p(support.y_true)
                                      - np.log1p(np.maximum(support.base_pred, 0))))
                         if k else 0.0)
                correction[station] = delta
                rows.append({"seed": int(seed), "k": k, "station": station,
                             "support_count": k, "support_correction": delta})
            deltas = selected.station.map(correction).to_numpy()
            np.testing.assert_allclose(deltas, selected.support_correction, rtol=1e-12)
            z = np.log1p(np.maximum(selected.base_pred.to_numpy(), 0))
            expected = np.maximum(np.expm1(z + selected.alpha.to_numpy() * deltas), 0)
            np.testing.assert_allclose(expected, selected.y_pred, rtol=1e-12)
    return pd.DataFrame(rows)


def station_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.assign(
        error=np.abs(frame.y_pred - frame.y_true),
        bias=frame.y_pred - frame.y_true,
    )
    station = work.groupby(["k", "station"], as_index=False).agg(
        mae=("error", "mean"), bias=("bias", "mean"),
        n_cells=("cell", "nunique"), n_seeds=("seed", "nunique"),
    )
    return station


def shuffle_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat the earlier single-per-seed permutation using this product only."""
    rows = []
    for (seed, k), group in frame[frame.k > 0].groupby(["seed", "k"]):
        station = group.groupby("station").support_correction.first().sort_index()
        rng = np.random.default_rng(9100 + 17 * int(seed) + int(k))
        shuffled = pd.Series(rng.permutation(station.to_numpy()), index=station.index)
        alpha = float(group.alpha.iloc[0])
        z = np.log1p(np.maximum(group.base_pred.to_numpy(), 0.0))
        z += alpha * group.station.map(shuffled).to_numpy()
        pred = np.maximum(np.expm1(z), 0.0)
        for condition, values in (("true_support", group.y_pred.to_numpy()),
                                  ("shuffled_support", pred)):
            rows.append({"seed": int(seed), "k": int(k), "condition": condition,
                         "mae": float(np.abs(values - group.y_true.to_numpy()).mean())})
    return pd.DataFrame(rows)


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_curve(summary: pd.DataFrame, shuffle: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.5), layout="constrained")
    ax = axes[0]
    ax.errorbar(summary.k, summary.mae, yerr=summary.seed_mae_sd,
                fmt="o-", color=BLUE, capsize=3, lw=1.8)
    for row in summary.itertuples():
        ax.annotate(f"{row.mae:.3f}", (row.k, row.mae),
                    xytext=(12 if row.k == 0 else 0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set(xticks=[0, 1, 3, 5], xlabel="Support months per station (K)",
           ylabel="DOC MAE (mg/L)", title="a  Fixed-query adaptation")
    ax.set_ylim(1.93, 2.58)
    ax.text(.02, .05, "Bars: training-seed SD (5 seeds)", transform=ax.transAxes,
            fontsize=7.5, color=GREY)
    ax = axes[1]
    ax.errorbar(summary.k, summary.reduction_pct,
                yerr=np.vstack([summary.reduction_pct - summary.reduction_lo,
                                summary.reduction_hi - summary.reduction_pct]),
                fmt="o-", color=ORANGE, capsize=3, lw=1.8)
    ax.axhline(0, color=GREY, lw=.8)
    ax.set(xticks=[0, 1, 3, 5], xlabel="Support months per station (K)",
           ylabel="MAE reduction from K=0 (%)", title="b  Paired station bootstrap")
    ax.text(.02, .96, "Bars: 95% CI; 43 station clusters", transform=ax.transAxes,
            fontsize=7.5, color=GREY, va="top")
    ax = axes[2]
    pooled = shuffle.groupby(["k", "condition"]).mae.mean().reset_index()
    for condition, color, label in (("true_support", BLUE, "Correct station"),
                                    ("shuffled_support", GREY, "Shuffled station")):
        data = pooled[pooled.condition == condition]
        ax.plot(data.k, data.mae, "o-", color=color, label=label, lw=1.8)
    ax.axhline(summary.mae.iloc[0], color=ORANGE, ls="--", lw=1, label="No support")
    ax.set(xticks=[1, 3, 5], xlabel="Support months per station (K)",
           ylabel="DOC MAE (mg/L)", title="c  Station-specific correction")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#dde2e6", lw=.6)
    save_figure(fig, out / "spatial_k_curve")


def plot_stations(table: pd.DataFrame, summary: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), layout="constrained")
    ax = axes[0]
    ax.scatter(np.abs(table.bias_k0), table.mae_reduction, s=30, color=BLUE,
               edgecolor="white", linewidth=.5)
    ax.axhline(0, color=GREY, lw=.8)
    ax.set(xlabel="Absolute K=0 station bias (mg/L)",
           ylabel="K=5 MAE reduction (mg/L)", title="a  Baseline error and calibration benefit")
    ax.text(.03, .94, f"Pearson r = {summary['bias_gain_pearson']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8.5)
    ax = axes[1]
    ordered = table.sort_values("mae_reduction")
    ax.bar(range(len(ordered)), ordered.mae_reduction,
           color=np.where(ordered.mae_reduction > 0, BLUE, ORANGE), width=.85)
    ax.axhline(0, color=GREY, lw=.8)
    ax.set(xticks=[], xlabel="Target station (sorted by error reduction)",
           ylabel="K=5 MAE reduction (mg/L)", title="b  Variation among target stations")
    ax.text(.03, .94, f"{summary['improved_stations']}/43 stations improve",
            transform=ax.transAxes, va="top", fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", color="#dde2e6", lw=.6)
        ax.set_axisbelow(True)
    save_figure(fig, out / "station_mechanism")


def project(lon, lat):
    """Spherical Lambert azimuthal equal-area projection, centered on CONUS."""
    lon, lat = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    lon0, lat0 = np.deg2rad([-96.0, 39.0])
    scale = np.sqrt(2 / (1 + np.sin(lat0) * np.sin(lat)
                         + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)))
    x = 6371 * scale * np.cos(lat) * np.sin(lon - lon0)
    y = 6371 * scale * (np.cos(lat0) * np.sin(lat)
                        - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0))
    return x, y


def plot_map(table: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame,
             out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.3), layout="constrained")
    coordinates = nodes.set_index("site_no")[["dec_long_va", "dec_lat_va"]]
    for row in edges.itertuples():
        pair = coordinates.loc[[row.source, row.target]]
        x, y = project(pair.dec_long_va, pair.dec_lat_va)
        ax.plot(x, y, color="#c8cfd5", alpha=.55, lw=.5, zorder=1)
    x, y = project(nodes.dec_long_va, nodes.dec_lat_va)
    ax.scatter(x, y, s=7, color="#c8cfd5", zorder=2)
    x, y = project(table.longitude, table.latitude)
    colors = LinearSegmentedColormap.from_list("gain", [ORANGE, "#f8f8f6", BLUE])
    norm = TwoSlopeNorm(vmin=min(-5, table.relative_reduction_pct.min()), vcenter=0,
                        vmax=max(5, table.relative_reduction_pct.max()))
    sizes = 25 + 90 * np.sqrt(table.n_cells / table.n_cells.max())
    points = ax.scatter(x, y, c=table.relative_reduction_pct, s=sizes,
                        cmap=colors, norm=norm, edgecolor="#343b40", linewidth=.45, zorder=3)
    cb = fig.colorbar(points, ax=ax, fraction=.035, pad=.025)
    cb.set_label("K=5 MAE reduction from K=0 (%)")
    ax.set(xlabel="East–west distance from map center (km)",
           ylabel="North–south distance from map center (km)",
           title="Station-level benefit of five target observations")
    ax.set_aspect("equal")
    ax.text(.015, .015,
            "43 target stations; size indicates query months\n"
            "Grey links: station graph, not river-channel geometry\n"
            "Lambert azimuthal equal-area; center 96°W, 39°N",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="#3e4850",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    save_figure(fig, out / "station_transfer_map")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "manuscript_evidence_v2")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PRODUCT / "paired_query_predictions.parquet")
    if (sorted(frame.seed.unique()) != [42, 43, 44, 45, 46]
            or sorted(frame.k.unique()) != [0, 1, 3, 5]
            or frame.cell.nunique() != 2316 or frame.station.nunique() != 43):
        raise ValueError("unexpected source-product population")
    support = verify_support_corrections(frame)
    support.to_csv(args.out_dir / "support_residuals.csv", index=False)
    reference = frame[frame.k == 0]
    rows = []
    for k, group in frame.groupby("k"):
        comparison = paired_station_comparison(group, reference,
                                               draws=args.bootstrap_draws, seed=1729 + int(k))
        errors = group.assign(error=np.abs(group.y_pred - group.y_true))
        rows.append({"k": int(k), "alpha": float(group.alpha.iloc[0]), **comparison,
                     "seed_mae_sd": float(errors.groupby("seed").error.mean().std()),
                     "rmse": float(errors.groupby("seed").apply(
                         lambda a: np.sqrt(np.mean((a.y_pred - a.y_true) ** 2)),
                         include_groups=False).mean())})
    summary = pd.DataFrame(rows)
    frozen = pd.read_csv(PRODUCT / "k_curve_summary.csv").set_index("k")
    np.testing.assert_allclose(summary.set_index("k").mae, frozen.mae, rtol=1e-12)
    summary.to_csv(args.out_dir / "main_results.csv", index=False)
    station = station_statistics(frame)
    station.to_csv(args.out_dir / "station_metrics.csv", index=False)
    macro = station.groupby("k", as_index=False).mae.mean().rename(
        columns={"mae": "station_equal_mae"})
    macro["delta_vs_k0"] = macro.station_equal_mae - macro.station_equal_mae.iloc[0]
    macro.to_csv(args.out_dir / "station_equal_results.csv", index=False)
    table = station[station.k == 0][["station", "mae", "bias", "n_cells"]].rename(
        columns={"mae": "mae_k0", "bias": "bias_k0"})
    table = table.merge(station[station.k == 5][["station", "mae"]].rename(
        columns={"mae": "mae_k5"}), on="station", validate="one_to_one")
    table["mae_reduction"] = table.mae_k0 - table.mae_k5
    table["relative_reduction_pct"] = 100 * table.mae_reduction / table.mae_k0
    nodes = pd.read_csv(NODES, dtype={"site_no": str})
    nodes["site_no"] = nodes.site_no.str.zfill(8)
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    np.testing.assert_array_equal(nodes.site_no,
                                  [str(site).zfill(8) for site in dataset["site_no"]])
    nodes["station"] = np.arange(len(nodes))
    table = table.merge(nodes[["station", "site_no", "dec_long_va", "dec_lat_va"]],
                        on="station", validate="one_to_one").rename(
                            columns={"dec_long_va": "longitude", "dec_lat_va": "latitude"})
    table.to_csv(args.out_dir / "station_gain.csv", index=False)
    bx, gy = np.abs(table.bias_k0), table.mae_reduction
    max_bias = bx.idxmax()
    keep = table.index != max_bias
    station_summary = {
        "n_stations": len(table), "improved_stations": int((gy > 0).sum()),
        "median_relative_reduction_pct": float(table.relative_reduction_pct.median()),
        "bias_gain_pearson": float(np.corrcoef(bx, gy)[0, 1]),
        "bias_gain_spearman": float(bx.corr(gy, method="spearman")),
        "bias_gain_pearson_excluding_largest_bias_station": float(
            np.corrcoef(bx[keep], gy[keep])[0, 1]),
        "largest_bias_station": str(table.loc[max_bias, "site_no"]),
        "interpretation": "descriptive shared-error association; not an independent mechanism test",
    }
    (args.out_dir / "station_summary.json").write_text(json.dumps(station_summary, indent=2) + "\n")
    shuffle = shuffle_diagnostic(frame)
    shuffle.to_csv(args.out_dir / "support_shuffle.csv", index=False)
    shuffle.groupby(["k", "condition"]).mae.agg(["mean", "std"]).reset_index().to_csv(
        args.out_dir / "support_shuffle_summary.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_curve(summary, shuffle, args.out_dir)
    plot_stations(table, station_summary, args.out_dir)
    edges = pd.read_csv(EDGES, dtype=str)
    for key in ("source", "target"):
        edges[key] = edges[key].str.zfill(8)
    plot_map(table, nodes, edges, args.out_dir)
    source_files = [PRODUCT / "paired_query_predictions.parquet", PRODUCT / "e3_full_grid.parquet",
                    PRODUCT / "k_curve_summary.csv", PRODUCT / "spec.json", NODES, EDGES, DATASET]
    manifest = {
        "source_product": str(PRODUCT), "model": "regional ExtraTrees, 120 trees, source pool 40",
        "seeds": [42, 43, 44, 45, 46], "query_cells": 2316, "stations": 43,
        "primary_estimand": "cell-weighted MAE, averaged losses over five seeds",
        "bootstrap": f"{args.bootstrap_draws} whole-station paired draws; per-draw cell weights and denominator",
        "station_statistics": "mean of per-seed losses, not loss of ensemble-mean prediction",
        "shuffle": "one station-residual permutation per seed/K; RNG 9100 + 17*seed + K; descriptive diagnostic",
        "no_retraining": True, "experiment_status": "development extension after E3 results were seen",
        "inputs": {str(p): sha256_file(p) for p in source_files},
        "code": {str(p): sha256_file(p) for p in [Path(__file__), Path(
            "src/river_graph/experiments/spatial_comparison.py")]},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    k5 = summary[summary.k == 5].iloc[0]
    (args.out_dir / "correction_note.md").write_text(
        "# Unified manuscript evidence\n\n"
        "All tables and figures in this directory use the same frozen 120-tree, "
        "five-seed regional product and fixed 2,316-cell query. No predictions "
        "were changed and no model was trained.\n\n"
        "## What was reconciled\n\n"
        "The previous main table was cell-weighted, whereas its bootstrap "
        "point was station-equal. The primary bootstrap now recomputes the "
        "cell-weighted statistic after sampling whole stations; station-equal "
        "results are a separately labelled secondary summary. The previous "
        "map, bias plot and shuffle diagnostic came from the separate 300-tree "
        "experiment. They are recomputed here from the 120-tree product.\n\n"
        f"K=5 MAE = {k5.mae:.6f} mg/L; reduction = {k5.reduction_pct:.4f}%; "
        f"delta = {k5.delta_mae:.6f}, 95% CI [{k5.delta_lo:.6f}, {k5.delta_hi:.6f}].\n\n"
        f"{station_summary['improved_stations']}/43 stations improve; median "
        f"station reduction = {station_summary['median_relative_reduction_pct']:.4f}%. "
        f"Bias/gain Pearson r = {station_summary['bias_gain_pearson']:.4f}. "
        "Bias and gain share query errors and this association is descriptive. "
        "Seasonal or hydroclimatic transfer is a modeling rationale rather than "
        "a mechanism established by these comparisons.\n\n"
        "Support candidates are the first, middle, last, first-quarter and "
        "third-quarter observed months, used as nested prefixes. Their labels "
        "only determine the log1p residual correction. Because support spans "
        "the record, this is retrospective reconstruction.\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(station_summary, indent=2))


if __name__ == "__main__":
    main()
