"""Recompute DOC hybrid manuscript evidence from saved seed predictions.

No model is trained or selected here. Main metrics average the five individual
seed metrics, and confidence intervals resample stations after averaging cell
losses across seeds. They are not errors of mean/median ensemble predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path("experiments/phase4_transfer/kgml_local_transport_v1")
PRODUCT = ROOT / "doc_hybrid_product_v1/all_test_predictions.parquet"
AFFINE = ROOT / "extra_trees_affine_v1/seed_family_metrics.csv"
CONTEXT = Path("experiments/phase4_transfer/rf_context_model_selection_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
MASKS = Path("experiments/masks_stcore_v1")
SPATIAL = Path("experiments/phase4_transfer/spatial_adaptation/manuscript_evidence_v2")
FAMILIES = {
    "e1_r20_seed42": "Random gaps",
    "e2a_strict": "Unobserved periods",
    "e2b_partial": "Observation-assisted periods",
    "e3_spatial_seed42": "Unmonitored stations",
}
SEEDS = (42, 43, 44, 45, 46)
MODELS = {
    "rf_context": "Random forest context",
    "et_context": "ExtraTrees context",
    "hybrid": "Selected hybrid",
    "local_base": "Local ExtraTrees base",
    "temporal_residual": "Temporal residual expert",
}
PALETTE = {"rf_context": "#398B86", "et_context": "#235B80", "hybrid": "#D08043"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    error = pred - y
    tail = y >= threshold
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / np.square(y - y.mean()).sum()),
        "log_mae": float(np.abs(np.log1p(np.maximum(pred, 0)) - np.log1p(y)).mean()),
        "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else np.nan,
        "q90_threshold_train": threshold,
        "q90_n_unique_cells": int(tail.sum()),
        "q90_unstable": bool(tail.sum() < 20),
    }


def paired_station_bootstrap(cell: pd.DataFrame, draws: int, seed: int) -> dict:
    """Preserve cell weighting while resampling whole station clusters."""
    station = cell.groupby("station", sort=True).agg(
        n=("cell", "size"), baseline_sum=("et_abs_error", "sum"),
        hybrid_sum=("hybrid_abs_error", "sum"),
    )
    n = station["n"].to_numpy()
    baseline = station["baseline_sum"].to_numpy()
    hybrid = station["hybrid_sum"].to_numpy()
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(station), (draws, len(station)))
    baseline_draw = baseline[sample].sum(axis=1) / n[sample].sum(axis=1)
    hybrid_draw = hybrid[sample].sum(axis=1) / n[sample].sum(axis=1)
    delta = hybrid_draw - baseline_draw
    reduction = 100 * (baseline_draw - hybrid_draw) / baseline_draw
    base = baseline.sum() / n.sum()
    final = hybrid.sum() / n.sum()
    return {
        "n_unique_cells": int(n.sum()), "n_stations": len(station),
        "et_mae": float(base), "hybrid_mae": float(final),
        "delta_mae_hybrid_minus_et": float(final - base),
        "delta_ci_low": float(np.quantile(delta, 0.025)),
        "delta_ci_high": float(np.quantile(delta, 0.975)),
        "relative_mae_reduction_pct": float(100 * (base - final) / base),
        "reduction_ci_low_pct": float(np.quantile(reduction, 0.025)),
        "reduction_ci_high_pct": float(np.quantile(reduction, 0.975)),
        "bootstrap_draws": draws, "bootstrap_seed": seed,
    }


def plot_results(summary: pd.DataFrame, paired: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.labelsize": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "text.color": "#292929",
        "axes.labelcolor": "#292929", "xtick.color": "#444444",
        "ytick.color": "#444444", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.8), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=0.29, right=0.99, bottom=0.28, top=0.86, wspace=0.24)
    ypos = np.arange(len(FAMILIES))
    models = ["rf_context", "et_context", "hybrid"]
    height = 0.23
    for j, model in enumerate(models):
        sub = summary.loc[summary.model.eq(model)].set_index("mask").loc[list(FAMILIES)]
        yy = ypos + (j - 1) * height
        axes[0].barh(yy, sub.mae, height=height * 0.86, color=PALETTE[model],
                     label=MODELS[model], hatch="//" if model == "hybrid" else None,
                     linewidth=0.6, edgecolor=PALETTE[model], zorder=3)
        axes[0].errorbar(sub.mae, yy, xerr=sub.mae_sd_seed, color="#333333",
                         fmt="none", capsize=2, linewidth=0.8, zorder=4)
    labels = []
    for mask, title in FAMILIES.items():
        row = paired.set_index("mask").loc[mask]
        if title == "Observation-assisted periods":
            title = "Observation-assisted\nperiods"
        labels.append(f"{title}\n{int(row.n_unique_cells):,} cells · {int(row.n_stations)} stations")
    axes[0].set_yticks(ypos, labels)
    axes[0].tick_params(axis="y", length=0, pad=10)
    axes[0].set_ylim(3.6, -0.7)
    axes[0].set_xlim(0, 3.2)
    axes[0].set_xlabel("DOC MAE (mg L$^{-1}$)")
    axes[0].set_title("a  Reconstruction performance", loc="left", pad=15)
    axes[0].grid(axis="x", color="#E8E8E8", linewidth=0.6, zorder=0)
    axes[0].spines["left"].set_visible(False)
    p = paired.set_index("mask").loc[list(FAMILIES)]
    x = p.relative_mae_reduction_pct.to_numpy()
    lo = p.reduction_ci_low_pct.to_numpy()
    hi = p.reduction_ci_high_pct.to_numpy()
    axes[1].axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].errorbar(x, ypos, xerr=np.vstack([x - lo, hi - x]),
                     fmt="o", color=PALETTE["hybrid"], ecolor=PALETTE["hybrid"],
                     markersize=6, capsize=4, linewidth=1.5, zorder=3)
    for i, value in enumerate(x):
        label = f"{value:.1f}%" if value != 0 else "0% (context route)"
        axes[1].text(value + 1.0, i - 0.17, label, fontsize=12.5, color="#333333")
    axes[1].set_xlim(min(-2, lo.min() - 2), max(32, hi.max() + 3))
    axes[1].set_ylim(3.6, -0.7)
    axes[1].set_yticks([])
    axes[1].set_xlabel("MAE reduction vs ExtraTrees (%)")
    axes[1].set_title("b  Paired improvement", loc="left", pad=15)
    axes[1].spines["left"].set_visible(False)
    axes[1].grid(axis="x", color="#E8E8E8", linewidth=0.6, zorder=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.6, 0.105), frameon=False, fontsize=12.5,
               handlelength=1.4, columnspacing=1.2)
    fig.text(0.29, 0.025,
             "Five training seeds. a: mean ± seed SD.\n"
             "b: 95% station-bootstrap intervals (5,000 draws).",
             fontsize=12.5, color="#555555", linespacing=1.35)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"doc_hybrid_performance.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def plot_spatial_publication_copies(out: Path) -> set[Path]:
    """Draw publication-sized copies without rewriting historical evidence."""
    paths = {
        SPATIAL / "main_results.csv", SPATIAL / "support_shuffle.csv",
        SPATIAL / "station_gain.csv", Path("data/processed/graph_nodes_graphfix_st357.csv"),
        Path("data/processed/graph_edges_graphfix_st357.csv"),
    }
    summary = pd.read_csv(SPATIAL / "main_results.csv").sort_values("k")
    shuffle = pd.read_csv(SPATIAL / "support_shuffle.csv")
    stations = pd.read_csv(SPATIAL / "station_gain.csv", dtype={"site_no": str})
    nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str})
    edges = pd.read_csv("data/processed/graph_edges_graphfix_st357.csv", dtype=str)
    blue, orange = "#235B80", "#D08043"
    style = {
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3),
                                 gridspec_kw={"width_ratios": [1.65, 1]})
        fig.subplots_adjust(left=0.09, right=0.985, top=0.85, bottom=0.22, wspace=0.32)
        ax = axes[0]
        baseline = float(summary.mae.iloc[0])
        ax.plot([0, 5.12], [baseline, baseline], color="#969CA0",
                linestyle=(0, (1.3, 2.4)), linewidth=0.9)
        ax.errorbar(summary.k, summary.mae, yerr=summary.seed_mae_sd,
                     fmt="o-", markersize=4.5, color=blue, capsize=2.5,
                     elinewidth=0.9, linewidth=1.5, zorder=3)
        shuffled = shuffle.loc[shuffle.condition.eq("shuffled_support")].groupby("k", as_index=False).agg(
            mae=("mae", "mean"), seed_sd=("mae", "std"),
        )
        ax.errorbar(shuffled.k, shuffled.mae, yerr=shuffled.seed_sd,
                     fmt="s--", markersize=4.3, color=orange, markerfacecolor="white",
                     markeredgewidth=1.1, capsize=2.5, elinewidth=0.9, linewidth=1.2, zorder=3)
        true_final = float(summary.loc[summary.k.eq(5), "mae"].iloc[0])
        shuffled_final = float(shuffled.loc[shuffled.k.eq(5), "mae"].iloc[0])
        for value, label, color in (
            (shuffled_final, "Shuffled station", "#333333"),
            (baseline, "No support", "#6A7075"),
            (true_final, "Correct station", "#333333"),
        ):
            ax.text(5.35, value, f"{label}\n{value:.3f}", color=color,
                    fontsize=8, va="center", linespacing=1.4)
        ax.set(xlabel="Target observations per station, K", ylabel="DOC MAE (mg L$^{-1}$)",
               xlim=(-0.25, 7.5), ylim=(1.90, 2.93), xticks=[0, 1, 3, 5],
               yticks=[2.0, 2.2, 2.4, 2.6, 2.8])
        ax.spines["bottom"].set_bounds(0, 5)
        ax.set_axisbelow(True)
        ax.text(-0.12, 1.08, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.set_title("Station-specific calibration", loc="left", pad=15,
                     fontsize=9, fontweight="normal")

        ax = axes[1]
        effects = summary.loc[summary.k.gt(0)].sort_values("k")
        positions = np.arange(len(effects))
        ax.axvline(0, color="#92989D", linestyle=(0, (2, 3)), linewidth=0.8)
        ax.errorbar(effects.reduction_pct, positions,
                     xerr=np.vstack([effects.reduction_pct - effects.reduction_lo,
                                     effects.reduction_hi - effects.reduction_pct]),
                     fmt="o", markersize=4.5, color=blue, capsize=3,
                     elinewidth=1.2, markeredgewidth=1, zorder=3)
        for ypos, effect in zip(positions, effects.reduction_pct, strict=True):
            ax.annotate(f"{effect:.1f}%", (effect, ypos), xytext=(0, 10),
                        textcoords="offset points", ha="center", color="#333333", fontsize=8.5)
        ax.set(xlabel="MAE reduction (%)", xlim=(-5, 34), ylim=(2.55, -0.55),
               xticks=[0, 10, 20, 30], yticks=positions,
               yticklabels=[f"K = {int(k)}" for k in effects.k])
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
        ax.text(-0.19, 1.08, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.set_title("Paired improvement", loc="left", pad=15,
                     fontsize=9, fontweight="normal")
        fig.text(0.09, 0.055, "43 stations  ·  2,316 fixed query cells  ·  5 training seeds",
                 fontsize=8, color="#6A7075")
        with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
            for suffix in ("pdf", "png", "svg"):
                fig.savefig(out / f"doc_spatial_support_publication.{suffix}", dpi=300, facecolor="white")
        plt.close(fig)

        def project(lon, lat):
            longitude, latitude = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
            lon0, lat0 = np.deg2rad([-96.0, 39.0])
            scale = np.sqrt(2 / (1 + np.sin(lat0) * np.sin(latitude)
                                 + np.cos(lat0) * np.cos(latitude) * np.cos(longitude - lon0)))
            x = 6371 * scale * np.cos(latitude) * np.sin(longitude - lon0)
            y = 6371 * scale * (np.cos(lat0) * np.sin(latitude)
                               - np.sin(lat0) * np.cos(latitude) * np.cos(longitude - lon0))
            return x, y

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        fig.subplots_adjust(left=0.13, right=0.90, bottom=0.23, top=0.91)
        coordinates = nodes.set_index("site_no")[["dec_long_va", "dec_lat_va"]]
        for edge in edges.itertuples():
            pair = coordinates.loc[[edge.source, edge.target]]
            x, y = project(pair.dec_long_va, pair.dec_lat_va)
            ax.plot(x, y, color="#C8CFD5", alpha=0.55, linewidth=0.5, zorder=1)
        x, y = project(nodes.dec_long_va, nodes.dec_lat_va)
        ax.scatter(x, y, s=7, color="#C8CFD5", zorder=2)
        x, y = project(stations.longitude, stations.latitude)
        colors = LinearSegmentedColormap.from_list("station_gain", [orange, "#F8F8F6", blue])
        norm = TwoSlopeNorm(vmin=min(-5, stations.relative_reduction_pct.min()), vcenter=0,
                            vmax=max(5, stations.relative_reduction_pct.max()))
        sizes = 22 + 75 * np.sqrt(stations.n_cells / stations.n_cells.max())
        points = ax.scatter(x, y, c=stations.relative_reduction_pct, s=sizes, cmap=colors,
                            norm=norm, edgecolor="#343B40", linewidth=0.45, zorder=3)
        colorbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.025)
        colorbar.set_label("K=5 MAE reduction from K=0 (%)", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=8)
        ax.set(xlabel="East–west distance from map center (km)",
               ylabel="North–south distance from map center (km)",
               title="Station-level benefit of five target observations")
        ax.set_aspect("equal")
        fig.text(0.13, 0.035,
                 "43 target stations; marker size represents query months.\n"
                 "Grey links show the station graph, not river-channel geometry.\n"
                 "Lambert azimuthal equal-area projection; center 96°W, 39°N.",
                 fontsize=8, color="#555555", linespacing=1.35)
        for suffix in ("pdf", "png"):
            fig.savefig(out / f"doc_station_response_publication.{suffix}", dpi=300, facecolor="white")
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "manuscript_evidence_v1")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = {PRODUCT, AFFINE, DATASET, Path(__file__)}
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    y_grid = data["y"].numpy()
    site_no = np.asarray(data["site_no"], dtype=str)
    months = np.asarray(data["months"], dtype=str)
    product = pd.read_parquet(PRODUCT)
    affine = pd.read_csv(AFFINE).set_index(["seed", "mask"])
    assert not product.duplicated(["seed", "mask", "cell"]).any()
    assert set(product.seed) == set(SEEDS)
    assert set(product["mask"]) == set(FAMILIES)
    seed_rows, paired_rows, query_rows, station_rows, selection_rows = [], [], [], [], []
    for mask, family in FAMILIES.items():
        mask_path = MASKS / f"{mask}.npz"
        sources.add(mask_path)
        with np.load(mask_path) as saved:
            test_cells = np.sort(saved["test"])
            threshold = float(np.quantile(y_grid.ravel()[saved["train"]], 0.9))
        family_losses = []
        for seed in SEEDS:
            frame = product.loc[product.seed.eq(seed) & product["mask"].eq(mask)].sort_values("cell").copy()
            np.testing.assert_array_equal(frame.cell, test_cells)
            truth = y_grid.ravel()[test_cells].astype(float)
            np.testing.assert_array_equal(frame.y_true, truth)
            context_root = CONTEXT / f"all_masks_seed{seed}"
            select_path = context_root / "validation_selection.csv"
            pred_path = context_root / "runs" / f"{mask}__seed{seed}/predictions.parquet"
            meta_path = pred_path.with_name("meta.json")
            sources.update([select_path, pred_path, meta_path])
            meta = json.loads(meta_path.read_text())
            assert meta["prediction_sha256"] == sha256(pred_path)
            assert meta["config"]["dataset_sha256"] == sha256(DATASET)
            assert meta["config"]["mask_sha256"] == sha256(mask_path)
            selected = pd.read_csv(select_path).set_index("mask").loc[mask, "selected_model"]
            assert str(selected).startswith("et_")
            source = pd.read_parquet(pred_path)
            et = source.loc[source.model.eq(selected) & source.role.eq("test")].sort_values("cell")
            rf = source.loc[source.model.eq("rf_default") & source.role.eq("test")].sort_values("cell")
            for source_frame in (et, rf):
                np.testing.assert_array_equal(source_frame.cell, frame.cell)
                np.testing.assert_array_equal(source_frame.y_true, frame.y_true)
            np.testing.assert_array_equal(et.y_pred, frame.context_pred)
            detail = affine.loc[(seed, mask)]
            if detail.selected_variant == "context":
                np.testing.assert_array_equal(frame.y_pred, frame.context_pred)
            elif detail.selected_variant == "log_affine":
                design = np.column_stack([
                    np.ones(len(frame)), np.log1p(frame.context_pred), np.log1p(frame.residual_pred),
                ])
                beta = detail[["beta0", "beta_context", "beta_residual"]].to_numpy(dtype=float)
                np.testing.assert_allclose(frame.y_pred, np.maximum(0, np.expm1(design @ beta)),
                                           rtol=1e-12, atol=1e-12)
                # Trace the deep expert to its matching saved prediction rows.
                residual_root = ROOT / ("extra_trees_residual_pilot" if seed == 42
                                        else f"extra_trees_residual_seed{seed}")
                residual_path = residual_root / "runs" / f"residual_nomsg__doc__{mask}__seed{seed}" / "test_predictions.parquet"
                sources.add(residual_path)
                residual_source = pd.read_parquet(residual_path).sort_values("cell")
                np.testing.assert_array_equal(residual_source.cell, frame.cell)
                np.testing.assert_array_equal(residual_source.y_true, frame.y_true)
                np.testing.assert_array_equal(residual_source.final_pred, frame.residual_pred)
            else:
                raise ValueError(f"unhandled saved route: {detail.selected_variant}")
            predictions = {"rf_context": rf.y_pred.to_numpy(), "et_context": frame.context_pred.to_numpy(),
                           "hybrid": frame.y_pred.to_numpy()}
            if frame.residual_pred.notna().all():
                predictions["local_base"] = residual_source.local_pred.to_numpy()
                predictions["temporal_residual"] = frame.residual_pred.to_numpy()
            for model, pred in predictions.items():
                assert np.isfinite(pred).all()
                result = scores(truth, pred, threshold)
                seed_rows.append({"mask": mask, "family": family, "seed": seed, "model": model,
                                  "n_unique_cells": len(frame), "n_stations": len(np.unique(test_cells // y_grid.shape[1])),
                                  **result})
                if model == "hybrid":
                    np.testing.assert_allclose(result["mae"], detail.test_mae, atol=1e-12)
                if model == "et_context":
                    np.testing.assert_allclose(result["mae"], detail.context_test_mae, atol=1e-12)
            selection_rows.append({"mask": mask, "seed": seed, "context_model": selected,
                                   "hybrid_route": detail.selected_variant,
                                   "beta0": detail.beta0, "beta_context": detail.beta_context,
                                   "beta_residual": detail.beta_residual})
            frame["et_abs_error"] = np.abs(truth - predictions["et_context"])
            frame["hybrid_abs_error"] = np.abs(truth - predictions["hybrid"])
            frame["rf_abs_error"] = np.abs(truth - predictions["rf_context"])
            family_losses.append(frame[["cell", "et_abs_error", "hybrid_abs_error", "rf_abs_error"]])
        cells = pd.concat(family_losses).groupby("cell", as_index=False).mean()
        idx = cells.cell.to_numpy(dtype=int)
        cells["station"] = site_no[idx // y_grid.shape[1]]
        cells["month"] = months[idx % y_grid.shape[1]]
        cells["y_true"] = y_grid.ravel()[idx]
        cells["mask"] = mask
        cells["family"] = family
        query_rows.append(cells)
        stat = cells.groupby("station", as_index=False).agg(
            n_unique_cells=("cell", "size"), et_mae=("et_abs_error", "mean"),
            hybrid_mae=("hybrid_abs_error", "mean"), rf_mae=("rf_abs_error", "mean"),
        )
        stat["mask"] = mask
        stat["delta_mae"] = stat.hybrid_mae - stat.et_mae
        station_rows.append(stat)
        paired_rows.append({"mask": mask, "family": family,
                            **paired_station_bootstrap(cells, args.bootstrap_draws, 42)})
    seed_table = pd.DataFrame(seed_rows)
    summary = seed_table.groupby(["mask", "family", "model"], sort=False, as_index=False).agg(
        n_seeds=("seed", "nunique"), n_unique_cells=("n_unique_cells", "first"),
        n_stations=("n_stations", "first"), mae=("mae", "mean"), mae_sd_seed=("mae", "std"),
        rmse=("rmse", "mean"), r2=("r2", "mean"), log_mae=("log_mae", "mean"),
        q90_mae=("q90_mae", "mean"), q90_threshold_train=("q90_threshold_train", "first"),
        q90_n_unique_cells=("q90_n_unique_cells", "first"), q90_unstable=("q90_unstable", "first"),
    )
    paired = pd.DataFrame(paired_rows)
    for mask in FAMILIES:
        expected = summary.loc[summary["mask"].eq(mask)].set_index("model")
        actual = paired.set_index("mask").loc[mask]
        np.testing.assert_allclose([actual.et_mae, actual.hybrid_mae],
                                   expected.loc[["et_context", "hybrid"], "mae"], atol=1e-12)
    seed_table.to_csv(out / "seed_metrics.csv", index=False)
    summary.to_csv(out / "family_metrics.csv", index=False)
    paired.to_csv(out / "paired_comparisons.csv", index=False)
    all_query_losses = pd.concat(query_rows, ignore_index=True)
    all_query_losses.to_parquet(out / "paired_cell_losses.parquet", index=False)
    pd.concat(station_rows).to_csv(out / "station_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "selected_routes.csv", index=False)
    component_rows = []
    temporal_masks = seed_table.loc[seed_table.model.eq("temporal_residual"), "mask"].unique()
    component_models = ("local_base", "et_context", "temporal_residual", "hybrid")
    for mask in temporal_masks:
        for seed in (*SEEDS, "mean"):
            if seed == "mean":
                block = summary.loc[summary["mask"].eq(mask)].set_index("model")
            else:
                block = seed_table.loc[seed_table["mask"].eq(mask) & seed_table.seed.eq(seed)].set_index("model")
            row = {"mask": mask, "family": FAMILIES[mask], "seed": seed,
                   "n_unique_cells": int(block.loc["hybrid", "n_unique_cells"]),
                   "n_stations": int(block.loc["hybrid", "n_stations"])}
            for model in component_models:
                for metric in ("mae", "rmse", "r2", "log_mae", "q90_mae"):
                    row[f"{model}_{metric}"] = float(block.loc[model, metric])
            for baseline in ("local_base", "et_context", "temporal_residual"):
                row[f"hybrid_gain_vs_{baseline}_pct"] = 100 * (
                    row[f"{baseline}_mae"] - row["hybrid_mae"]
                ) / row[f"{baseline}_mae"]
            row["temporal_residual_gain_vs_local_base_pct"] = 100 * (
                row["local_base_mae"] - row["temporal_residual_mae"]
            ) / row["local_base_mae"]
            component_rows.append(row)
    pd.DataFrame(component_rows).to_csv(out / "component_metrics.csv", index=False)
    pooled_rows = []
    hybrid_pooled_mae = float(all_query_losses.hybrid_abs_error.mean())
    for model, column in (("rf_context", "rf_abs_error"), ("et_context", "et_abs_error"),
                          ("hybrid", "hybrid_abs_error")):
        pooled_mae = float(all_query_losses[column].mean())
        pooled_rows.append({
            "model": model, "mae": pooled_mae, "n_seeds": len(SEEDS),
            "n_family_cell_occurrences": len(all_query_losses),
            "n_distinct_cells_across_families": int(all_query_losses.cell.nunique()),
            "n_distinct_stations_across_families": int(all_query_losses.station.nunique()),
            "hybrid_gain_vs_model_pct": 100 * (pooled_mae - hybrid_pooled_mae) / pooled_mae,
        })
    pd.DataFrame(pooled_rows).to_csv(out / "pooled_metrics.csv", index=False)
    plot_results(summary, paired, out)
    sources.update(plot_spatial_publication_copies(out))
    manifest = {
        "analysis": "Saved DOC hybrid and matched RF/ExtraTrees context predictions",
        "training_performed": False, "seeds": list(SEEDS),
        "main_estimator": "Arithmetic mean of individual-seed cell-weighted metrics; not ensemble prediction error",
        "ci": "5,000 paired station bootstrap draws of seed-averaged cell absolute errors; cell weighting preserved",
        "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42,
        "tail": "At or above the train-label 90th percentile; fewer than20 unique cells flagged unstable",
        "source_sha256": {str(p): sha256(p) for p in sorted(sources)},
        "outputs_sha256": {str(p.name): sha256(p) for p in sorted(out.iterdir())
                           if p.is_file() and p.name not in {"manifest.json", "verification.md"}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "verification.md").write_text(
        "# DOC hybrid manuscript evidence\n\n"
        "All results were recomputed from saved per-seed predictions; no training occurred. "
        "Query membership and truth were checked against the dataset and original masks. "
        "Context predictions match the validation-selected source model exactly; RF uses "
        "the fixed `rf_default` from the same feature/split/seed run. Source sidecars match "
        "the actual dataset, mask, and prediction hashes. Temporal residual rows match the "
        "saved no-message residual expert, and the affine predictions are independently "
        "reconstructed from the saved coefficients. Main MAEs reproduce the existing "
        "five-seed affine report.\n\n"
        "## Estimands and interpretation\n\n"
        "- MAE, RMSE and R2 are means of the five seed-specific metrics, not metrics of an "
        "ensemble mean or median prediction. Cell and station counts are unique within each scenario.\n"
        "- Paired 95% intervals resample whole stations after averaging cell losses across "
        "seeds, recomputing cell-weighted MAE on each draw. They are descriptive intervals "
        "for one repeatedly used development mask per family, not independent external confirmation.\n"
        "- Random gaps and unmonitored stations use the context route exactly, so zero "
        "hybrid improvement there is a routing identity, not a measured neural increment.\n"
        "- The temporal residual expert is no-message. These gains establish value of the "
        "selected temporal residual combination, not an independent river-message contribution.\n"
        "- The hybrid combines complete context and residual-expert predictions in log-space. "
        "It is not numerically identical to adding a raw temporal correction to the context forest.\n"
        "- Final test inference uses train+val+context observations; fitting and validation "
        "inference use train+context. This matches the existing protocol.\n"
        "- Q90 is based only on train labels. Temporal tail samples are small; see the "
        "unique-cell counts and unstable flags rather than treating five seed repetitions "
        "as additional ecological samples.\n"
        "- `component_metrics.csv` compares the matched local ExtraTrees base (`local_pred` "
        "in the residual source file, in raw DOC units), context expert, complete temporal "
        "residual expert, and selected affine hybrid on identical query cells for each temporal "
        "family and seed, plus a mean-seed row. Local-to-residual improvement measures the "
        "increment of the full deep residual branch; it is not an attention or graph ablation.\n"
        "- `pooled_metrics.csv` weights family-cell occurrences equally after averaging seed "
        "losses. The 11,046 occurrences overlap across scenarios and are not 11,046 independent "
        "or distinct observations; distinct counts are reported separately.\n\n"
        "## Reproduction\n\n"
        "`uv run --no-sync python scripts/analyze_doc_hybrid_manuscript.py`\n\n"
        "Figure: `doc_hybrid_performance.pdf` (vector) and `.png` (300dpi). "
        "Panel a error bars are seed SD; panel b intervals are station bootstrap CIs. "
        "`doc_spatial_support_publication` and `doc_station_response_publication` are "
        "6.5-inch publication-sized copies of verified spatial evidence, with 8–9pt "
        "type at full-width inclusion; historical spatial figures and numbers are unchanged.\n"
    )
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
