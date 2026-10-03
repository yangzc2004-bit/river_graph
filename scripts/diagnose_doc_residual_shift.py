"""Describe source OOF versus validation DOC context residuals; no test labels.

Positive residual means underprediction. This reused-source-validation
diagnostic neither fits models nor chooses weights, strata or candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_unified_doc_spatial import sha256
from analyze_unified_doc_spatial_v2 import SEEDS, SPLITS
from analyze_unified_doc_spatial_v3 import read_bound_json
from run_unified_doc_spatial import verify_files
from run_unified_doc_spatial_v2 import read_source

from river_graph.experiments.unified_spatial_protocol import support_query_cells

ROOT = Path("experiments/phase4_transfer/doc_flow_interaction_v1")
ROLES = ("source_oof", "source_validation")
STRATA = ("all", "flow_observed", "flow_unobserved", "upstream_support", "no_upstream_support")


def upstream_fraction(visible, edges):
    """Current-month visible DOC fraction on distinct directed upstream edges."""
    n, _ = visible.shape
    counts = np.zeros(visible.shape, dtype=float)
    unique = np.unique(np.asarray(edges, dtype=np.int64), axis=1)
    if unique.size:
        src, dst = unique
        np.add.at(counts, dst, visible[src].astype(float))
        degree = np.bincount(dst, minlength=n)
        counts /= np.maximum(degree, 1)[:, None]
    return counts


def source_support(split, folds, shape, edges):
    """Use each held station's actual OOF visibility, not full-fit source support."""
    n, months = shape
    visible = np.zeros(n * months, dtype=bool)
    visible[np.asarray(split["train"], dtype=int)] = True
    if len(split["context"]):
        raise ValueError("This diagnostic expects the saved spatial protocol with train-only visibility")
    visible = visible.reshape(shape)
    all_support = upstream_fraction(visible, edges)
    oof_support = np.full(shape, np.nan)
    held_total = []
    for record in folds:
        held = np.asarray(record["held_station_ids"], dtype=int)
        view = visible.copy()
        view[held] = False
        oof_support[held] = upstream_fraction(view, edges)[held]
        held_total.extend(held.tolist())
    source_ids = np.unique(np.asarray(split["train"], dtype=int) // months)
    np.testing.assert_array_equal(np.sort(held_total), source_ids)
    if not np.isfinite(oof_support.ravel()[split["train"]]).all():
        raise ValueError("Source fold support does not cover all training cells")
    return oof_support, all_support


def read_cells(root):
    frames, sizes, sources = [], [], []
    for split_seed in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split_seed}_seed{seed}"
            config = json.loads((run / "config.json").read_text())
            verify_files(run, "complete.json", config)
            if (config["split_seed"], config["seed"]) != (split_seed, seed):
                raise ValueError("Run identity differs")
            source_run, oof_run = Path(config["source_run"]), Path(config["oof_run"])
            for prefix, path in (("source", source_run), ("oof", oof_run)):
                if sha256(path / "complete.json") != config[f"{prefix}_completion_hash"]:
                    raise ValueError(f"Changed {prefix} dependency")
            source_config, _, dataset, split, full = read_source(source_run)
            oof_config = json.loads((oof_run / "config.json").read_text())
            completion = verify_files(oof_run, "complete.json", oof_config)
            folds = read_bound_json(oof_run, "oof_folds.json", completion, sources)
            with np.load(oof_run / "source_oof.npz", allow_pickle=False) as saved:
                oof_z = saved["pred_z"].copy()
            shape = tuple(dataset["y"].shape)
            months = shape[1]
            source_cells = np.asarray(split["train"], dtype=int)
            _, validation_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
            if (np.intersect1d(source_cells, split["test"]).size
                    or np.intersect1d(validation_cells, split["test"]).size
                    or not np.isin(validation_cells, split["val"]).all()):
                raise ValueError("Diagnostic cells overlap outer test or leave source/validation roles")
            oof_support, val_support = source_support(split, folds, shape, np.asarray(dataset["edge_index"]))
            flow_observed = np.asarray(dataset["x_mask"])[..., 1].astype(bool)
            threshold = float(config["q90_threshold_train"])
            if threshold != float(source_config["q90_threshold_train"]):
                raise ValueError("Source Q90 threshold differs")
            # Only these source/validation indices are ever read from the DOC label tensor.
            truth = dataset["y"].reshape(-1)
            for role, cells, base, support in (
                ("source_oof", source_cells, np.maximum(0, np.expm1(oof_z.ravel()[source_cells])), oof_support),
                ("source_validation", validation_cells, full.context_pred.to_numpy()[validation_cells], val_support),
            ):
                y = np.asarray(truth[cells], dtype=float)
                if not np.isfinite(y).all() or not np.isfinite(base).all() or (y < 0).any() or (base < 0).any():
                    raise ValueError("Invalid selected DOC labels or base predictions")
                identity = full.iloc[cells][["cell", "station", "month"]].copy()
                identity["split_seed"], identity["seed"], identity["role"] = split_seed, seed, role
                identity["y_true"], identity["base_pred"] = y, base
                identity["residual"] = y - base
                identity["is_q90"] = y >= threshold
                identity["q90_threshold_train"] = threshold
                identity["flow_observed"] = flow_observed.ravel()[cells]
                identity["upstream_fraction"] = support.ravel()[cells]
                frames.append(identity)
            full_stations = len(np.unique(source_cells // months))
            for fold in folds:
                sizes.append({"split_seed": split_seed, "seed": seed, "fold": fold["fold"],
                              "oof_fit_stations": len(fold["training_station_ids"]),
                              "oof_fit_cells": fold["n_train_cells"],
                              "held_source_stations": len(fold["held_station_ids"]),
                              "held_source_cells": fold["n_oof_cells"],
                              "full_source_fit_stations": full_stations, "full_source_fit_cells": len(source_cells),
                              "validation_query_cells": len(validation_cells),
                              "validation_query_stations": len(np.unique(validation_cells // months))})
            for path in (run / "config.json", run / "complete.json", source_run / "complete.json",
                         source_run / "config.json", source_run / "full_grid.parquet",
                         oof_run / "complete.json", oof_run / "config.json", oof_run / "source_oof.npz",
                         Path(source_config["dataset_path"]), Path(source_config["mask_path"])):
                sources.append({"path": str(path), "sha256": sha256(path)})
    frame = pd.concat(frames, ignore_index=True)
    if set(frame.role) != set(ROLES) or frame.duplicated(["split_seed", "seed", "role", "cell"]).any():
        raise ValueError("Invalid diagnostic population")
    return frame, pd.DataFrame(sizes), sources


def distribution_tables(frame):
    rows = []
    for (split, seed, role), group in frame.groupby(["split_seed", "seed", "role"]):
        selections = {"all": np.ones(len(group), dtype=bool),
                      "flow_observed": group.flow_observed.to_numpy(),
                      "flow_unobserved": ~group.flow_observed.to_numpy(),
                      "upstream_support": group.upstream_fraction.to_numpy() > 0,
                      "no_upstream_support": group.upstream_fraction.to_numpy() == 0}
        for stratum in STRATA:
            for region in ("overall", "q90", "nontail"):
                selected = selections[stratum].copy()
                if region != "overall":
                    selected &= group.is_q90.to_numpy() if region == "q90" else ~group.is_q90.to_numpy()
                sub = group.loc[selected]
                record = {"split_seed": split, "seed": seed, "role": role, "stratum": stratum,
                          "region": region, "n_cells": len(sub), "n_stations": sub.station.nunique(),
                          "unstable": len(sub) < 20, "q90_threshold_train": group.q90_threshold_train.iloc[0]}
                for column in ("y_true", "base_pred", "residual"):
                    values = sub[column].to_numpy()
                    record[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
                    record[f"{column}_sd"] = float(values.std()) if len(values) else np.nan
                    for q in (10, 50, 90, 95):
                        record[f"{column}_q{q}"] = float(np.percentile(values, q)) if len(values) else np.nan
                record["mae"] = float(sub.residual.abs().mean()) if len(sub) else np.nan
                record["underprediction_fraction"] = float((sub.residual > 0).mean()) if len(sub) else np.nan
                record["q90_fraction"] = float(sub.is_q90.mean()) if len(sub) else np.nan
                rows.append(record)
    runs = pd.DataFrame(rows)
    metrics = [name for name in runs if name.startswith(("y_true_", "base_pred_", "residual_"))]
    metrics += ["mae", "underprediction_fraction", "q90_fraction"]
    partitions = runs.groupby(["split_seed", "role", "stratum", "region"], as_index=False).agg(
        **{name: (name, "mean") for name in metrics}, n_seeds=("seed", "size"),
        n_nonempty_seeds=("n_cells", lambda values: int((values > 0).sum())),
        mean_n_cells=("n_cells", "mean"), min_n_cells=("n_cells", "min"), max_n_cells=("n_cells", "max"),
        mean_n_stations=("n_stations", "mean"), unstable_any_seed=("unstable", "any"))
    partitions.loc[partitions.n_nonempty_seeds.ne(len(SEEDS)), metrics] = np.nan
    summary = partitions.groupby(["role", "stratum", "region"], as_index=False).agg(
        **{name: (name, "mean") for name in metrics}, n_partitions=("split_seed", "size"),
        n_complete_partitions=("n_nonempty_seeds", lambda values: int((values == len(SEEDS)).sum())),
        mean_n_cells=("mean_n_cells", "mean"), mean_n_stations=("mean_n_stations", "mean"),
        unstable_any=("unstable_any_seed", "any"))
    summary.loc[summary.n_complete_partitions.ne(len(SPLITS)), metrics] = np.nan
    return runs, partitions, summary


def write_report(out, summary, partitions, sizes):
    lines = ["# DOC context residual shift: source OOF and source-validation", "",
             "This diagnostic compares the actual station-blocked OOF environmental predictions used",
             "for residual training with full-source environmental predictions on fixed validation",
             "query cells. It reads no outer-test query labels and fits no model. Validation has already",
             "been used for checkpoint and adapter selection, so these are selection-set diagnostics.", "",
             "Residual = observed DOC − context prediction, in mg/L; positive values mean",
             "underprediction. Q90 is the original source-training threshold. The source and validation",
             "populations contain different stations and concentration distributions. OOF forests also",
             "use fewer source stations than the full-source validation predictor. These differences",
             "cannot isolate training-set size, station composition or missing process information as",
             "the cause. No in-sample source residual comparator is used.", "",
             "Seeds are averaged within partitions and partitions are weighted equally. Displayed",
             "means and quantiles are averages of per-run summaries, not pooled-sample quantiles.",
             "Exact per-run cell/station counts and partition summaries are saved separately.", "",
             "## Overall and concentration-specific distributions", "",
             "| Role | Region | Mean cells per partition | Mean DOC | Mean base | Mean residual | Median residual | Residual Q90 | MAE |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary[summary.stratum.eq("all")].itertuples():
        lines.append(f"| {row.role} | {row.region} | {row.mean_n_cells:.0f} | {row.y_true_mean:.3f} | "
                     f"{row.base_pred_mean:.3f} | {row.residual_mean:.3f} | {row.residual_q50:.3f} | "
                     f"{row.residual_q90:.3f} | {row.mae:.3f} |")
    lines += ["", "## Partition-level Q90 residuals", "",
              "| Partition | Role | Cells | Mean DOC | Mean base | Mean residual | MAE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in partitions[partitions.stratum.eq("all") & partitions.region.eq("q90")].itertuples():
        lines.append(f"| {row.split_seed} | {row.role} | {row.mean_n_cells:.0f} | {row.y_true_mean:.3f} | "
                     f"{row.base_pred_mean:.3f} | {row.residual_mean:.3f} | {row.mae:.3f} |")
    lines += ["", "## Fixed information strata: Q90", "",
              "| Stratum | Role | Mean cells | Mean DOC | Mean residual | MAE | Small group flag |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in summary[summary.stratum.ne("all") & summary.region.eq("q90")].itertuples():
        lines.append(f"| {row.stratum} | {row.role} | {row.mean_n_cells:.1f} | {row.y_true_mean:.3f} | "
                     f"{row.residual_mean:.3f} | {row.mae:.3f} | {bool(row.unstable_any)} |")
    lines += ["", "Flow strata use the current-month discharge observation flag. Upstream support",
              "means at least one currently visible DOC observation on a true upstream edge; zero",
              "also includes stations with no upstream neighbor. Source support uses each station's",
              "saved OOF fold with that entire fold hidden. Validation support uses training observations.",
              "The two stratification axes are marginal, not a new cross-classified experiment matrix.",
              "A group with fewer than 20 cells in any run is flagged; absent groups are not silently pooled.", "",
              "## Context training size", ""]
    for split, group in sizes.groupby("split_seed"):
        lines.append(f"- Partition {split}: OOF context fits use {group.oof_fit_stations.min()}–"
                     f"{group.oof_fit_stations.max()} stations and {group.oof_fit_cells.min():,}–"
                     f"{group.oof_fit_cells.max():,} labels; full-source context uses "
                     f"{group.full_source_fit_stations.iloc[0]} stations and "
                     f"{group.full_source_fit_cells.iloc[0]:,} labels.")
    lines += ["", "## Scientific reading", ""]
    all_groups = summary[summary.stratum.eq("all")].set_index(["role", "region"])
    source, validation = all_groups.loc[("source_oof", "q90")], all_groups.loc[("source_validation", "q90")]
    lines.append(f"Both populations' high-DOC distributions are shown explicitly: source Q90 mean "
                 f"DOC {source.y_true_mean:.3f} and mean residual {source.residual_mean:.3f}; validation "
                 f"Q90 mean DOC {validation.y_true_mean:.3f} and mean residual {validation.residual_mean:.3f} mg/L.")
    tail_parts = partitions[partitions.stratum.eq("all") & partitions.region.eq("q90")]
    lines += ["", "The partition pattern matters:"]
    for split in SPLITS:
        pair = tail_parts[tail_parts.split_seed.eq(split)].set_index("role")
        src, val = pair.loc["source_oof"], pair.loc["source_validation"]
        lines.append(f"- Partition {split}: validation minus source mean tail residual = "
                     f"{val.residual_mean - src.residual_mean:+.3f} mg/L; mean observed DOC changes by "
                     f"{val.y_true_mean - src.y_true_mean:+.3f}, while mean context prediction changes by "
                     f"{val.base_pred_mean - src.base_pred_mean:+.3f} mg/L. These are an arithmetic "
                     "decomposition of the residual shift, not independent causal explanations.")
    lines += ["", "A large positive tail residual establishes underprediction by the context base,",
              "but does not establish that a larger tail weight will improve transfer. Mean residual",
              "differences accompany different target distributions and fitted context populations.",
              "The observed validation early-stopping behavior and the fixed tail/non-tail comparisons",
              "should be read alongside these distributions before changing the objective. This",
              "diagnostic makes no model, weight, threshold or station selection.", ""]
    (out / "residual_shift_findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    cells, sizes, sources = read_cells(args.root)
    runs, partitions, summary = distribution_tables(cells)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("runs", runs), ("partitions", partitions), ("summary", summary), ("training_sizes", sizes)):
        frame.to_csv(out / f"residual_shift_{name}.csv", index=False)
    write_report(out, summary, partitions, sizes)
    for path in (Path(__file__), Path("scripts/run_unified_doc_spatial_v2.py"),
                 Path("src/river_graph/experiments/unified_spatial_protocol.py")):
        sources.append({"path": str(path), "sha256": sha256(path)})
    unique = {row["path"]: row for row in sources}
    (out / "residual_shift_sources.json").write_text(json.dumps(list(unique.values()), indent=2) + "\n")
    print(json.dumps({"output": str(out), "n_rows": len(cells), "roles": sorted(cells.role.unique()),
                      "outer_test_labels_used": False, "fits_performed": 0}, indent=2))


if __name__ == "__main__":
    main()
