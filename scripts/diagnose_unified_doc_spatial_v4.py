"""Describe remaining DOC errors on reused source-validation queries only.

No outer-test prediction table or outer-query label is used. Saved adapters and
representations are replayed against held-station fusion coefficients. This is
a selection-set diagnostic, not an independent performance estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_unified_doc_spatial import digest, verify_files
from run_unified_doc_spatial_v2 import read_source
from run_unified_doc_spatial_v3 import load_fusion

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v4")
SPLITS = (142, 143, 144)
SEEDS = (42, 43, 44)
ARMS = {
    "fusion_base_k0": ("constant", 0),
    "fusion_constant_k5": ("constant", 5),
    "fusion_gru_tuned_anchor_k5": ("gru_tuned_anchor", 5),
    "fusion_tree_episodic_k5": ("tree_episodic", 5),
}
LABELS = {
    "fusion_base_k0": "Unadapted fusion (K=0)",
    "fusion_constant_k5": "Constant correction (K=5)",
    "fusion_gru_tuned_anchor_k5": "Updated GRU shape (K=5)",
    "fusion_tree_episodic_k5": "Tree shape (K=5)",
}


def decompose(frame):
    """Exact cell-weighted residual sum-of-squares partition, for one run/arm."""
    out = {"n_query": len(frame), "n_stations": frame.station.nunique(),
           "q90_n": int(frame.is_q90.sum()), "q90_cell_fraction": frame.is_q90.mean()}
    counts = frame.groupby(["station", "calendar_month"]).cell.transform("size")
    groups = frame.groupby(["station", "calendar_month"]).size()
    out["calendar_singleton_cell_fraction"] = (counts == 1).mean()
    out["calendar_singleton_group_fraction"] = (groups == 1).mean()
    for scale, residual in (("raw", "residual"), ("log", "log_residual")):
        r = frame[residual].to_numpy()
        station_mean = frame.groupby("station")[residual].transform("mean").to_numpy()
        calendar_mean = frame.groupby(["station", "calendar_month"])[residual].transform("mean").to_numpy()
        mse = np.mean(r**2)
        bias = np.mean(station_mean**2)
        within = np.mean((r - station_mean)**2)
        calendar = np.mean((calendar_mean - station_mean)**2)
        remaining = np.mean((r - calendar_mean)**2)
        np.testing.assert_allclose(mse, bias + within, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(within, calendar + remaining, rtol=1e-12, atol=1e-12)
        tail = r[frame.is_q90.to_numpy()]
        tail_mse = np.sum(tail**2) / len(r)
        tail_absolute = np.sum(np.abs(tail)) / len(r)
        out.update({f"{scale}_mae": np.mean(np.abs(r)), f"{scale}_mse": mse,
                    f"{scale}_station_bias_mse": bias, f"{scale}_within_station_mse": within,
                    f"{scale}_calendar_pattern_mse": calendar,
                    f"{scale}_within_calendar_mse": remaining,
                    f"{scale}_q90_error_mass": tail_mse,
                    f"{scale}_q90_absolute_error_mass": tail_absolute,
                    f"{scale}_q90_absolute_error_share": tail_absolute / np.mean(np.abs(r)),
                    f"{scale}_q90_mean_residual": np.mean(tail) if len(tail) else np.nan,
                    f"{scale}_bias_share": bias / mse if mse else 0,
                    f"{scale}_within_share": within / mse if mse else 0,
                    f"{scale}_q90_error_share": tail_mse / mse if mse else 0})
    return out


def station_covariates(dataset, full, support, query, months):
    """Covariates use feature visibility, calendar dates and the support design."""
    dates = np.asarray(dataset["months"], dtype="datetime64[M]")
    calendar = dates.astype(np.int64)
    month_number = calendar % 12 + 1
    hydro = np.asarray(dataset["x_mask"], dtype=float)
    rows = []
    for station_id in np.unique(query // months):
        q = query[query // months == station_id]
        s = support[support // months == station_id]
        qm, sm = q % months, s % months
        gaps = np.abs(calendar[qm, None] - calendar[sm][None, :]).min(axis=1)
        if len(s) != 5:
            raise ValueError("diagnostic requires five reserved supports per station")
        rows.append({
            "station": str(full.iloc[q[0]].station), "station_index": station_id,
            "n_query": len(q), "support_count": len(s),
            "support_span_months": int(np.ptp(calendar[sm])),
            "support_unique_calendar_months": len(np.unique(month_number[sm])),
            "support_unique_seasons": len(np.unique((month_number[sm] % 12) // 3)),
            "nearest_support_gap_mean_months": float(gaps.mean()),
            "nearest_support_gap_median_months": float(np.median(gaps)),
            "nearest_support_gap_max_months": int(gaps.max()),
            "temperature_observed_fraction": float(hydro[station_id, qm, 0].mean()),
            "discharge_observed_fraction": float(hydro[station_id, qm, 1].mean()),
            "hydro_observed_fraction": float(hydro[station_id, qm].mean()),
            "hydro_timeline_observed_fraction": float(hydro[station_id].mean()),
            "ecological_novelty": float(full.iloc[q].ecological_novelty.mean()),
            "upstream_support": float(full.iloc[q].upstream_support.mean()),
        })
    return pd.DataFrame(rows)


def read_validation(run):
    config = json.loads((run / "config.json").read_text())
    verify_files(run, "complete.json", config)
    source = Path(config["source_run"])
    _, _, dataset, split, full = read_source(source)
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    support, query = support_query_cells(split, target_role="val", k=5, n_months=months)
    _, zero_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    np.testing.assert_array_equal(query, zero_query)
    if (not np.isin(np.r_[support, query], split["val"]).all()
            or np.intersect1d(support, query).size
            or np.intersect1d(np.r_[support, query], split["test"]).size):
        raise ValueError("only disjoint source-validation support/query cells are permitted")
    # Only validation labels are extracted; all other positions stay NaN.
    truth = np.full(np.prod(shape), np.nan)
    val = np.asarray(split["val"], dtype=np.int64)
    truth[val] = np.asarray(dataset["y"]).reshape(-1)[val]
    context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
    fusion_run = Path(config["fusion_run"])
    _, _, held_fusion = load_fusion(fusion_run, context, temporal, split, months)
    states = json.loads((run / "adapters.json").read_text())
    covariates = station_covariates(dataset, full, support, query, months)
    dates = np.asarray(dataset["months"], dtype="datetime64[M]")
    threshold = config["q90_threshold_train"]
    frames, metrics, station_metrics = [], [], []
    with np.load(run / "representations.npz", allow_pickle=False) as shapes:
        for name, (head, k) in ARMS.items():
            selected_support = support if k else np.empty(0, dtype=np.int64)
            adapter = SupportShapeAdapter.from_dict(states[f"fusion_{head}"])
            basis = shapes[head]
            prediction = adapter.adapt(
                held_fusion[query], query, held_fusion[selected_support], selected_support,
                truth[selected_support], query_basis=basis[query],
                support_basis=basis[selected_support], k=k,
            )
            f = full.iloc[query][["cell", "station", "month"]].copy()
            f["station"] = f.station.astype(str)
            f["split_seed"], f["seed"] = config["split_seed"], config["seed"]
            f["model_name"], f["k"] = name, k
            f["visibility_role"] = "source_validation_query"
            f["y_true"], f["y_pred"] = truth[query], prediction
            f["residual"] = prediction - truth[query]
            f["log_residual"] = np.log1p(prediction) - np.log1p(truth[query])
            f["calendar_month"] = dates[query % months].astype(np.int64) % 12 + 1
            f["is_q90"] = truth[query] >= threshold
            f["q90_threshold_train"] = threshold
            if not np.isfinite(f[["y_true", "y_pred", "residual", "log_residual"]].to_numpy()).all():
                raise ValueError("nonfinite validation residual")
            identity = {"split_seed": config["split_seed"], "seed": config["seed"], "model_name": name, "k": k}
            metrics.append({**identity, **decompose(f)})
            for station, group in f.groupby("station"):
                row = {**identity, "station": station,
                       "residual_mean": group.residual.mean(),
                       "log_residual_mean": group.log_residual.mean(),
                       "raw_mae": group.residual.abs().mean(),
                       "log_mae": group.log_residual.abs().mean(),
                       "raw_mse": np.mean(group.residual.to_numpy()**2),
                       "log_mse": np.mean(group.log_residual.to_numpy()**2),
                       "q90_n": int(group.is_q90.sum())}
                station_metrics.append(row)
            frames.append(f)
    stations = pd.DataFrame(station_metrics).merge(covariates, on="station", validate="many_to_one")
    sources = [run / "complete.json", run / "config.json", run / "adapters.json",
               run / "representations.npz", source / "full_grid.parquet",
               fusion_run / "fusion.json", Path(config["dataset_path"]), Path(config["mask_path"])]
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics), stations, sources


def summarize(metrics):
    numeric = metrics.select_dtypes(include="number").columns.difference(["split_seed", "seed", "k"])
    split = metrics.groupby(["split_seed", "model_name", "k"])[numeric].mean().reset_index()
    summary = split.groupby(["model_name", "k"])[numeric].mean().reset_index()
    # Shares are ratios of equally weighted component masses, not mean ratios.
    for scale in ("raw", "log"):
        for key, mass in (("bias_share", "station_bias_mse"), ("within_share", "within_station_mse"),
                          ("q90_error_share", "q90_error_mass")):
            summary[f"{scale}_{key}"] = summary[f"{scale}_{mass}"] / summary[f"{scale}_mse"]
        summary[f"{scale}_q90_absolute_error_share"] = (
            summary[f"{scale}_q90_absolute_error_mass"] / summary[f"{scale}_mae"])
    return split, summary


def paired_effects(metrics):
    rows = []
    for reference in ("fusion_base_k0", "fusion_constant_k5"):
        baseline = metrics[metrics.model_name.eq(reference)].set_index(["split_seed", "seed"])
        for candidate in ARMS:
            if candidate == reference or (reference.endswith("k5") and candidate.endswith("k0")):
                continue
            current = metrics[metrics.model_name.eq(candidate)].set_index(["split_seed", "seed"])
            for metric in ("raw_mae", "log_mae", "raw_mse", "log_mse",
                           "raw_station_bias_mse", "raw_within_station_mse",
                           "log_station_bias_mse", "log_within_station_mse"):
                delta = current[metric] - baseline[metric]
                partition_delta = delta.groupby("split_seed").mean()
                a = current[metric].groupby("split_seed").mean().mean()
                b = baseline[metric].groupby("split_seed").mean().mean()
                rows.append({"candidate": candidate, "reference": reference, "metric": metric,
                             "candidate_mean": a, "reference_mean": b, "delta": a - b,
                             "relative_reduction_pct": 100 * (b - a) / b,
                             "improved_partitions": int((partition_delta < 0).sum()),
                             "improved_runs": int((delta < 0).sum()), "n_partitions": 3, "n_runs": 9})
    return pd.DataFrame(rows)


def station_correlations(stations):
    covariates = ("nearest_support_gap_mean_months", "support_span_months", "support_unique_seasons",
                  "hydro_observed_fraction", "ecological_novelty", "n_query")
    # Average seeds first: repeated predictions do not create more stations.
    names = ["raw_mae", "log_mae", *covariates]
    panel = stations.groupby(["split_seed", "model_name", "station"])[names].mean().reset_index()
    rows = []
    for (split, model), group in panel.groupby(["split_seed", "model_name"]):
        for metric in ("raw_mae", "log_mae"):
            for covariate in covariates:
                varying = group[covariate].nunique() > 1 and group[metric].nunique() > 1
                rho = group[covariate].rank().corr(group[metric].rank()) if varying else np.nan
                rows.append({"split_seed": split, "model_name": model, "metric": metric,
                             "covariate": covariate, "spearman_rho": rho, "n_stations": len(group),
                             "status": "descriptive" if varying else "constant_covariate_or_metric"})
    return pd.DataFrame(rows)


def markdown(summary, effects, correlations, queries):
    by_model = summary.set_index("model_name")
    lines = ["# Where DOC reconstruction error remains", "",
             ("This is a **reused source-validation selection diagnostic**. It replays saved adapters "
             "on validation support/query cells using held-station fusion coefficients. These validation "
             "labels previously selected models, checkpoints and adapter settings. No outer-test query "
             "label or outer-test prediction table is used, and no model is fitted or selected here."), "",
             (f"The panel contains nine runs, three station partitions and "
             f"{queries[['station', 'month']].drop_duplicates().shape[0]:,} distinct validation station-months "
             f"at {queries.station.nunique()} distinct stations. Repeated seeds/partitions are not independent samples. "
             "Run metrics average cells, then seeds within partition, then the three partitions equally."), "",
             "## Exact residual decomposition", "",
             ("For residual e = prediction − observation, MSE = cell-weighted mean(station mean(e)²) "
             "+ cell-weighted mean((e − station mean(e))²). This is an exact accounting identity, "
             "not a statement about causes or irreducible error."), "",
             "| Predictor | MAE (mg/L) | Raw MSE: station bias / within station | Log MSE: station bias / within station | Q90 share of raw squared / absolute error |",
             "|---|---:|---:|---:|---:|"]
    for name in ARMS:
        r = by_model.loc[name]
        lines.append(f"| {LABELS[name]} | {r.raw_mae:.4f} | {100*r.raw_bias_share:.1f}% / "
                     f"{100*r.raw_within_share:.1f}% | {100*r.log_bias_share:.1f}% / "
                     f"{100*r.log_within_share:.1f}% | {100*r.raw_q90_error_share:.1f}% / "
                     f"{100*r.raw_q90_absolute_error_share:.1f}% |")
    r = by_model.loc["fusion_gru_tuned_anchor_k5"]
    dominant = "within-station temporal variation" if r.raw_within_share > .5 else "station mean bias"
    lines += ["", (f"The remaining raw squared error in the updated-GRU adapter is dominated by **{dominant}**. "
              f"Its within-station fraction is {100*r.raw_within_share:.1f}% in raw space and "
              f"{100*r.log_within_share:.1f}% in log space. Source-training-Q90 cells make up "
              f"{100*r.q90_cell_fraction:.1f}% of validation queries but account for "
              f"{100*r.raw_q90_error_share:.1f}% of raw squared error "
              f"({100*r.log_q90_error_share:.1f}% of log squared error). "
              f"They contribute {100*r.raw_q90_absolute_error_share:.1f}% of raw absolute error "
              f"and {100*r.log_q90_absolute_error_share:.1f}% of log absolute error. "
              f"Their signed mean residual is {r.raw_q90_mean_residual:.3f} mg/L "
              f"({r.log_q90_mean_residual:.3f} in log1p space); negative means underprediction. "
              "Absolute-error shares are directly relevant to MAE, whereas squared-error shares "
              "give greater weight to extreme residuals."), "",
              "## What adaptation changes on the selection set", "",
              "| Candidate versus constant correction | Raw MAE reduction | Station-bias MSE reduction | Within-station MSE reduction |",
              "|---|---:|---:|---:|"]
    for name in ("fusion_gru_tuned_anchor_k5", "fusion_tree_episodic_k5"):
        e = effects[effects.candidate.eq(name) & effects.reference.eq("fusion_constant_k5")].set_index("metric")
        lines.append(f"| {LABELS[name]} | {e.loc['raw_mae', 'relative_reduction_pct']:.2f}% | "
                     f"{e.loc['raw_station_bias_mse', 'relative_reduction_pct']:.2f}% | "
                     f"{e.loc['raw_within_station_mse', 'relative_reduction_pct']:.2f}% |")
    lines += ["", ("Positive reductions favor the candidate. These differences describe the reused validation "
              "set; they do not replace the held-out comparison and do not justify another hyperparameter choice."), "",
              "## Calendar pattern and observation coverage", "",
              (f"A further descriptive split assigns {100*r.raw_calendar_pattern_mse/r.raw_mse:.1f}% of total "
              f"raw MSE to differences among each station's calendar-month residual means, and "
              f"{100*r.raw_within_calendar_mse/r.raw_mse:.1f}% to variation remaining within those groups. "
              f"However, {100*r.calendar_singleton_group_fraction:.1f}% of station×calendar-month groups "
              f"are singletons ({100*r.calendar_singleton_cell_fraction:.1f}% of cells). This in-sample "
              "grouping is descriptive; it cannot identify a learnable seasonal fraction or an irreducible ceiling."), "",
              ("Station diagnostics retain the five-support date span, represented calendar months and "
              "meteorological seasons (DJF/MAM/JJA/SON), nearest-support gaps, query/timeline hydro visibility "
              "and ecological novelty. Correlations average seeds within each partition/station and use "
              "Spearman ranks without significance tests."), "",
              "| Covariate versus updated-GRU log MAE | Split 142 | Split 143 | Split 144 |",
              "|---|---:|---:|---:|"]
    chosen = correlations[correlations.model_name.eq("fusion_gru_tuned_anchor_k5") & correlations.metric.eq("log_mae")]
    for name, group in chosen.groupby("covariate", sort=False):
        values = group.set_index("split_seed").spearman_rho
        cells = [f"{values.loc[s]:.2f}" if np.isfinite(values.loc[s]) else "not identifiable" for s in SPLITS]
        lines.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines += ["", ("These associations are descriptive and may reflect record length, station DOC variability "
              "or other differences; they do not establish that a missing covariate caused an error."), "",
              "## Research implication", "",
              ("Prioritize the dominant component and its high-value records when reviewing the existing "
              "data: inspect timing and hydro coverage of large residuals, then assess whether the present "
              "monthly inputs represent those events. This diagnostic does not establish that a larger "
              "network, more observations or higher-frequency drivers would necessarily solve the remaining error. "
              "It provides a concrete target for the next scientific question while preserving the current model."), "",
              "## Reproduction", "",
              ("Run `uv run python scripts/diagnose_unified_doc_spatial_v4.py`. "
              "`validation_queries.parquet` contains only source-validation queries. "
              "`run_decomposition.csv` and `decomposition.csv` retain exact masses/shares; "
              "`station_diagnostics.csv`, `paired_effects.csv` and `covariate_correlations.csv` retain "
              "the descriptive evidence. Source hashes are in `sources.json`."), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root / "diagnostics"
    frames, metrics, stations, sources = [], [], [], set()
    for split in SPLITS:
        for seed in SEEDS:
            run = args.root / "runs" / f"split{split}_seed{seed}"
            f, m, s, paths = read_validation(run)
            if not f.split_seed.eq(split).all() or not f.seed.eq(seed).all():
                raise ValueError("run identities differ")
            frames.append(f)
            metrics.append(m)
            stations.append(s)
            sources.update(paths)
    queries = pd.concat(frames, ignore_index=True)
    run_metrics = pd.concat(metrics, ignore_index=True)
    station_metrics = pd.concat(stations, ignore_index=True)
    split_metrics, summary = summarize(run_metrics)
    effects = paired_effects(run_metrics)
    correlations = station_correlations(station_metrics)
    output.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(output / "validation_queries.parquet", index=False)
    for name, table in (("run_decomposition", run_metrics), ("split_decomposition", split_metrics),
                        ("decomposition", summary), ("station_diagnostics", station_metrics),
                        ("paired_effects", effects), ("covariate_correlations", correlations)):
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "diagnostic.md").write_text(markdown(summary, effects, correlations, queries))
    sources.add(Path(__file__))
    sources.update(Path(__file__).with_name(name) for name in (
        "run_unified_doc_spatial.py", "run_unified_doc_spatial_v2.py", "run_unified_doc_spatial_v3.py"))
    (output / "sources.json").write_text(json.dumps({
        "role": "reused source-validation selection diagnostic; no outer-test query labels",
        "n_runs": 9, "residual_sign": "prediction_minus_observation",
        "weighting": "cell-weighted per run, mean seeds within partition, mean partitions",
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(sources)],
    }, indent=2) + "\n")
    run_configs = [json.loads((args.root / "runs" / f"split{split}_seed{seed}" / "config.json").read_text())
                   for split in SPLITS for seed in SEEDS]
    dataset_hashes = {item["dataset_hash"] for item in run_configs}
    if len(dataset_hashes) != 1:
        raise ValueError("validation diagnostic requires a common dataset")
    masks = {str(item["split_seed"]): item["mask_hash"] for item in run_configs}
    config = {"artifact": "derived_source_validation_predictions", "models": ARMS,
              "splits": SPLITS, "seeds": SEEDS, "visibility_role": "source_validation_query",
              "source_config_hashes": [digest(item) for item in run_configs],
              "diagnostic_script_sha256": sha256_file(__file__)}
    (output / "validation_queries.meta.json").write_text(json.dumps({
        "config": config, "config_hash": digest(config),
        "dataset_hash": next(iter(dataset_hashes)), "mask_hash": digest(masks),
        "mask_hash_kind": "aggregate_split_to_mask_content_hash", "mask_hashes_by_split": masks,
        "source_manifest_sha256": sha256_file(output / "sources.json"),
        "runtime_snapshot_hashes": sorted({item["runtime_snapshot_hash"] for item in run_configs}),
        "prediction_sha256": sha256_file(output / "validation_queries.parquet"),
        "rows": len(queries), "role": "reused source-validation selection diagnostic",
    }, indent=2) + "\n")
    print(output / "diagnostic.md")


if __name__ == "__main__":
    main()
