"""Analyze new-partition DOC hybrid and station-calibration predictions.

The estimand is a mean over partitions, after averaging individual-seed
losses within each partition. Whole station IDs are bootstrapped jointly
across partitions, preserving within-partition cell weighting. This script
does not fit models, choose adapters, or change stored predictions.
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

DEFAULT_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation")
SPLITS = (142, 143, 144)
SEEDS = (42, 43, 44)
KS = (0, 1, 3, 5)
MODELS = ("extra_trees", "extra_trees_calibrated", "hybrid", "hybrid_calibrated")
LABELS = {
    "extra_trees": "ExtraTrees", "extra_trees_calibrated": "ExtraTrees + calibration",
    "hybrid": "Hybrid", "hybrid_calibrated": "Hybrid + calibration",
}
COLORS = {name: "#235B80" if name.startswith("extra") else "#D08043" for name in MODELS}
COMPARISONS = (
    ("hybrid_vs_et_k5_calibrated", "hybrid_calibrated", 5, "extra_trees_calibrated", 5),
    ("hybrid_vs_et_k0", "hybrid", 0, "extra_trees", 0),
    ("et_calibration_k5_vs_k0", "extra_trees_calibrated", 5, "extra_trees", 0),
    ("hybrid_calibration_k5_vs_k0", "hybrid_calibrated", 5, "hybrid", 0),
)
COMPARISON_LABELS = {
    "hybrid_vs_et_k5_calibrated": "Hybrid vs ExtraTrees\nBoth calibrated, K = 5",
    "hybrid_vs_et_k0": "Hybrid vs ExtraTrees\nNo target support, K = 0",
    "et_calibration_k5_vs_k0": "ExtraTrees calibration\nK = 5 vs K = 0",
    "hybrid_calibration_k5_vs_k0": "Hybrid calibration\nK = 5 vs K = 0",
}
METRICS = ("mae", "rmse", "r2", "log_mae", "q90_mae")
REQUIRED = {
    "split_seed", "seed", "k", "model_name", "cell", "station", "month",
    "y_true", "y_pred", "context_pred", "temporal_pred", "hybrid_pred",
    "calibration_delta", "ecological_novelty", "upstream_support",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_digest(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def metric_values(y: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    error = pred - y
    denominator = np.square(y - y.mean()).sum()
    tail = y >= threshold
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else np.nan,
        "log_mae": float(np.abs(np.log1p(pred) - np.log1p(y)).mean()),
        "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else np.nan,
        "q90_threshold_train": float(threshold), "q90_n": int(tail.sum()),
        "q90_unstable": bool(tail.sum() < 20), "n_query_cells": len(y),
    }


def validate_prediction_panel(frame: pd.DataFrame) -> None:
    """Reject unequal query populations or accidental changes of the base by K."""
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    keys = ["split_seed", "seed", "k", "model_name", "cell"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate model/query rows")
    numeric = ["y_true", "y_pred", "context_pred", "temporal_pred", "hybrid_pred",
               "calibration_delta", "ecological_novelty", "upstream_support"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite prediction, truth or diagnostic")
    if (frame[["y_true", "y_pred", "context_pred", "temporal_pred", "hybrid_pred"]] < 0).any().any():
        raise ValueError("DOC concentrations must be nonnegative")
    if frame[["station", "month"]].isna().any().any():
        raise ValueError("station/month identity is absent")
    if not frame.upstream_support.between(0, 1).all():
        raise ValueError("upstream_support must be a fraction in [0, 1]")
    identities = frame.groupby("cell")[["station", "month", "y_true"]].nunique()
    if (identities != 1).any().any():
        raise ValueError("global station-month identity/truth changes across partitions")
    for split, sub in frame.groupby("split_seed"):
        reference = None
        for (seed, k, model), part in sub.groupby(["seed", "k", "model_name"]):
            identity = part.sort_values("cell")[["cell", "station", "month", "y_true"]].reset_index(drop=True)
            if reference is None:
                reference = identity
            else:
                pd.testing.assert_frame_equal(reference, identity, check_dtype=False,
                                              check_exact=True, obj=f"fixed query split {split}")
        for seed, run in sub.groupby("seed"):
            combinations = set(map(tuple, run[["model_name", "k"]].drop_duplicates().to_numpy()))
            if combinations != {(m, k) for m in MODELS for k in KS}:
                raise ValueError(f"incomplete four-arm K panel: split {split}, seed {seed}")
            pivot = run.pivot(index="cell", columns=["model_name", "k"], values="y_pred")
            for model in ("extra_trees", "hybrid"):
                for k in KS:
                    np.testing.assert_array_equal(pivot[(model, 0)], pivot[(model, k)])
                np.testing.assert_allclose(pivot[(model, 0)], pivot[(model + "_calibrated", 0)],
                                           rtol=1e-12, atol=1e-12)


def load_predictions(root: Path, allow_partial: bool) -> tuple[pd.DataFrame, dict, list, dict]:
    expected = {(s, r) for s in SPLITS for r in SEEDS}
    frames, thresholds, sources, adapters, available, budgets = [], {}, [], {}, set(), []
    incomplete = []
    for run in sorted((root / "runs").glob("split*_seed*")):
        paths = [run / name for name in ("predictions.parquet", "config.json", "adapter.json")]
        completion = run / "complete.json"
        if not completion.is_file() or not all(path.is_file() for path in paths):
            incomplete.append(run.name)
            continue
        config = json.loads(paths[1].read_text())
        completed = json.loads(completion.read_text())
        if completed.get("config_hash") != config_digest(config):
            raise ValueError(f"completion configuration identity differs: {run}")
        consumed_hashes = {path.name: sha256(path) for path in paths}
        for name, content_hash in consumed_hashes.items():
            if completed.get("files", {}).get(name) != content_hash:
                raise ValueError(f"completed analysis input changed or unbound: {run / name}")
        frame = pd.read_parquet(paths[0])
        if frame.empty:
            raise ValueError(f"empty predictions: {run}")
        pairs = frame[["split_seed", "seed"]].drop_duplicates()
        if len(pairs) != 1:
            raise ValueError(f"multiple identities in {run}")
        pair = tuple(int(v) for v in pairs.iloc[0])
        if pair not in expected or pair in available:
            raise ValueError(f"unexpected or duplicate split/seed: {pair}")
        if run.name != f"split{pair[0]}_seed{pair[1]}":
            raise ValueError(f"directory and prediction identities differ: {run}")
        if (int(config["split_seed"]), int(config["seed"])) != pair:
            raise ValueError(f"configuration and prediction identities differ: {run}")
        if frame.cell.nunique() != int(config["query_cells"]):
            raise ValueError(f"configuration and query counts differ: {run}")
        threshold = float(config["q90_threshold_train"])
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"invalid training Q90 threshold: {run}")
        thresholds[pair] = threshold
        adapters[pair] = json.loads(paths[2].read_text())
        sources.extend({"path": str(path), "sha256": consumed_hashes[path.name]} for path in paths)
        sources.append({"path": str(completion), "sha256": sha256(completion)})
        budgets.append({"split_seed": pair[0], "seed": pair[1],
                        "n_estimators": config.get("n_estimators"),
                        "max_epochs": config.get("max_epochs"), "patience": config.get("patience")})
        frames.append(frame)
        available.add(pair)
    missing = sorted(expected - available)
    if missing and not allow_partial:
        raise ValueError(f"requires all nine completed runs; missing {missing}")
    status = {
        "complete": not missing, "available_runs": len(available), "expected_runs": 9,
        "missing_runs": [{"split_seed": s, "seed": r} for s, r in missing],
        "incomplete_directories": incomplete,
        "seeds_per_split": {str(s): sorted(r for ss, r in available if ss == s) for s in SPLITS},
        "training_budgets": budgets,
        "confirmation_budget": bool(budgets) and all(
            b["n_estimators"] == 300 and b["max_epochs"] == 20 and b["patience"] == 5 for b in budgets),
    }
    if not frames:
        return pd.DataFrame(), thresholds, sources, {"status": status, "adapters": adapters}
    data = pd.concat(frames, ignore_index=True)
    validate_prediction_panel(data)
    return data, thresholds, sources, {"status": status, "adapters": adapters}


def paired_cells(frame: pd.DataFrame, candidate: str, ck: int,
                 reference: str, rk: int) -> pd.DataFrame:
    keys = ["split_seed", "seed", "cell", "station", "month"]
    keep = keys + ["y_true", "y_pred", "ecological_novelty", "upstream_support"]
    cand = frame.loc[frame.model_name.eq(candidate) & frame.k.eq(ck), keep]
    ref = frame.loc[frame.model_name.eq(reference) & frame.k.eq(rk), keys + ["y_true", "y_pred"]]
    pair = cand.merge(ref, on=keys, validate="one_to_one", suffixes=("_candidate", "_reference"))
    if len(pair) != len(cand) or len(pair) != len(ref):
        raise ValueError("paired comparison loses query cells")
    np.testing.assert_array_equal(pair.y_true_candidate, pair.y_true_reference)
    pair["candidate_error"] = np.abs(pair.y_pred_candidate - pair.y_true_candidate)
    pair["reference_error"] = np.abs(pair.y_pred_reference - pair.y_true_reference)
    pair["delta_mae"] = pair.candidate_error - pair.reference_error
    return pair


def joint_station_bootstrap(pair: pd.DataFrame, draws: int = 5000, seed: int = 42) -> dict:
    """Joint station resampling; seed-mean cell loss, then equal split means.

    One multinomial station multiplicity is used in every partition. A station
    occurring in two partitions is therefore resampled together. Replicates
    with no sampled cell in a partition are redrawn, not silently averaged over
    fewer partitions. Seeds and partitions are not ecological replicates.
    """
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    cells = pair.groupby(["split_seed", "station", "cell"], as_index=False).agg(
        candidate_error=("candidate_error", "mean"), reference_error=("reference_error", "mean"))
    station = cells.groupby(["split_seed", "station"], as_index=False).agg(
        n=("cell", "size"), candidate_sum=("candidate_error", "sum"),
        reference_sum=("reference_error", "sum"))
    splits = sorted(station.split_seed.unique())
    ids = sorted(station.station.unique())
    matrices = {name: station.pivot(index="station", columns="split_seed", values=name)
                .reindex(index=ids, columns=splits).fillna(0).to_numpy(dtype=float)
                for name in ("n", "candidate_sum", "reference_sum")}
    counts = matrices["n"]
    base = float(np.mean(matrices["reference_sum"].sum(0) / counts.sum(0)))
    final = float(np.mean(matrices["candidate_sum"].sum(0) / counts.sum(0)))
    rng = np.random.default_rng(seed)
    delta_draws, gain_draws = [], []
    obtained, rejected, attempts = 0, 0, 0
    while obtained < draws:
        batch = min(512, draws - obtained)
        weight = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=batch)
        denominator = weight @ counts
        good = (denominator > 0).all(axis=1)
        rejected += int((~good).sum())
        attempts += batch
        if attempts > max(100000, draws * 100):
            raise ValueError("too few stations to bootstrap all partitions jointly")
        weight, denominator = weight[good], denominator[good]
        if not len(weight):
            continue
        reference_draw = np.mean((weight @ matrices["reference_sum"]) / denominator, axis=1)
        candidate_draw = np.mean((weight @ matrices["candidate_sum"]) / denominator, axis=1)
        delta_draws.append(candidate_draw - reference_draw)
        gain_draws.append(np.divide(100 * (reference_draw - candidate_draw), reference_draw,
                                   out=np.full(len(weight), np.nan), where=reference_draw > 0))
        obtained += len(weight)
    delta, gain = np.concatenate(delta_draws), np.concatenate(gain_draws)
    finite_gain = gain[np.isfinite(gain)]
    gain_ci = np.quantile(finite_gain, [.025, .975]) if len(finite_gain) else [np.nan, np.nan]
    return {
        "reference_mae": base, "candidate_mae": final, "delta_mae": final - base,
        "delta_ci_low": float(np.quantile(delta, .025)),
        "delta_ci_high": float(np.quantile(delta, .975)),
        "relative_gain_pct": 100 * (base - final) / base if base > 0 else np.nan,
        "gain_ci_low_pct": float(gain_ci[0]), "gain_ci_high_pct": float(gain_ci[1]),
        "n_splits": len(splits), "n_stations_unique": len(ids),
        "n_station_months_unique": int(cells[["station", "cell"]].drop_duplicates().shape[0]),
        "n_split_cell_occurrences": len(cells), "bootstrap_draws": draws,
        "bootstrap_seed": seed, "bootstrap_empty_split_redraws": rejected,
    }


def comparison_products(frame: pd.DataFrame, draws: int) -> tuple[pd.DataFrame, ...]:
    summaries, consistency, station_rows, strata_rows = [], [], [], []
    for name, candidate, ck, reference, rk in COMPARISONS:
        pair = paired_cells(frame, candidate, ck, reference, rk)
        summary = {"comparison": name, "candidate": candidate, "candidate_k": ck,
                   "reference": reference, "reference_k": rk, **joint_station_bootstrap(pair, draws)}
        per_seed = pair.groupby(["split_seed", "seed"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
        per_seed["delta_mae"] = per_seed.candidate_mae - per_seed.reference_mae
        per_seed["relative_gain_pct"] = 100 * (1 - per_seed.candidate_mae / per_seed.reference_mae)
        per_seed["comparison"] = name
        consistency.append(per_seed)
        split_delta = per_seed.groupby("split_seed").delta_mae.mean()
        summary.update(n_split_seed_pairs=len(per_seed),
                       improved_split_seed_pairs=int((per_seed.delta_mae < 0).sum()),
                       improved_splits=int((split_delta < 0).sum()))
        summaries.append(summary)
        cells = pair.groupby(["split_seed", "station", "cell"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
            ecological_novelty=("ecological_novelty", "mean"),
            upstream_support=("upstream_support", "mean"))
        station = cells.groupby(["split_seed", "station"], as_index=False).agg(
            candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
            n_query_cells=("cell", "size"), ecological_novelty=("ecological_novelty", "mean"),
            upstream_support=("upstream_support", "mean"))
        station["comparison"] = name
        station["delta_mae"] = station.candidate_mae - station.reference_mae
        station["relative_gain_pct"] = 100 * (1 - station.candidate_mae / station.reference_mae)
        station_rows.append(station)
        # Predictor-only descriptive bins, determined once over split/station
        # occurrences. These are not trained thresholds or confirmatory tests.
        q1, q2 = station.ecological_novelty.quantile([1 / 3, 2 / 3]).to_numpy()
        cells["novelty_stratum"] = np.select(
            [cells.ecological_novelty <= q1, cells.ecological_novelty <= q2],
            ["lower", "middle"], default="higher")
        cells["support_stratum"] = np.where(cells.upstream_support > 0, "visible upstream", "no visible upstream")
        for column, variable in (("novelty_stratum", "ecological_novelty"),
                                 ("support_stratum", "upstream_support")):
            for (split, level), group in cells.groupby(["split_seed", column]):
                base, cand = float(group.reference_mae.mean()), float(group.candidate_mae.mean())
                strata_rows.append({
                    "comparison": name, "split_seed": split, "variable": variable, "stratum": level,
                    "n_query_cells": len(group), "n_stations": group.station.nunique(),
                    "reference_mae": base, "candidate_mae": cand, "delta_mae": cand - base,
                    "relative_gain_pct": 100 * (base - cand) / base if base > 0 else np.nan,
                    "novelty_q33": q1, "novelty_q67": q2,
                    "novelty_bins_degenerate": bool(q1 == q2),
                })
    return (pd.DataFrame(summaries), pd.concat(consistency, ignore_index=True),
            pd.concat(station_rows, ignore_index=True), pd.DataFrame(strata_rows))


def plot_results(curve: pd.DataFrame, comparisons: pd.DataFrame,
                 stations: pd.DataFrame, out: Path, partial: bool,
                 implementation_check: bool = False) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#555555", "text.color": "#292929",
                         "axes.labelcolor": "#292929", "pdf.fonttype": 42})
    suffix = "\nImplementation check" if implementation_check else ("\nPartial results" if partial else "")

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    for model in MODELS:
        part = curve.loc[curve.model_name.eq(model)].sort_values("k")
        calibrated = model.endswith("calibrated")
        errors = part.mae_sd_split if calibrated and part.mae_sd_split.notna().any() else None
        ax.errorbar(part.k, part.mae, yerr=errors,
                    color=COLORS[model], linestyle="-" if calibrated else "--",
                    marker=("o" if model.startswith("extra") else "s") if calibrated else None,
                    markersize=5, linewidth=1.5, capsize=3, label=LABELS[model])
    ax.set(xlabel="Target observations per station (K)", ylabel="MAE (mg/L)", xticks=KS,
           title="Spatial reconstruction with sparse calibration" + suffix)
    ax.grid(axis="y", color="#eeeeee", linewidth=.6)
    ax.legend(frameon=False, fontsize=9)
    caption = ("Equal partition means; error bars show partition SD"
               if curve.mae_sd_split.notna().any()
               else "One completed partition; partition SD unavailable")
    fig.supxlabel(caption, fontsize=9)
    save(fig, "k_curves")

    fig, ax = plt.subplots(figsize=(6.5, 4.7), layout="constrained")
    for i, row in enumerate(comparisons.itertuples()):
        color = "#D08043" if i < 2 else "#235B80"
        ax.hlines(i, row.gain_ci_low_pct, row.gain_ci_high_pct, color=color, linewidth=1.6)
        ax.plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], "|", color=color, markersize=7)
        ax.plot(row.relative_gain_pct, i, "o", color=color, markersize=6)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=.9)
    ax.set(yticks=range(len(comparisons)),
           yticklabels=[COMPARISON_LABELS[c] for c in comparisons.comparison],
           xlabel="Relative MAE reduction (%)", title="Paired spatial comparisons" + suffix)
    ax.invert_yaxis()
    fig.supxlabel("95% intervals: joint station bootstrap across partitions", fontsize=9)
    save(fig, "paired_improvements")

    fig, ax = plt.subplots(figsize=(6.5, 4.7), layout="constrained")
    subset = stations.loc[stations.comparison.eq(COMPARISONS[0][0])]
    for split, marker in zip(sorted(subset.split_seed.unique()), ("o", "s", "^")):
        part = subset.loc[subset.split_seed.eq(split)]
        ax.scatter(part.ecological_novelty, part.relative_gain_pct,
                   c=np.where(part.relative_gain_pct > 0, "#235B80", "#D08043"),
                   marker=marker, s=26, alpha=.75, linewidths=.35, edgecolors="white",
                   label=f"Partition {split}")
    ax.axhline(0, color="#777777", linewidth=.9)
    ax.set(xlabel="Ecological novelty (distance to training stations)",
           ylabel="Station MAE reduction (%)",
           title="Hybrid vs ExtraTrees after K = 5 calibration" + suffix)
    ax.legend(frameon=False, fontsize=9)
    fig.supxlabel("One point per station–partition; positive favors the hybrid", fontsize=9)
    save(fig, "station_heterogeneity")


def fallback_diagnostics(frame: pd.DataFrame, adapters: dict) -> pd.DataFrame:
    rows = []
    for (split, seed), group in frame.loc[frame.k.eq(0) & frame.model_name.eq("hybrid")].groupby(["split_seed", "seed"]):
        diff = np.abs(group.hybrid_pred.to_numpy() - group.context_pred.to_numpy())
        record = {"split_seed": split, "seed": seed,
                  "exact_context_fallback": bool(np.allclose(diff, 0, rtol=0, atol=1e-10)),
                  "mean_abs_hybrid_context_difference": float(diff.mean()),
                  "max_abs_hybrid_context_difference": float(diff.max())}
        adapter = adapters[(int(split), int(seed))]
        fusion = adapter.get("fusion", {})
        record["selected_fusion"] = fusion.get("name", "not_recorded")
        coefficients = fusion.get("coefficients", [np.nan, np.nan, np.nan])
        if len(coefficients) == 3:
            record.update(zip(("intercept", "context_weight", "temporal_weight"), coefficients))
        if record["selected_fusion"] == "context" and not record["exact_context_fallback"]:
            raise ValueError("context-only adapter disagrees with saved hybrid prediction")
        # Preserve metadata without guessing whether a numeric weight is a
        # neural gate, an affine coefficient or a station-calibration alpha.
        record["adapter_json"] = json.dumps(adapter, sort_keys=True)
        rows.append(record)
    return pd.DataFrame(rows)


def write_findings(out: Path, status: dict, comparisons: pd.DataFrame,
                   curves: pd.DataFrame, fallback: pd.DataFrame) -> None:
    state = "COMPLETE" if status["complete"] else "PARTIAL"
    lines = [f"# Unified DOC spatial reconstruction: {state}", "",
             (f"Completed runs: {status['available_runs']}/9. "
             "Three planned spatial partitions (142–144), three training seeds (42–44), "
             "four support budgets (0, 1, 3, 5), and four paired arms."), "",
             "## Main comparisons", "",
             ("MAEs average individual-seed losses within each partition and then weight partitions equally. "
             "Negative paired ΔMAE and positive relative reduction favor the candidate. "
             "Intervals resample global station IDs jointly across partitions; they preserve each "
             "partition's cell weighting and the repeated-station dependence."), "",
             "| Comparison | Reference MAE | Candidate MAE | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions |", 
             "|---|---:|---:|---:|---:|---:|"]
    if not status["confirmation_budget"]:
        lines[2:2] = [("**Implementation check only.** These runs do not use the planned "
                      "300-tree, 20-epoch confirmation budget. Their numbers test the reporting "
                      "pipeline and are not confirmation evidence."), ""]
    for row in comparisons.itertuples():
        lines.append(f"| {COMPARISON_LABELS[row.comparison].replace(chr(10), '; ')} | "
                     f"{row.reference_mae:.4f} | {row.candidate_mae:.4f} | "
                     f"{row.delta_mae:.4f} [{row.delta_ci_low:.4f}, {row.delta_ci_high:.4f}] | "
                     f"{row.relative_gain_pct:.2f}% [{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}] | "
                     f"{row.improved_splits}/{row.n_splits} |")
    primary = comparisons.iloc[0]
    lines.extend(["", "## Interpretation", ""])
    if not status["confirmation_budget"]:
        lines.append("The reduced-budget outputs were used only to check paired table construction, "
                     "bootstrap execution, metadata handling and figure rendering. No scientific "
                     "performance conclusion is assigned to this implementation check.")
    elif primary.delta_ci_high < 0:
        lines.append("At K = 5, the calibrated hybrid improves on equally calibrated ExtraTrees in "
                     "the evaluated partitions; the paired station interval is below zero.")
    elif primary.delta_ci_low > 0:
        lines.append("At K = 5, calibrated ExtraTrees outperforms the calibrated hybrid in "
                     "the evaluated partitions; the paired station interval is above zero.")
    else:
        lines.append("At K = 5, the paired interval spans zero; the evaluated partitions do not "
                     "establish an additional hybrid improvement beyond calibrating ExtraTrees.")
    fallback_count = int(fallback.exact_context_fallback.sum())
    lines.append(f"The hybrid equals its context prediction in {fallback_count}/{len(fallback)} completed "
                 "split–seed runs (numerical tolerance 1e-10 mg/L). "
                 "Calibration gains in such runs are not neural gains. The component comparison "
                 "does not isolate any one neural feature or river-message mechanism.")
    if not status["complete"]:
        lines.append("This is an interim analysis. Available seeds are averaged within each completed "
                     "partition; missing runs can change all pooled estimates and intervals. "
                     f"Missing pairs: {status['missing_runs']}.")
    lines.extend(["", "## K curves", "", "| Model | K | MAE | Partition SD |", "|---|---:|---:|---:|"])
    for row in curves.itertuples():
        sd = "NA" if pd.isna(row.mae_sd_split) else f"{row.mae_sd_split:.4f}"
        lines.append(f"| {LABELS[row.model_name]} | {row.k} | {row.mae:.4f} | {sd} |")
    lines.extend(["", "## Supporting analyses", "",
                  ("- `run_metrics.csv`: MAE, RMSE, R², log-space MAE and train-Q90 tail MAE. "
                  "Tail counts are unique query cells per run; n < 20 is marked unstable."),
                  "- `split_seed_consistency.csv`: every paired split/seed outcome, including failures.",
                  ("- `station_heterogeneity.csv`: within-partition seed-mean station effects. "
                  "A station repeated in partitions is not an independent ecological replicate."),
                  ("- `stratified_by_split.csv` and `stratified_summary.csv`: predictor-only descriptive "
                  "novelty tertiles and presence/absence of visible upstream support. Empty strata "
                  "are not assigned a synthetic zero; the summary records contributing partitions."),
                  "- `fallback_diagnostics.csv`: numerical context equivalence and saved adapter metadata.",
                  "", ("The common support/query design evaluates retrospective reconstruction. "
                  "These are new partitions of the existing cohort, not independent external basins. "
                  "Station-bootstrap intervals are conditional on the selected partitions and fitted models."), ""])
    (out / "findings.md").write_text("\n".join(lines))


def analyze(root: Path, allow_partial: bool = False, draws: int = 5000) -> Path:
    data, thresholds, sources, metadata = load_predictions(root, allow_partial)
    out = root / ("analysis" if metadata["status"]["complete"] else "analysis_partial")
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(metadata["status"], indent=2) + "\n")
    if data.empty:
        (out / "findings.md").write_text("# Unified DOC spatial reconstruction: PARTIAL\n\n"
                                         "No prediction/config/adapter set with complete.json is available.\n")
    else:
        rows = []
        for (split, seed, k, model), group in data.groupby(["split_seed", "seed", "k", "model_name"]):
            rows.append({"split_seed": split, "seed": seed, "k": k, "model_name": model,
                         "n_stations": group.station.nunique(),
                         **metric_values(group.y_true, group.y_pred, thresholds[(int(split), int(seed))])})
        metrics = pd.DataFrame(rows)
        metrics.to_csv(out / "run_metrics.csv", index=False)
        split_metrics = metrics.groupby(["split_seed", "k", "model_name"], as_index=False).agg(
            **{name: (name, "mean") for name in METRICS}, mae_sd_seed=("mae", "std"),
            n_seeds=("seed", "size"), n_query_cells=("n_query_cells", "first"),
            n_stations=("n_stations", "first"), q90_n=("q90_n", "first"),
            q90_unstable=("q90_unstable", "any"))
        split_metrics.to_csv(out / "split_metrics.csv", index=False)
        curves = split_metrics.groupby(["k", "model_name"], as_index=False).agg(
            **{name: (name, "mean") for name in METRICS}, mae_sd_split=("mae", "std"),
            n_splits=("split_seed", "size"))
        curves.to_csv(out / "k_curves.csv", index=False)
        comparisons, consistency, stations, strata = comparison_products(data, draws)
        comparisons.to_csv(out / "primary_comparisons.csv", index=False)
        consistency.to_csv(out / "split_seed_consistency.csv", index=False)
        stations.to_csv(out / "station_heterogeneity.csv", index=False)
        strata.to_csv(out / "stratified_by_split.csv", index=False)
        strata.groupby(["comparison", "variable", "stratum"], as_index=False).agg(
            reference_mae=("reference_mae", "mean"), candidate_mae=("candidate_mae", "mean"),
            delta_mae=("delta_mae", "mean"), n_splits=("split_seed", "nunique"),
            n_split_cell_occurrences=("n_query_cells", "sum"),
            n_station_split_occurrences=("n_stations", "sum"),
        ).to_csv(out / "stratified_summary.csv", index=False)
        fallback = fallback_diagnostics(data, metadata["adapters"])
        fallback.to_csv(out / "fallback_diagnostics.csv", index=False)
        plot_results(curves, comparisons, stations, out, not metadata["status"]["complete"],
                     not metadata["status"]["confirmation_budget"])
        write_findings(out, metadata["status"], comparisons, curves, fallback)
    # Lightweight source list, written last. Original run products are untouched.
    (out / "source_hashes.json").write_text(json.dumps({
        "sources": sources, "analysis_script": str(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": draws, "bootstrap_seed": 42,
        "estimand": "seed mean within partition; cell weighted; partition equal",
        "complete": metadata["status"]["complete"],
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    print(analyze(args.root, args.allow_partial, args.bootstrap_draws))


if __name__ == "__main__":
    main()
