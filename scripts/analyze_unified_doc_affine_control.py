"""Additional affine and local-forest ablations; primary results unchanged.

Fit context-only affine recalibration and support shrinkage using source
validation, then compare against the frozen hybrid on identical outer queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_unified_doc_spatial import (
    DEFAULT_ROOT,
    KS,
    joint_station_bootstrap,
    load_predictions,
    metric_values,
    paired_cells,
    sha256,
)

from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.station_adapted_hybrid import (
    CalibrationEpisode,
    StationAdaptedHybrid,
    station_residual_correction,
)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTROL = "context_affine_control"
FOREST_CONTROL = "context_local_forest_control"
PROTOCOL = Path("experiments/phase4_transfer/unified_doc_spatial_v1/context_affine_control.md")


def inverse(z):
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.maximum(np.expm1(z), 0.0)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite affine prediction; no test-derived upper clipping is allowed")
    return values


def fit_context_affine(prediction, truth):
    """Fit and select on source validation only; return JSON-compatible choices."""
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if (prediction.ndim != 1 or prediction.shape != truth.shape or not len(truth)
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()
            or (prediction < 0).any() or (truth < 0).any()):
        raise ValueError("validation predictions and truth must be aligned finite DOC values")
    identity = {"name": "identity", "coefficients": [0.0, 1.0], "valid": True,
                "validation_mae": float(np.abs(prediction - truth).mean())}
    design = np.column_stack([np.ones(len(prediction)), np.log1p(prediction)])
    coefficients = np.linalg.lstsq(design, np.log1p(truth), rcond=None)[0]
    affine = {"name": "log_affine", "coefficients": coefficients.tolist()}
    try:
        fitted = inverse(design @ coefficients)
        affine.update(valid=True, validation_mae=float(np.abs(fitted - truth).mean()))
    except ValueError:
        affine.update(valid=False, validation_mae=None)
    candidates = [identity, affine]
    selected = min((row for row in candidates if row["valid"]),
                   key=lambda row: row["validation_mae"])
    return {"selection_role": "source_validation", "selected": selected, "candidates": candidates}


def apply_context_affine(prediction, selection):
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("context predictions must be finite nonnegative DOC values")
    if selection["selected"]["name"] == "identity":
        return prediction.copy()
    intercept, slope = selection["selected"]["coefficients"]
    return inverse(intercept + slope * np.log1p(prediction))


def select_shrinkage(base, truth, split, n_months):
    """Choose independent control alphas on the source-validation station task."""
    choices, scores = {0: 0.0}, []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=n_months)
        for alpha in ((0.0,) if k == 0 else ALPHAS):
            predicted = station_residual_correction(
                base[query], query, base[support], support, truth[support],
                n_months=n_months, alpha=alpha,
            )
            scores.append({"k": k, "alpha": alpha,
                           "validation_mae": float(np.abs(predicted - truth[query]).mean()),
                           "validation_query_cells": len(query), "validation_support_cells": len(support)})
        choices[k] = min((row for row in scores if row["k"] == k),
                         key=lambda row: row["validation_mae"])["alpha"]
    return choices, scores


def fit_local_forest_fusion(context, local, truth, split, n_months):
    """Replace the complete residual expert R by its uncorrected local forest B.

    This is precisely the primary fusion/calibration class, fit on precisely
    the same source-validation query and support populations. Nothing is fitted
    on the target stations, apart from the explicit support-only correction.
    """
    _, query = support_query_cells(split, target_role="val", k=0, n_months=n_months)
    model = StationAdaptedHybrid(n_months=n_months)
    model.fit_fusion(context[query], local[query], truth[query], selection_role="source_validation")
    episodes = []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=n_months)
        episodes.append(CalibrationEpisode(
            k=k, query_cells=query, query_values=truth[query],
            context_query_prediction=context[query], temporal_query_prediction=local[query],
            support_cells=support, support_values=truth[support],
            context_support_prediction=context[support], temporal_support_prediction=local[support],
        ))
    model.fit_calibration(episodes, selection_role="source_validation")
    return model


def run_control(root, pair, primary, dataset_cache):
    split_seed, seed = pair
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    full_path = run / "full_grid.parquet"
    full_hash = sha256(full_path)
    if complete["files"].get(full_path.name) != full_hash:
        raise ValueError(f"unbound or altered frozen full-grid prediction: {full_path}")
    data_path, mask_path = Path(config["dataset_path"]), Path(config["mask_path"])
    if str(data_path) not in dataset_cache:
        dataset_cache[str(data_path)] = (sha256(data_path), torch.load(
            data_path, map_location="cpu", weights_only=False))
    data_hash, dataset = dataset_cache[str(data_path)]
    if data_hash != config["dataset_hash"] or sha256(mask_path) != config["mask_hash"]:
        raise ValueError(f"dataset or mask changed since expert fitting: {run}")
    with np.load(mask_path, allow_pickle=False) as archive:
        split = {role: archive[role] for role in ("train", "val", "test", "context")}
    n, months = dataset["y"].shape
    values = np.asarray(dataset["y"], dtype=np.float64).ravel()
    full = pd.read_parquet(full_path, columns=["cell", "context_pred", "local_pred"])
    np.testing.assert_array_equal(full["cell"].to_numpy(), np.arange(n * months))
    context = full["context_pred"].to_numpy()
    local = full["local_pred"].to_numpy()
    _, validation_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    selection = fit_context_affine(context[validation_query], values[validation_query])
    control = apply_context_affine(context, selection)
    choices, shrinkage_scores = select_shrinkage(control, values, split, months)
    selection.update(split_seed=split_seed, seed=seed, alpha_by_k=choices,
                     calibration_scores=shrinkage_scores, validation_query_cells=len(validation_query))
    forest_fusion = fit_local_forest_fusion(context, local, values, split, months)
    forest_control = forest_fusion.predict_components(context, local)["hybrid"]
    selection["local_forest_control"] = forest_fusion.to_dict()
    rows = []
    # All source-validation choices are frozen before target support correction.
    for k in KS:
        support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
        template = primary.loc[primary["k"].eq(k) & primary.model_name.eq("extra_trees")].sort_values("cell")
        np.testing.assert_array_equal(template["cell"].to_numpy(), query)
        np.testing.assert_array_equal(template.context_pred.to_numpy(), context[query])
        for name, base in ((CONTROL, control), (FOREST_CONTROL, forest_control)):
            for calibrated in (False, True):
                if not calibrated:
                    predicted = base[query].copy()
                elif name == FOREST_CONTROL:
                    predicted = forest_fusion.adapt(
                        base[query], query, base[support], support, values[support], k=k, arm="hybrid",
                    )
                else:
                    predicted = station_residual_correction(
                        base[query], query, base[support], support, values[support],
                        n_months=months, alpha=choices[k],
                    )
                frame = template.copy()
                frame["model_name"] = name + ("_calibrated" if calibrated else "")
                frame["y_pred"] = predicted
                frame["control_pred"] = base[query]
                frame["calibration_delta"] = np.log1p(predicted) - np.log1p(base[query])
                frame["support_count"] = k if calibrated else 0
                rows.append(frame)
    sources = [{"path": str(path), "sha256": digest} for path, digest in
               ((full_path, full_hash), (data_path, data_hash), (mask_path, config["mask_hash"]))]
    return pd.concat(rows, ignore_index=True), selection, sources


def compare(data, draws):
    definitions = [(f"hybrid_vs_affine_context_k{k}", "hybrid_calibrated", k,
                    CONTROL + "_calibrated", k) for k in KS]
    definitions.extend([
        ("affine_context_vs_original_et_k0", CONTROL, 0, "extra_trees", 0),
        ("affine_context_vs_original_et_calibrated_k5", CONTROL + "_calibrated", 5,
         "extra_trees_calibrated", 5),
        ("hybrid_vs_context_local_forest_k0", "hybrid", 0, FOREST_CONTROL, 0),
        ("hybrid_vs_context_local_forest_calibrated_k5", "hybrid_calibrated", 5,
         FOREST_CONTROL + "_calibrated", 5),
    ])
    summaries, per_run = [], []
    for name, candidate, ck, reference, rk in definitions:
        paired = paired_cells(data, candidate, ck, reference, rk)
        result = joint_station_bootstrap(paired, draws=draws)
        run = paired.groupby(["split_seed", "seed"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
        run["comparison"] = name
        run["delta_mae"] = run.candidate_mae - run.reference_mae
        run["relative_gain_pct"] = 100 * (1 - run.candidate_mae / run.reference_mae)
        result.update(comparison=name, candidate=candidate, candidate_k=ck,
                      reference=reference, reference_k=rk,
                      improved_split_seed_pairs=int((run.delta_mae < 0).sum()),
                      n_split_seed_pairs=len(run),
                      improved_splits=int((run.groupby("split_seed").delta_mae.mean() < 0).sum()))
        summaries.append(result)
        per_run.append(run)
    return pd.DataFrame(summaries), pd.concat(per_run, ignore_index=True)


def write_report(out, status, summary, selections):
    lines = ["# Affine and local-forest component ablations", "",
             "Additional controls; original four-arm primary results are unchanged.", "",
             (f"Completed runs: {status['available_runs']}/9. "
              "Source validation selects identity versus context-only log-affine recalibration "
              "and then independently selects the control's support shrinkage. "
              "A second control replaces the temporal residual expert R with its local forest B "
              "and fits C+B using the primary fusion/calibration class and validation populations."), ""]
    if not status["confirmation_budget"]:
        lines.extend([("**Implementation check only.** Reduced-budget smoke outputs verify the "
                      "analysis workflow. They do not establish scientific performance."), ""])
    elif not status["complete"]:
        lines.extend([("**Partial additional ablation.** Missing formal runs can change the estimates. "
                      "No replacement of the primary comparisons or final conclusion is made."), ""])
    lines.extend(["| Comparison | Reference MAE | Candidate MAE | Reduction (%) [95% CI] |",
                  "|---|---:|---:|---:|"])
    for row in summary.itertuples():
        lines.append(f"| {row.comparison} | {row.reference_mae:.4f} | {row.candidate_mae:.4f} | "
                     f"{row.relative_gain_pct:.2f} [{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}] |")
    selected_count = sum(row["selected"]["name"] == "log_affine" for row in selections)
    lines.extend(["", f"Context-only affine recalibration was selected in {selected_count}/{len(selections)} runs.",
                  "", ("MAEs average individual-seed losses within partition, then weight partitions equally. "
                  "Confidence intervals jointly resample station IDs across partitions and retain each "
                  "partition's cell weighting. Positive reduction favors the named candidate."), "",
                  ("At K > 0, both compared models use their own source-validation-selected station "
                  "calibration. K = 0 compares unadapted predictions. The additional ablation separates "
                  "the complete temporal expert's contribution from simple global recalibration; "
                  "it does not isolate an individual neural mechanism."), "",
                  ("The C+R versus C+B comparison at K = 0 and K = 5 isolates the addition of the "
                  "trained temporal residual to the local forest, while allowing both combinations "
                  "the same fusion candidates and station-calibration selection. The saved "
                  "`local_forest_control` selection contains all C+B validation candidate scores. "
                  "No additional forests or recurrent models were trained."), "",
                  ("Support can postdate query months. The experiment concerns retrospective station "
                  "reconstruction in the known cohort. Selection scores are training/selection "
                  "diagnostics, not independent performance estimates."), ""])
    (out / "findings.md").write_text("\n".join(lines))


def analyze(root, allow_partial=False, draws=5000):
    primary, thresholds, sources, metadata = load_predictions(root, allow_partial)
    if primary.empty:
        raise ValueError("no completed frozen run is available for the affine ablation")
    controls, selections, dataset_cache = [], [], {}
    for pair, frame in primary.groupby(["split_seed", "seed"]):
        prediction, selection, extra_sources = run_control(root, tuple(map(int, pair)), frame, dataset_cache)
        controls.append(prediction)
        selections.append(selection)
        sources.extend(extra_sources)
    control = pd.concat(controls, ignore_index=True)
    panel = pd.concat([primary, control], ignore_index=True)
    summary, consistency = compare(panel, draws)
    metrics = []
    for (split, seed, k, name), group in panel.groupby(["split_seed", "seed", "k", "model_name"]):
        metrics.append({"split_seed": split, "seed": seed, "k": k, "model_name": name,
                        **metric_values(group.y_true, group.y_pred, thresholds[(int(split), int(seed))])})
    out = root / "context_affine_control"
    out.mkdir(parents=True, exist_ok=True)
    control.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(out / "run_metrics.csv", index=False)
    summary.to_csv(out / "comparisons.csv", index=False)
    consistency.to_csv(out / "split_seed_consistency.csv", index=False)
    (out / "selection.json").write_text(json.dumps(selections, indent=2, allow_nan=False) + "\n")
    (out / "status.json").write_text(json.dumps(metadata["status"], indent=2) + "\n")
    write_report(out, metadata["status"], summary, selections)
    (out / "sources.json").write_text(json.dumps({
        "role": "additional affine and local-forest component ablations; not primary replacements",
        "sources": sources, "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "protocol": str(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
        "bootstrap_draws": draws, "bootstrap_seed": 42,
        "estimand": "seed mean within partition; cell weighted; partition equal",
    }, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    print(analyze(args.root, allow_partial=args.allow_partial, draws=args.bootstrap_draws))


if __name__ == "__main__":
    main()
