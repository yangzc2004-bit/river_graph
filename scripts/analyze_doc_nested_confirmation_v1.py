"""Evaluate six fixed DOC support-calibration procedures on fresh station roles.

All fitting and representation choices precede this script. The two primary
comparisons are nested versus joint calibration at K3/K5; eight additional
comparisons preserve the legacy, availability, general and tree controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_chemistry_support_v1 import (
    attach_availability,
    availability_profiles,
    bound_file,
    full_grid_availability,
)
from analyze_doc_encoder_residual_v1 import compare
from analyze_doc_nested_chemistry_v1 import (
    check_state_predictions,
    read_product,
    state_records,
)
from analyze_doc_selective_residual_v1 import classification_metrics, recall_comparisons
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import config_digest, sha256
from analyze_unified_doc_spatial_v3 import read_bound_json
from run_doc_nested_chemistry_v1 import MODELS

ROOT = Path("experiments/phase4_transfer/doc_nested_confirmation_v1")
SPLITS, SEEDS, KS = (342, 343, 344), (42, 43, 44), (0, 1, 3, 5)
GENERAL, JOINT, LEGACY, TREE, NESTED, MASKS = MODELS
EXTRA_COLUMNS = ("ph_available", "ec_available", "aux_available", "chemical_delta",
                 "chemical_support_count", "visibility_role")
ANALYSIS_DEPENDENCIES = (
    "analyze_doc_chemistry_confirmation_v1", "analyze_doc_chemistry_support_v1",
    "analyze_doc_encoder_residual_v1", "analyze_doc_nested_chemistry_v1",
    "analyze_doc_selective_residual_v1", "analyze_doc_tail_residual_v1",
    "analyze_unified_doc_spatial", "analyze_unified_doc_spatial_v2",
    "analyze_unified_doc_spatial_v3", "run_doc_nested_chemistry_v1",
)


def comparison_definitions():
    """Two primary plus eight fixed diagnostic contrasts, without ranking."""
    return [(f"nested_vs_{name}_k{k}", NESTED, k, reference, k, role)
            for name, reference, role in (
                ("joint", JOINT, "primary_nested_vs_joint"),
                ("legacy", LEGACY, "increment_beyond_legacy"),
                ("general", GENERAL, "general_model_reference"),
                ("masks", MASKS, "chemical_values_beyond_availability"),
                ("tree", TREE, "chemical_tree_reference")) for k in (3, 5)]


def validate_panel(frame, *, require_nine=True):
    """Check complete fixed queries and the increment's exact parent fallback."""
    required = {"split_seed", "seed", "model_name", "k", "cell", "station", "month",
                "y_true", "y_pred", "aux_available", "ph_available", "ec_available",
                "chemical_delta", "chemical_support_count"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing validation columns: {sorted(missing)}")
    keys = ["split_seed", "seed", "model_name", "k", "cell"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Empty or duplicate query panel")
    numeric = ["y_true", "y_pred", "chemical_delta", "chemical_support_count",
               "ecological_novelty", "upstream_support"]
    if not set(numeric).issubset(frame) or not frame.upstream_support.between(0, 1).all():
        raise ValueError("Missing or invalid query descriptors")
    if (not np.isfinite(frame[numeric].to_numpy(float)).all()
            or (frame[["y_true", "y_pred", "chemical_support_count"]] < 0).any().any()
            or frame.cell.dtype.kind not in "iu" or (frame.cell < 0).any()
            or frame[["station", "month"]].isna().any().any()):
        raise ValueError("Invalid query values or cell identities")
    for field in ("ph_available", "ec_available", "aux_available"):
        if not np.isin(frame[field].to_numpy(), (False, True)).all():
            raise ValueError("Nonbinary chemical availability")
    active = frame.ph_available.to_numpy(bool) | frame.ec_available.to_numpy(bool)
    if not np.array_equal(frame.aux_available.to_numpy(bool), active):
        raise ValueError("Chemical availability does not match input masks")
    if "visibility_role" in frame and not frame.visibility_role.eq("test").all():
        raise ValueError("Only held-target query roles may enter this confirmation analysis")
    count = frame.chemical_support_count.to_numpy(float)
    if ((count != np.floor(count)).any() or (count > frame.k.to_numpy()).any()):
        raise ValueError("Invalid chemical support count")
    identity = ["station", "month", "y_true", "aux_available", "ph_available", "ec_available"]
    if frame.groupby("cell")[identity].nunique().ne(1).any().any():
        raise ValueError("Cell identity, truth or availability changes between panels")
    if frame.groupby(["station", "month"]).cell.nunique().ne(1).any():
        raise ValueError("Station/month maps to multiple cells")
    pairs = set(map(tuple, frame[["split_seed", "seed"]].drop_duplicates().to_numpy()))
    if require_nine and pairs != {(split, seed) for split in SPLITS for seed in SEEDS}:
        raise ValueError("Exactly nine fresh confirmation packages are required")
    expected = {(model, k) for model in MODELS for k in KS}
    for split, part in frame.groupby("split_seed"):
        reference = None
        for (_, _, _), group in part.groupby(["seed", "model_name", "k"]):
            query = group.sort_values("cell")[["cell", *identity]].reset_index(drop=True)
            if reference is None:
                reference = query
            elif not reference.equals(query):
                raise ValueError(f"Fixed query differs across model/K/seed in split {split}")
        for seed, run in part.groupby("seed"):
            actual = set(map(tuple, run[["model_name", "k"]].drop_duplicates().to_numpy()))
            if actual != expected:
                raise ValueError(f"Incomplete model/K panel: split{split}, seed{seed}")
            for k in KS:
                parent = run[run.model_name.eq(LEGACY) & run.k.eq(k)].sort_values("cell")
                for model in (NESTED, MASKS):
                    candidate = run[run.model_name.eq(model) & run.k.eq(k)].sort_values("cell")
                    zero = candidate.chemical_delta.eq(0).to_numpy()
                    inactive = ~candidate.aux_available.to_numpy(bool)
                    no_donors = candidate.chemical_support_count.to_numpy() < 2
                    if k in (0, 1):
                        zero_required = np.ones(len(candidate), dtype=bool)
                    else:
                        zero_required = inactive | no_donors
                    if not zero[zero_required].all():
                        raise ValueError("Chemical increment must be zero without available support")
                    if not np.array_equal(candidate.y_pred.to_numpy()[zero], parent.y_pred.to_numpy()[zero]):
                        raise ValueError("Zero chemical increment changed the legacy parent")
                    expected_prediction = np.maximum(0., np.expm1(
                        np.log1p(parent.y_pred.to_numpy()[~zero])
                        + candidate.chemical_delta.to_numpy()[~zero]))
                    if not np.allclose(candidate.y_pred.to_numpy()[~zero], expected_prediction,
                                       rtol=1e-12, atol=1e-12):
                        raise ValueError("Chemical increment does not reproduce final predictions")



def validate_config(config, split, seed, snapshot):
    if ((config["split_seed"], config["seed"]) != (split, seed)
            or config["experiment"] != "doc_nested_confirmation_v1" or config["smoke"]
            or tuple(config["models"]) != MODELS or tuple(config["k_values"]) != KS
            or config["selection_role"] != "source_validation" or config["inference_roles"] != ["train"]
            or config["historical_fitted_models_reused"] is not False
            or config["source_neural_is_oof"] is not False or config["source_forest_is_oof"] is not True
            or config["runtime_snapshot_hash"] != config_digest(snapshot)):
        raise ValueError("Unexpected fresh confirmation configuration")
    if (config["initial_recipe"]["initial"]["max_epochs"] != 20
            or config["initial_recipe"]["projector"]["epochs"] != 100
            or config["initial_recipe"]["memory"]["epochs"] != 30
            or config["native_recipe"]["initializer_epochs"] != 30
            or config["native_recipe"]["off_epochs"] != 120
            or config["chemical_recipe"]["epochs"] != 120):
        raise ValueError("Smoke or changed training budgets cannot enter confirmation")
    recipe = config["nested_recipe"]
    if (recipe["ridge_values"] != [.1, 1., 10., 100.]
            or recipe["strength_values"] != [0., .25, .5, 1.]
            or recipe["shared_selection_k"] != [3, 5]
            or recipe["selection_folds"] != 5 or recipe["fold_seed"] != 4100 + split
            or recipe["selection_role"] != "source_validation"
            or recipe["source_role"] != "source_training"
            or recipe["parent_parameters_reselected_by_nested_branch"] is not False):
        raise ValueError("Changed nested calibration recipe")


def check_batch(root, snapshot, sources):
    path = root / "batch_complete.json"
    batch = json.loads(path.read_text())
    expected = {f"split{s}_seed{r}" for s in SPLITS for r in SEEDS}
    actual = {p.name for p in (root / "runs").glob("split*_seed*") if p.is_dir()}
    if (actual != expected or set(batch["runs"]) != expected or batch["smoke"]
            or tuple(batch["split_seeds"]) != SPLITS or tuple(batch["seeds"]) != SEEDS
            or batch["runtime_snapshot_hash"] != config_digest(snapshot)):
        raise ValueError("Exactly nine completed fresh production packages are required")
    for run, digest in batch["runs"].items():
        if sha256(root / "runs" / run / "complete.json") != digest:
            raise ValueError("Batch completion identity changed")
    sources.append({"path": str(path), "sha256": sha256(path)})


def load_panel(root):
    """Read complete products and archived execution identity; never fit models."""
    sources, frames, thresholds, choices, scores, folds, validation_rows = [], [], {}, [], [], [], []
    reference_choices, reference_checks = [], []
    snapshot_path = root / "runtime_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources.append({"path": str(snapshot_path), "sha256": sha256(snapshot_path)})
    for name, digest in snapshot.items():
        archived = root / "code_snapshot" / name
        if sha256(archived) != digest:
            raise ValueError(f"Execution archive changed: {archived}")
        sources.append({"path": str(archived), "sha256": digest})
    check_batch(root, snapshot, sources)
    common_grid, deployment = None, None
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            validate_config(config, split, seed, snapshot)
            for key in ("dataset", "mask", "protocol", "daily_features", "daily_metadata"):
                path_key, hash_key = f"{key}_path", f"{key}_hash"
                if path_key in config:
                    path = Path(config[path_key])
                    if sha256(path) != config[hash_key]:
                        raise ValueError(f"Changed configured data: {path}")
                    sources.append({"path": str(path), "sha256": config[hash_key]})
            # Completion binds stage configs, snapshots and small/large fitted files.
            for name in completion["files"]:
                bound_file(run, name, completion, sources)
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            extra = pd.read_parquet(run / "predictions.parquet", columns=list(EXTRA_COLUMNS))
            for column in extra:
                frame[column] = extra[column].to_numpy()
            validation = read_product(run, "validation.parquet", config, completion, sources)
            cross_validation = read_product(run, "cv_validation.parquet", config, completion, sources)
            identity = {"split_seed": split, "seed": seed}
            reference = pd.read_parquet(bound_file(run, "chemical/predictions.parquet", completion, sources),
                columns=["model_name", "k", "cell", "y_pred"])
            for model in (GENERAL, JOINT, LEGACY, TREE):
                keys = ["k", "cell"]
                a = frame[frame.model_name.eq(model)].sort_values(keys)
                b = reference[reference.model_name.eq(model)].sort_values(keys)
                np.testing.assert_array_equal(a[[*keys, "y_pred"]].to_numpy(), b[[*keys, "y_pred"]].to_numpy())
                reference_checks.append({**identity, "model_name": model, "exact_fresh_reference": True})
            selection = read_bound_json(run, "chemical/basis_selection.json", completion, sources)
            if selection["selection_role"] != "source_validation":
                raise ValueError("Reference representation selection is not source-validation only")
            for pipeline in ("neural_chemistry_integrated", "tree_chemistry"):
                for k in KS:
                    choice = selection["choices"][pipeline][str(k)]
                    reference_choices.append({**identity, "pipeline": pipeline, "k": k,
                        "basis": choice["basis"], "active_mae": choice["active_mae"]})
            for mode, model in (("chemistry", NESTED), ("masks", MASKS)):
                state = read_bound_json(run, f"nested_{mode}.json", completion, sources)
                if (state["fold_seed"] != 4100 + split
                        or state["source_role"] != "source_training"
                        or state["selection_role"] != "source_validation"):
                    raise ValueError("Invalid calibration selection role or station folds")
                choice, candidates, held_folds = state_records(state, identity, mode)
                choices.append(choice)
                scores.extend(candidates)
                folds.extend(held_folds)
                check_state_predictions(state, validation, cross_validation, model)
            summary = pd.read_csv(bound_file(run, "source_validation.csv", completion, sources))
            expected = {(model, k) for model in MODELS for k in KS}
            if (summary.duplicated(["model_name", "k"]).any()
                    or set(map(tuple, summary[["model_name", "k"]].to_numpy())) != expected):
                raise ValueError("Incomplete source-validation summary")
            for row in summary.itertuples():
                group = validation[validation.model_name.eq(row.model_name) & validation.k.eq(row.k)]
                error = np.abs(group.y_pred.to_numpy() - group.y_true.to_numpy())
                active = group.aux_available.to_numpy(bool)
                if row.n != len(group) or row.n_active != active.sum():
                    raise ValueError("Source-validation denominator differs")
                np.testing.assert_allclose(row.mae, error.mean(), atol=1e-12, rtol=0)
                np.testing.assert_allclose(row.active_mae, error[active].mean() if active.any() else np.nan,
                                           atol=1e-12, rtol=0, equal_nan=True)
            validation_rows.append(summary.assign(**identity))
            cols = ["cell", "station", "month", "ph_available", "ec_available", "doc_observed"]
            full = pd.read_parquet(bound_file(run, "full_grid.parquet", completion, sources), columns=cols).sort_values("cell")
            np.testing.assert_array_equal(full.cell, np.arange(len(full)))
            if common_grid is not None and not full.equals(common_grid):
                raise ValueError("Full-grid identity or chemistry availability changed across refits")
            observed = full.set_index("cell").loc[frame.cell]
            np.testing.assert_array_equal(observed[cols[1:-1]].to_numpy(), frame[cols[1:-1]].to_numpy())
            if not observed.doc_observed.all():
                raise ValueError("Query has no observed DOC truth")
            if common_grid is None:
                common_grid, deployment = full, full_grid_availability(attach_availability(full))
            frames.append(frame)
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source threshold changes across training seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel)
    return attach_availability(panel), thresholds, sources, {
        "reference_basis_choices": pd.DataFrame(reference_choices),
        "reference_checks": pd.DataFrame(reference_checks),
        "nested_choices": pd.DataFrame(choices), "nested_candidate_scores": pd.DataFrame(scores),
        "conditional_validation_folds": pd.DataFrame(folds),
        "source_validation": pd.concat(validation_rows, ignore_index=True),
        "deployment_availability": deployment}


def summarize(panel, thresholds, draws):
    """Shared scientific estimand, with no model-selection step."""
    tables = {}
    tables["metrics_by_run"], tables["metrics_by_partition"], tables["k_curves"] = metric_summary(panel, thresholds)
    tables["error_profiles_by_run"], tables["error_profiles_by_partition"], tables["error_profiles"] = error_profiles(panel, thresholds)
    tables["classification_by_run"], tables["classification_by_partition"], tables["classification"] = classification_metrics(panel, thresholds)
    names = ("paired_effects", "directions_by_seed", "directions_by_partition", "station_responses",
             "global_station_contributions", "gain_loss_concentration")
    tables.update(zip(names, compare(panel, thresholds, draws, comparison_definitions()), strict=True))
    tables["paired_effects"] = pd.concat((tables["paired_effects"], recall_comparisons(
        panel, thresholds, draws, comparison_definitions())), ignore_index=True)
    tables.update(zip(("availability_by_run", "availability_by_partition", "availability_summary", "query_population"),
                      availability_profiles(panel, thresholds), strict=True))
    return tables


def write_report(out, tables, draws):
    curves, effects = tables["k_curves"], tables["paired_effects"]
    lines = ["# Nested chemistry calibration: fresh station-role confirmation", "",
        "Six fixed procedures are compared on station-role seeds 342–344 and training seeds 42–44.",
        "Every fitted predictor is rebuilt using its fresh source roles. These partitions reuse the ST357 cohort;",
        "they are not external-basin validation or new independent measurements. This script selects no model.", "",
        "The main comparison is the separate chemical increment versus the previously joint-selected support basis.",
        "Legacy calibration, the availability-only increment, the general model and chemical trees remain controls.",
        "The neural decoder and ecological/temporal pipeline use the retained empty-edge spatial self path.",
        "Source forests are station-blocked OOF; the complete source neural pipeline is not OOF.",
        "Support is retrospective and may postdate a query. Current-month pH/conductance are allowed covariates.", "",
        "Each run uses cell-weighted error. Seeds are averaged within a partition, then partitions receive equal weight.",
        f"The ten fixed comparisons use {draws:,} joint whole-station bootstrap draws, with repeated stations sampled",
        "jointly across partitions. Seed repeats do not increase ecological sample size. The intervals are unadjusted",
        "95% percentile intervals; a crossing-zero interval does not establish equivalence. RMSE/R² are averages",
        "of run metrics, not statistics calculated after pooling predictions.", "",
        "## Full support curves", "", "| Procedure | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 log MAE |",
        "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        rows = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | " + " | ".join(f"{rows.loc[k, 'mae']:.6f}" for k in KS)
                     + f" | {rows.loc[5, 'log_mae']:.6f} |")
    lines += ["", "## All fixed comparisons", "", "Negative error deltas favor the nested candidate.", "",
        "| Comparison | ΔMAE [95% CI] | Relative gain % [95% CI] | Q90 ΔMAE [95% CI] | Ordinary ΔMAE [95% CI] | Better partitions / runs |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, *_ in comparison_definitions():
        group = effects[effects.comparison.eq(name) & effects.metric.eq("mae")].set_index("region")
        row = group.loc["overall"]
        def fmt(region, group=group):
            r = group.loc[region]
            return (f"{r.delta_value:+.6f} [{r.delta_ci_low:+.6f}, {r.delta_ci_high:+.6f}]"
                    if r.status == "estimated" else "not estimable in all partitions")
        lines.append(f"| {name} | {fmt('overall')} | {row.relative_gain_pct:+.3f} "
            f"[{row.gain_ci_low_pct:+.3f}, {row.gain_ci_high_pct:+.3f}] | {fmt('q90')} | {fmt('nontail')} | "
            f"{int(row.improved_splits)}/3; {int(row.improved_split_seed_pairs)}/9 |")
    lines += ["", "## Interpretation and coverage", "",
        "Q90 means DOC at or above its source-training 90th percentile, including ties. Tail groups with fewer",
        "than 20 unique query cells per partition are flagged unstable. Error-profile and classification CSVs",
        "report tail/ordinary MAE, signed bias, precision, recall and false-positive rates together.",
        "Positive signed bias means overprediction. Low K0/K1 nested increments are exactly zero by design;",
        "their unchanged predictions are algebraic negative controls, not a statistical power finding.", "",
        "The availability appendix retains both, pH-only, conductance-only and neither groups, including empty",
        "groups. Full-grid covariate coverage at genuinely missing DOC cells is separate from accuracy on observed",
        "held queries. No accuracy is inferred for cells without DOC truth.", "",
        "Source-validation choices and conditional fold diagnostics are recorded separately. Those conditional folds",
        "assess the incremental adapter after its parent already used all source validation; only the fresh target",
        "queries provide the present held-station comparison. No K-specific winner is assembled from target results.", "",
        "All six curves, station gain/loss concentration and every partition/seed direction remain in the output tables.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, metadata = load_panel(args.root)
    tables = {**summarize(panel, thresholds, args.bootstrap_draws), **metadata}
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    write_report(out, tables, args.bootstrap_draws)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in
        (Path(__file__), *(Path(f"scripts/{name}.py") for name in ANALYSIS_DEPENDENCIES)))
    replay_path = args.root / "verification" / "replay_checks.json"
    replay_status = "pending"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        replay_status = replay.get("status", "unknown")
        expected_runs = {f"split{s}_seed{r}" for s in SPLITS for r in SEEDS}
        if (replay_status == "verified" and
                {row.get("run") for row in replay.get("results", [])} != expected_runs):
            replay_status = "partial"
        sources.append({"path": str(replay_path), "sha256": sha256(replay_path)})
    catalog = {row["path"]: row for row in sources}
    outputs = [*(out / f"{name}.csv" for name in tables), out / "findings.md"]
    manifest = {"complete": True, "packages": 9, "partitions": list(SPLITS), "training_seeds": list(SEEDS),
        "models": list(MODELS), "k_values": list(KS), "bootstrap_draws": args.bootstrap_draws,
        "comparisons": comparison_definitions(), "verification_status": replay_status,
        "scope": "fresh source refit and station roles on the same ST357 cohort; not external validation",
        "estimator": "cell MAE per run; seed mean within partition; equal partitions; joint station bootstrap",
        "sources": [catalog[name] for name in sorted(catalog)],
        "outputs": {path.name: sha256(path) for path in outputs}}
    (out / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"analysis": str(out), "packages": 9, "models": len(MODELS),
                      "comparisons": len(comparison_definitions()), "bootstrap_draws": args.bootstrap_draws}))


if __name__ == "__main__":
    main()
