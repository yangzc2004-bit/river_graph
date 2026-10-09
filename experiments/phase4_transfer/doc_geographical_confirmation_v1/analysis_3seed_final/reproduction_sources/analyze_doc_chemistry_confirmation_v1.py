"""Evaluate the fixed DOC chemistry recipe on fresh ST357 station assignments.

No fitting or target-driven selection occurs here. The nine packages must be
complete production refits, and all twenty predeclared curves remain visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_chemistry_support_v1 import (
    BASES,
    PIPELINES,
    attach_availability,
    availability_profiles,
    bound_file,
    full_grid_availability,
    selection_records,
    verify_pca_state,
)
from analyze_doc_encoder_residual_v1 import compare
from analyze_doc_selective_residual_v1 import classification_metrics, recall_comparisons
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import config_digest, sha256
from analyze_unified_doc_spatial_v2 import COLUMNS
from analyze_unified_doc_spatial_v3 import read_bound_json

ROOT = Path("experiments/phase4_transfer/doc_chemistry_confirmation_v1")
SPLITS, SEEDS, KS = (242, 243, 244), (42, 43, 44), (0, 1, 3, 5)
VARIANTS = (*BASES, "selected")
CONTROLS = ("context", "point", "tree_prior", "neural_no_aux", "neural_no_aux_integrated",
            "neural_masks", "neural_masks_integrated")
MODELS = tuple(f"{p}_{b}" for p in PIPELINES for b in VARIANTS) + (
    "point_integrated_legacy", *(f"{name}_legacy" for name in CONTROLS))
METRICS = ("mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2", "signed_bias", "log_signed_bias", "q90_mae")


def comparison_definitions():
    """Thirty fixed contrasts; no outcome-based ranking or selection."""
    rows = []
    for k in (0, 3, 5):
        for reference, role in (("point_integrated_legacy", "accepted_vs_general"),
                                ("tree_chemistry_selected", "accepted_neural_vs_chemical_tree")):
            rows.append((f"accepted_vs_{reference}_k{k}", "neural_chemistry_integrated_selected", k,
                         reference, k, role))
    for pipeline in PIPELINES:
        for k in (3, 5):
            for candidate, reference, role in (("selected", "legacy", "selected_support_coordinates"),
                    ("chemistry_aug", "masks_aug", "chemical_values_beyond_mask_coordinates")):
                rows.append((f"{pipeline}_{candidate}_vs_{reference}_k{k}", f"{pipeline}_{candidate}", k,
                             f"{pipeline}_{reference}", k, role))
    # Matched legacy coordinates isolate decoder inputs from basis selection.
    for suffix in ("", "_integrated"):
        for control in ("no_aux", "masks"):
            for k in (0, 3, 5):
                rows.append((f"chemistry_vs_{control}{suffix}_k{k}", f"neural_chemistry{suffix}_legacy", k,
                             f"neural_{control}{suffix}_legacy", k, "matched_decoder_information"))
    return rows


def validate_panel(frame, *, expected_models=MODELS):
    """Require new partitions, complete model/K panels and identical queries."""
    missing = set(COLUMNS) | {"ph_available", "ec_available"}
    missing -= set(frame)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    keys = ["split_seed", "seed", "model_name", "k", "cell"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Empty or duplicate prediction panel")
    numeric = ["y_true", "y_pred", "ecological_novelty", "upstream_support"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions or descriptors")
    if (frame[["y_true", "y_pred"]] < 0).any().any() or not frame.upstream_support.between(0, 1).all():
        raise ValueError("Invalid DOC concentration or support fraction")
    if frame.cell.dtype.kind not in "iu" or (frame.cell < 0).any() or frame[["station", "month"]].isna().any().any():
        raise ValueError("Invalid station/month/cell identity")
    for field in ("ph_available", "ec_available"):
        if not np.isin(frame[field], [False, True]).all():
            raise ValueError("Nonbinary chemistry availability")
    identity_fields = ["station", "month", "y_true", "ph_available", "ec_available"]
    if frame.groupby("cell")[identity_fields].nunique().ne(1).any().any():
        raise ValueError("Cell identity, truth or auxiliary availability changes across panels")
    if frame.groupby(["station", "month"]).cell.nunique().ne(1).any():
        raise ValueError("Station/month pair maps to multiple cells")
    pairs = set(map(tuple, frame[["split_seed", "seed"]].drop_duplicates().to_numpy()))
    if pairs != {(split, seed) for split in SPLITS for seed in SEEDS}:
        raise ValueError("Exactly nine fresh partition-by-seed packages are required")
    expected = {(model, k) for model in expected_models for k in KS}
    for split, part in frame.groupby("split_seed"):
        reference = None
        for (_, _, _), group in part.groupby(["seed", "model_name", "k"]):
            query = group.sort_values("cell")[["cell", *identity_fields]].reset_index(drop=True)
            if reference is None:
                reference = query
            elif not query.equals(reference):
                raise ValueError(f"Fixed query differs across model/K/seeds in partition {split}")
        for seed, run in part.groupby("seed"):
            actual = set(map(tuple, run[["model_name", "k"]].drop_duplicates().to_numpy()))
            if actual != expected:
                raise ValueError(f"Incomplete model/K panel in split {split}, seed {seed}")


def metric_summary(panel, thresholds):
    """Average run metrics over seeds, then weight the three partitions equally."""
    rows = []
    for (split, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        y, p = group.y_true.to_numpy(float), group.y_pred.to_numpy(float)
        z, zp = np.log1p(y), np.log1p(p)
        error, log_error = p-y, zp-z
        tail = y >= thresholds[(int(split), int(seed))]
        variance, log_variance = np.square(y-y.mean()).sum(), np.square(z-z.mean()).sum()
        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
            "n_query_cells": len(y), "n_stations": group.station.nunique(), "q90_n": int(tail.sum()),
            "q90_unstable": int(tail.sum()) < 20,
            "mae": np.abs(error).mean(), "rmse": np.sqrt(np.square(error).mean()),
            "r2": 1-np.square(error).sum()/variance if variance else np.nan,
            "log_mae": np.abs(log_error).mean(), "log_rmse": np.sqrt(np.square(log_error).mean()),
            "log_r2": 1-np.square(log_error).sum()/log_variance if log_variance else np.nan,
            "signed_bias": error.mean(), "log_signed_bias": log_error.mean(),
            "q90_mae": np.abs(error[tail]).mean() if tail.any() else np.nan})
    runs = pd.DataFrame(rows)
    parts = runs.groupby(["split_seed", "model_name", "k"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS}, n_seeds=("seed", "size"),
        mae_sd_seed=("mae", "std"), n_query_cells=("n_query_cells", "first"),
        n_stations=("n_stations", "first"), q90_n=("q90_n", "first"), q90_unstable=("q90_unstable", "any"))
    curves = parts.groupby(["model_name", "k"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS}, n_splits=("split_seed", "size"),
        mae_sd_split=("mae", "std"), q90_nonempty_partitions=("q90_mae", "count"),
        r2_defined_partitions=("r2", "count"), log_r2_defined_partitions=("log_r2", "count"))
    for metric, count in (("q90_mae", "q90_nonempty_partitions"), ("r2", "r2_defined_partitions"),
                           ("log_r2", "log_r2_defined_partitions")):
        curves.loc[curves[count].ne(len(SPLITS)), metric] = np.nan
    return runs, parts, curves


def check_selected_curves(frame, selection):
    """Check that reported selected curves copy source-chosen curves, exactly."""
    rows = []
    for pipeline in PIPELINES:
        for k in KS:
            panels = {basis: frame[frame.model_name.eq(f"{pipeline}_{basis}") & frame.k.eq(k)]
                      .sort_values("cell") for basis in VARIANTS}
            chosen = selection["choices"][pipeline][str(k)]["basis"]
            if chosen not in BASES:
                raise ValueError("Unknown source-selected representation")
            np.testing.assert_array_equal(panels["selected"].cell, panels[chosen].cell)
            np.testing.assert_array_equal(panels["selected"].y_pred, panels[chosen].y_pred)
            if k in (0, 1):
                for basis in BASES:
                    np.testing.assert_array_equal(panels[basis].y_pred, panels["legacy"].y_pred)
            for basis in VARIANTS:
                inactive = ~(panels[basis].ph_available | panels[basis].ec_available)
                np.testing.assert_array_equal(panels[basis].loc[inactive, "y_pred"].to_numpy(),
                    panels["legacy"].loc[~(panels["legacy"].ph_available | panels["legacy"].ec_available), "y_pred"].to_numpy())
            rows.append({"pipeline": pipeline, "k": k, "selected_basis": chosen,
                         "selected_exact": True, "inactive_fallback_exact": True})
    return rows


def load_panel(root):
    frames, sources, thresholds, states, controls, training, validations, pca = [], [], {}, [], [], [], [], []
    expected = {f"split{s}_seed{r}" for s in SPLITS for r in SEEDS}
    if {p.name for p in (root / "runs").glob("split*_seed*") if p.is_dir()} != expected:
        raise ValueError("Exactly the nine fresh production run directories are required")
    snapshot_path = root / "runtime_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources.append({"path": str(snapshot_path), "sha256": sha256(snapshot_path)})
    for name, recorded in snapshot.items():
        archived = root / "code_snapshot" / name
        if sha256(archived) != recorded:
            raise ValueError(f"Execution archive changed: {archived}")
        sources.append({"path": str(archived), "sha256": recorded})
    common_grid, deployment = None, None
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if ((config["split_seed"], config["seed"]) != (split, seed)
                    or config["experiment"] != "doc_chemistry_confirmation_v1" or config["smoke"]
                    or set(config["models"]) != set(MODELS) or tuple(config["k_values"]) != KS
                    or config["selection_role"] != "source_validation" or config["inference_roles"] != ["train"]
                    or config["historical_fitted_models_reused"] is not False
                    or config["source_neural_is_oof"] is not False or config["source_forest_is_oof"] is not True
                    or config["runtime_snapshot_hash"] != config_digest(snapshot)):
                raise ValueError(f"Unexpected fresh confirmation settings: {run}")
            if (config["initial_recipe"]["initial"]["max_epochs"] != 20
                    or config["initial_recipe"]["projector"]["epochs"] != 100
                    or config["initial_recipe"]["memory"]["epochs"] != 30
                    or config["native_recipe"]["initializer_epochs"] != 30
                    or config["native_recipe"]["off_epochs"] != 120 or config["chemical_recipe"]["epochs"] != 120):
                raise ValueError("Smoke or changed training budgets cannot enter the confirmation report")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            identity = {"split_seed": split, "seed": seed}
            availability = pd.read_parquet(run / "predictions.parquet", columns=["ph_available", "ec_available"])
            for column in availability:
                frame[column] = availability[column].to_numpy()
            selection = read_bound_json(run, "chemical/basis_selection.json", completion, sources)
            stage = read_bound_json(run, "stage_summary.json", completion, sources)
            validation = pd.read_csv(bound_file(run, "source_validation.csv", completion, sources))
            if (len(validation) != len(MODELS)*len(KS) or validation.duplicated(["model_name", "k"]).any()
                    or set(validation.model_name) != set(MODELS) or set(validation.k) != set(KS)
                    or not np.isfinite(validation[["mae", "active_mae"]]).all().all()):
                raise ValueError("Incomplete or nonfinite source-validation curves")
            state = {**identity, "config": {**config, "gamma_k0": stage["gamma_k0"]},
                "selection": selection, "validation": validation,
                "adapters": read_bound_json(run, "chemical/adapters.json", completion, sources),
                "mixers": read_bound_json(run, "chemical/mixers.json", completion, sources)}
            states.append(state)
            validations.append(validation.assign(**identity))
            controls.extend({**identity, **row} for row in check_selected_curves(frame, selection))
            definitions = read_bound_json(run, "chemical/basis_definition.json", completion, sources)
            for name, definition in definitions.items():
                np.testing.assert_array_equal(definition["source_station_indices"], config["source_station_ids"])
                pca.append({**identity, "basis": name, **verify_pca_state(definition)})
            for mode in ("no_aux", "masks", "chemistry"):
                model = read_bound_json(run, f"chemical/neural_{mode}.json", completion, sources)
                if model["model_class"] != "NonlinearChemistryHead" or model["selection_role"] != "source_validation":
                    raise ValueError("Wrong matched chemical head or checkpoint selection role")
                training.append({**identity, "mode": mode, **{name: model[name] for name in (
                    "best_epoch", "epochs_run", "selected_scale", "initial_validation_mae",
                    "trainable_parameter_count", "phi_parameter_distance")},
                    "selected_validation_mae": next(row["validation_mae"] for row in model["trace"]
                        if row["epoch"] == model["best_epoch"])})
            columns = ["cell", "station", "month", "ph_available", "ec_available", "doc_observed"]
            full = pd.read_parquet(bound_file(run, "full_grid.parquet", completion, sources), columns=columns).sort_values("cell")
            if full.cell.duplicated().any() or (common_grid is not None and not full.equals(common_grid)):
                raise ValueError("Full-grid identity or auxiliary availability differs across refits")
            observed = full.set_index("cell").loc[frame.cell, ["station", "month", "ph_available", "ec_available", "doc_observed"]]
            np.testing.assert_array_equal(observed[["station", "month", "ph_available", "ec_available"]].to_numpy(),
                frame[["station", "month", "ph_available", "ec_available"]].to_numpy())
            if not observed.doc_observed.all():
                raise ValueError("Evaluation query is not an observed DOC cell")
            if common_grid is None:
                common_grid, deployment = full, full_grid_availability(attach_availability(full))
            frames.append(frame)
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across training seeds for a fixed role assignment")
    selection_tables = selection_records(states)
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel)
    return (attach_availability(panel), thresholds, sources, selection_tables, pd.DataFrame(controls),
            pd.DataFrame(training), pd.concat(validations, ignore_index=True), pd.DataFrame(pca), deployment)


def write_report(out, tables, draws):
    curves, effects, profiles = tables["k_curves"], tables["paired_effects"], tables["error_profiles"]
    lines = ["# Fresh station assignments: chemical-state DOC reconstruction", "",
        "Nine complete production packages refit the accepted DOC recipe on station-partition seeds242–244",
        "and training seeds42–44. These are fresh role assignments on the same ST357 Mississippi cohort,",
        "not an independent external basin or a new set of field measurements. Historical fitted predictors",
        "are not reused. All twenty fixed curves and all K values are retained; target results select none.", "",
        "The model combines local/environmental prediction, a temporal residual expert, ecological transfer",
        "and a nonlinear current-month chemistry decoder. The retained spatial self path has no river messages.",
        "Source forests are station-blocked OOF; the complete source neural pipeline is not OOF.",
        "Checkpoint, mixing and support-coordinate choices use source-validation data. Station support is",
        "retrospective and can postdate a query. Known pH/conductance at a DOC-held station is a different",
        "information setting from a completely unmonitored station.", "",
        "Metrics average three training seeds within each partition, then weight the three partitions equally.",
        "R² and RMSE are averaged run metrics, not metrics computed after pooling all predictions.",
        f"The {len(comparison_definitions())} fixed contrasts use {draws:,} paired whole-station bootstrap draws.",
        "A global station's sampled multiplicity is shared across partitions. Repeated seeds and overlapping",
        "role assignments do not count as additional ecological observations. Intervals are descriptive",
        "paired intervals without multiplicity adjustment; a crossing-zero interval does not establish equivalence.", "",
        "## Whole K curves", "",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² | K5 log MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | " + " | ".join(f"{values.loc[k, 'mae']:.6f}" for k in KS)
            + f" | {values.loc[5, 'rmse']:.6f} | {values.loc[5, 'r2']:.6f} | {values.loc[5, 'log_mae']:.6f} |")
    lines += ["", "## Fixed paired comparisons", "", "Negative error deltas favor the candidate. Positive bias means overprediction.", "",
        "| Contrast | ΔMAE [95% CI] | Relative gain % [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Better partitions / packages |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, *_ in comparison_definitions():
        group = effects[effects.comparison.eq(name)]
        primary = group[group.metric.eq("mae") & group.region.eq("overall")].iloc[0]
        def format_effect(region, group=group):
            row = group[group.metric.eq("mae") & group.region.eq(region)].iloc[0]
            if row.status != "estimated":
                return "not estimable across all partitions"
            return f"{row.delta_value:+.6f} [{row.delta_ci_low:+.6f}, {row.delta_ci_high:+.6f}]"
        lines.append(f"| {name} | {format_effect('overall')} | {primary.relative_gain_pct:+.3f} "
            f"[{primary.gain_ci_low_pct:+.3f}, {primary.gain_ci_high_pct:+.3f}] | {format_effect('q90')} | "
            f"{format_effect('nontail')} | {int(primary.improved_splits)}/3; {int(primary.improved_split_seed_pairs)}/9 |")
    lines += ["", "## Tail and ordinary DOC", "",
        "Q90 thresholds come from source-training DOC within each partition, including threshold ties.",
        "Counts below20 are marked unstable. Bias is prediction minus observation; negative tail bias",
        "indicates underprediction. See classification tables for recall, precision and false-high rates.", "",
        "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias |",
        "|---|---:|---:|---:|---:|---:|"]
    main = ("point_integrated_legacy", "neural_chemistry_integrated_legacy", "neural_chemistry_integrated_selected", "tree_chemistry_selected")
    for model in main:
        for k in (0, 3, 5):
            rows = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            lines.append(f"| {model} | {k} | {rows.loc['q90', 'mae']:.6f} | {rows.loc['q90', 'signed_bias']:+.6f} | "
                f"{rows.loc['nontail', 'mae']:.6f} | {rows.loc['nontail', 'signed_bias']:+.6f} |")
    lines += ["", "## Chemistry availability", "",
        "The availability appendix reports both/one/neither auxiliary measurement, including empty strata.",
        "Observed-DOC test availability and genuinely missing-DOC grid availability are shown separately.",
        "This distinguishes measured chemistry as an added input from availability-only information.", "",
        "| Cohort | Availability | Cells | Fraction |", "|---|---|---:|---:|"]
    for row in tables["deployment_availability"].itertuples():
        lines.append(f"| {row.cohort} | {row.aux_available} | {row.n_station_months} | {100*row.fraction:.3f}% |")
    lines += ["", "## Source-validation selections", "",
        "Representation choice is part of the method, not a target-selected splice across K values.",
        "Raw legacy, availability-coordinate and chemistry-coordinate curves stay in every table.", "",
        "| Pipeline | K | Legacy / masks / chemistry choices | Mean selected active-validation MAE |",
        "|---|---:|---:|---:|"]
    for (pipeline, k), group in tables["basis_choices"].groupby(["pipeline", "k"]):
        count = "/".join(str(int(group.basis.eq(b).sum())) for b in BASES)
        lines.append(f"| {pipeline} | {k} | {count} | {group.active_mae.mean():.6f} |")
    lines += ["", "Station responses, positive-gain concentration, all seed/partition directions and the full",
              "native/log metrics are saved as adjacent CSV files. No automatic model-promotion rule is applied.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, selections, controls, training, validation, pca, deployment = load_panel(args.root)
    tables = {}
    tables["metrics_by_run"], tables["metrics_by_partition"], tables["k_curves"] = metric_summary(panel, thresholds)
    tables["error_profiles_by_run"], tables["error_profiles_by_partition"], tables["error_profiles"] = error_profiles(panel, thresholds)
    tables["classification_by_run"], tables["classification_by_partition"], tables["classification"] = classification_metrics(panel, thresholds)
    names = ("paired_effects", "directions_by_seed", "directions_by_partition", "station_responses",
             "global_station_contributions", "gain_loss_concentration")
    tables.update(zip(names, compare(panel, thresholds, args.bootstrap_draws, comparison_definitions()), strict=True))
    tables["paired_effects"] = pd.concat((tables["paired_effects"], recall_comparisons(
        panel, thresholds, args.bootstrap_draws, comparison_definitions())), ignore_index=True)
    tables.update(zip(("adapter_choices", "mixing_choices", "mixing_candidate_scores", "adapter_candidate_scores",
                       "basis_choices", "basis_candidate_scores"), selections, strict=True))
    tables.update(zip(("availability_by_run", "availability_by_partition", "availability_summary", "query_population"),
                      availability_profiles(panel, thresholds), strict=True))
    tables.update(control_checks=controls, training_selections=training, source_validation=validation,
                  chemical_pca=pca, deployment_availability=deployment)
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    write_report(out, tables, args.bootstrap_draws)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in (
        Path(__file__), Path("scripts/analyze_doc_chemistry_support_v1.py"),
        Path("scripts/analyze_doc_encoder_residual_v1.py"), Path("scripts/analyze_doc_tail_residual_v1.py"),
        Path("scripts/analyze_unified_doc_spatial.py"), Path("scripts/analyze_unified_doc_spatial_v2.py"),
        Path("scripts/analyze_unified_doc_spatial_v3.py"), Path("scripts/analyze_doc_selective_residual_v1.py")))
    catalog = {item["path"]: item for item in sources}
    (out / "sources.json").write_text(json.dumps({"complete": True, "packages": 9, "partitions": list(SPLITS),
        "training_seeds": list(SEEDS), "bootstrap_draws": args.bootstrap_draws,
        "scope": "fresh role assignment on the same ST357 cohort; not external validation",
        "estimator": "seed-mean within partition, equal partition weights; joint station bootstrap",
        "sources": [catalog[key] for key in sorted(catalog)]}, indent=2)+"\n")
    print(json.dumps({"analysis": str(out), "packages": 9, "models": len(MODELS),
                      "comparisons": len(comparison_definitions()), "bootstrap_draws": args.bootstrap_draws}))


if __name__ == "__main__":
    main()
