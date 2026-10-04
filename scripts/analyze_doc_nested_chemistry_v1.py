"""Describe nested chemical station calibration on source validation only.

The completed parent already used validation labels for model selection.
Held-station folds therefore assess the added adapter conditionally, rather
than providing an independent confirmation of the full reconstruction model.
This script reads only the new validation products and small fitted states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary

from river_graph.experiments.provenance import run_identity_sha256
from river_graph.models.nested_chemical_adapter import NestedChemicalAdapter

ROOT = Path("experiments/phase4_transfer/doc_nested_chemistry_v1")
SPLITS, SEEDS, KS = (242, 243, 244), (42, 43, 44), (0, 1, 3, 5)
GENERAL = "point_integrated_legacy"
JOINT = "neural_chemistry_integrated_selected"
LEGACY = "neural_chemistry_integrated_legacy"
TREE = "tree_chemistry_selected"
NESTED = "neural_chemistry_integrated_nested"
MASKS = "neural_chemistry_integrated_nested_masks"
MODELS = (GENERAL, JOINT, LEGACY, TREE, NESTED, MASKS)
REGIONS = ("overall", "chemistry_available", "q90", "ordinary")
EXTRA_METRICS = ("active_mae", "station_equal_mae", "q90_signed_bias", "ordinary_mae")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def config_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def comparison_definitions():
    """Eight fixed adapter contrasts, without selecting a winning K or model."""
    return [{"comparison": f"nested_vs_{role}_k{k}", "candidate": NESTED,
             "reference": reference, "k": k, "comparison_role": role}
            for k in (3, 5) for reference, role in (
                (LEGACY, "legacy"), (JOINT, "joint"), (GENERAL, "general"),
                (MASKS, "availability_only"))]


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
        raise ValueError("Empty or duplicate validation panel")
    numeric = ["y_true", "y_pred", "chemical_delta", "chemical_support_count"]
    if (not np.isfinite(frame[numeric].to_numpy(float)).all()
            or (frame[["y_true", "y_pred", "chemical_support_count"]] < 0).any().any()
            or frame.cell.dtype.kind not in "iu" or (frame.cell < 0).any()
            or frame[["station", "month"]].isna().any().any()):
        raise ValueError("Invalid validation values or cell identities")
    for field in ("ph_available", "ec_available", "aux_available"):
        if not np.isin(frame[field].to_numpy(), (False, True)).all():
            raise ValueError("Nonbinary chemical availability")
    active = frame.ph_available.to_numpy(bool) | frame.ec_available.to_numpy(bool)
    if not np.array_equal(frame.aux_available.to_numpy(bool), active):
        raise ValueError("Chemical availability does not match input masks")
    if "visibility_role" in frame and not frame.visibility_role.eq("val").all():
        raise ValueError("Only source-validation roles may enter this development analysis")
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
        raise ValueError("Exactly nine development packages are required")
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


def summarize(panel, thresholds):
    """Native run metrics: seed mean within partition, then equal partitions."""
    runs, _, _ = metric_summary(panel, thresholds)
    extra = []
    for (split, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        error = group.y_pred.to_numpy(float) - group.y_true.to_numpy(float)
        active = group.aux_available.to_numpy(bool)
        high = group.y_true.to_numpy(float) >= thresholds[(int(split), int(seed))]
        station_error = pd.DataFrame({"station": group.station.to_numpy(), "error": np.abs(error)})
        extra.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
            "active_mae": np.abs(error[active]).mean() if active.any() else np.nan,
            "station_equal_mae": station_error.groupby("station").error.mean().mean(),
            "q90_signed_bias": error[high].mean() if high.any() else np.nan,
            "ordinary_mae": np.abs(error[~high]).mean() if (~high).any() else np.nan,
            "n_active": int(active.sum())})
    runs = runs.merge(pd.DataFrame(extra), on=["split_seed", "seed", "model_name", "k"],
                      validate="one_to_one")
    metrics = ("mae", "rmse", "r2", "log_mae", "log_rmse", "log_r2", "signed_bias",
               "log_signed_bias", "q90_mae", *EXTRA_METRICS)
    parts = runs.groupby(["split_seed", "model_name", "k"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in metrics}, n_seeds=("seed", "nunique"),
        mae_sd_seed=("mae", "std"), n_query_cells=("n_query_cells", "first"),
        n_stations=("n_stations", "first"), n_active=("n_active", "first"),
        q90_n=("q90_n", "first"), q90_unstable=("q90_unstable", "any"))
    curves = parts.groupby(["model_name", "k"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in metrics}, n_splits=("split_seed", "nunique"),
        mae_sd_split=("mae", "std"), q90_unstable_any=("q90_unstable", "any"),
        **{f"{metric}_defined_partitions": (metric, "count") for metric in metrics})
    for metric in metrics:
        curves.loc[curves[f"{metric}_defined_partitions"].ne(len(SPLITS)), metric] = np.nan
    return runs, parts, curves


def compare(panel, thresholds):
    """Descriptive paired effects, preserving equal partition/seed weights."""
    run_rows, station_rows = [], []
    for definition in comparison_definitions():
        a = panel[panel.model_name.eq(definition["candidate"]) & panel.k.eq(definition["k"])]
        b = panel[panel.model_name.eq(definition["reference"]) & panel.k.eq(definition["k"])]
        paired = a.merge(b[["split_seed", "seed", "cell", "y_pred"]].rename(
            columns={"y_pred": "reference_pred"}), on=["split_seed", "seed", "cell"],
            how="inner", validate="one_to_one")
        if len(paired) != len(a) or len(a) != len(b):
            raise ValueError("Unmatched validation comparison")
        for (split, seed), full in paired.groupby(["split_seed", "seed"]):
            high = full.y_true.to_numpy(float) >= thresholds[(int(split), int(seed))]
            regions = (np.ones(len(full), dtype=bool), full.aux_available.to_numpy(bool), high, ~high)
            for region, selected in zip(REGIONS, regions, strict=True):
                group = full.loc[selected]
                if group.empty:
                    run_rows.append({**definition, "split_seed": split, "seed": seed, "region": region,
                        "n": 0, "candidate_mae": np.nan, "reference_mae": np.nan,
                        "delta_mae": np.nan, "candidate_better": False, "q90_unstable": region == "q90"})
                    continue
                candidate = np.abs(group.y_pred.to_numpy() - group.y_true.to_numpy())
                reference = np.abs(group.reference_pred.to_numpy() - group.y_true.to_numpy())
                delta = candidate.mean() - reference.mean()
                run_rows.append({**definition, "split_seed": split, "seed": seed, "region": region,
                    "n": len(group), "candidate_mae": candidate.mean(), "reference_mae": reference.mean(),
                    "delta_mae": delta, "candidate_better": delta < 0,
                    "q90_unstable": region == "q90" and len(group) < 20})
                if region == "overall":
                    temporary = group[["station"]].assign(gain=reference-candidate)
                    for station, sub in temporary.groupby("station"):
                        station_rows.append({**definition, "split_seed": split, "seed": seed,
                            "station": station, "n_station_cells": len(sub), "n_partition_cells": len(group),
                            "mae_reduction": sub.gain.mean(), "weighted_gain": sub.gain.sum()/len(group)})
    runs = pd.DataFrame(run_rows)
    keys = ["comparison", "candidate", "reference", "k", "comparison_role", "region"]
    parts = runs.groupby([*keys, "split_seed"], as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
        delta_mae=("delta_mae", "mean"), n=("n", "first"),
        improved_seeds=("candidate_better", "sum"), n_seeds=("seed", "nunique"),
        q90_unstable=("q90_unstable", "any"))
    parts["candidate_better"] = parts.delta_mae < 0
    effects = parts.groupby(keys, as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
        delta_mae=("delta_mae", "mean"), improved_partitions=("candidate_better", "sum"),
        n_partitions=("split_seed", "nunique"), improved_seed_fits=("improved_seeds", "sum"),
        n_seed_fits=("n_seeds", "sum"), q90_unstable_any=("q90_unstable", "any"),
        n_nonempty_partitions=("n", lambda values: int(values.gt(0).sum())))
    effects["relative_gain_pct"] = np.where(effects.reference_mae > 0,
                                           -100*effects.delta_mae/effects.reference_mae, np.nan)
    effects["status"] = np.select([effects.n_nonempty_partitions.eq(0),
        effects.n_nonempty_partitions.eq(len(SPLITS))], ["empty_region", "complete"], default="missing_partition")
    for field in ("candidate_mae", "reference_mae", "delta_mae", "relative_gain_pct"):
        effects.loc[effects.status.ne("complete"), field] = np.nan
    stations = pd.DataFrame(station_rows)
    weighted = stations.groupby(["comparison", "k", "station", "split_seed"], as_index=False).agg(
        weighted_gain=("weighted_gain", "mean"), mae_reduction=("mae_reduction", "mean"),
        n_station_cells=("n_station_cells", "first"))
    weighted["weighted_gain"] /= len(SPLITS)
    global_stations = weighted.groupby(["comparison", "k", "station"], as_index=False).agg(
        weighted_gain=("weighted_gain", "sum"), n_role_assignments=("split_seed", "nunique"))
    concentration = []
    for (name, k), group in global_stations.groupby(["comparison", "k"]):
        positive, harm = group.weighted_gain.clip(lower=0), (-group.weighted_gain).clip(lower=0)
        concentration.append({"comparison": name, "k": k, "net_mae_reduction": group.weighted_gain.sum(),
            "positive_gain_mass": positive.sum(), "harm_mass": harm.sum(),
            "stations_improved": int((group.weighted_gain > 0).sum()),
            "stations_worsened": int((group.weighted_gain < 0).sum()),
            "stations_equal": int((group.weighted_gain == 0).sum()), "n_stations_unique": len(group),
            "top5_positive_gain_share": positive.nlargest(5).sum()/positive.sum() if positive.sum() else np.nan,
            "top5_harm_share": harm.nlargest(5).sum()/harm.sum() if harm.sum() else np.nan})
    return effects, runs, parts, stations, global_stations, pd.DataFrame(concentration)


def correction_diagnostics(panel):
    rows = []
    for (split, seed, model, k), group in panel[panel.model_name.isin((NESTED, MASKS))].groupby(
            ["split_seed", "seed", "model_name", "k"]):
        delta = np.abs(group.chemical_delta.to_numpy(float))
        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
            "n": len(group), "changed_fraction": float((delta != 0).mean()),
            "chemical_log_increment_mean_abs": float(delta.mean()),
            "chemical_log_increment_q95_abs": float(np.quantile(delta, .95)),
            "chemical_log_increment_max_abs": float(delta.max()),
            "chemical_support_count_mean": float(group.chemical_support_count.mean()),
            "two_or_more_chemical_support_fraction": float(group.chemical_support_count.ge(2).mean())})
    return pd.DataFrame(rows)


def bound_file(run, name, completion, sources):
    path = run / name
    recorded = completion["files"].get(name)
    actual = sha256(path)
    if actual != recorded:
        raise ValueError(f"Changed or unbound validation artifact: {path}")
    sources.append({"path": str(path), "sha256": actual})
    return path


def read_product(run, name, config, completion, sources):
    """Check the product identity without loading any parent forest caches."""
    meta_name = Path(name).with_suffix(".meta.json").name
    metadata = json.loads(bound_file(run, meta_name, completion, sources).read_text())
    path = bound_file(run, name, completion, sources)
    digest = config_digest(config)
    expected = {"config_hash": digest, "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "run_identity_sha256": run_identity_sha256(digest, config["started_at"], config["runtime_snapshot_hash"]),
        "prediction_sha256": sha256(path), "selection_role": "source_validation"}
    if (config_digest(metadata["config"]) != digest
            or any(metadata.get(key) != value for key, value in expected.items())):
        raise ValueError(f"Validation sidecar identity differs: {path}")
    if not metadata.get("model_files"):
        raise ValueError(f"Validation sidecar lacks fitted state bindings: {path}")
    for file_name, recorded in metadata.get("model_files", {}).items():
        if sha256(bound_file(run, file_name, completion, sources)) != recorded:
            raise ValueError(f"Changed small fitted state: {run / file_name}")
    frame = pd.read_parquet(path)
    if (len(frame) != metadata["rows"] or not frame.split_seed.eq(config["split_seed"]).all()
            or not frame.seed.eq(config["seed"]).all()):
        raise ValueError(f"Validation row/run identity differs: {path}")
    return frame


def state_records(state, identity, mode):
    """Expose candidate choices without mistaking tuning scores for replication."""
    NestedChemicalAdapter.from_dict(state)
    chosen = state["selection"]
    full = {**identity, "mode": mode, **chosen,
        "n_validation_stations": len(state["validation_stations"]),
        "n_source_active_rows": state["n_source_active_rows"],
        "selection_mae_station_equal_k_equal": next(row["mae"] for row in state["selection_scores"]
            if row["strength"] == chosen["strength"] and row["ridge_strength"] == chosen["ridge_strength"])}
    candidates = [{**identity, "mode": mode, **row,
                   "mae_k3": None if row["mae_by_k"] is None else row["mae_by_k"]["3"],
                   "mae_k5": None if row["mae_by_k"] is None else row["mae_by_k"]["5"]}
                  for row in state["selection_scores"]]
    folds = []
    for fold in state["fold_diagnostics"]:
        selected = [row for row in fold["candidate_scores"]
                    if row["strength"] == fold["strength"] and row["ridge_strength"] == fold["ridge_strength"]]
        if len(selected) != 1:
            raise ValueError("Fold choice is absent or repeated in candidate scores")
        expected = min(fold["candidate_scores"], key=lambda row: (
            row["selection_mae"], row["strength"], -row["ridge_strength"]))
        if expected != selected[0]:
            raise ValueError("Fold choice differs from the shared K3/K5 selection rule")
        if (fold["choice"] != {"strength": fold["strength"], "ridge_strength": fold["ridge_strength"]}
                or fold["selected_zero"] != (fold["strength"] == 0)
                or not np.isclose(fold["selection_mae"], selected[0]["selection_mae"], rtol=0, atol=1e-12)
                or not np.isclose(fold["held_mae"], np.mean(list(fold["held_mae_by_k"].values())),
                                  rtol=0, atol=1e-12)):
            raise ValueError("Fold shared-choice or summary metadata differs")
        folds.append({**identity, "mode": mode, "fold": fold["fold"],
            "strength": fold["strength"], "ridge_strength": fold["ridge_strength"],
            "selected_zero": fold["selected_zero"],
            "selection_mae_station_equal_k_equal": selected[0]["selection_mae"],
            "selection_mae_station_equal_k3": fold["selection_mae_by_k"]["3"],
            "selection_mae_station_equal_k5": fold["selection_mae_by_k"]["5"],
            "held_mae_station_equal_k3": fold["held_mae_by_k"]["3"],
            "held_mae_station_equal_k5": fold["held_mae_by_k"]["5"],
            "n_selection_stations": len(fold["selection_stations"]),
            "n_held_stations": len(fold["held_stations"]),
            "held_stations": json.dumps(fold["held_stations"]),
            "selection_stations": json.dumps(fold["selection_stations"])})
    return full, candidates, folds


def check_state_predictions(state, validation, cross_validation, model):
    """Check held-fold assignment and station-equal scores against saved rows."""
    months = state["n_months"]
    station_ids = set(state["validation_stations"])
    candidate = validation[validation.model_name.eq(model) & validation.k.isin((3, 5))]
    scores = {}
    for k, group in candidate.groupby("k"):
        stations = group.cell.to_numpy() // months
        if set(stations) != station_ids:
            raise ValueError("Saved adapter and validation station populations differ")
        errors = pd.DataFrame({"station_index": stations,
            "error": np.abs(group.y_pred.to_numpy()-group.y_true.to_numpy())})
        scores[str(k)] = float(errors.groupby("station_index").error.mean().mean())
    chosen = next(row for row in state["selection_scores"] if row["strength"] == state["selection"]["strength"]
                  and row["ridge_strength"] == state["selection"]["ridge_strength"])
    if not np.allclose([scores["3"], scores["5"]], [chosen["mae_by_k"]["3"], chosen["mae_by_k"]["5"]],
                       rtol=0, atol=1e-12):
        raise ValueError("Saved full-validation scores do not match prediction rows")
    candidate = cross_validation[cross_validation.model_name.eq(model) & cross_validation.k.isin((3, 5))]
    if "conditional_fold" not in candidate:
        raise ValueError("Conditional CV prediction lacks its held-fold identity")
    covered = []
    for fold in state["fold_diagnostics"]:
        if set(fold["held_stations"]) & set(fold["selection_stations"]):
            raise ValueError("Adapter selection and held station folds overlap")
        if set(fold["held_stations"]) | set(fold["selection_stations"]) != station_ids:
            raise ValueError("Adapter station fold is incomplete")
        covered.extend(fold["held_stations"])
        group = candidate[np.isin(candidate.cell.to_numpy()//months, fold["held_stations"])]
        if not group.conditional_fold.eq(fold["fold"]).all():
            raise ValueError("CV rows use the wrong held-station fold")
        for k in (3, 5):
            local = group[group.k.eq(k)]
            errors = pd.DataFrame({"station_index": local.cell.to_numpy()//months,
                "error": np.abs(local.y_pred.to_numpy()-local.y_true.to_numpy())})
            mae = float(errors.groupby("station_index").error.mean().mean())
            if not np.isclose(mae, fold["held_mae_by_k"][str(k)], rtol=0, atol=1e-12):
                raise ValueError("Held-fold CV scores do not match prediction rows")
    if sorted(covered) != sorted(station_ids):
        raise ValueError("Held folds do not cover each validation station exactly once")


def load_panels(root):
    panels, thresholds, sources, choices, candidates, folds = {"validation_tuning": [], "conditional_cv": []}, {}, [], [], [], []
    snapshot_path = root / "runtime_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources.append({"path": str(snapshot_path), "sha256": sha256(snapshot_path)})
    for name, recorded in snapshot.items():
        path = root / "code_snapshot" / name
        if sha256(path) != recorded:
            raise ValueError(f"Changed archived execution source: {path}")
        sources.append({"path": str(path), "sha256": recorded})
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            completion_path = run / "complete.json"
            if not completion_path.exists():
                raise ValueError(f"All nine completed packages are required: {completion_path}")
            completion = json.loads(completion_path.read_text())
            sources.append({"path": str(completion_path), "sha256": sha256(completion_path)})
            config = json.loads(bound_file(run, "config.json", completion, sources).read_text())
            if (completion["config_hash"] != config_digest(config)
                    or (config["split_seed"], config["seed"]) != (split, seed)
                    or config["experiment"] != "doc_nested_chemistry_v1"
                    or set(config["models"]) != set(MODELS) or tuple(config["k_values"]) != KS
                    or config["runtime_snapshot_hash"] != config_digest(snapshot)
                    or config["selection_role"] != "source_validation" or config["target_evaluation"] is not False
                    or config["parent_frozen"] is not True
                    or config["new_neural_fits"] != 0 or config["new_forest_fits"] != 0):
                raise ValueError(f"Unexpected nested calibration development config: {run}")
            parent = Path(config["parent_run"])
            if sha256(parent / "complete.json") != config["parent_completion_hash"]:
                raise ValueError("Parent completion binding changed")
            sources.append({"path": str(parent / "complete.json"), "sha256": config["parent_completion_hash"]})
            for name, recorded in config["parent_bindings"].items():
                path = parent / name
                if sha256(path) != recorded:
                    raise ValueError(f"Changed parent covariate/calibration artifact: {path}")
                sources.append({"path": str(path), "sha256": recorded})
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            for scope, filename in (("validation_tuning", "validation.parquet"),
                                    ("conditional_cv", "cv_predictions.parquet")):
                product = read_product(run, filename, config, completion, sources)
                kind = "conditional_adapter_cv" if scope == "conditional_cv" else "validation_tuning"
                if not product.evaluation_kind.eq(kind).all():
                    raise ValueError("Validation tuning and conditional CV product roles differ")
                panels[scope].append(product)
            for mode in ("chemistry", "masks"):
                state = json.loads(bound_file(run, f"nested_{mode}.json", completion, sources).read_text())
                choice, rows, fold_rows = state_records(state, {"split_seed": split, "seed": seed}, mode)
                choices.append(choice)
                candidates.extend(rows)
                folds.extend(fold_rows)
                check_state_predictions(state, panels["validation_tuning"][-1], panels["conditional_cv"][-1],
                                        NESTED if mode == "chemistry" else MASKS)
    frames = {scope: pd.concat(rows, ignore_index=True) for scope, rows in panels.items()}
    for frame in frames.values():
        validate_panel(frame)
    inherited = [GENERAL, JOINT, LEGACY, TREE]
    a = frames["validation_tuning"][frames["validation_tuning"].model_name.isin(inherited)]
    b = frames["conditional_cv"][frames["conditional_cv"].model_name.isin(inherited)]
    keys = ["split_seed", "seed", "model_name", "k", "cell"]
    a, b = a.sort_values(keys), b.sort_values(keys)
    if not np.array_equal(a[[*keys, "y_pred"]].to_numpy(), b[[*keys, "y_pred"]].to_numpy()):
        raise ValueError("Conditional CV changed an inherited reference model")
    return frames, thresholds, sources, pd.DataFrame(choices), pd.DataFrame(candidates), pd.DataFrame(folds)


def write_findings(output, tables, panels, choices):
    lines = ["# Nested chemical station calibration: development findings", "",
        "Only source-validation predictions are analyzed. No outer target query labels or scores are read.",
        "The backbone, chemistry decoder, legacy support calibration and ecological mixture stay fixed.",
        "The added linear chemical2 increment uses a separately selected ridge and strength shared by K3/K5.", "",
        "## Interpretation and weighting", "",
        "Full-validation scores use the same observations that select the chemical increment and are tuning evidence.",
        "Conditional station CV selects the increment on other validation stations. The parent has already used",
        "validation for checkpoint/calibration selection, so this is not a fully OOF or independent confirmation",
        "of the complete model. No model promotion or new station-transfer claim is made by this report.", "",
        "Reported reconstruction MAE pools query cells within each fitted seed, averages seeds within partition,",
        "then weights partitions equally. Adapter selection instead weights K3/K5 equally and stations equally;",
        "both objectives are exposed explicitly. Seeds repeat ecological observations and do not add sample size.", ""]
    cv = tables["conditional_cv_comparisons"]
    lines += ["## Main development result", ""]
    for k in (3, 5):
        parent = cv[cv.k.eq(k) & cv.region.eq("overall") & cv.comparison_role.eq("legacy")].iloc[0]
        joint = cv[cv.k.eq(k) & cv.region.eq("overall") & cv.comparison_role.eq("joint")].iloc[0]
        lines.append(f"At K{k}, the nested increment changes legacy MAE by {parent.delta_mae:+.6f} mg/L "
            f"({parent.relative_gain_pct:+.3f}% relative gain), with gains in {parent.improved_partitions}/"
            f"{parent.n_partitions} partitions and {parent.improved_seed_fits}/{parent.n_seed_fits} seed fits. "
            f"Against the existing joint-coordinate procedure, its MAE difference is {joint.delta_mae:+.6f} mg/L.")
    lines += ["", "These are descriptive conditional-CV results, without an independent-confirmation claim or",
              "confidence-interval threshold. The existing joint procedure remains an explicit reference.", ""]
    for scope, title in (("conditional_cv", "Conditional station CV"), ("validation_tuning", "Full-validation tuning")):
        curves = tables[f"{scope}_kcurves"]
        effects = tables[f"{scope}_comparisons"]
        lines += [f"## {title}", "", "| Model | K | MAE | Q90 MAE | Q90 bias | Station-equal MAE |", "|---|---:|---:|---:|---:|---:|"]
        for row in curves.itertuples():
            lines.append(f"| {row.model_name} | {row.k} | {row.mae:.6f} | {row.q90_mae:.6f} | "
                         f"{row.q90_signed_bias:+.6f} | {row.station_equal_mae:.6f} |")
        lines += ["", "| Nested comparison | K | ΔMAE | Relative gain | Improved partitions | Improved seed fits |", "|---|---:|---:|---:|---:|---:|"]
        for row in effects[effects.region.eq("overall")].itertuples():
            lines.append(f"| {row.comparison_role} | {row.k} | {row.delta_mae:+.6f} | "
                f"{row.relative_gain_pct:+.3f}% | {row.improved_partitions}/{row.n_partitions} | "
                f"{row.improved_seed_fits}/{row.n_seed_fits} |")
        rows = panels[scope].drop_duplicates("cell")
        occurrences = panels[scope].drop_duplicates(["split_seed", "cell"])
        lines += ["", f"Population: {len(rows):,} unique validation station-months at {rows.station.nunique():,} stations;",
                  f"{len(occurrences):,} partition-cell occurrences. Negative ΔMAE and positive relative gain favor the nested adapter.", ""]
    lines += ["## Source-selected strength", "", "| Mode | Zero-strength packages | Nonzero-strength packages |", "|---|---:|---:|"]
    for mode, group in choices.groupby("mode"):
        zero = int(group.strength.eq(0).sum())
        lines.append(f"| {mode} | {zero} | {len(group)-zero} |")
    lines += ["", "K0/K1 are exact legacy-parent controls. Inactive query chemistry, fewer than two active support rows",
        "and zero chemical increments preserve the complete parent prediction. Availability-only increments",
        "control the added coordinate information; the shared parent itself already uses measured chemistry.", "",
        "`fold_choices.csv` retains station-held choices and their station-equal K3/K5 scores. Correction sizes,",
        "Q90 bias and station gain/harm concentration remain visible even when overall gains are small.",
        "Tail counts below20 are flagged; absent strata are not silently averaged over fewer partitions.", "",
        "## Research use", "",
        "These development results inform whether a separately regularized chemical increment merits a fixed",
        "recipe replication under fresh ST357 station roles 342/343/344, with training seeds 42/43/44. That",
        "replication must refit the complete source package: source/target roles change, so earlier forest, neural,",
        "ecological and chemical-projection states cannot stand in for newly fitted models. The current 242–244",
        "target results already motivated this design and remain development information for the next recipe.",
        "No target-selected legacy/joint/nested splice is created. An external-basin claim requires separate data.", ""]
    (output / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    panels, thresholds, sources, choices, candidates, folds = load_panels(args.root)
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    tables = {"selection_choices": choices, "selection_candidates": candidates, "fold_choices": folds}
    for scope, panel in panels.items():
        runs, parts, curves = summarize(panel, thresholds)
        tables.update({f"{scope}_run_metrics": runs, f"{scope}_partition_metrics": parts,
                       f"{scope}_kcurves": curves, f"{scope}_correction_diagnostics": correction_diagnostics(panel)})
        effects, direction, split_effects, stations, global_stations, concentration = compare(panel, thresholds)
        tables.update({f"{scope}_comparisons": effects, f"{scope}_seed_effects": direction,
            f"{scope}_partition_effects": split_effects, f"{scope}_station_responses": stations,
            f"{scope}_global_station_contributions": global_stations,
            f"{scope}_gain_loss_concentration": concentration})
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    write_findings(output, tables, panels, choices)
    manifest = {"study_role": "source-validation development; conditional adapter CV only",
        "target_query_results_read": False, "model_promoted": False,
        "report_estimator": "query-cell mean per seed; seed mean per partition; equal partitions",
        "selection_estimator": "K3/K5 equal; validation stations equal",
        "sources": {row["path"]: row["sha256"] for row in sources},
        "analysis_script_sha256": sha256(__file__),
        "metric_helper_sha256": sha256(Path(__file__).with_name("analyze_doc_chemistry_confirmation_v1.py"))}
    (output / "sources.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(f"Analyzed nine source-validation packages; report: {output / 'findings.md'}")


if __name__ == "__main__":
    main()
