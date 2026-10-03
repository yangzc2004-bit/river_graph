"""Evaluate fixed daily/monthly fallback on fixed DOC station holdouts.

All pipeline variants and the same query cells are reported. The analysis never
fits a model or chooses a fallback, support adapter, or ecological mixture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_encoder_residual_v1 import check_source_identity, compare
from analyze_doc_selective_residual_v1 import classification_metrics, recall_comparisons
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import config_digest, sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import read_bound_json, summarize_metrics

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_fallback_v1")
ARMS = ("monthly", "daily", "hybrid")
SHAPES = ("constant", "gru_tuned_anchor")
REFERENCE_BASES = ("prior_encoder", "prior_encoder_integrated", "prior_ecological_affine")
REFERENCES = tuple(f"{base}_{shape}" for base in REFERENCE_BASES for shape in SHAPES)
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS + REFERENCES


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in ((5,) if shape == "constant" else (0, 5)):
            for suffix in ("", "_integrated"):
                for reference in ("daily", "monthly"):
                    rows.append((f"hybrid_vs_{reference}{suffix}_{shape}_k{k}",
                                 f"hybrid{suffix}_{shape}", k, f"{reference}{suffix}_{shape}", k,
                                 "hybrid_fallback"))
                rows.append((f"hybrid{suffix}_vs_prior_ecological_{shape}_k{k}",
                             f"hybrid{suffix}_{shape}", k, f"prior_ecological_affine_{shape}", k,
                             "previous_overall_reference"))
            for arm in ARMS:
                rows.append((f"{arm}_vs_context_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"context_{shape}", k, "fixed_context_control"))
                rows.append((f"{arm}_integrated_vs_direct_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"{arm}_{shape}", k, "ecological_integration"))
    return rows


def read_full_grid(run, config, completion, sources):
    """Verify the component product before using it for routing diagnostics."""
    path = run / "full_grid.parquet"
    identity = sha256(path)
    meta = read_bound_json(run, "full_grid.meta.json", completion, sources)
    if (completion["files"].get(path.name) != identity or meta["prediction_sha256"] != identity
            or meta["config_hash"] != config_digest(config) or config_digest(meta["config"]) != config_digest(config)
            or meta["dataset_sha256"] != config["dataset_hash"] or meta["mask_sha256"] != config["mask_hash"]
            or meta["runtime_snapshot_hash"] != config["runtime_snapshot_hash"]
            or meta["selection_role"] != "source_validation" or not meta["model_files"]):
        raise ValueError("Changed full-grid component identity")
    for name, value in meta["model_files"].items():
        if sha256(run / name) != value or completion["files"].get(name) != value:
            raise ValueError("Changed full-grid model dependency")
    full = pd.read_parquet(path)
    if len(full) != meta["rows"] or full.cell.duplicated().any():
        raise ValueError("Full-grid cell identity differs")
    sources.append({"path": str(path), "sha256": identity})
    return full.set_index("cell", drop=False)


def load_panel(root):
    frames, thresholds, sources, states, controls, routing = [], {}, [], [], [], []
    daily_identity = None
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if ((config["split_seed"], config["seed"]) != (split, seed)
                    or config["experiment"] != "doc_daily_hydro_fallback_v1"
                    or config["expert_retraining"] is not False
                    or set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS
                    or tuple(config["k_values"]) != KS or config["inference_roles"] != ["train"]
                    or config["gamma_grid"] != [0, .25, .5, 1]
                    or config["alpha_grid"] != [0, .25, .5, .75, 1]
                    or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
                raise ValueError(f"Unexpected fallback experiment settings: {run}")
            verification = Path(config["parent_verification_path"])
            if sha256(verification) != config["parent_verification_hash"]:
                raise ValueError("Changed parent replay verification")
            sources.append({"path": str(verification), "sha256": config["parent_verification_hash"]})
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            prior = Path(config["prior_run"])
            if sha256(prior / "complete.json") != config["prior_completion_hash"]:
                raise ValueError("Changed parent daily-flow package")
            previous, old_config, old_completion = read_predictions(prior, sources)
            check_source_identity(config, old_config, (
                "split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train", "query_cells",
                "source_run", "basis_run", "oof_run", "memory_run", "memory_file_hash",
                "source_completion_hash", "basis_completion_hash", "oof_completion_hash",
                "daily_features_path", "daily_features_hash", "daily_metadata_path", "daily_metadata_hash"))
            paths = (Path(config["daily_features_path"]), Path(config["daily_metadata_path"]))
            hashes = (config["daily_features_hash"], config["daily_metadata_hash"])
            if any(sha256(path) != value for path, value in zip(paths, hashes, strict=True)):
                raise ValueError("Daily hydrological feature artifact changed")
            if daily_identity is not None and hashes != daily_identity:
                raise ValueError("Daily hydrological artifacts differ across packages")
            if daily_identity is None:
                sources.extend({"path": str(path), "sha256": value}
                               for path, value in zip(paths, hashes, strict=True))
                daily_identity = hashes
            metadata = json.loads(paths[1].read_text())
            if metadata["validity_feature_indices"] != [5, 6, 7] or metadata["dataset_hash"] != config["dataset_hash"]:
                raise ValueError("Daily validity/data identity differs")
            route = read_bound_json(run, "router.json", completion, sources)
            if (route != config["route_definition"] or route["model_class"] != "DailyHydroRouter"
                    or route["version"] != 1 or route["fitted_parameters"] is not False
                    or route["rule"] != "any_numeric_descriptor_valid_selects_daily_else_monthly"
                    or route["validity_feature_indices"] != [5, 6, 7]
                    or route["daily_feature_names"] != metadata["feature_names"]
                    or route["target_label_dependency"] != "none"):
                raise ValueError("Routing rule differs from the fixed label-free fallback")
            for name in ("source", "basis", "oof", "memory"):
                path = Path(config[f"{name}_run"]) / "complete.json"
                actual = sha256(path)
                if actual != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed frozen {name} dependency")
                sources.append({"path": str(path), "sha256": actual})
            memory_run = Path(config["memory_run"])
            memory_completion = json.loads((memory_run / "complete.json").read_text())
            memory = read_bound_json(memory_run, "ecological_affine.json", memory_completion, sources)
            if (sha256(memory_run / "ecological_affine.json") != config["memory_file_hash"]
                    or memory["mode"] != "ecological_affine" or memory["selection_role"] != "source_validation"):
                raise ValueError("Changed ecological memory")
            # Refitted controls must reproduce their previous validation choices and queries.
            copied = (*REFERENCES, *(f"{base}_{shape}" for base in
                      ("context", "monthly", "daily", "monthly_integrated", "daily_integrated") for shape in SHAPES))
            for name in copied:
                current = frame[frame.model_name.eq(name)].sort_values(["k", "cell"])
                old = previous[previous.model_name.eq(name)].sort_values(["k", "cell"])
                for field in ("k", "cell", "y_true", "y_pred"):
                    np.testing.assert_array_equal(current[field], old[field])
                controls.append({"split_seed": split, "seed": seed, "model_name": name,
                                 "query_rows_all_k": len(current), "predictions_bitwise_equal": True})
            full = read_full_grid(run, config, completion, sources)
            if not np.isfinite(full[["context_pred", "monthly_pred", "daily_pred", "hybrid_pred", "ecological_memory"]]).all().all():
                raise ValueError("Nonfinite native component grid")
            old_full = read_full_grid(prior, old_config, old_completion, sources)
            np.testing.assert_array_equal(full.index, old_full.index)
            for field in ("context_pred", "monthly_pred", "daily_pred", "ecological_memory", *metadata["feature_names"]):
                np.testing.assert_array_equal(full[field], old_full[field])
            flags = full[metadata["feature_names"][5:8]].to_numpy()
            if not np.isin(flags, [0, 1]).all():
                raise ValueError("Nonbinary daily descriptor validity")
            count = flags.sum(axis=1).astype(int)
            np.testing.assert_array_equal(full.daily_numeric_valid_count, count)
            np.testing.assert_array_equal(full.uses_daily, count > 0)
            expected = np.where(count > 0, full.daily_pred, full.monthly_pred)
            np.testing.assert_array_equal(full.hybrid_pred, expected)
            raw_availability = pd.read_parquet(run / "predictions.parquet", columns=[
                "cell", "daily_numeric_valid_count", "uses_daily", *metadata["feature_names"]])
            for field in ("daily_numeric_valid_count", "uses_daily", *metadata["feature_names"]):
                np.testing.assert_array_equal(raw_availability[field], full.loc[raw_availability.cell, field])
            frame["daily_numeric_valid_count"] = frame.cell.map(full.daily_numeric_valid_count)
            if frame.daily_numeric_valid_count.isna().any():
                raise ValueError("Missing query availability")
            no_daily = frame.daily_numeric_valid_count.eq(0)
            for shape in SHAPES:
                a = frame[no_daily & frame.k.eq(0) & frame.model_name.eq(f"hybrid_{shape}")].sort_values("cell")
                b = frame[no_daily & frame.k.eq(0) & frame.model_name.eq(f"monthly_{shape}")].sort_values("cell")
                np.testing.assert_array_equal(a.cell, b.cell)
                np.testing.assert_array_equal(a.y_pred, b.y_pred)
            # Direct K0 must expose its native base, including on partially valid cells.
            for arm in ARMS:
                a = frame[frame.k.eq(0) & frame.model_name.eq(f"{arm}_constant")].sort_values("cell")
                np.testing.assert_array_equal(a.y_pred, full.loc[a.cell, f"{arm}_pred"])
            routing.extend(k0_routing_profiles(frame, full, threshold))
            adapters = read_bound_json(run, "adapters.json", completion, sources)
            mixers = read_bound_json(run, "mixers.json", completion, sources)
            old_adapters = read_bound_json(prior, "adapters.json", old_completion, sources)
            old_mixers = read_bound_json(prior, "mixers.json", old_completion, sources)
            for name in old_adapters:
                if name.startswith(("context_", "monthly_", "daily_")) and adapters[name] != old_adapters[name]:
                    raise ValueError("Refitted control adapter differs from its parent")
            for name in old_mixers:
                if name.startswith(("monthly_", "daily_")) and mixers[name] != old_mixers[name]:
                    raise ValueError("Refitted control mixer differs from its parent")
            validation_path = run / "source_validation.csv"
            if completion["files"].get(validation_path.name) != sha256(validation_path):
                raise ValueError("Changed source-validation diagnostic")
            sources.append({"path": str(validation_path), "sha256": sha256(validation_path)})
            states.append({"split_seed": split, "seed": seed, "config": config, "route": route,
                           "adapters": adapters, "mixers": mixers,
                           "validation": pd.read_csv(validation_path)})
            frames.append(frame)
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across training seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    if (panel.groupby("cell").daily_numeric_valid_count.nunique() != 1).any():
        raise ValueError("Daily availability differs across runs")
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        pivot = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS, *(f"{arm}_integrated" for arm in ARMS), *REFERENCE_BASES):
            np.testing.assert_array_equal(pivot[f"{base}_constant"], pivot[f"{base}_gru_tuned_anchor"])
    return panel, thresholds, sources, states, pd.DataFrame(controls), pd.DataFrame(routing)


def selection_records(states, panel):
    choices, mixing, scores, validation = [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        validation.append(run["validation"].assign(**identity))
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != set(DIRECT_MODELS) or set(run["mixers"]) != set(MIXED_MODELS):
            raise ValueError("Unexpected adapter/mixer names")
        for model, state in run["adapters"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong support selection role or K")
            choices.extend({**identity, "model_name": model, "k": int(k), **selection}
                           for k, selection in state["selection_by_k"].items())
        for model, state in run["mixers"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong mixture selection role or K")
            gamma_k0 = state["gamma_k0"]
            initial = [r for r in state["gamma_scores"] if r["k"] == 0]
            if gamma_k0 != min(initial, key=lambda r: (r["mae"], r["gamma"]))["gamma"]:
                raise ValueError("K0 gamma differs from saved validation selection")
            for k in KS:
                selected = state["selection_by_k"][str(k)]
                candidates = [r for r in state["gamma_scores"] if r["k"] == k]
                eligible = [r for r in candidates if k != 0 or r["gamma"] == gamma_k0]
                chosen = min(eligible, key=lambda r: (r["mae"], r["gamma"] != 0,
                                                      r["gamma"] != gamma_k0, r["gamma"]))
                if selected != {**chosen, "locked": k == 0}:
                    raise ValueError("Mixture choice differs from source-validation trace")
                if selected["gamma"] == 0:
                    direct = model.replace("_integrated_", "_")
                    a = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                    b = current[current.model_name.eq(direct) & current.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
                mixing.append({**identity, "model_name": model, "gamma_k0": gamma_k0, **selected})
                scores.extend({**identity, "model_name": model, **row} for row in candidates)
    return (*tuple(pd.DataFrame(rows) for rows in (choices, mixing, scores)),
            pd.concat(validation, ignore_index=True))






AVAILABILITY_GROUPS = (("none_valid", (0,)), ("partially_valid", (1, 2)), ("all_three_valid", (3,)))


def k0_routing_profiles(frame, full, threshold):
    """Keep native, direct and integrated no-support routing effects explicit."""
    query = frame[frame.model_name.eq("context_constant") & frame.k.eq(0)].sort_values("cell")
    cells = query.cell.to_numpy()
    truth = query.y_true.to_numpy()
    count = full.loc[cells, "daily_numeric_valid_count"].to_numpy()
    identity = {name: int(query[name].iloc[0]) for name in ("split_seed", "seed")}
    rows = []
    for stage in ("native", "direct_k0", "integrated_k0"):
        predictions = {}
        for arm in ARMS:
            if stage == "native":
                predictions[arm] = full.loc[cells, f"{arm}_pred"].to_numpy()
            else:
                suffix = "_integrated" if stage == "integrated_k0" else ""
                chosen = frame[frame.k.eq(0) & frame.model_name.eq(f"{arm}{suffix}_constant")].sort_values("cell")
                np.testing.assert_array_equal(chosen.cell, cells)
                predictions[arm] = chosen.y_pred.to_numpy()
        for label, values in AVAILABILITY_GROUPS:
            selected = np.isin(count, values)
            for arm, pred in predictions.items():
                error = pred[selected] - truth[selected]
                high = truth[selected] >= threshold
                difference = pred[selected] - predictions["monthly"][selected]
                rows.append({**identity, "stage": stage, "arm": arm, "daily_availability": label,
                             "n_query_cells": int(selected.sum()), "n_q90_cells": int(high.sum()),
                             "mae": float(np.abs(error).mean()) if error.size else np.nan,
                             "q90_mae": float(np.abs(error[high]).mean()) if high.any() else np.nan,
                             "nontail_mae": float(np.abs(error[~high]).mean()) if (~high).any() else np.nan,
                             "signed_bias": float(error.mean()) if error.size else np.nan,
                             "mean_abs_prediction_difference_from_monthly": float(np.abs(difference).mean()) if difference.size else np.nan,
                             "max_abs_prediction_difference_from_monthly": float(np.abs(difference).max()) if difference.size else np.nan,
                             "prediction_equals_monthly": bool(np.array_equal(pred[selected], predictions["monthly"][selected])) if selected.any() else None})
    return rows


def summarize_routing(profiles):
    keys = ["stage", "arm", "daily_availability"]
    means = ["mae", "q90_mae", "nontail_mae", "signed_bias", "mean_abs_prediction_difference_from_monthly"]
    parts = profiles.groupby(["split_seed", *keys], as_index=False).agg(
        **{name: (name, "mean") for name in means}, n_query_cells=("n_query_cells", "first"),
        max_abs_prediction_difference_from_monthly=("max_abs_prediction_difference_from_monthly", "max"))
    summary = parts.groupby(keys, as_index=False).agg(
        **{name: (name, "mean") for name in means}, n_nonempty_partitions=("mae", "count"),
        max_abs_prediction_difference_from_monthly=("max_abs_prediction_difference_from_monthly", "max"))
    return parts, summary


def availability_profiles(panel, thresholds):
    """Descriptive fixed strata; no bootstrap, new threshold, or selection."""
    rows = []
    for (split, seed, model, k), frame in panel[panel.k.isin((0, 5))].groupby(
            ["split_seed", "seed", "model_name", "k"]):
        for label, values in AVAILABILITY_GROUPS:
            group = frame[frame.daily_numeric_valid_count.isin(values)]
            y, prediction = group.y_true.to_numpy(), group.y_pred.to_numpy()
            error = prediction - y
            high = y >= thresholds[(int(split), int(seed))]
            rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                         "daily_availability": label,
                         "n_query_cells": len(group), "n_stations": group.station.nunique(),
                         "n_q90_cells": int(high.sum()), "n_nontail_cells": int((~high).sum()),
                         "mae": float(np.abs(error).mean()) if len(group) else np.nan,
                         "rmse": float(np.sqrt(np.square(error).mean())) if len(group) else np.nan,
                         "log_mae": float(np.abs(np.log1p(prediction)-np.log1p(y)).mean()) if len(group) else np.nan,
                         "signed_bias": float(error.mean()) if len(group) else np.nan,
                         "q90_mae": float(np.abs(error[high]).mean()) if high.any() else np.nan,
                         "nontail_mae": float(np.abs(error[~high]).mean()) if (~high).any() else np.nan,
                         "q90_unstable": int(high.sum()) < 20})
    runs = pd.DataFrame(rows)
    scores = ("mae", "rmse", "log_mae", "signed_bias", "q90_mae", "nontail_mae")
    counts = ("n_query_cells", "n_stations", "n_q90_cells", "n_nontail_cells")
    keys = ["model_name", "k", "daily_availability"]
    splits = runs.groupby(["split_seed", *keys], as_index=False).agg(
        **{name: (name, "mean") for name in scores}, **{name: (name, "first") for name in counts},
        q90_unstable=("q90_unstable", "any"), n_seeds_nonempty=("mae", "count"))
    summary = splits.groupby(keys, as_index=False).agg(
        **{name: (name, "mean") for name in scores}, n_nonempty_partitions=("mae", "count"),
        n_q90_nonempty_partitions=("q90_mae", "count"), q90_unstable_any=("q90_unstable", "any"))
    populations = []
    for label, values in AVAILABILITY_GROUPS:
        subset = panel[panel.daily_numeric_valid_count.isin(values)].drop_duplicates("cell")
        occurrences = panel[panel.daily_numeric_valid_count.isin(values)].drop_duplicates(["split_seed", "cell"])
        populations.append({"daily_availability": label,
                            "n_station_months_unique": len(subset), "n_stations_unique": subset.station.nunique(),
                            "n_split_cell_occurrences": len(occurrences)})
    summary = summary.merge(pd.DataFrame(populations), on="daily_availability", validate="many_to_one")
    return runs, splits, summary



def write_availability_report(out, profiles, routing):
    lines = ["# Routing and daily availability: descriptive appendix", "",
             "Groups are fixed by numeric descriptor-validity flags 5, 6, 7: no valid descriptor,",
             "one or two valid descriptors, and all three valid. Coverage fractions do not determine",
             "the route. Any valid descriptor selects daily; zero valid descriptors select monthly.",
             "The routing rule was specified after the preceding daily study, before these new fits.", "",
             "These are descriptive strata, without additional bootstrap endpoints or model selection.",
             "Cell errors are pooled within seed, seed means averaged within partition, and nonempty",
             "partitions equally weighted. Repeated predictions do not multiply ecological sample sizes.", "",
             "| Availability | Unique station-months | Unique stations | Partition-cell occurrences |",
             "|---|---:|---:|---:|"]
    for row in profiles.drop_duplicates("daily_availability").itertuples():
        lines.append(f"| {row.daily_availability} | {row.n_station_months_unique} | {row.n_stations_unique} | {row.n_split_cell_occurrences} |")
    lines += ["", "## Native versus direct/integrated K0", "",
              "Routing guarantees the native and direct K0 hybrid equals monthly on zero-valid cells.",
              "The integrated K0 output may differ there because ecological gamma is selected separately",
              "for the complete hybrid pipeline. Such a difference is shown rather than treated as a routing failure.", "",
              "| Stage | Arm | Availability | MAE | Q90 MAE | Ordinary MAE | Mean absolute prediction difference from monthly |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in routing.itertuples():
        lines.append(f"| {row.stage} | {row.arm} | {row.daily_availability} | {row.mae:.4f} | "
                     f"{row.q90_mae:.4f} | {row.nontail_mae:.4f} | {row.mean_abs_prediction_difference_from_monthly:.6f} |")
    lines += ["", "## Direct and integrated products, GRU support basis", "",
              "| Model | K | Availability | MAE | Q90 MAE | Ordinary MAE | Bias | Nonempty partitions | Q90 unstable |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    selected = profiles[profiles.model_name.isin(
        {f"{arm}{suffix}_gru_tuned_anchor" for arm in ARMS for suffix in ("", "_integrated")})]
    for row in selected.itertuples():
        lines.append(f"| {row.model_name} | {row.k} | {row.daily_availability} | {row.mae:.4f} | "
                     f"{row.q90_mae:.4f} | {row.nontail_mae:.4f} | {row.signed_bias:+.4f} | "
                     f"{row.n_nonempty_partitions} | {row.q90_unstable_any} |")
    lines += ["", "All context and historical references, both support paths and empty strata remain in the CSVs.", ""]
    (out / "availability_appendix.md").write_text("\n".join(lines))



def write_report(out, curves, profiles, classification, effects, mixing, controls, draws):
    lines = ["# Availability-aware daily/monthly fallback", "",
             "The monthly and daily neural experts remain frozen. A hybrid combines their native",
             "predictions using the fixed label-free availability rule: any valid daily descriptor selects daily; otherwise monthly.",
             "Support adapters and ecological mixing use source validation only, on the same fixed",
             "episodes and candidate grids. The route never reads DOC labels; its fixed availability rule",
             "was motivated by the preceding experiment, before inspecting this experiment’s target results.", "",
             "Direct and ecological-integrated pipelines are reported separately. A change in",
             "the integrated product can include a changed ecological mixture or support adapter;",
             "it is not solely the effect of switching the underlying expert.", "",
             "All 20 saved models and all K values are retained, including historical references.",
             "The copied monthly/daily controls and original references must reproduce exactly.",
             f"The reference checks contain {len(controls)} model/run comparisons, all verified exactly.", "",
             f"Intervals use {draws:,} paired whole-station bootstrap draws, jointly resampling repeated",
             "station identities across partitions. Seed means are averaged within partition; partitions",
             "have equal weight. Negative error or false-positive deltas favor the candidate; positive",
             "recall deltas indicate improved detection. Q90 uses source-training labels and includes threshold ties.",
             "These are previously examined development partitions, and source validation has been reused.",
             "No model, threshold, support count or mixture is selected using these target comparisons.", "",
             "## Complete K curves", "",
             "| Model (GRU support basis) | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        selected = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | " + " | ".join(f"{selected.loc[k, 'mae']:.4f}" for k in KS)
                     + f" | {selected.loc[5, 'rmse']:.4f} | {selected.loc[5, 'r2']:.4f} |")
    lines += ["", "Constant-only support adaptation is retained in all CSVs. Its K0 predictions duplicate",
              "the GRU-basis K0 predictions, so duplicate K0 bootstrap comparisons are omitted.", "",
              "## Tail accuracy, ordinary error and classification", "",
              "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Q90 recall | Q90 precision | False-Q90 rate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        for k in (0, 5):
            profile = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            rates = classification[classification.model_name.eq(model) & classification.k.eq(k)].iloc[0]
            tail, ordinary = profile.loc["q90"], profile.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:+.4f} | {ordinary.mae:.4f} | "
                         f"{ordinary.signed_bias:+.4f} | {100*rates.q90_recall:.2f}% | {100*rates.q90_precision:.2f}% | {100*rates.q90_false_positive_rate:.2f}% |")
    lines += ["", "Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.",
              "A lower false-positive rate alone is not better selectivity if recall also collapses. Precision is descriptive; paired intervals are reported for recall and false-positive rate.", ""]
    for role, title in (("hybrid_fallback", "Hybrid versus frozen monthly and daily pipelines"),
                        ("previous_overall_reference", "Direct and integrated hybrid versus the fixed ecological reference"),
                        ("ecological_integration", "Ecological integration versus direct residuals"),
                        ("fixed_context_control", "Direct residuals versus context")):
        lines += [f"## {title}", "", "| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |",
                  "|---|---:|---:|---:|---:|---:|"]
        selected = effects[effects.comparison_role.eq(role) & effects.region.eq("overall") & effects.metric.eq("mae")]
        for row in selected.itertuples():
            group = effects[effects.comparison.eq(row.comparison)]

            def pick(region, metric, group=group):
                return group[group.region.eq(region) & group.metric.eq(metric)].iloc[0]

            def interval(value, multiplier=1):
                if value.status != "estimated":
                    return "not estimable"
                return (f"{multiplier*value.delta_value:+.4f} "
                        f"[{multiplier*value.delta_ci_low:+.4f}, {multiplier*value.delta_ci_high:+.4f}]")

            tail, ordinary = pick("q90", "mae"), pick("nontail", "mae")
            recall, false = pick("q90", "q90_recall"), pick("nontail", "q90_false_positive_rate")
            lines.append(f"| {row.comparison} | {interval(row)} | {interval(tail)} | {interval(ordinary)} | "
                         f"{interval(recall, 100)} | {interval(false, 100)} |")
        lines.append("")
    lines += ["## Source-validation choices", "",
              "The expert-training checkpoints are frozen; this experiment does not train another backbone.",
              "The fixed router is serialized in router.json; fitted support and ecological choices are saved in CSVs.", "",
              "| Integrated model | K | Selected gamma counts |", "|---|---:|---|"]
    for (model, k), group in mixing.groupby(["model_name", "k"], sort=False):
        counts = ", ".join(f"{value:g}: {count}" for value, count in group.gamma.value_counts().sort_index().items())
        lines.append(f"| {model} | {k} | {counts} |")
    lines += ["", "Interpret direct feature contrasts before integrated ones: the latter also include changes",
              "in source-validation-selected ecological mixing and positive-K support calibration.",
              "A narrow or sign-changing contrast does not establish equivalence. Station gain/harm",
              "concentration and partition/seed directions are retained alongside the overall means.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, states, controls, routing_runs = load_panel(args.root)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_splits, profiles = error_profiles(panel, thresholds)
    class_runs, class_splits, classification = classification_metrics(panel, thresholds)
    availability_runs, availability_splits, availability = availability_profiles(panel, thresholds)
    routing_splits, routing = summarize_routing(routing_runs)
    definitions = comparison_definitions()
    effects, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws, definitions=definitions)
    effects = pd.concat([effects, recall_comparisons(panel, thresholds, args.bootstrap_draws, definitions)],
                        ignore_index=True)
    choices, mixing, gamma_scores, validation = selection_records(states, panel)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    outputs = (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
               ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
               ("error_profiles", profiles), ("classification_by_run", class_runs),
               ("classification_by_partition", class_splits), ("classification", classification),
               ("comparisons", effects), ("directions_by_seed", directions),
               ("directions_by_partition", partitions), ("station_responses", stations),
               ("global_station_contributions", global_stations), ("gain_loss_concentration", concentration),
               ("adapter_choices", choices), ("mixing_choices", mixing), ("gamma_scores", gamma_scores),
               ("source_validation_by_run", validation), ("reference_replication", controls),
               ("availability_profiles_by_run", availability_runs),
               ("availability_profiles_by_partition", availability_splits), ("availability_profiles", availability),
               ("k0_routing_profiles_by_run", routing_runs), ("k0_routing_profiles_by_partition", routing_splits),
               ("k0_routing_profiles", routing))
    for name, frame in outputs:
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, classification, effects, mixing, controls, args.bootstrap_draws)
    write_availability_report(out, availability, routing)
    output_hashes = {f"{name}.csv": sha256(out / f"{name}.csv") for name, _ in outputs}
    for name in ("findings.md", "availability_appendix.md"):
        output_hashes[name] = sha256(out / name)
    for name in ("analyze_doc_encoder_residual_v1.py", "analyze_doc_selective_residual_v1.py",
                 "analyze_doc_tail_residual_v1.py", "analyze_unified_doc_spatial.py",
                 "analyze_unified_doc_spatial_v2.py", "analyze_unified_doc_spatial_v3.py"):
        path = Path(__file__).parent / name
        sources.append({"path": str(path), "sha256": sha256(path)})
    (out / "analysis_manifest.json").write_text(json.dumps({
        "analysis_script": str(Path(__file__)), "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws, "models": MODELS, "k_values": KS,
        "comparison_count": len(definitions), "sources": sources, "outputs": output_hashes,
        "role": "same-cohort development; fixed availability route inspired by the preceding experiment",
        "estimand": "cell-pooled within seed; seed mean within partition; equal partition mean",
        "availability_strata": {label: list(values) for label, values in AVAILABILITY_GROUPS},
    }, indent=2) + "\n")
    print(curves[curves.k.isin((0, 5))][["model_name", "k", "mae", "rmse"]].to_string(index=False))
    print(f"Saved all fallback products and {len(definitions)} fixed contrasts to {out}")


if __name__ == "__main__":
    main()
