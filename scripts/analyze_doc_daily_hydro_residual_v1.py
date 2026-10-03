"""Compare daily hydrological information in the existing DOC residual encoder.

The fixed monthly, availability-only, and full-daily arms are all reported.
Target outcomes never select checkpoints, residual scales, support fits or mixtures.
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
from analyze_unified_doc_spatial import sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import read_bound_json, summarize_metrics

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
ARMS = ("monthly", "availability", "daily")
SHAPES = ("constant", "gru_tuned_anchor")
REFERENCE_BASES = ("prior_encoder", "prior_encoder_integrated", "prior_ecological_affine")
REFERENCES = tuple(f"{base}_{shape}" for base in REFERENCE_BASES for shape in SHAPES)
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS + REFERENCES
INTERACTIONS = [0, 2, 4, 28, 30, 31, 32]


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in ((5,) if shape == "constant" else (0, 5)):
            for suffix in ("", "_integrated"):
                for candidate, reference in (("daily", "monthly"), ("availability", "monthly"),
                                             ("daily", "availability")):
                    rows.append((f"{candidate}_vs_{reference}{suffix}_{shape}_k{k}",
                                 f"{candidate}{suffix}_{shape}", k, f"{reference}{suffix}_{shape}", k,
                                 "daily_information"))
            for arm in ARMS:
                rows.append((f"{arm}_vs_context_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"context_{shape}", k, "fixed_context_control"))
                rows.append((f"{arm}_integrated_vs_direct_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"{arm}_{shape}", k, "ecological_integration"))
                rows.append((f"{arm}_integrated_vs_prior_ecological_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"prior_ecological_affine_{shape}", k, "previous_overall_reference"))
    return rows


def load_panel(root, expected_epochs=120):
    """Load all completed products, preserving the original query estimand."""
    frames, thresholds, sources, states, controls = [], {}, [], [], []
    daily_identity = None
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if ((config["split_seed"], config["seed"]) != (split, seed)
                    or config["experiment"] != "doc_daily_hydro_residual_v1"
                    or set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS
                    or tuple(config["k_values"]) != KS or config["inference_roles"] != ["train"]
                    or config["tail_weight"] != 2 or config["extra_dim"] != 38
                    or config["epochs"] != expected_epochs or config["patience"] != 5
                    or config["interaction_indices"] != INTERACTIONS or config["train_memory"] is not True
                    or config["encoder_learning_rate"] != 1e-5 or config["encoder_mode"] != "last_self_ecology"
                    or config["gamma_grid"] != [0, .25, .5, 1]
                    or config["alpha_grid"] != [0, .25, .5, .75, 1]
                    or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
                raise ValueError(f"Unexpected daily-hydro experiment settings: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            paths = (Path(config["daily_features_path"]), Path(config["daily_metadata_path"]))
            hashes = (config["daily_features_hash"], config["daily_metadata_hash"])
            if any(sha256(path) != value for path, value in zip(paths, hashes, strict=True)):
                raise ValueError("Daily hydrological artifact changed")
            if daily_identity is not None and hashes != daily_identity:
                raise ValueError("Daily hydrological artifact differs across runs")
            if daily_identity is None:
                sources.extend({"path": str(path), "sha256": value}
                               for path, value in zip(paths, hashes, strict=True))
                daily_identity = hashes
            metadata = json.loads(paths[1].read_text())
            if (metadata["dataset_hash"] != config["dataset_hash"]
                    or metadata["value_feature_indices"] != [0, 1, 2]
                    or metadata["availability_feature_indices"] != [3, 4, 5, 6, 7]
                    or metadata["validity_feature_indices"] != [5, 6, 7]):
                raise ValueError("Daily hydrological dataset or column identity differs")
            prior, memory_run = Path(config["prior_run"]), Path(config["memory_run"])
            for name, path in (("prior", prior), ("memory", memory_run)):
                if sha256(path / "complete.json") != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed bound {name} package")
            previous, old_config, _ = read_predictions(prior, sources)
            check_source_identity(config, old_config, (
                "split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train",
                "source_run", "basis_run", "oof_run", "memory_run", "memory_file_hash", "patience",
                "source_completion_hash", "basis_completion_hash", "oof_completion_hash"))
            for name in ("source", "basis", "oof"):
                path = Path(config[f"{name}_run"]) / "complete.json"
                actual = sha256(path)
                if actual != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed frozen {name} expert dependency")
                sources.append({"path": str(path), "sha256": actual})
            mapping = {f"prior_encoder_{shape}": f"last_self_ecology_{shape}" for shape in SHAPES}
            mapping.update({f"prior_encoder_integrated_{shape}": f"last_self_ecology_integrated_{shape}"
                            for shape in SHAPES})
            mapping.update({f"prior_ecological_affine_{shape}": f"prior_ecological_affine_{shape}"
                            for shape in SHAPES})
            mapping.update({f"context_{shape}": f"context_{shape}" for shape in SHAPES})
            for current_name, old_name in mapping.items():
                current = frame[frame.model_name.eq(current_name)].sort_values(["k", "cell"])
                old = previous[previous.model_name.eq(old_name)].sort_values(["k", "cell"])
                for field in ("k", "cell", "y_true", "y_pred"):
                    np.testing.assert_array_equal(current[field], old[field])
                controls.append({"split_seed": split, "seed": seed, "model_name": current_name,
                                 "reference_model": old_name, "query_rows_all_k": len(current),
                                 "predictions_bitwise_equal": True})
            memory_completion = json.loads((memory_run / "complete.json").read_text())
            memory = read_bound_json(memory_run, "ecological_affine.json", memory_completion, sources)
            if (sha256(memory_run / "ecological_affine.json") != config["memory_file_hash"]
                    or memory["mode"] != "ecological_affine" or memory["selection_role"] != "source_validation"):
                raise ValueError("Changed ecological memory profile")
            sources.append({"path": str(memory_run / "complete.json"),
                            "sha256": sha256(memory_run / "complete.json")})
            features = read_bound_json(run, "feature_definition.json", completion, sources)
            if (features != config["feature_definition"] or len(features["feature_names"]) != 38
                    or features["feature_names"][:30] != old_config["feature_definition"]["feature_names"]
                    or features["feature_names"][30:] != features["daily_feature_names"]
                    or features["normalization"] != old_config["feature_definition"]["normalization"]
                    or features["daily_value_feature_indices"] != [30, 31, 32]
                    or features["daily_availability_feature_indices"] != [33, 34, 35, 36, 37]
                    or features["combined_interaction_indices"] != INTERACTIONS
                    or features["daily_feature_names"] != metadata["feature_names"]
                    or features["daily_feature_policy"] != metadata["policy"]
                    or features["source_station_ids"] != old_config["feature_definition"]["source_station_ids"]):
                raise ValueError("Changed daily readout feature definition")
            # Canonical prediction readers intentionally discard auxiliary fields.
            # Restore only fixed hydrological validity; every arm/K carries the same flags.
            flag_names = metadata["feature_names"][5:8]
            if len(flag_names) != 3:
                raise ValueError("Three daily descriptor-validity flags are required")
            raw_flags = pd.read_parquet(run / "predictions.parquet", columns=["cell", *flag_names])
            if (not np.isin(raw_flags[flag_names].to_numpy(), [0, 1]).all()
                    or (raw_flags.groupby("cell")[flag_names].nunique().to_numpy() != 1).any()):
                raise ValueError("Daily validity differs across arms/K or is not binary")
            flags = raw_flags.drop_duplicates("cell").set_index("cell")[flag_names].eq(1).all(axis=1)
            frame["daily_all_descriptors_valid"] = frame.cell.map(flags)
            if frame.daily_all_descriptors_valid.isna().any():
                raise ValueError("Missing query hydrological availability")
            training = {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in ARMS}
            states.append({"split_seed": split, "seed": seed, "config": config, "training": training,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "mixers": read_bound_json(run, "mixers.json", completion, sources)})
            frames.append(frame)
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across training seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    if (panel.groupby("cell").daily_all_descriptors_valid.nunique() != 1).any():
        raise ValueError("Daily hydrological availability changes across seeds/partitions")
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        pivot = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS, *(f"{arm}_integrated" for arm in ARMS), *REFERENCE_BASES):
            np.testing.assert_array_equal(pivot[f"{base}_constant"], pivot[f"{base}_gru_tuned_anchor"])
    return panel, thresholds, sources, states, pd.DataFrame(controls)


def selection_records(states, panel):
    training, choices, mixing, scores = [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != set(DIRECT_MODELS) or set(run["mixers"]) != set(MIXED_MODELS):
            raise ValueError("Unexpected adapter/mixer names")
        for model, state in run["adapters"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong support selection role or K")
            choices.extend({**identity, "model_name": model, "k": int(k), **selection}
                           for k, selection in state["selection_by_k"].items())
        for arm, state in run["training"].items():
            cfg, trace = state["config"], state["trace"]
            chosen = [r for r in trace if r["epoch"] == state["best_epoch"]]
            if (state["model_class"] != "EncoderNativeResidual" or cfg["tail_weight"] != 2
                    or cfg["encoder_mode"] != "last_self_ecology" or cfg["encoder_learning_rate"] != 1e-5
                    or cfg["extra_dim"] != 38 or cfg["interaction_indices"] != INTERACTIONS
                    or cfg["learning_rate"] != 1e-4 or cfg["head_learning_rate"] != 1e-3
                    or cfg["lookback"] != 12 or cfg["batch_size"] != 512
                    or cfg["scales"] != [0, .25, .5, 1] or cfg["seed"] != run["seed"]
                    or cfg["epochs"] != run["config"]["epochs"] or cfg["patience"] != 5 or not cfg.get("train_memory", True)
                    or state["protocol"]["selection_role"] != "source_validation"
                    or len(chosen) != 1 or chosen[0]["validation_scale"] != state["selected_scale"]):
                raise ValueError("Unexpected saved daily-hydro encoder training definition")
            training.append({**identity, "arm": arm, "tail_weight": cfg["tail_weight"],
                             "epoch_ceiling": cfg["epochs"],
                             "best_epoch": state["best_epoch"], "epochs_run": state["epochs_run"],
                             "selected_scale": state["selected_scale"],
                             "initial_validation_mae": trace[0]["validation_mae"],
                             "selected_validation_mae": chosen[0]["validation_mae"],
                             **{key: state[key] for key in ("trainable_parameter_count", "n_source_cells",
                                 "n_source_tail_cells", "n_validation_query", "temporal_parameter_distance",
                                 "decay_parameter_distance", "head_parameter_norm", "encoder_trainable_parameter_count",
                                 "spatial_parameter_distance", "last_self_parameter_distance", "ecology_parameter_distance")}})
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
    return tuple(pd.DataFrame(rows) for rows in (training, choices, mixing, scores))




def availability_profiles(panel, thresholds):
    """Descriptive fixed strata; no bootstrap, new threshold, or selection."""
    rows = []
    for (split, seed, model, k), frame in panel[panel.k.isin((0, 5))].groupby(
            ["split_seed", "seed", "model_name", "k"]):
        for valid in (True, False):
            group = frame[frame.daily_all_descriptors_valid.eq(valid)]
            y, prediction = group.y_true.to_numpy(), group.y_pred.to_numpy()
            error = prediction - y
            high = y >= thresholds[(int(split), int(seed))]
            rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                         "daily_availability": "all_three_valid" if valid else "not_all_three_valid",
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
    for valid in (True, False):
        subset = panel[panel.daily_all_descriptors_valid.eq(valid)].drop_duplicates("cell")
        occurrences = panel[panel.daily_all_descriptors_valid.eq(valid)].drop_duplicates(["split_seed", "cell"])
        populations.append({"daily_availability": "all_three_valid" if valid else "not_all_three_valid",
                            "n_station_months_unique": len(subset), "n_stations_unique": subset.station.nunique(),
                            "n_split_cell_occurrences": len(occurrences)})
    summary = summary.merge(pd.DataFrame(populations), on="daily_availability", validate="many_to_one")
    return runs, splits, summary


def write_availability_report(out, profiles):
    lines = ["# Descriptive daily-hydrology availability appendix", "",
             "Strata are fixed from the three descriptor-validity flags (local daily columns 5, 6, 7):",
             "all three flags equal one versus not all three. Coverage fractions at columns 3 and 4",
             "are not validity flags. The same availability values are carried by every arm and reference;",
             "these columns describe the data, not the zeroed inputs of the matched control arms.", "",
             "This is an appendix, without a new primary endpoint, bootstrap test or model selection.",
             "Cell errors are pooled within seed, seeds averaged within partition, and nonempty",
             "partitions equally weighted. These conditional populations differ from the full query",
             "population. Distinct ecological counts are deduplicated across repeated seeds and partitions.",
             "Q90 is the existing source-training threshold; fewer than 20 true high cells in any partition",
             "is marked unstable. Empty strata are recorded as undefined, not assigned zero error.", "",
             "| Availability | Unique station-months | Unique stations | Partition-cell occurrences |",
             "|---|---:|---:|---:|"]
    for row in profiles.drop_duplicates("daily_availability").itertuples():
        lines.append(f"| {row.daily_availability} | {row.n_station_months_unique} | {row.n_stations_unique} | {row.n_split_cell_occurrences} |")
    lines += ["", "## Direct and integrated residuals (GRU support basis)", "",
              "| Model | K | Availability | MAE | Q90 MAE | Ordinary MAE | Signed bias | Nonempty partitions | Q90 unstable |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    names = {f"{arm}{suffix}_gru_tuned_anchor" for arm in ARMS for suffix in ("", "_integrated")}
    for row in profiles[profiles.model_name.isin(names)].itertuples():
        lines.append(f"| {row.model_name} | {row.k} | {row.daily_availability} | {row.mae:.4f} | "
                     f"{row.q90_mae:.4f} | {row.nontail_mae:.4f} | {row.signed_bias:+.4f} | "
                     f"{row.n_nonempty_partitions} | {row.q90_unstable_any} |")
    lines += ["", "All context and historical references and both support paths are included in the CSVs.", ""]
    (out / "availability_appendix.md").write_text("\n".join(lines))


def write_report(out, curves, profiles, classification, effects, training, mixing, controls, draws,
                 expected_epochs=120):
    lines = ["# Daily hydrology in the existing DOC residual encoder", "",
             "Three matched arms share the same partially trainable ecological encoder, original",
             f"GRU initialization, 38-channel readout, source OOF context base, {expected_epochs}-epoch cap and patience 5.",
             "All retain cell-weighted native MAE with weight two at or above source Q90.", "",
             "- monthly: all eight appended daily-flow channels are zero.",
             "- availability: daily numerical values are zero, with availability flags and fractions retained.",
             "- daily: all eight daily-flow channels are supplied.", "",
             "The daily-minus-availability comparison separates daily numerical information from",
             "hydrological availability metadata. Daily-minus-monthly measures their joint contribution.",
             "The three numerical daily channels interact with recurrent state; the feature count",
             "and trainable head dimensions are identical across arms. The context forests, source OOF",
             "predictions, source folds, ecological memory and GRU support basis remain fixed.",
             "Current-month daily summaries serve retrospective monthly reconstruction, not month-start",
             "forecasting. The monthly discharge footprint and coverage rules are fixed in the bound",
             "daily feature metadata; no new DOC target label is used to create these inputs.", "",
             "Checkpoint and residual scale use unweighted pooled source-validation MAE. Support",
             "and ecological mixing are fitted separately using identical validation episodes and grids.",
             "The no-message self path remains in use; this experiment does not establish river-message gain.", "",
             "All 20 saved models and all K values are retained. The expanded monthly head is newly",
             "trained and need not reproduce the earlier 30-channel head trajectory. The carried",
             "context, prior encoder and ecological-affine reference predictions must reproduce exactly.",
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
    for role, title in (("daily_information", "Daily numerical information and availability controls"),
                        ("previous_overall_reference", "Integrated models versus the fixed ecological reference"),
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
              "| Hydrology arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |",
              "|---|---:|---:|---|---:|"]
    for arm, group in training.groupby("arm", sort=False):
        scales = ", ".join(f"{scale:g}: {count}" for scale, count in group.selected_scale.value_counts().sort_index().items())
        lines.append(f"| {arm} | {group.best_epoch.min()}–{group.best_epoch.max()} | {group.epochs_run.sum()} | {scales} | {group.selected_validation_mae.mean():.6f} |")
    lines += ["", "All three arms use the same source-training objective. Encoder/GRU parameter movements and support choices are",
              "saved in the training and adapter CSVs.", "",
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
    parser.add_argument("--expected-epochs", type=int, default=120)
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or args.expected_epochs < 1:
        raise ValueError("Bootstrap draws and epoch ceiling must be positive")
    panel, thresholds, sources, states, controls = load_panel(args.root, args.expected_epochs)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_splits, profiles = error_profiles(panel, thresholds)
    class_runs, class_splits, classification = classification_metrics(panel, thresholds)
    availability_runs, availability_splits, availability = availability_profiles(panel, thresholds)
    definitions = comparison_definitions()
    effects, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws, definitions=definitions)
    effects = pd.concat([effects, recall_comparisons(panel, thresholds, args.bootstrap_draws, definitions)],
                        ignore_index=True)
    training, choices, mixing, gamma_scores = selection_records(states, panel)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    outputs = (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
               ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
               ("error_profiles", profiles), ("classification_by_run", class_runs),
               ("classification_by_partition", class_splits), ("classification", classification),
               ("comparisons", effects), ("directions_by_seed", directions),
               ("directions_by_partition", partitions), ("station_responses", stations),
               ("global_station_contributions", global_stations), ("gain_loss_concentration", concentration),
               ("training_choices", training), ("adapter_choices", choices),
               ("mixing_choices", mixing), ("gamma_scores", gamma_scores), ("reference_replication", controls),
               ("availability_profiles_by_run", availability_runs),
               ("availability_profiles_by_partition", availability_splits), ("availability_profiles", availability))
    for name, frame in outputs:
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, classification, effects, training, mixing, controls,
                 args.bootstrap_draws, args.expected_epochs)
    write_availability_report(out, availability)
    output_hashes = {f"{name}.csv": sha256(out / f"{name}.csv") for name, _ in outputs}
    output_hashes["findings.md"] = sha256(out / "findings.md")
    output_hashes["availability_appendix.md"] = sha256(out / "availability_appendix.md")
    # Bind shared analytical implementations as well as this study-specific entry point.
    helpers = ("analyze_doc_encoder_residual_v1.py", "analyze_doc_selective_residual_v1.py",
               "analyze_doc_tail_residual_v1.py", "analyze_unified_doc_spatial.py",
               "analyze_unified_doc_spatial_v2.py", "analyze_unified_doc_spatial_v3.py")
    for name in helpers:
        path = Path(__file__).parent / name
        sources.append({"path": str(path), "sha256": sha256(path)})
    (out / "analysis_manifest.json").write_text(json.dumps({
        "analysis_script": str(Path(__file__)), "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws, "models": MODELS, "k_values": KS,
        "expected_epochs": args.expected_epochs, "comparison_count": len(definitions),
        "sources": sources, "outputs": output_hashes,
        "role": "same-cohort model development; no target-based model selection",
        "estimand": "cell-pooled within seed; seed mean within partition; equal partition mean",
    }, indent=2) + "\n")
    print(curves[curves.k.isin((0, 5))][["model_name", "k", "mae", "rmse"]].to_string(index=False))
    print(f"Saved all three daily-hydro arms and {len(definitions)} fixed contrasts to {out}")


if __name__ == "__main__":
    main()
