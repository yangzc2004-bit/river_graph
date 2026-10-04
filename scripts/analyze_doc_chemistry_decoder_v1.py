"""Evaluate a compact nonlinear chemistry decoder on frozen DOC representations.

Fixed direct and integrated contrasts distinguish chemistry values from their
availability masks. No target model or information-availability route is selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_daily_hydro_fallback_v1 import read_full_grid
from analyze_doc_encoder_residual_v1 import check_source_identity, compare
from analyze_doc_selective_residual_v1 import classification_metrics, recall_comparisons
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import config_digest, sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import read_bound_json, summarize_metrics

ROOT = Path("experiments/phase4_transfer/doc_chemistry_decoder_v1")
NEURAL_ARMS = ("neural_no_aux", "neural_masks", "neural_chemistry")
TREE_ARMS = ("tree_no_aux", "tree_masks", "tree_chemistry")
SHAPE = "gru_tuned_anchor"
DIRECT_MODELS = tuple(f"{base}_{SHAPE}" for base in ("context", "point", "tree_prior", *NEURAL_ARMS, *TREE_ARMS, "linear_chemistry"))
MIXED_MODELS = tuple(f"{arm}_integrated_{SHAPE}" for arm in ("point", *NEURAL_ARMS, "linear_chemistry"))
MODELS = DIRECT_MODELS + MIXED_MODELS
ACTIVE_MODELS = (*NEURAL_ARMS, *TREE_ARMS, "linear_chemistry")
AVAILABILITY = ("both", "ph_only", "ec_only", "neither")


def comparison_definitions():
    rows = []
    for k in (0, 5):
        for suffix in ("", "_integrated"):
            rows.append((f"neural_chemistry_vs_no_aux{suffix}_k{k}",
                f"neural_chemistry{suffix}_{SHAPE}", k, f"neural_no_aux{suffix}_{SHAPE}", k,
                "neural_auxiliary_information" + ("_integrated" if suffix else "_direct")))
        for reference, role in (("neural_masks", "chemistry_values_beyond_masks"),
                                ("point", "retained_point_reference")):
            rows.append((f"neural_chemistry_vs_{reference}_integrated_k{k}",
                f"neural_chemistry_integrated_{SHAPE}", k, f"{reference}_integrated_{SHAPE}", k, role))
        rows.append((f"tree_chemistry_vs_no_aux_k{k}", f"tree_chemistry_{SHAPE}", k,
                     f"tree_no_aux_{SHAPE}", k, "tree_auxiliary_information"))
        rows.append((f"neural_integrated_vs_tree_chemistry_k{k}", f"neural_chemistry_integrated_{SHAPE}", k,
                     f"tree_chemistry_{SHAPE}", k, "neural_vs_matched_tree_information"))
    for k in (0, 5):
        for suffix in ("", "_integrated"):
            rows.append((f"nonlinear_vs_linear_chemistry{suffix}_k{k}",
                f"neural_chemistry{suffix}_{SHAPE}", k, f"linear_chemistry{suffix}_{SHAPE}", k,
                "nonlinear_chemistry_decoder" + ("_integrated" if suffix else "_direct")))
    return rows


def bound_file(run, name, completion, sources):
    path = run / name
    value = sha256(path)
    if completion["files"].get(name) != value:
        raise ValueError(f"Unbound or changed artifact: {path}")
    sources.append({"path": str(path), "sha256": value})
    return path


def selection_records(states, panel):
    choices, mixing, scores, adapter_scores = [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != set(DIRECT_MODELS) or set(run["mixers"]) != set(MIXED_MODELS):
            raise ValueError("Unexpected direct or integrated support-model names")
        for model, state in run["adapters"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong support selection role or K")
            expected_ridge = ["infinity"] if model.endswith("_constant") else [.1, 1, 10, "infinity"]
            if state["alpha_values"] != [0, .25, .5, .75, 1] or state["ridge_strengths"] != expected_ridge:
                raise ValueError("Changed support-calibration grid")
            for k in KS:
                candidates = [v for v in state["selection_scores"] if v["k"] == k and v["valid"]]
                if not candidates:
                    raise ValueError("No finite validation candidate for support adaptation")
                best = min(candidates, key=lambda v: (v["mae"], v["alpha"],
                    -float("inf") if v["ridge_strength"] == "infinity" else -v["ridge_strength"]))
                selected = state["selection_by_k"][str(k)]
                if selected != {key: best[key] for key in ("alpha", "ridge_strength")}:
                    raise ValueError("Support choice differs from source-validation candidate trace")
                validation = run["validation"]
                row = validation[validation.model_name.eq(model) & validation.k.eq(k)].iloc[0]
                metric = "active_mae" if model.startswith(ACTIVE_MODELS) else "mae"
                np.testing.assert_allclose(best["mae"], row[metric], rtol=0, atol=1e-12)
                choices.append({**identity, "model_name": model, "k": k, **selected,
                                "validation_selection_mae": best["mae"], "n_validation_query": best["n_query_cells"],
                                "score_scope": "auxiliary_active" if model.startswith(ACTIVE_MODELS) else "all_validation_query"})
            adapter_scores.extend({**identity, "model_name": model, **v} for v in state["selection_scores"])
        for model, state in run["mixers"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong integrated selection role or K")
            gamma_k0 = state["gamma_k0"]
            initial = [r for r in state["gamma_scores"] if r["k"] == 0]
            if gamma_k0 != min(initial, key=lambda r: (r["mae"], r["gamma"]))["gamma"]:
                raise ValueError("K0 gamma differs from its source-validation choice")
            for k in KS:
                selected = state["selection_by_k"][str(k)]
                candidates = [r for r in state["gamma_scores"] if r["k"] == k]
                eligible = [r for r in candidates if k != 0 or r["gamma"] == gamma_k0]
                chosen = min(eligible, key=lambda r: (r["mae"], r["gamma"] != 0,
                                                      r["gamma"] != gamma_k0, r["gamma"]))
                if selected != {**chosen, "locked": k == 0}:
                    raise ValueError("Integrated choice differs from recorded validation losses")
                if selected["gamma"] == 0:
                    direct = model.replace("_integrated_", "_")
                    a = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                    b = current[current.model_name.eq(direct) & current.k.eq(k)].sort_values("cell")
                    if model.startswith((*NEURAL_ARMS, "linear_chemistry")):
                        a, b = a[a.aux_available.ne("neither")], b[b.aux_available.ne("neither")]
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
                validation = run["validation"]
                row = validation[validation.model_name.eq(model) & validation.k.eq(k)].iloc[0]
                metric = "active_mae" if model.startswith((*NEURAL_ARMS, "linear_chemistry")) else "mae"
                np.testing.assert_allclose(chosen["mae"], row[metric], rtol=0, atol=1e-12)
                mixing.append({**identity, "model_name": model, "gamma_k0": gamma_k0, **selected,
                    "score_scope": "auxiliary_active" if model.startswith((*NEURAL_ARMS, "linear_chemistry")) else "all_validation_query"})
                scores.extend({**identity, "model_name": model, **row} for row in candidates)
    return tuple(pd.DataFrame(rows) for rows in (choices, mixing, scores, adapter_scores))


def attach_availability(frame):
    """Name the four observed auxiliary-input states without reading DOC values."""
    result = frame.copy()
    for field in ("ph_available", "ec_available"):
        if not np.isin(result[field].to_numpy(), (False, True)).all():
            raise ValueError(f"Nonbinary auxiliary availability: {field}")
    ph = result.ph_available.to_numpy(dtype=bool)
    ec = result.ec_available.to_numpy(dtype=bool)
    result["aux_available"] = np.select(
        [ph & ec, ph, ec], ["both", "ph_only", "ec_only"], default="neither")
    return result


def availability_profiles(panel, thresholds):
    """Descriptive fixed K0/K5 appendix; zero-cell strata stay explicit."""
    rows, populations = [], []
    for label in AVAILABILITY:
        selected = panel[panel.aux_available.eq(label)]
        unique = selected.drop_duplicates("cell")
        occurrences = selected.drop_duplicates(["split_seed", "cell"])
        populations.append({"aux_available": label, "n_station_months_unique": len(unique),
            "n_stations_unique": unique.station.nunique(), "n_split_cell_occurrences": len(occurrences)})
    for (split, seed, model, k), run in panel[panel.k.isin((0, 5))].groupby(
            ["split_seed", "seed", "model_name", "k"]):
        threshold = thresholds[(int(split), int(seed))]
        for label in AVAILABILITY:
            group = run[run.aux_available.eq(label)]
            y, pred = group.y_true.to_numpy(), group.y_pred.to_numpy()
            high, predicted_high = y >= threshold, pred >= threshold
            error = pred - y
            true_positive = int((high & predicted_high).sum())
            rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                "aux_available": label, "n_query_cells": len(group), "n_stations": group.station.nunique(),
                "n_q90_cells": int(high.sum()), "n_nontail_cells": int((~high).sum()),
                "mae": float(np.abs(error).mean()) if len(group) else np.nan,
                "rmse": float(np.sqrt(np.square(error).mean())) if len(group) else np.nan,
                "log_mae": float(np.abs(np.log1p(pred) - np.log1p(y)).mean()) if len(group) else np.nan,
                "signed_bias": float(error.mean()) if len(group) else np.nan,
                "q90_mae": float(np.abs(error[high]).mean()) if high.any() else np.nan,
                "nontail_mae": float(np.abs(error[~high]).mean()) if (~high).any() else np.nan,
                "q90_recall": true_positive / int(high.sum()) if high.any() else np.nan,
                "q90_precision": true_positive / int(predicted_high.sum()) if predicted_high.any() else np.nan,
                "q90_false_positive_rate": float(predicted_high[~high].mean()) if (~high).any() else np.nan,
                "q90_unstable": int(high.sum()) < 20})
    runs = pd.DataFrame(rows)
    keys = ["model_name", "k", "aux_available"]
    scores = ("mae", "rmse", "log_mae", "signed_bias", "q90_mae", "nontail_mae", "q90_recall",
              "q90_precision", "q90_false_positive_rate")
    parts = runs.groupby(["split_seed", *keys], as_index=False).agg(
        **{name: (name, "mean") for name in scores},
        **{name: (name, "first") for name in ("n_query_cells", "n_stations", "n_q90_cells", "n_nontail_cells")},
        n_nonempty_seeds=("mae", "count"), q90_unstable=("q90_unstable", "any"))
    summary = parts.groupby(keys, as_index=False).agg(
        **{name: (name, "mean") for name in scores}, n_nonempty_partitions=("mae", "count"),
        n_nonempty_q90_partitions=("q90_mae", "count"), q90_unstable_any=("q90_unstable", "any"))
    populations = pd.DataFrame(populations)
    summary = summary.merge(populations, on="aux_available", validate="many_to_one")
    return runs, parts, summary, populations


def full_grid_availability(full):
    """Deployment coverage is separate from observed DOC evaluation coverage."""
    if not np.isin(full.doc_observed.to_numpy(), (False, True)).all():
        raise ValueError("DOC observed indicator must be binary")
    rows = []
    for cohort, mask in (("all_station_months", np.ones(len(full), dtype=bool)),
                         ("doc_observed", full.doc_observed.to_numpy(dtype=bool)),
                         ("doc_genuinely_missing", ~full.doc_observed.to_numpy(dtype=bool))):
        selected = full.loc[mask]
        for label in AVAILABILITY:
            group = selected[selected.aux_available.eq(label)]
            rows.append({"cohort": cohort, "aux_available": label,
                "n_station_months": len(group), "n_stations": group.station.nunique(),
                "cohort_denominator": len(selected),
                "fraction": len(group) / len(selected) if len(selected) else np.nan})
    return pd.DataFrame(rows)


def write_report(out, curves, profiles, classification, effects, training, trees, controls,
                 availability, populations, deployment, draws):
    lines = ["# Nonlinear auxiliary chemistry decoding for sparse DOC reconstruction", "",
        "The experiment replaces the preceding linear chemistry correction with a small nonlinear",
        "eight-state chemistry decoder and a native residual readout. The underlying DOC encoder,",
        "GRU, point predictor and support basis stay frozen. Three decoder arms share parameter",
        "count and initialization; only the auxiliary-value/mask mode changes. Three previously fitted",
        "ExtraTrees controls and the old linear chemistry correction are copied without retraining.",
        "All fifteen curves and sixteen fixed contrasts are reported. The nonlinear-versus-linear",
        "comparison changes both the feature form and parameter count; it is not proof that",
        "nonlinearity alone causes any observed difference.", "",
        "This is reconstruction conditional on same-month pH/conductivity availability, not a forecast",
        "before those measurements arrive. Chemistry is not imputed from DOC or target evaluation",
        "labels. The source-validation data choose head checkpoint, support parameters and ecological",
        "mixture; target outcomes do not select a model or an availability-dependent route.", "",
        f"Intervals use {draws:,} paired whole-station bootstrap draws, with shared station multiplicities",
        "across overlapping partitions, seed averaging within partition and equal partition weights.",
        "The station holdouts are reused development partitions. Repeated seeds are not new ecological samples.",
        "Q90 includes ties at the source-training threshold. The availability appendix is descriptive",
        "and does not introduce a new performance threshold or model-selection rule.", "",
        f"The loader verifies {len(controls)} copied reference-model comparisons.", "",
        "## Complete K curves", "",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | " + " | ".join(f"{values.loc[k, 'mae']:.6f}" for k in KS)
                     + f" | {values.loc[5, 'rmse']:.6f} | {values.loc[5, 'r2']:.6f} |")
    lines += ["", "## All sixteen fixed contrasts", "",
        "Negative error deltas favor the candidate; classification deltas are percentage points.", "",
        "| Contrast | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in effects[effects.metric.eq("mae") & effects.region.eq("overall")].itertuples():
        group = effects[effects.comparison.eq(row.comparison)]

        def take(region, metric, group=group):
            return group[group.region.eq(region) & group.metric.eq(metric)].iloc[0]

        def fmt(value, scale=1):
            if value.status != "estimated":
                return "not estimable"
            return f"{scale * value.delta_value:+.6f} [{scale * value.delta_ci_low:+.6f}, {scale * value.delta_ci_high:+.6f}]"

        lines.append(f"| {row.comparison} | {fmt(row)} | {fmt(take('q90', 'mae'))} | "
                     f"{fmt(take('nontail', 'mae'))} | {fmt(take('q90', 'q90_recall'), 100)} | "
                     f"{fmt(take('nontail', 'q90_false_positive_rate'), 100)} |")
    lines += ["", "## High and ordinary DOC", "",
        "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for k in (0, 5):
            f = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            c = classification[classification.model_name.eq(model) & classification.k.eq(k)].iloc[0]
            lines.append(f"| {model} | {k} | {f.loc['q90', 'mae']:.6f} | {f.loc['q90', 'signed_bias']:+.6f} | "
                f"{f.loc['nontail', 'mae']:.6f} | {f.loc['nontail', 'signed_bias']:+.6f} | "
                f"{100 * c.q90_recall:.3f}% | {100 * c.q90_precision:.3f}% | {100 * c.q90_false_positive_rate:.3f}% |")
    lines += ["", "## Availability and intended reconstruction coverage", "",
        "Observed-DOC evaluation and genuinely missing-DOC reconstruction have different auxiliary",
        "data coverage. Success in the observed evaluation cannot be assumed to cover the missing grid.", "",
        "| Full-grid cohort | Auxiliary availability | Cells | Denominator | Fraction |",
        "|---|---|---:|---:|---:|"]
    for row in deployment.itertuples():
        lines.append(f"| {row.cohort} | {row.aux_available} | {row.n_station_months} | "
                     f"{row.cohort_denominator} | {100 * row.fraction:.3f}% |")
    lines += ["", "| Evaluation availability | Unique query cells | Unique stations | Split-cell occurrences |",
              "|---|---:|---:|---:|"]
    for row in populations.itertuples():
        lines.append(f"| {row.aux_available} | {row.n_station_months_unique} | {row.n_stations_unique} | "
                     f"{row.n_split_cell_occurrences} |")
    lines += ["", "## Source-validation choices", "",
        "The complete checkpoint, support alpha/ridge and ecological mixing scores are saved separately.",
        "The fixed ExtraTrees probes were trained in the preceding study and are copied with their original support choices.",
        "Each arm keeps its own validation-fitted support parameters; no calibration setting is transferred",
        "from the nonlinear chemistry arm to its controls. New-arm support and ecological choices are fitted",
        "on auxiliary-active validation queries; inactive queries use an identical fixed parent fallback",
        "for every candidate, so excluding their constant errors preserves full-product candidate ranking.",
        "Head checkpoint selection still evaluates the complete validation query. Native source predictions",
        "combine a forest OOF base with a source-trained frozen neural correction; the neural base is not OOF.", ""]
    if not training.empty:
        lines += ["| Neural arm | Selected epochs | Mean initial validation MAE | Mean selected validation MAE |",
                  "|---|---:|---:|---:|"]
        for arm, group in training.groupby("arm", sort=False):
            lines.append(f"| {arm} | {group.best_epoch.min()}–{group.best_epoch.max()} | "
                         f"{group.initial_validation_mae.mean():.6f} | {group.selected_validation_mae.mean():.6f} |")
    lines += ["", "Partition/seed directions and station gain/harm concentration accompany the aggregate estimates.",
              "Intervals crossing zero do not demonstrate equality. No new K-specific route is inferred.", ""]
    (out / "findings.md").write_text("\n".join(lines))
    appendix = ["# Descriptive auxiliary-availability strata", "",
        "Fixed K0/K5 groups; empty strata remain explicit. Metrics average seeds within partition and",
        "then nonempty partitions equally. Q90 samples below 20 are marked unstable. No new bootstrap",
        "or selection gate is introduced by this appendix.", "",
        "| Model | K | Availability | MAE | Q90 MAE | Ordinary MAE | Recall | False-Q90 | Nonempty partitions |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in availability.itertuples():
        appendix.append(f"| {row.model_name} | {row.k} | {row.aux_available} | {row.mae:.6f} | "
            f"{row.q90_mae:.6f} | {row.nontail_mae:.6f} | {row.q90_recall:.6f} | "
            f"{row.q90_false_positive_rate:.6f} | {row.n_nonempty_partitions} |")
    (out / "availability_findings.md").write_text("\n".join(appendix) + "\n")


def training_records(states, expected_epochs=120, expected_patience=10):
    rows, traces, scale_scores = [], [], []
    for run in states:
        for arm, state in run["training"].items():
            cfg, definition, trace = state["config"], state["definition"], state["trace"]
            if (state["model_class"] != "NonlinearChemistryHead" or state["schema_version"] != 1
                    or cfg["n_features"] != 554 or cfg["epochs"] != expected_epochs or cfg["patience"] != expected_patience
                    or cfg["batch_size"] != 512 or cfg["learning_rate"] != .001 or cfg["seed"] != run["seed"]
                    or state["selection_role"] != "source_validation" or state["trainable_parameter_count"] != 1111
                    or definition["objective"] != "tail2_weighted_source_native_mae"
                    or definition["normalization_role"] != "all_source_loss_rows"
                    or definition["selection"] != "unweighted_pooled_source_validation_native_mae"
                    or definition["correction_scales"] != [0, .25, .5, 1]
                    or state["tail_threshold"] != run["config"]["q90_threshold_train"]):
                raise ValueError("Unexpected auxiliary native head training definition")
            if [row["epoch"] for row in trace] != list(range(state["epochs_run"] + 1)):
                raise ValueError("Incomplete native head training trace")
            chosen = min(trace, key=lambda row: (row["validation_mae"], row["selected_scale"], row["epoch"]))
            selected_keys = ("validation_mae", "selected_scale", "scale_scores")
            if (chosen["epoch"] != state["best_epoch"] or chosen["selected_scale"] != state["selected_scale"]
                    or {key: chosen[key] for key in selected_keys} != state["validation_metrics"]
                    or state["initial_validation_mae"] != trace[0]["validation_mae"]
                    or state["initial_validation_mae"] != state["baseline_validation_mae"]
                    or trace[0]["selected_scale"] != 0
                    or not np.isfinite([[row[key] for key in ("training_loss", "validation_mae")] for row in trace]).all()):
                raise ValueError("Native checkpoint differs from source-validation selection or zero initialization")
            architecture = state["architecture"]
            if (architecture["input_features"] != 554 or architecture["original_features"] != 550
                    or architecture["hidden_features"] != 64 or architecture["auxiliary_features"] != 4
                    or architecture["chemistry_basis"] != 8 or architecture["expanded_features"] != 1070
                    or architecture["phi"] != "SiLU(Linear(4,8))" or architecture["dtype"] != "float64"
                    or architecture["dropout"] != 0):
                raise ValueError("Changed nonlinear chemistry architecture")
            phi = state["phi_initialization"]
            if (phi["seed"] != run["seed"] or np.asarray(phi["weight"]).shape != (8, 4)
                    or np.asarray(phi["bias"]).shape != (8,)
                    or not np.isfinite([*np.asarray(phi["weight"]).ravel(), *phi["bias"]]).all()
                    or any(not np.isfinite(state[key]) or state[key] < 0 for key in
                           ("phi_parameter_distance", "phi_weight_norm", "output_weight_norm"))):
                raise ValueError("Invalid chemistry decoder initialization or movement")
            norm = state["normalization"]
            mean, std, scale = [np.asarray(norm[key]) for key in ("feature_mean", "feature_raw_std", "feature_scale")]
            if (any(value.shape != (554,) or not np.isfinite(value).all() for value in (mean, std, scale))
                    or (std < 0).any() or (scale <= 0).any()
                    or norm["unit_scale_feature_count"] != int((std < 1e-6).sum())):
                raise ValueError("Invalid source native-feature normalization")
            np.testing.assert_array_equal(scale, np.where(std < 1e-6, 1., std))
            identity = {"split_seed": run["split_seed"], "seed": run["seed"], "arm": arm}
            for row in trace:
                scores = row["scale_scores"]
                if [score["scale"] for score in scores] != [0, .25, .5, 1]:
                    raise ValueError("Changed native-head blend grid")
                winner = min(scores, key=lambda value: (value["mae"], value["scale"]))
                if (winner["mae"] != row["validation_mae"] or winner["scale"] != row["selected_scale"]
                        or scores[0]["mae"] != state["baseline_validation_mae"]):
                    raise ValueError("Native blend differs from candidate validation scores")
                traces.append({**identity, **{key: value for key, value in row.items() if key != "scale_scores"}})
                scale_scores.extend({**identity, "epoch": row["epoch"], **value} for value in scores)
            rows.append({**identity, **{key: state[key] for key in (
                "trainable_parameter_count", "n_source_cells", "n_source_tail_cells", "n_source_active",
                "n_validation_query", "n_validation_active", "source_weight_mean", "source_weight_sum",
                "best_epoch", "epochs_run", "selected_scale", "optimizer_steps", "baseline_validation_mae",
                "initial_validation_mae", "selected_source_weighted_mae", "selected_source_full_scale_weighted_mae")},
                "selected_validation_mae": chosen["validation_mae"], "exact_point_fallback": state["selected_scale"] == 0,
                "unit_scale_feature_count": norm["unit_scale_feature_count"],
                "phi_parameter_distance": state["phi_parameter_distance"], "phi_weight_norm": state["phi_weight_norm"],
                "output_weight_norm": state["output_weight_norm"]})
    return tuple(pd.DataFrame(values) for values in (rows, traces, scale_scores))


def source_states(root, *, expected_epochs=120, expected_patience=10, require_nine=True):
    """Read source/validation metadata and artifact hashes, never target rows."""
    sources, states, validation_rows = [], [], []
    snapshot_path = root / "runtime_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources.append({"path": str(snapshot_path), "sha256": sha256(snapshot_path)})
    for name, value in snapshot.items():
        path = root / "code_snapshot" / name
        if sha256(path) != value:
            raise ValueError("Execution archive differs from its saved snapshot")
        sources.append({"path": str(path), "sha256": value})
    paths = sorted((root / "runs").glob("split*_seed*"))
    expected_names = {f"split{split}_seed{seed}" for split in SPLITS for seed in SEEDS}
    if require_nine and {p.name for p in paths} != expected_names:
        raise ValueError("Exactly nine completed production packages are required")
    if not paths:
        raise ValueError("No run packages found")
    for run in paths:
        completion_path = run / "complete.json"
        completion = json.loads(completion_path.read_text())
        sources.append({"path": str(completion_path), "sha256": sha256(completion_path)})
        config = read_bound_json(run, "config.json", completion, sources)
        if (completion["config_hash"] != config_digest(config)
                or config["runtime_snapshot_hash"] != config_digest(snapshot)
                or config["experiment"] != "doc_chemistry_decoder_v1"
                or config["head_class"] != "NonlinearChemistryHead" or config["activation"] != "SiLU"
                or set(config["models"]) != set(MODELS) or config["basis_names"] != [SHAPE]
                or tuple(config["neural_arms"]) != NEURAL_ARMS or tuple(config["tree_arms"]) != TREE_ARMS
                or tuple(config["k_values"]) != KS or config["inference_roles"] != ["train"]
                or config["selection_role"] != "source_validation" or config["backbone_retraining"] is not False
                or config["source_neural_is_oof"] is not False or config["new_tree_fits"] != 0
                or config["new_head_fits"] != 3 or config["trees_reused_fixed_controls"] is not True
                or config["epochs"] != expected_epochs or config["patience"] != expected_patience
                or config["feature_dim"] != 554 or config["chemistry_embedding_dim"] != 8
                or config["head_parameters"] != 1111 or config["tail_weight"] != 2
                or config["batch_size"] != 512 or config["learning_rate"] != .001
                or config["correction_scales"] != [0, .25, .5, 1]
                or config["support_selection_query"] != "aux-active val only; constant parent fallback elsewhere"):
            raise ValueError("Unexpected nonlinear decoder experiment settings")
        if run.name != f"split{config['split_seed']}_seed{config['seed']}":
            raise ValueError("Run-directory/config identity mismatch")
        old_run = Path(config["chemistry_run"])
        old_completion = json.loads((old_run / "complete.json").read_text())
        old_config = read_bound_json(old_run, "config.json", old_completion, sources)
        check_source_identity(config, old_config, ("split_seed", "seed", "dataset_hash", "mask_hash", "query_cells",
            "q90_threshold_train", "source_run", "basis_run", "oof_run", "source_completion_hash", "basis_completion_hash",
            "oof_completion_hash", "prior_run", "prior_completion_hash", "daily_features_hash", "daily_metadata_hash",
            "auxiliary_hashes", "auxiliary_provenance_hashes", "auxiliary_feature_hash", "auxiliary_active_hash",
            "neural_parent_hash", "tree_parent_hash", "auxiliary_policy", "availability_audit_hash"))
        if old_config["experiment"] != "doc_auxiliary_chemistry_v1":
            raise ValueError("Linear chemistry reference has wrong study identity")
        for path, value in ((old_run / "complete.json", config["chemistry_completion_hash"]),
                (old_run / "full_grid.parquet", config["chemistry_full_grid_hash"]),
                (old_run / "predictions.parquet", config["chemistry_prediction_hash"]),
                (Path(config["availability_audit_path"]), config["availability_audit_hash"]),
                (Path(config["dataset_path"]), config["dataset_hash"]),
                (Path(config["daily_features_path"]), config["daily_features_hash"]),
                (Path(config["daily_metadata_path"]), config["daily_metadata_hash"])):
            if sha256(path) != value:
                raise ValueError("Changed inherited input/reference artifact")
            sources.append({"path": str(path), "sha256": value})
        for name in ("prior", "source", "basis", "oof"):
            path = Path(config[f"{name}_run"]) / "complete.json"
            if sha256(path) != config[f"{name}_completion_hash"]:
                raise ValueError("Changed ancestor completion")
            sources.append({"path": str(path), "sha256": sha256(path)})
        for name in ("ph", "ec"):
            path = Path(config["auxiliary_paths"][name])
            for p, value in ((path, config["auxiliary_hashes"][name]),
                            (path.with_suffix(".provenance.json"), config["auxiliary_provenance_hashes"][name])):
                if sha256(p) != value:
                    raise ValueError("Changed auxiliary input or provenance")
                sources.append({"path": str(p), "sha256": value})
        adapters = read_bound_json(run, "adapters.json", completion, sources)
        mixers = read_bound_json(run, "mixers.json", completion, sources)
        old_adapters = read_bound_json(old_run, "adapters.json", old_completion, sources)
        old_mixers = read_bound_json(old_run, "mixers.json", old_completion, sources)
        if (set(adapters) != set(DIRECT_MODELS) - {f"linear_chemistry_{SHAPE}"}
                or set(mixers) != set(MIXED_MODELS) - {f"linear_chemistry_integrated_{SHAPE}"}):
            raise ValueError("Incorrect new model-selection set")
        adapters[f"linear_chemistry_{SHAPE}"] = old_adapters[f"neural_chemistry_{SHAPE}"]
        mixers[f"linear_chemistry_integrated_{SHAPE}"] = old_mixers[f"neural_chemistry_integrated_{SHAPE}"]
        for name in ("point", "context", "tree_prior", *TREE_ARMS):
            if adapters[f"{name}_{SHAPE}"] != old_adapters[f"{name}_{SHAPE}"]:
                raise ValueError("Fixed reference adapter differs from old study")
        if mixers[f"point_integrated_{SHAPE}"] != old_mixers[f"point_integrated_{SHAPE}"]:
            raise ValueError("Fixed point integration differs from old study")
        inputs = read_bound_json(run, "input_definition.json", completion, sources)
        old_inputs = read_bound_json(old_run, "input_definition.json", old_completion, sources)
        for key in ("source_original_feature_hash", "validation_original_feature_hash"):
            if inputs[key] != old_inputs[key]:
                raise ValueError("Frozen original feature digest differs")
        if set(inputs["modes"]) != {"no_aux", "masks", "chemistry"}:
            raise ValueError("Missing mode feature digest")
        source_path = bound_file(run, "source_training.npz", completion, sources)
        old_source = bound_file(old_run, "source_training.npz", old_completion, sources)
        with np.load(source_path, allow_pickle=False) as actual, np.load(old_source, allow_pickle=False) as old:
            if set(actual.files) != set(old.files):
                raise ValueError("Source archive keys differ from unchanged source views")
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], old[key])
        training = {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in NEURAL_ARMS}
        for arm in NEURAL_ARMS:
            bound_file(run, f"{arm}.pt", completion, sources)
            bound_file(run, f"{arm}_trace.csv", completion, sources)
        tree_states = {}
        for arm in TREE_ARMS:
            bound_file(old_run, f"{arm}.joblib", old_completion, sources)
            tree_states[arm] = read_bound_json(old_run, f"{arm}.json", old_completion, sources)
        bound_file(old_run, "neural_chemistry.pt", old_completion, sources)
        read_bound_json(old_run, "neural_chemistry.json", old_completion, sources)
        validation_path = bound_file(run, "source_validation.csv", completion, sources)
        validation = pd.read_csv(validation_path)
        if (len(validation) != 60 or validation[["model_name", "k"]].duplicated().any()
                or set(validation.model_name) != set(MODELS) or set(validation.k) != set(KS)
                or not np.isfinite(validation[["mae", "active_mae"]]).all().all()
                or (validation.n_active > validation.n).any() or (validation.n_active <= 0).any()):
            raise ValueError("Incomplete final-product source-validation table")
        states.append({"run": run, "split_seed": config["split_seed"], "seed": config["seed"], "config": config,
            "completion": completion, "old_run": old_run, "old_completion": old_completion, "old_config": old_config,
            "adapters": adapters, "mixers": mixers, "training": training, "validation": validation, "trees": tree_states})
        validation_rows.extend(validation.assign(split_seed=config["split_seed"], seed=config["seed"]).to_dict("records"))
    training_records(states, expected_epochs, expected_patience)
    empty = pd.DataFrame(columns=["split_seed", "seed", "model_name", "k", "cell", "aux_available", "y_pred"])
    selection_records(states, empty)
    return states, sources, pd.DataFrame(validation_rows)


def load_panel(root):
    states, sources, validation = source_states(root)
    frames, thresholds, controls, trees = [], {}, [], []
    common_descriptors, deployment = None, None
    for state in states:
        run, old_run = state["run"], state["old_run"]
        split, seed, config = state["split_seed"], state["seed"], state["config"]
        frame, checked_config, completion = read_predictions(run, sources)
        if checked_config != config:
            raise ValueError("Source and target metadata changed within analysis")
        previous, old_config, old_completion = read_predictions(old_run, sources)
        extras = ("base_pred", "adaptation_delta", "regional_gamma", "candidate_y_pred", "aux_fallback",
                  "ph_available", "ec_available", "aux_available", "doc_observed")
        for panel, directory in ((frame, run), (previous, old_run)):
            raw = pd.read_parquet(directory / "predictions.parquet", columns=["cell", "model_name", "k", *extras])
            for name in ("cell", "model_name", "k"):
                np.testing.assert_array_equal(panel[name], raw[name])
            for name in extras:
                panel[name] = raw[name].to_numpy()
        full = read_full_grid(run, config, completion, sources)
        old_full = read_full_grid(old_run, old_config, old_completion, sources)
        np.testing.assert_array_equal(full.index, old_full.index)
        threshold = float(config["q90_threshold_train"])
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid source threshold")
        thresholds[(split, seed)] = threshold
        descriptor_columns = ("cell", "station", "month", "ph_available", "ec_available", "aux_available", "doc_observed")
        pd.testing.assert_frame_equal(full[list(descriptor_columns)], old_full[list(descriptor_columns)], check_exact=True)
        np.testing.assert_array_equal(full.aux_available, full.ph_available | full.ec_available)
        np.testing.assert_array_equal(frame.aux_available, frame.ph_available | frame.ec_available)
        for field in ("ph_available", "ec_available", "doc_observed"):
            np.testing.assert_array_equal(frame[field], full.loc[frame.cell, field])
        if not frame.doc_observed.all():
            raise ValueError("Evaluation unexpectedly includes naturally unobserved DOC")
        if common_descriptors is None:
            common_descriptors = full[list(descriptor_columns)].copy()
            audit = json.loads(Path(config["availability_audit_path"]).read_text())
            if int(full.doc_observed.sum()) != audit["doc_observed_cells"] or len(full) != int(np.prod(audit["shape"])):
                raise ValueError("Observed/missing population differs from data audit")
            deployment = full_grid_availability(attach_availability(full))
        else:
            pd.testing.assert_frame_equal(full[list(descriptor_columns)], common_descriptors, check_exact=True)
        frame, full = attach_availability(frame), attach_availability(full)
        mapping = {f"{name}_{SHAPE}": f"{name}_{SHAPE}" for name in
            ("point", "context", "tree_prior", *TREE_ARMS, "point_integrated")}
        mapping.update({f"linear_chemistry{suffix}_{SHAPE}": f"neural_chemistry{suffix}_{SHAPE}"
                        for suffix in ("", "_integrated")})
        for name, old_name in mapping.items():
            a = frame[frame.model_name.eq(name)].sort_values(["k", "cell"])
            b = previous[previous.model_name.eq(old_name)].sort_values(["k", "cell"])
            for field in ("cell", "k", "y_true", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                np.testing.assert_array_equal(a[field], b[field])
            controls.append({"split_seed": split, "seed": seed, "model_name": name, "parent_model": old_name,
                "control_role": "copied_reference", "n_query_rows": len(a), "bitwise_exact": True})
        for name in ("context_pred", "point_pred", "tree_prior_pred", "ecological_memory", "point_integrated_k0_pred",
                     *(f"{arm}_pred" for arm in TREE_ARMS)):
            np.testing.assert_array_equal(full[name], old_full[name])
        for name, old_name in (("linear_chemistry_pred", "neural_chemistry_pred"),
                ("linear_chemistry_integrated_k0_pred", "neural_chemistry_integrated_k0_pred")):
            np.testing.assert_array_equal(full[name], old_full[old_name])
        inactive = full.aux_available.eq("neither")
        for arm in (*NEURAL_ARMS, *TREE_ARMS, "linear_chemistry"):
            reference = "tree_prior" if arm in TREE_ARMS else "point"
            np.testing.assert_array_equal(full.loc[inactive, f"{arm}_pred"], full.loc[inactive, f"{reference}_pred"])
            for suffix in (("",) if arm in TREE_ARMS else ("", "_integrated")):
                a = frame[frame.model_name.eq(f"{arm}{suffix}_{SHAPE}")].sort_values(["k", "cell"])
                b = frame[frame.model_name.eq(f"{reference}{suffix}_{SHAPE}")].sort_values(["k", "cell"])
                absent = a.aux_available.eq("neither").to_numpy()
                np.testing.assert_array_equal(a.aux_fallback, absent)
                for field in ("y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    np.testing.assert_array_equal(a[field].to_numpy()[absent], b[field].to_numpy()[absent])
                np.testing.assert_array_equal(a.y_pred.to_numpy()[~absent], a.candidate_y_pred.to_numpy()[~absent])
        for name in MODELS:
            rows = frame[frame.model_name.eq(name) & frame.k.eq(0)]
            base = name.removesuffix(f"_{SHAPE}")
            field = f"{base}_k0_pred" if "_integrated" in base else f"{base}_pred"
            np.testing.assert_array_equal(rows.y_pred, full.loc[rows.cell, field])
        for arm, head in state["training"].items():
            if head["selected_scale"] == 0:
                np.testing.assert_array_equal(full[f"{arm}_pred"], full.point_pred)
        checks = read_bound_json(run, "reference_checks.json", completion, sources)
        if len(checks) != 36 or not all(row["bitwise_exact"] for row in checks):
            raise ValueError("Incomplete exact copied-reference checks")
        for arm, tree in state["trees"].items():
            if tree["n_features"] != 151 or tree["hyperparameter_search"] is not False:
                raise ValueError("Incorrect copied tree definition")
            trees.append({"split_seed": split, "seed": seed, "arm": arm, "copied_fixed_control": True,
                          **{key: tree[key] for key in ("mode", "n_features", "source_cells", "validation_mae")}})
        frames.append(frame)
    for split in SPLITS:
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    return panel, thresholds, sources, states, pd.DataFrame(controls), deployment, validation, pd.DataFrame(trees)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--expected-epochs", type=int, default=120)
    parser.add_argument("--expected-patience", type=int, default=10)
    args = parser.parse_args()
    if args.source_only:
        states, sources, validation = source_states(args.root, expected_epochs=args.expected_epochs, expected_patience=args.expected_patience, require_nine=False)
        training, _, _ = training_records(states, args.expected_epochs, args.expected_patience)
        print(f"Source-only checks passed: {len(states)} packages, {len(training)} heads, {len(validation)} validation rows, {len(sources)} bindings")
        print(training[["split_seed", "seed", "arm", "best_epoch", "selected_scale", "selected_validation_mae"]].to_string(index=False))
        return
    if args.expected_epochs != 120 or args.expected_patience != 10:
        raise ValueError("Fixed target analysis requires the prespecified 120-epoch budget")
    if args.bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, states, controls, deployment, source_validation, tree_records = load_panel(args.root)
    choices, mixing, gamma_scores, adapter_scores = selection_records(states, panel)
    training, traces, scale_scores = training_records(states)
    runs, parts, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_parts, profiles = error_profiles(panel, thresholds)
    class_runs, class_parts, classification = classification_metrics(panel, thresholds)
    strata_runs, strata_parts, strata, populations = availability_profiles(panel, thresholds)
    definitions = comparison_definitions()
    effects, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws, definitions=definitions)
    effects = pd.concat([effects, recall_comparisons(panel, thresholds, args.bootstrap_draws, definitions)], ignore_index=True)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    outputs = (("metrics_by_run", runs), ("metrics_by_partition", parts), ("k_curves", curves),
        ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_parts), ("error_profiles", profiles),
        ("classification_by_run", class_runs), ("classification_by_partition", class_parts), ("classification", classification),
        ("comparisons", effects), ("directions_by_seed", directions), ("directions_by_partition", partitions),
        ("station_responses", stations), ("global_station_contributions", global_stations), ("gain_loss_concentration", concentration),
        ("adapter_choices", choices), ("adapter_scores", adapter_scores), ("mixing_choices", mixing), ("gamma_scores", gamma_scores),
        ("parent_replication", controls), ("training_choices", training), ("head_training_traces", traces), ("head_scale_scores", scale_scores),
        ("tree_records", tree_records), ("source_validation_final_products", source_validation),
        ("availability_by_run", strata_runs), ("availability_by_partition", strata_parts), ("availability_profiles", strata),
        ("availability_query_population", populations), ("availability_full_grid", deployment))
    for name, frame in outputs:
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, classification, effects, training, tree_records, controls,
                 strata, populations, deployment, args.bootstrap_draws)
    hashes = {f"{name}.csv": sha256(out / f"{name}.csv") for name, _ in outputs}
    for name in ("findings.md", "availability_findings.md"):
        hashes[name] = sha256(out / name)
    for name in ("analyze_doc_daily_hydro_fallback_v1.py", "analyze_doc_encoder_residual_v1.py",
                 "analyze_doc_selective_residual_v1.py", "analyze_doc_tail_residual_v1.py",
                 "analyze_unified_doc_spatial.py", "analyze_unified_doc_spatial_v2.py", "analyze_unified_doc_spatial_v3.py"):
        path = Path(__file__).parent / name
        sources.append({"path": str(path), "sha256": sha256(path)})
    (out / "analysis_manifest.json").write_text(json.dumps({
        "analysis_script": str(Path(__file__)), "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws, "models": MODELS, "k_values": KS, "comparison_count": len(definitions),
        "sources": sources, "outputs": hashes,
        "role": "same-cohort nonlinear-chemistry-decoder development; no target model or route selection",
        "estimand": "cell-pooled within seed; seed mean within partition; equal partition mean"}, indent=2) + "\n")
    print(curves[curves.k.isin((0, 5))][["model_name", "k", "mae", "rmse"]].to_string(index=False))
    print(f"Saved all 15 nonlinear-chemistry models and {len(definitions)} fixed contrasts to {out}")



if __name__ == "__main__":
    main()
