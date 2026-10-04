"""Evaluate source-fitted chemical support coordinates with frozen DOC experts.

Only support calibration and its validation-selected representation change.
Native K0 models and the fixed chemistry-tree expert are not retrained.
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

ROOT = Path("experiments/phase4_transfer/doc_chemistry_support_v1")
PIPELINES = ("neural_chemistry", "neural_chemistry_integrated", "tree_chemistry")
BASES = ("legacy", "masks_aug", "chemistry_aug")
VARIANTS = (*BASES, "selected")
MODELS = tuple(f"{pipeline}_{basis}" for pipeline in PIPELINES for basis in VARIANTS) + ("point_integrated_legacy",)
AVAILABILITY = ("both", "ph_only", "ec_only", "neither")


def comparison_definitions():
    rows = []
    for pipeline in PIPELINES:
        for k in (3, 5):
            for candidate, reference, role in (("chemistry_aug", "legacy", "chemical_coordinates"),
                    ("chemistry_aug", "masks_aug", "chemical_values_beyond_mask_coordinates"),
                    ("selected", "legacy", "validation_selected_coordinates")):
                rows.append((f"{pipeline}_{candidate}_vs_{reference}_k{k}", f"{pipeline}_{candidate}", k,
                             f"{pipeline}_{reference}", k, role))
    for k in (3, 5):
        for reference, role in (("point_integrated_legacy", "retained_general_reference"),
                                ("tree_chemistry_selected", "selected_neural_vs_tree")):
            rows.append((f"selected_neural_integrated_vs_{reference}_k{k}", "neural_chemistry_integrated_selected", k,
                         reference, k, role))
    return rows


def bound_file(run, name, completion, sources):
    path = run / name
    value = sha256(path)
    if completion["files"].get(name) != value:
        raise ValueError(f"Unbound or changed artifact: {path}")
    sources.append({"path": str(path), "sha256": value})
    return path


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
    """Descriptive all-K appendix; zero-cell strata stay explicit."""
    rows, populations = [], []
    for label in AVAILABILITY:
        selected = panel[panel.aux_available.eq(label)]
        unique = selected.drop_duplicates("cell")
        occurrences = selected.drop_duplicates(["split_seed", "cell"])
        populations.append({"aux_available": label, "n_station_months_unique": len(unique),
            "n_stations_unique": unique.station.nunique(), "n_split_cell_occurrences": len(occurrences)})
    for (split, seed, model, k), run in panel.groupby(
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



def write_report(out, curves, profiles, classification, effects, controls, basis_choices,
                 availability, populations, deployment, draws):
    lines = ["# Chemical-state coordinates for station support adaptation", "",
        "The selected nonlinear chemistry expert and chemistry-tree native predictions are frozen.",
        "Three calibration representations are compared: the legacy two-dimensional GRU basis,",
        "that basis plus two source-fitted mask-state principal components, and that basis plus two",
        "source-fitted measured-chemistry-state components. A fourth curve chooses the representation",
        "using source-validation final-pipeline MAE at each K, with deterministic legacy/masks/chemistry ties.",
        "No neural network or forest is retrained; no target score chooses a representation.", "",
        "K0 is required to preserve the old native predictions. One-support centered shape correction",
        "has rank zero; K1 invariance is a useful algebraic check, not a low-power significance result.",
        "Augmented coordinates use a fixed source-global center and row-local chemical measurements.",
        "Inactive chemistry retains the previous final prediction. The PCA fit uses only active months",
        "at source stations, without DOC response values or target-fitted normalization.", "",
        f"All thirteen curves are reported; the twenty-two fixed K3/K5 contrasts use {draws:,} paired whole-station",
        "bootstrap draws, joint station multiplicities across overlapping partitions, seed means within",
        "partition and equal partition weights. These are reused development station partitions.",
        "Repeated training seeds do not add ecological samples. Q90 includes threshold ties.", "",
        f"The loader verifies {len(controls)} copied-reference or K-invariance comparisons.", "",
        "## Complete K curves", "",
        "| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | " + " | ".join(f"{values.loc[k, 'mae']:.6f}" for k in KS)
                     + f" | {values.loc[5, 'rmse']:.6f} | {values.loc[5, 'r2']:.6f} |")
    lines += ["", "## All twenty-two fixed contrasts", "",
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
    lines += ["", "## High and ordinary DOC at primary K values", "",
        "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for k in (3, 5):
            f = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            c = classification[classification.model_name.eq(model) & classification.k.eq(k)].iloc[0]
            lines.append(f"| {model} | {k} | {f.loc['q90', 'mae']:.6f} | {f.loc['q90', 'signed_bias']:+.6f} | "
                f"{f.loc['nontail', 'mae']:.6f} | {f.loc['nontail', 'signed_bias']:+.6f} | "
                f"{100 * c.q90_recall:.3f}% | {100 * c.q90_precision:.3f}% | {100 * c.q90_false_positive_rate:.3f}% |")
    lines += ["", "## Availability and interpretation", "",
        "Observed-DOC queries and the genuinely missing-DOC grid have different chemistry availability.",
        "The new coordinates cannot imply validated improvement where neither auxiliary indicator is observed.", "",
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
    lines += ["", "## Source-validation representation selection", "",
        "Representation selection is part of the evaluated method. Every raw-basis result is still shown,",
        "so an apparent selected-curve gain cannot hide an unfavorable chemistry-coordinate result.",
        "The full candidate source-validation scores and chosen representations are saved separately.",
        "No target-driven K splice is added. Intervals crossing zero do not establish equivalence.", ""]
    if not basis_choices.empty:
        lines += ["| Pipeline | K | Legacy / masks / chemistry selected | Mean selected validation MAE |",
                  "|---|---:|---:|---:|"]
        for (pipeline, k), group in basis_choices.groupby(["pipeline", "k"], sort=False):
            counts = "/".join(str(int(group.basis.eq(b).sum())) for b in BASES)
            lines.append(f"| {pipeline} | {k} | {counts} | {group.mae.mean():.6f} |")
        lines.append("")
    (out / "findings.md").write_text("\n".join(lines))
    appendix = ["# Descriptive auxiliary-availability strata", "",
        "All K values are descriptive strata. Empty groups stay explicit and Q90 counts below20 are unstable.",
        "Metrics average seeds within partition, then nonempty partitions equally. No new hypothesis gate.", "",
        "| Model | K | Availability | MAE | Q90 MAE | Ordinary MAE | Recall | False-Q90 | Nonempty partitions |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in availability.itertuples():
        appendix.append(f"| {row.model_name} | {row.k} | {row.aux_available} | {row.mae:.6f} | "
            f"{row.q90_mae:.6f} | {row.nontail_mae:.6f} | {row.q90_recall:.6f} | "
            f"{row.q90_false_positive_rate:.6f} | {row.n_nonempty_partitions} |")
    (out / "availability_findings.md").write_text("\n".join(appendix) + "\n")


def verify_pca_state(state):
    """Check saved source-only coordinates without inspecting response values."""
    if (state["model_class"] != "ChemicalSupportBasis" or state["schema_version"] != 1
            or state["n_components"] != 2 or state["phi_dim"] != 8 or state["source_role"] != "source_training"
            or state["mode"] not in ("masks", "chemistry") or state["eigenvalue_floor"] != 1e-8
            or state["definition"]["target_label_dependency"] != "none"):
        raise ValueError("Unexpected chemical PCA definition")
    ids = np.asarray(state["source_station_indices"])
    counts = np.asarray(state["source_active_counts"])
    means = np.asarray(state["source_station_active_means"])
    center, covariance = np.asarray(state["source_global_mean"]), np.asarray(state["covariance"])
    components, eigenvalues = np.asarray(state["components"]), np.asarray(state["eigenvalues"])
    scale, identified = np.asarray(state["whitening_scale"]), np.asarray(state["identified"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(np.diff(ids) <= 0)
            or counts.shape != ids.shape or counts.dtype.kind not in "iu" or (counts < 0).any()
            or means.shape != (len(ids), 8) or center.shape != (8,) or covariance.shape != (8, 8)
            or components.shape != (2, 8) or eigenvalues.shape != (2,) or (eigenvalues < 0).any()
            or scale.shape != (2,) or identified.shape != (2,)
            or not all(np.isfinite(v).all() for v in (means, center, covariance, components, eigenvalues, scale))
            or state["n_source_stations"] != len(ids) or state["n_source_active_stations"] != int((counts > 0).sum())
            or state["n_source_active_rows"] != int(counts.sum())
            or state["identified_components"] != int(identified.sum())):
        raise ValueError("Invalid PCA dimensions or source population")
    np.testing.assert_allclose(center, (means * counts[:, None]).sum(0) / max(int(counts.sum()), 1), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(components @ components.T, np.eye(2), rtol=0, atol=1e-10)
    np.testing.assert_allclose(covariance @ components.T, components.T * eigenvalues, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(scale, np.sqrt(np.maximum(eigenvalues, 1e-8)))
    np.testing.assert_array_equal(identified, eigenvalues > 1e-8)
    return {"mode": state["mode"], "n_source_stations": len(ids), "n_source_active_stations": int((counts > 0).sum()),
        "n_source_active_rows": int(counts.sum()), "identified_components": int(identified.sum()),
        "eigenvalue_0": eigenvalues[0], "eigenvalue_1": eigenvalues[1],
        "whitening_scale_0": scale[0], "whitening_scale_1": scale[1]}


def selection_records(states):
    choices, mixing, gamma_scores, adapter_scores, basis_choices, basis_scores = [], [], [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        validation = run["validation"]
        for model, state in run["adapters"].items():
            if (state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS)
                    or state["alpha_values"] != [0, .25, .5, .75, 1]
                    or state["ridge_strengths"] != [.1, 1, 10, "infinity"]):
                raise ValueError("Changed direct support-calibration role/grid")
            for k in KS:
                candidates = [row for row in state["selection_scores"] if row["k"] == k and row["valid"]]
                best = min(candidates, key=lambda row: (row["mae"], row["alpha"],
                    -float("inf") if row["ridge_strength"] == "infinity" else -row["ridge_strength"]))
                selected = state["selection_by_k"][str(k)]
                if selected != {key: best[key] for key in ("alpha", "ridge_strength")}:
                    raise ValueError("Support choice differs from validation candidate trace")
                row = validation[validation.model_name.eq(model) & validation.k.eq(k)].iloc[0]
                np.testing.assert_allclose(best["mae"], row.active_mae, rtol=0, atol=1e-12)
                choices.append({**identity, "model_name": model, "k": k, **selected,
                    "validation_active_mae": best["mae"], "n_validation_query": best["n_query_cells"]})
            adapter_scores.extend({**identity, "model_name": model, **row} for row in state["selection_scores"])
        for model, state in run["mixers"].items():
            if (state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS)
                    or state["gamma_k0"] != run["config"]["gamma_k0"] or state["gamma_values"] != [0, .25, .5, 1]):
                raise ValueError("Changed integration role/grid or K0 lock")
            for k in KS:
                candidates = [row for row in state["gamma_scores"] if row["k"] == k]
                eligible = [row for row in candidates if k != 0 or row["gamma"] == state["gamma_k0"]]
                best = min(eligible, key=lambda row: (row["mae"], row["gamma"] != 0,
                                                     row["gamma"] != state["gamma_k0"], row["gamma"]))
                selected = state["selection_by_k"][str(k)]
                if selected != {**best, "locked": k == 0}:
                    raise ValueError("Integrated support choice differs from validation candidates")
                row = validation[validation.model_name.eq(model) & validation.k.eq(k)].iloc[0]
                np.testing.assert_allclose(best["mae"], row.active_mae, rtol=0, atol=1e-12)
                mixing.append({**identity, "model_name": model, "gamma_k0": state["gamma_k0"], **selected})
                gamma_scores.extend({**identity, "model_name": model, **row} for row in candidates)
        selection = run["selection"]
        if (selection["version"] != 1 or selection["selection_role"] != "source_validation"
                or selection["score"] != "final_active_validation_mae" or tuple(selection["tie_order"]) != BASES
                or set(selection["choices"]) != set(PIPELINES)):
            raise ValueError("Changed representation-selection rule")
        for pipeline in PIPELINES:
            for k in KS:
                chosen = selection["choices"][pipeline][str(k)]
                scores = chosen["scores"]
                if [row["basis"] for row in scores] != list(BASES):
                    raise ValueError("Incomplete or reordered representation-selection candidates")
                for candidate in scores:
                    row = validation[validation.model_name.eq(f"{pipeline}_{candidate['basis']}") & validation.k.eq(k)].iloc[0]
                    for key in ("mae", "active_mae", "n", "n_active"):
                        np.testing.assert_allclose(candidate[key], row[key], rtol=0, atol=1e-12)
                    basis_scores.append({**identity, "pipeline": pipeline, "k": k, **candidate})
                best = min(scores, key=lambda row: (row["active_mae"], BASES.index(row["basis"])))
                if (chosen["basis"] != best["basis"] or chosen["active_mae"] != best["active_mae"]
                        or (k in (0, 1) and chosen["basis"] != "legacy")):
                    raise ValueError("Representation choice or low-K tie differs from validation scores")
                a = validation[validation.model_name.eq(f"{pipeline}_selected") & validation.k.eq(k)].iloc[0]
                b = validation[validation.model_name.eq(f"{pipeline}_{best['basis']}") & validation.k.eq(k)].iloc[0]
                np.testing.assert_array_equal(a[["mae", "active_mae", "n", "n_active"]], b[["mae", "active_mae", "n", "n_active"]])
                basis_choices.append({**identity, "pipeline": pipeline, "k": k, **best})
    return tuple(pd.DataFrame(rows) for rows in (choices, mixing, gamma_scores, adapter_scores, basis_choices, basis_scores))


def source_states(root, *, require_nine=True):
    """Inspect source/validation states and content identities, never target scores."""
    sources, states, validation_rows, pca_rows = [], [], [], []
    snapshot_path = root / "runtime_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources.append({"path": str(snapshot_path), "sha256": sha256(snapshot_path)})
    for name, value in snapshot.items():
        path = root / "code_snapshot" / name
        if sha256(path) != value:
            raise ValueError("Execution archive differs from the saved snapshot")
        sources.append({"path": str(path), "sha256": value})
    paths = sorted((root / "runs").glob("split*_seed*"))
    if require_nine and {p.name for p in paths} != {f"split{s}_seed{r}" for s in SPLITS for r in SEEDS}:
        raise ValueError("Exactly nine completed production packages are required")
    if not paths:
        raise ValueError("No run packages found")
    for run in paths:
        completion_path = run / "complete.json"
        completion = json.loads(completion_path.read_text())
        sources.append({"path": str(completion_path), "sha256": sha256(completion_path)})
        config = read_bound_json(run, "config.json", completion, sources)
        if (completion["config_hash"] != config_digest(config) or config["runtime_snapshot_hash"] != config_digest(snapshot)
                or config["experiment"] != "doc_chemistry_support_v1" or set(config["models"]) != set(MODELS)
                or tuple(config["pipelines"]) != PIPELINES or tuple(config["basis_names"]) != VARIANTS
                or tuple(config["k_values"]) != KS or config["selection_role"] != "source_validation"
                or config["inference_roles"] != ["train"] or config["new_neural_fits"] != 0
                or config["new_forest_fits"] != 0 or config["backbone_retraining"] is not False
                or config["basis_dimensions"] != {"legacy": 2, "masks_aug": 4, "chemistry_aug": 4}
                or config["pca_components"] != 2 or config["pca_eigenvalue_floor"] != 1e-8
                or config["pca_fit_role"] != "within source-station active-aux months; no DOC labels"
                or config["pca_projection_center"] != "source-global active mean; target row-local"
                or config["representation_selection"] != "active source-validation MAE; ties legacy,masks_aug,chemistry_aug"
                or run.name != f"split{config['split_seed']}_seed{config['seed']}"):
            raise ValueError("Unexpected chemical-support protocol or run identity")
        old_run = Path(config["prior_run"])
        old_completion = json.loads((old_run / "complete.json").read_text())
        old_config = read_bound_json(old_run, "config.json", old_completion, sources)
        check_source_identity(config, old_config, ("split_seed", "seed", "dataset_hash", "mask_hash", "query_cells",
            "q90_threshold_train", "basis_run", "basis_completion_hash", "auxiliary_paths", "auxiliary_hashes",
            "auxiliary_provenance_hashes", "auxiliary_feature_hash", "auxiliary_active_hash", "availability_audit_path"))
        if old_config["experiment"] != "doc_chemistry_decoder_v1":
            raise ValueError("Wrong frozen chemistry parent")
        for path, value in ((old_run / "complete.json", config["prior_completion_hash"]),
                (old_run / "full_grid.parquet", config["prior_full_grid_hash"]),
                (old_run / "predictions.parquet", config["prior_prediction_hash"]),
                (old_run / "neural_chemistry.pt", config["prior_checkpoint_hash"]),
                (Path(config["prior_replay_path"]), config["prior_replay_hash"]),
                (Path(config["dataset_path"]), config["dataset_hash"]),
                (Path(config["mask_path"]), config["mask_hash"]),
                (Path(config["basis_run"]) / "complete.json", config["basis_completion_hash"]),
                (Path(config["availability_audit_path"]), old_config["availability_audit_hash"])):
            if sha256(path) != value:
                raise ValueError(f"Changed frozen source artifact: {path}")
            sources.append({"path": str(path), "sha256": value})
        for name in ("ph", "ec"):
            path = Path(config["auxiliary_paths"][name])
            for p, value in ((path, config["auxiliary_hashes"][name]),
                             (path.with_suffix(".provenance.json"), config["auxiliary_provenance_hashes"][name])):
                if sha256(p) != value:
                    raise ValueError("Changed auxiliary input or provenance")
                sources.append({"path": str(p), "sha256": value})
        if sha256(bound_file(run, "full_grid.parquet", completion, sources)) != config["prior_full_grid_hash"]:
            raise ValueError("Full grid must be the exact frozen parent component file")
        definitions = read_bound_json(run, "basis_definition.json", completion, sources)
        selection = read_bound_json(run, "basis_selection.json", completion, sources)
        adapters = read_bound_json(run, "adapters.json", completion, sources)
        mixers = read_bound_json(run, "mixers.json", completion, sources)
        if (set(definitions) != {"masks_aug", "chemistry_aug"}
                or set(adapters) != {f"{p}_{v}" for p in ("neural_chemistry", "tree_chemistry") for v in BASES}
                or set(mixers) != {f"neural_chemistry_integrated_{v}" for v in BASES}):
            raise ValueError("Incomplete source coordinate/calibrator collection")
        old_adapters = read_bound_json(old_run, "adapters.json", old_completion, sources)
        old_mixers = read_bound_json(old_run, "mixers.json", old_completion, sources)
        if config["gamma_k0"] != old_mixers["neural_chemistry_integrated_gru_tuned_anchor"]["gamma_k0"]:
            raise ValueError("K0 integration differs from frozen chemistry parent")
        for p in ("neural_chemistry", "tree_chemistry"):
            if adapters[f"{p}_legacy"] != old_adapters[f"{p}_gru_tuned_anchor"]:
                raise ValueError("Legacy direct calibration changed")
        if mixers["neural_chemistry_integrated_legacy"] != old_mixers["neural_chemistry_integrated_gru_tuned_anchor"]:
            raise ValueError("Legacy integrated calibration changed")
        source_path = bound_file(old_run, "source_training.npz", old_completion, sources)
        with np.load(source_path, allow_pickle=False) as source:
            source_ids = source["source_station_ids"].copy()
        np.testing.assert_array_equal(source_ids, config["source_station_ids"])
        archive = bound_file(run, "representations.npz", completion, sources)
        basis_run = Path(config["basis_run"])
        basis_completion = json.loads((basis_run / "complete.json").read_text())
        old_archive = bound_file(basis_run, "representations.npz", basis_completion, sources)
        with np.load(archive, allow_pickle=False) as reps, np.load(old_archive, allow_pickle=False) as old_reps:
            if set(reps.files) != {"legacy", "active", "source_station_ids", "masks_aug", "chemistry_aug", "masks_chemical", "chemistry_chemical"}:
                raise ValueError("Unexpected chemical-coordinate archive keys")
            np.testing.assert_array_equal(reps["legacy"], old_reps["gru_tuned_anchor"])
            np.testing.assert_array_equal(reps["source_station_ids"], source_ids)
            cells = len(reps["active"])
            if reps["active"].shape != (cells,) or reps["active"].dtype != bool or reps["legacy"].shape != (cells, 2):
                raise ValueError("Invalid coordinate or availability shape")
            for name, mode in (("masks_aug", "masks"), ("chemistry_aug", "chemistry")):
                state = definitions[name]
                if state["mode"] != mode:
                    raise ValueError("Coordinate definition mode differs from archive key")
                np.testing.assert_array_equal(state["source_station_indices"], source_ids)
                row = verify_pca_state(state)
                pca_rows.append({"split_seed": config["split_seed"], "seed": config["seed"], "basis": name, **row})
                values = reps[f"{mode}_chemical"]
                if values.shape != (cells, 2) or not np.isfinite(values).all():
                    raise ValueError("Invalid source-projected chemical coordinates")
                np.testing.assert_array_equal(values[~reps["active"]], np.zeros((int((~reps["active"]).sum()), 2)))
                np.testing.assert_array_equal(values[:, ~np.asarray(state["identified"], dtype=bool)],
                    np.zeros((cells, 2 - state["identified_components"])))
                np.testing.assert_array_equal(reps[name], np.concatenate((reps["legacy"], values), axis=1))
        validation_path = bound_file(run, "source_validation.csv", completion, sources)
        validation = pd.read_csv(validation_path)
        if (len(validation) != 52 or validation[["model_name", "k"]].duplicated().any()
                or set(validation.model_name) != set(MODELS) or set(validation.k) != set(KS)
                or not np.isfinite(validation[["mae", "active_mae"]]).all().all()
                or (validation.n_active > validation.n).any() or (validation.n_active <= 0).any()):
            raise ValueError("Incomplete source-validation products")
        old_validation = pd.read_csv(bound_file(old_run, "source_validation.csv", old_completion, sources))
        for pipeline in (*PIPELINES, "point_integrated"):
            a = validation[validation.model_name.eq(f"{pipeline}_legacy")].sort_values("k")
            b = old_validation[old_validation.model_name.eq(f"{pipeline}_gru_tuned_anchor")].sort_values("k")
            np.testing.assert_array_equal(a.k, b.k)
            np.testing.assert_allclose(a[["mae", "active_mae", "n", "n_active"]], b[["mae", "active_mae", "n", "n_active"]], rtol=0, atol=1e-12)
        states.append({"run": run, "split_seed": config["split_seed"], "seed": config["seed"], "config": config,
            "completion": completion, "old_run": old_run, "old_completion": old_completion, "old_config": old_config,
            "adapters": adapters, "mixers": mixers, "selection": selection, "definitions": definitions, "validation": validation})
        validation_rows.extend(validation.assign(split_seed=config["split_seed"], seed=config["seed"]).to_dict("records"))
    selection_records(states)
    return states, sources, pd.DataFrame(validation_rows), pd.DataFrame(pca_rows)


def load_panel(root):
    states, sources, validation, pca = source_states(root)
    frames, thresholds, controls = [], {}, []
    common_descriptors, deployment = None, None
    fields = ("cell", "k", "y_true", "y_pred", "base_pred", "adaptation_delta", "regional_gamma")
    for state in states:
        run, old_run = state["run"], state["old_run"]
        split, seed, config = state["split_seed"], state["seed"], state["config"]
        frame, checked_config, completion = read_predictions(run, sources)
        if checked_config != config:
            raise ValueError("Source and target metadata changed within analysis")
        previous, _, _ = read_predictions(old_run, sources)
        extras = ("base_pred", "adaptation_delta", "regional_gamma", "candidate_y_pred", "aux_fallback",
                  "ph_available", "ec_available", "aux_available", "doc_observed")
        for panel, directory in ((frame, run), (previous, old_run)):
            raw = pd.read_parquet(directory / "predictions.parquet", columns=["cell", "model_name", "k", *extras])
            for name in ("cell", "model_name", "k"):
                np.testing.assert_array_equal(panel[name], raw[name])
            for name in extras:
                panel[name] = raw[name].to_numpy()
        named = pd.read_parquet(run / "predictions.parquet", columns=["basis_name", "selected_basis"])
        for field in named:
            frame[field] = named[field].to_numpy()
        full = read_full_grid(run, config, completion, sources)
        threshold = float(config["q90_threshold_train"])
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid source-training Q90 threshold")
        thresholds[(split, seed)] = threshold
        descriptor_columns = ("cell", "station", "month", "ph_available", "ec_available", "aux_available", "doc_observed")
        np.testing.assert_array_equal(full.aux_available, full.ph_available | full.ec_available)
        np.testing.assert_array_equal(frame.aux_available, frame.ph_available | frame.ec_available)
        for field in ("ph_available", "ec_available", "doc_observed"):
            np.testing.assert_array_equal(frame[field], full.loc[frame.cell, field])
        if not frame.doc_observed.all():
            raise ValueError("Evaluation contains naturally missing DOC")
        if common_descriptors is None:
            common_descriptors = full[list(descriptor_columns)].copy()
            audit = json.loads(Path(config["availability_audit_path"]).read_text())
            if int(full.doc_observed.sum()) != audit["doc_observed_cells"] or len(full) != int(np.prod(audit["shape"])):
                raise ValueError("DOC population differs from availability audit")
            deployment = full_grid_availability(attach_availability(full))
        else:
            pd.testing.assert_frame_equal(full[list(descriptor_columns)], common_descriptors, check_exact=True)
        for pipeline in (*PIPELINES, "point_integrated"):
            name, old_name = f"{pipeline}_legacy", f"{pipeline}_gru_tuned_anchor"
            for k in KS:
                a = frame[frame.model_name.eq(name) & frame.k.eq(k)].sort_values("cell")
                b = previous[previous.model_name.eq(old_name) & previous.k.eq(k)].sort_values("cell")
                for field in fields:
                    np.testing.assert_array_equal(a[field], b[field])
                controls.append({"split_seed": split, "seed": seed, "model_name": name, "k": k,
                    "reference": old_name, "control_role": "copied_parent", "n_query_rows": len(a), "bitwise_exact": True})
        for pipeline in PIPELINES:
            for k in KS:
                selected_name = state["selection"]["choices"][pipeline][str(k)]["basis"]
                a = frame[frame.model_name.eq(f"{pipeline}_selected") & frame.k.eq(k)].sort_values("cell")
                b = frame[frame.model_name.eq(f"{pipeline}_{selected_name}") & frame.k.eq(k)].sort_values("cell")
                if not a.selected_basis.eq(selected_name).all() or not a.basis_name.eq("selected").all():
                    raise ValueError("Product basis name differs from source-validation choice")
                for field in (*fields, "candidate_y_pred", "aux_fallback"):
                    np.testing.assert_array_equal(a[field], b[field])
                for variant in VARIANTS:
                    name = f"{pipeline}_{variant}"
                    a = frame[frame.model_name.eq(name) & frame.k.eq(k)].sort_values("cell")
                    b = frame[frame.model_name.eq(f"{pipeline}_legacy") & frame.k.eq(k)].sort_values("cell")
                    if k in (0, 1) and variant != "legacy":
                        for field in fields:
                            np.testing.assert_array_equal(a[field], b[field])
                        controls.append({"split_seed": split, "seed": seed, "model_name": name, "k": k,
                            "reference": f"{pipeline}_legacy", "control_role": "low_k_invariance", "n_query_rows": len(a), "bitwise_exact": True})
                    inactive = ~a.aux_available.to_numpy(dtype=bool)
                    np.testing.assert_array_equal(a.aux_fallback, inactive)
                    for field in fields[3:]:
                        np.testing.assert_array_equal(a[field].to_numpy()[inactive], b[field].to_numpy()[inactive])
                    np.testing.assert_array_equal(a.y_pred.to_numpy()[~inactive], a.candidate_y_pred.to_numpy()[~inactive])
                    if k == 0:
                        column = f"{pipeline}_k0_pred" if pipeline.endswith("integrated") else f"{pipeline}_pred"
                        np.testing.assert_array_equal(a.y_pred, full.loc[a.cell, column])
        for name, model in state["mixers"].items():
            for k in KS:
                if model["selection_by_k"][str(k)]["gamma"] == 0:
                    a = frame[frame.model_name.eq(name) & frame.k.eq(k) & frame.aux_available].sort_values("cell")
                    b = frame[frame.model_name.eq(name.replace("_integrated_", "_")) & frame.k.eq(k) & frame.aux_available].sort_values("cell")
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
        checks = read_bound_json(run, "reference_checks.json", completion, sources)
        if (checks["role"] != "test" or len(checks["checks"]) != 34
                or not all(row.get("reference_exact", row.get("low_k_exact", False)) for row in checks["checks"])):
            raise ValueError("Incomplete saved invariance checks")
        frames.append(attach_availability(frame))
    for split in SPLITS:
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    return panel, thresholds, sources, states, pd.DataFrame(controls), deployment, validation, pca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.source_only:
        states, sources, validation, pca = source_states(args.root, require_nine=False)
        print(f"Source-only checks passed: {len(states)} packages, {len(validation)} validation rows, {len(pca)} PCA states, {len(sources)} bindings")
        return
    if args.bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, states, controls, deployment, source_validation, pca = load_panel(args.root)
    choices, mixing, gamma_scores, adapter_scores, basis_choices, basis_scores = selection_records(states)
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
        ("parent_replication", controls), ("basis_choices", basis_choices), ("basis_scores", basis_scores),
        ("pca_diagnostics", pca), ("source_validation_final_products", source_validation),
        ("availability_by_run", strata_runs), ("availability_by_partition", strata_parts), ("availability_profiles", strata),
        ("availability_query_population", populations), ("availability_full_grid", deployment))
    for name, frame in outputs:
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, classification, effects, controls, basis_choices,
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
        "role": "same-cohort chemical support-coordinate development; no target model or route selection",
        "estimand": "cell-pooled within seed; seed mean within partition; equal partition mean"}, indent=2) + "\n")
    print(curves[["model_name", "k", "mae", "rmse"]].to_string(index=False))
    print(f"Saved all 13 chemical-support models and {len(definitions)} fixed contrasts to {out}")


if __name__ == "__main__":
    main()
