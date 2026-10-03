"""Evaluate episodic recurrent updates with matched normalization controls.

Completed predictions are analyzed as saved. Checkpoints, adapter parameters and
comparison populations are never selected from outer-test performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_unified_doc_spatial import (
    config_digest,
    joint_station_bootstrap,
    paired_cells,
    sha256,
)
from analyze_unified_doc_spatial_v2 import (
    KS,
    SEEDS,
    SPLITS,
    canonical_panel,
    validate_panel,
)
from analyze_unified_doc_spatial_v3 import (
    add_directions,
    read_bound_json,
    summarize_metrics,
)

DEFAULT_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v4")
HEADS = ("constant", "gru_episodic", "tree_episodic", "gru_frozen_anchor", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{head}" for base in ("context", "fusion") for head in HEADS)
LABELS = {
    f"{base}_{head}": f"{base_label} + {head_label}"
    for base, base_label in (("context", "Environmental"), ("fusion", "Frozen fusion"))
    for head, head_label in (("constant", "constant correction"), ("gru_episodic", "prior GRU projection"),
                             ("tree_episodic", "prior tree projection"),
                             ("gru_frozen_anchor", "frozen GRU, anchor normalized"),
                             ("gru_tuned_anchor", "updated GRU, anchor normalized"))
}


def comparison_definitions():
    comparisons = []
    for base in ("context", "fusion"):
        for k in (3, 5):
            comparisons.append((f"{base}_tuned_vs_frozen_anchor_k{k}",
                                f"{base}_gru_tuned_anchor", k, f"{base}_gru_frozen_anchor", k,
                                "direct_recurrent_update"))
            for head, label in (("gru_episodic", "prior_gru"), ("tree_episodic", "tree"),
                                ("constant", "constant")):
                comparisons.append((f"{base}_tuned_vs_{label}_k{k}",
                                    f"{base}_gru_tuned_anchor", k, f"{base}_{head}", k,
                                    "performance_control"))
            comparisons.append((f"{base}_frozen_anchor_vs_prior_gru_k{k}",
                                f"{base}_gru_frozen_anchor", k, f"{base}_gru_episodic", k,
                                "normalization_control"))
    return comparisons


def load_panel(root):
    frames, thresholds, sources, states = [], {}, [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            completion_path = run / "complete.json"
            if not completion_path.is_file():
                raise ValueError(f"All nine completed runs are required; missing {completion_path}")
            completion = json.loads(completion_path.read_text())
            config = read_bound_json(run, "config.json", completion, sources)
            if completion["config_hash"] != config_digest(config):
                raise ValueError(f"Changed configuration identity: {run}")
            if (config["split_seed"], config["seed"]) != (split, seed):
                raise ValueError(f"Run/configuration identity differs: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid training Q90 threshold: {run}")
            thresholds[(split, seed)] = threshold
            prediction_path = run / "predictions.parquet"
            content_hash = sha256(prediction_path)
            if completion["files"].get(prediction_path.name) != content_hash:
                raise ValueError(f"Unbound or changed predictions: {run}")
            frame = canonical_panel(pd.read_parquet(prediction_path))
            if not frame.split_seed.eq(split).all() or not frame.seed.eq(seed).all():
                raise ValueError(f"Run/prediction identity differs: {run}")
            frames.append(frame)
            sources.extend({"path": str(path), "sha256": sha256(path)}
                           for path in (prediction_path, completion_path))
            states.append({"split_seed": split, "seed": seed,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "memory": read_bound_json(run, "memory.json", completion, sources)})
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError(f"Training Q90 must define the same tail across seeds in partition {split}")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    for (_, _), group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        zero = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", "fusion"):
            for head in HEADS[1:]:
                np.testing.assert_array_equal(zero[f"{base}_constant"], zero[f"{base}_{head}"])
    return panel, thresholds, sources, states


def compare(panel, draws):
    effects, directions, station_responses = [], [], []
    for name, candidate, ck, reference, rk, role in comparison_definitions():
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        effect = {"comparison": name, "comparison_role": role, "candidate": candidate,
                  "candidate_k": ck, "reference": reference, "reference_k": rk,
                  **joint_station_bootstrap(pairs, draws=draws)}
        per_seed = pairs.groupby(["split_seed", "seed"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
        per_seed["comparison"], per_seed["comparison_role"] = name, role
        add_directions(per_seed)
        split_delta = per_seed.groupby("split_seed").delta_mae.mean()
        effect.update(improved_split_seed_pairs=int((per_seed.delta_mae < 0).sum()),
                      n_split_seed_pairs=len(per_seed), improved_splits=int((split_delta < 0).sum()),
                      equal_splits=int((split_delta == 0).sum()))
        stations = pairs.groupby(["split_seed", "station", "cell"], as_index=False).agg(
            candidate_error=("candidate_error", "mean"), reference_error=("reference_error", "mean"),
            ecological_novelty=("ecological_novelty", "mean"), upstream_support=("upstream_support", "mean"))
        stations = stations.groupby(["split_seed", "station"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
            n_query_cells=("cell", "size"), ecological_novelty=("ecological_novelty", "mean"),
            upstream_support=("upstream_support", "mean"))
        stations["comparison"], stations["comparison_role"] = name, role
        add_directions(stations)
        effects.append(effect)
        directions.append(per_seed)
        station_responses.append(stations)
    directions = pd.concat(directions, ignore_index=True)
    partitions = directions.groupby(["comparison", "comparison_role", "split_seed"], as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
        n_seeds=("seed", "size"))
    add_directions(partitions)
    return pd.DataFrame(effects), directions, partitions, pd.concat(station_responses, ignore_index=True)


def tail_comparisons(panel, thresholds, draws):
    """Secondary conditional-tail estimands; never silently drop a partition."""
    summaries, counts = [], []
    definitions = [item for item in comparison_definitions()
                   if item[2] == 5 and ("tuned_vs_frozen_anchor" in item[0] or "tuned_vs_tree" in item[0])]
    for name, candidate, ck, reference, rk, _ in definitions:
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        threshold = np.array([thresholds[(int(s), int(r))]
                              for s, r in zip(pairs.split_seed, pairs.seed, strict=True)])
        tail = pairs[pairs.y_true_candidate.to_numpy() >= threshold].copy()
        unique = tail.drop_duplicates(["split_seed", "cell"])
        current_counts = []
        for split in SPLITS:
            part = unique[unique.split_seed.eq(split)]
            current_counts.append({"comparison": f"q90_{name}", "split_seed": split,
                                   "q90_threshold_train": thresholds[(split, SEEDS[0])],
                                   "n_tail_cells": len(part), "n_tail_stations": part.station.nunique(),
                                   "unstable": len(part) < 20})
        counts.extend(current_counts)
        summary = {"comparison": f"q90_{name}", "comparison_role": "secondary_conditional_Q90",
                   "candidate": candidate, "candidate_k": ck, "reference": reference, "reference_k": rk,
                   "unstable_any_partition": any(row["unstable"] for row in current_counts),
                   "n_partitions_with_tail": sum(row["n_tail_cells"] > 0 for row in current_counts)}
        if all(row["n_tail_cells"] > 0 for row in current_counts):
            summary.update(status="estimated", **joint_station_bootstrap(tail, draws=draws))
        else:
            summary.update(status="not_estimable_missing_partition",
                           **{key: np.nan for key in (
                               "reference_mae", "candidate_mae", "delta_mae", "delta_ci_low", "delta_ci_high",
                               "relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct")})
        summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(counts)


def selection_records(states):
    adapters, training = [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        if set(run["adapters"]) != set(MODELS):
            raise ValueError("The ten saved adapter choices are incomplete")
        for model, state in run["adapters"].items():
            if set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError(f"Adapter choices do not cover all K: {model}")
            for k, choice in state["selection_by_k"].items():
                k = int(k)
                ridge = choice["ridge_strength"]
                constant = ridge == "infinity" or np.isinf(float(ridge))
                adapters.append({**identity, "model_name": model, "k": k, "alpha": choice["alpha"],
                                 "ridge_strength": "infinity" if constant else float(ridge),
                                 "constant_shape_selected": constant,
                                 "finite_shape_enabled": bool(not constant and k > 1),
                                 "selection_role": state["selection_role"]})
        memory = run["memory"]
        trace, best_epoch = memory["trace"], int(memory["best_epoch"])
        if not trace or trace[0]["epoch"] != 0:
            raise ValueError("Memory trace must retain the unchanged epoch-0 model")
        selected = [row for row in trace if row["epoch"] == best_epoch]
        if len(selected) != 1:
            raise ValueError("Selected memory epoch is missing or duplicated")
        initial, best = float(trace[0]["validation_mae"]), float(memory["validation_metrics"]["validation_mae"])
        losses = [row[key] for row in trace for key in ("training_mae", "validation_mae") if row[key] is not None]
        if not np.isfinite(losses).all() or best != float(selected[0]["validation_mae"]):
            raise ValueError("Invalid memory losses or checkpoint selection")
        distances = {}
        for key in ("temporal_parameter_distance", "decay_parameter_distance"):
            distance = float(memory[key])
            if (not np.isfinite(distance) or distance < 0
                    or not np.isclose(distance, selected[0][key], rtol=1e-8, atol=1e-10)):
                raise ValueError(f"Selected memory {key} differs from its training trace")
            if best_epoch == 0 and distance != 0:
                raise ValueError("Epoch-0 checkpoint must preserve initial recurrent parameters")
            distances[key] = distance
        training.append({
            **identity, "best_epoch": best_epoch, "epochs_run": int(memory["epochs_run"]),
            "initial_validation_mae": initial, "best_validation_mae": best,
            "validation_mae_change": best - initial, **distances,
            "initial_checkpoint_selected": best_epoch == 0,
            "trainable_parameter_count": int(memory["trainable_parameter_count"]),
            "hidden_size": int(memory["hidden_size"]),
            "n_source_stations": int(memory["n_source_stations"]),
            "n_validation_stations": int(memory["n_validation_stations"]),
            "n_validation_query": int(memory["n_validation_query"]),
            "n_source_calendar_anchors": len(memory["source_anchor_months"]),
            "n_validation_calendar_anchors": len(memory["validation_anchor_months"]),
            "finite_recorded_losses": True,
            "gradient_finite_policy": memory["protocol"]["gradient_finite_policy"],
            "selected_validation_anchor_floor_hits": memory["selected_validation_anchor_floor_hits"],
            "selected_epoch_source_batch_anchor_floor_hits": memory["selected_source_anchor_floor_hits"],
            "selected_validation_anchor_rms_min": selected[0]["validation_anchor_rms_min"],
            "selected_validation_anchor_rms_max": selected[0]["validation_anchor_rms_max"],
            "selection_role": memory["protocol"]["selection_role"],
        })
    return pd.DataFrame(adapters), pd.DataFrame(training)


def write_report(out, curves, comparisons, tails, training, choices, draws):
    lines = ["# DOC spatial adaptation: updating recurrent states", "",
             "All nine packages compare the updated GRU with its frozen counterpart under identical",
             "calendar-anchor normalization. Environmental experts, spatial encoders, fixed readouts",
             "and fusion predictions remain unchanged. Partitions 142–144 are previously examined",
             "same-cohort station holdouts, so this is a development comparison.", "",
             "Negative paired MAE difference and positive relative reduction favor the candidate.",
             f"Intervals use {draws:,} joint whole-station bootstrap draws. Seed losses are averaged",
             "within partitions, followed by equal partition weighting; repeated station identities",
             "are resampled together. No checkpoint or adapter is selected from these test results.", ""]
    for role, title in (("direct_recurrent_update", "Primary: recurrent update with matched normalization"),
                        ("performance_control", "Performance against existing adapters"),
                        ("normalization_control", "Normalization change with recurrence frozen")):
        lines += [f"## {title}", "",
                  "| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in comparisons[comparisons.comparison_role.eq(role)].itertuples():
            lines.append(f"| {row.comparison} | {row.reference_mae:.4f} | {row.candidate_mae:.4f} | "
                         f"{row.delta_mae:.4f} [{row.delta_ci_low:.4f}, {row.delta_ci_high:.4f}] | "
                         f"{row.relative_gain_pct:.2f}% [{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}] | "
                         f"{row.improved_splits}/3; {row.improved_split_seed_pairs}/9 |")
        lines.append("")
    lines += ["## Error profiles", "",
              "| Predictor | K | MAE | RMSE | R² | Log MAE | Training-Q90 MAE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in curves[curves.k.isin((0, 5))].itertuples():
        lines.append(f"| {LABELS[row.model_name]} | {row.k} | {row.mae:.4f} | {row.rmse:.4f} | "
                     f"{row.r2:.4f} | {row.log_mae:.4f} | {row.q90_mae:.4f} |")
    lines += ["", "## Secondary conditional Q90 comparisons", "",
              "These estimates condition on query DOC at or above each partition's source-training Q90.",
              "They are tail diagnostics, not primary endpoints. The same paired station bootstrap is",
              "applied to this tail population. Counts exclude repeated seed predictions; n < 20 in",
              "any partition is marked unstable. A missing tail partition is not silently dropped.", "",
              "| Comparison | Reference tail MAE | Candidate tail MAE | Difference [95% CI] | Reduction [95% CI] | Status; unstable |",
              "|---|---:|---:|---:|---:|---|"]
    for row in tails.itertuples():
        lines.append(f"| {row.comparison} | {row.reference_mae:.4f} | {row.candidate_mae:.4f} | "
                     f"{row.delta_mae:.4f} [{row.delta_ci_low:.4f}, {row.delta_ci_high:.4f}] | "
                     f"{row.relative_gain_pct:.2f}% [{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}] | "
                     f"{row.status}; {bool(row.unstable_any_partition)} |")
    lines += ["", "## Source training and selected adaptation", "",
              "| Split | Seed | Best / run epoch | Initial / best validation MAE | GRU / decay distance | Trainable parameters |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in training.itertuples():
        lines.append(f"| {row.split_seed} | {row.seed} | {row.best_epoch} / {row.epochs_run} | "
                     f"{row.initial_validation_mae:.4f} / {row.best_validation_mae:.4f} | "
                     f"{row.temporal_parameter_distance:.5f} / {row.decay_parameter_distance:.5f} | "
                     f"{row.trainable_parameter_count:,} |")
    lines += ["", f"Epoch 0 was selected in {int(training.initial_checkpoint_selected.sum())}/9 runs.",
              "Distances measure selected GRU/decay parameter change from initialization; recorded losses",
              "are finite. `memory_training.csv` retains the training code's nonfinite-gradient policy",
              "and anchor-scale floor counts. Source anchor counts describe batches during the selected",
              "epoch, not a common checkpoint, and are absent for epoch 0; validation counts describe",
              "the selected checkpoint. These are source diagnostics, not outer-test performance.", "",
              "| Updated adapter | K | Finite shape enabled | Constant selected |",
              "|---|---:|---:|---:|"]
    selected = choices[choices.model_name.str.endswith("gru_tuned_anchor") & choices.k.isin((3, 5))]
    for (model, k), group in selected.groupby(["model_name", "k"]):
        lines.append(f"| {LABELS[model]} | {k} | {int(group.finite_shape_enabled.sum())}/9 | "
                     f"{int(group.constant_shape_selected.sum())}/9 |")
    lines += ["", "## Reading the evidence", "",
              "Updated versus frozen-anchor GRU is the direct recurrent-training comparison. Updated",
              "versus prior GRU combines recurrent training with the normalization change; frozen-anchor",
              "versus prior GRU reports that normalization change separately. Each complete adapter",
              "selects its own alpha and ridge on source validation, so these are not fixed-alpha or",
              "fixed-ridge ablations. Finite ridge enables a shape correction but does not guarantee a",
              "nonzero correction at every station. The tree representation has the same two-dimensional",
              "output and K support observations, but is not matched for trainable parameter count.", "",
              "Only the existing GRUCell and observation-decay parameters are updated. K = 0 remains",
              "the frozen base. Source environmental residuals use station-fold-fitted forests, while",
              "spatial weights and normalization retain earlier source training. Source validation selects",
              "the recurrent epoch and final adapter parameters; this is conditional development evaluation.", "",
              "Each recurrent window respects temporal order, but calendar-anchor normalization can use",
              "later covariates and support observations can follow query dates. The overall product is",
              "retrospective reconstruction. No attention, river-message or statistical-causal claim follows",
              "from this comparison. Seeds and repeated stations are not additional ecological samples.", "",
              "All K curves, run/partition metrics, paired directions, station responses, adapter selections",
              "and tail counts are saved with the report. Historical results remain available unchanged.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def analyze(root, *, draws=5000):
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, states = load_panel(root)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    comparisons, directions, partition_directions, stations = compare(panel, draws)
    tails, tail_counts = tail_comparisons(panel, thresholds, draws)
    choices, training = selection_records(states)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("run_metrics", runs), ("split_metrics", splits), ("k_curves", curves),
                        ("comparisons", comparisons), ("split_seed_directions", directions),
                        ("partition_directions", partition_directions), ("station_heterogeneity", stations),
                        ("q90_comparisons", tails), ("q90_partition_counts", tail_counts),
                        ("adapter_choices", choices), ("memory_training", training)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, comparisons, tails, training, choices, draws)
    (out / "status.json").write_text(json.dumps({
        "complete": True, "available_runs": 9, "expected_runs": 9,
        "bootstrap_draws": draws, "models": list(MODELS), "n_comparisons": len(comparisons),
        "n_primary_direct_comparisons": 4, "n_secondary_q90_comparisons": len(tails),
        "scientific_role": "seen-partition development evaluation; not independent confirmation",
    }, indent=2) + "\n")
    helper_names = ("analyze_unified_doc_spatial.py", "analyze_unified_doc_spatial_v2.py",
                    "analyze_unified_doc_spatial_v3.py")
    (out / "sources.json").write_text(json.dumps({
        "sources": sources, "analysis_script": str(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)), "bootstrap_seed": 42,
        "analysis_helpers": [{"path": str(Path(__file__).with_name(name)),
                              "sha256": sha256(Path(__file__).with_name(name))} for name in helper_names],
        "estimand": "individual-seed mean within partition; query-cell weighted; partition equal",
        "q90_estimand": "same aggregation conditional on DOC >= source-training Q90 within each partition",
    }, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(analyze(args.root, draws=args.draws))


if __name__ == "__main__":
    main()
