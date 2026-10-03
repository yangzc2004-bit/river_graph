"""Evaluate saved episodic DOC representations against matched PCA controls.

All comparisons use completed, source-selected predictions. This script does
not fit projectors/adapters or select configurations from outer-test outcomes.
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
    metric_values,
    paired_cells,
    sha256,
)
from analyze_unified_doc_spatial_v2 import (
    KS,
    METRICS,
    SEEDS,
    SPLITS,
    canonical_panel,
    validate_panel,
)

DEFAULT_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v3")
HEADS = ("constant", "gru_pca", "gru_episodic", "tree_pca", "tree_episodic")
MODELS = tuple(f"{base}_{head}" for base in ("context", "fusion") for head in HEADS)
LABELS = {
    f"{base}_{head}": f"{base_label} + {head_label}"
    for base, base_label in (("context", "Environmental"), ("fusion", "Frozen fusion"))
    for head, head_label in (("constant", "constant correction"), ("gru_pca", "GRU PCA"),
                             ("gru_episodic", "episodic GRU"), ("tree_pca", "tree PCA"),
                             ("tree_episodic", "episodic tree"))
}


def comparison_definitions():
    comparisons = []
    for base in ("context", "fusion"):
        for representation in ("gru", "tree"):
            for k in (3, 5):
                comparisons.append((f"{base}_{representation}_episodic_vs_pca_k{k}",
                                    f"{base}_{representation}_episodic", k,
                                    f"{base}_{representation}_pca", k))
        for k in (3, 5):
            comparisons.append((f"{base}_gru_vs_tree_episodic_k{k}",
                                f"{base}_gru_episodic", k, f"{base}_tree_episodic", k))
        for representation in ("gru", "tree"):
            comparisons.append((f"{base}_{representation}_episodic_vs_constant_k5",
                                f"{base}_{representation}_episodic", 5, f"{base}_constant", 5))
    return comparisons


def read_bound_json(run: Path, name: str, completion: dict, sources: list) -> dict:
    path = run / name
    content_hash = sha256(path)
    if completion["files"].get(name) != content_hash:
        raise ValueError(f"Unbound or changed completed analysis input: {path}")
    sources.append({"path": str(path), "sha256": content_hash})
    return json.loads(path.read_text())


def load_panel(root: Path):
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
            states.append({
                "split_seed": split, "seed": seed,
                "adapters": read_bound_json(run, "adapters.json", completion, sources),
                "projectors": {name: read_bound_json(run, f"{name}_projector.json", completion, sources)
                               for name in ("gru", "tree")},
            })
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    for (_, _), group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        zero = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", "fusion"):
            for head in HEADS[1:]:
                np.testing.assert_array_equal(zero[f"{base}_constant"], zero[f"{base}_{head}"])
    return panel, thresholds, sources, states


def summarize_metrics(panel, thresholds):
    rows = []
    for (split, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                     "n_stations": group.station.nunique(),
                     **metric_values(group.y_true, group.y_pred, thresholds[(int(split), int(seed))])})
    runs = pd.DataFrame(rows)
    partitions = runs.groupby(["split_seed", "model_name", "k"], as_index=False).agg(
        **{name: (name, "mean") for name in METRICS}, n_seeds=("seed", "size"),
        mae_sd_seed=("mae", "std"), n_stations=("n_stations", "first"),
        n_query_cells=("n_query_cells", "first"), q90_n=("q90_n", "first"),
        q90_unstable=("q90_unstable", "any"))
    curves = partitions.groupby(["model_name", "k"], as_index=False).agg(
        **{name: (name, "mean") for name in METRICS}, n_splits=("split_seed", "size"),
        mae_sd_split=("mae", "std"))
    return runs, partitions, curves


def add_directions(frame):
    frame["delta_mae"] = frame.candidate_mae - frame.reference_mae
    frame["relative_gain_pct"] = 100 * (1 - frame.candidate_mae / frame.reference_mae)
    frame["direction"] = np.select([frame.delta_mae < 0, frame.delta_mae > 0],
                                    ["candidate_better", "reference_better"], default="equal")
    return frame


def compare(panel: pd.DataFrame, draws: int):
    effects, directions, station_responses = [], [], []
    for name, candidate, ck, reference, rk in comparison_definitions():
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        effect = {"comparison": name, "candidate": candidate, "candidate_k": ck,
                  "reference": reference, "reference_k": rk,
                  **joint_station_bootstrap(pairs, draws=draws)}
        per_seed = pairs.groupby(["split_seed", "seed"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
        per_seed["comparison"] = name
        add_directions(per_seed)
        split_delta = per_seed.groupby("split_seed").delta_mae.mean()
        effect.update(improved_split_seed_pairs=int((per_seed.delta_mae < 0).sum()),
                      n_split_seed_pairs=len(per_seed), improved_splits=int((split_delta < 0).sum()),
                      equal_splits=int((split_delta == 0).sum()))
        station = pairs.groupby(["split_seed", "station", "cell"], as_index=False).agg(
            candidate_error=("candidate_error", "mean"), reference_error=("reference_error", "mean"),
            ecological_novelty=("ecological_novelty", "mean"), upstream_support=("upstream_support", "mean"))
        station = station.groupby(["split_seed", "station"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
            n_query_cells=("cell", "size"), ecological_novelty=("ecological_novelty", "mean"),
            upstream_support=("upstream_support", "mean"))
        station["comparison"] = name
        add_directions(station)
        effects.append(effect)
        directions.append(per_seed)
        station_responses.append(station)
    directions = pd.concat(directions, ignore_index=True)
    partitions = directions.groupby(["comparison", "split_seed"], as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
        n_seeds=("seed", "size"))
    add_directions(partitions)
    return (pd.DataFrame(effects), directions, partitions,
            pd.concat(station_responses, ignore_index=True))


def adapter_choices(states):
    rows = []
    for run in states:
        if set(run["adapters"]) != set(MODELS):
            raise ValueError("The ten saved adapter choices are incomplete")
        for model, state in run["adapters"].items():
            if set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError(f"Adapter choices do not cover all K: {model}")
            for k, choice in state["selection_by_k"].items():
                k = int(k)
                ridge = choice["ridge_strength"]
                constant = ridge == "infinity" or np.isinf(float(ridge))
                rows.append({
                    "split_seed": run["split_seed"], "seed": run["seed"], "model_name": model,
                    "k": k, "alpha": choice["alpha"],
                    "ridge_strength": "infinity" if constant else float(ridge),
                    "constant_shape_selected": constant,
                    "finite_shape_enabled": bool(not constant and k > 1),
                    "selection_role": state["selection_role"],
                })
    return pd.DataFrame(rows)


def projector_training(states):
    rows = []
    for run in states:
        for representation, state in run["projectors"].items():
            trace = state["trace"]
            if not trace or trace[0]["epoch"] != 0:
                raise ValueError("Projector trace must retain the initial PCA validation result")
            best_epoch = int(state["best_epoch"])
            best_rows = [entry for entry in trace if entry["epoch"] == best_epoch]
            if len(best_rows) != 1:
                raise ValueError("Saved projector best epoch is missing or duplicated")
            initial = float(trace[0]["validation_mae"])
            best = float(state["validation_metrics"]["validation_mae"])
            if best != float(best_rows[0]["validation_mae"]) or not np.isfinite([initial, best]).all():
                raise ValueError("Projector validation selection differs from its trace")
            projection = np.asarray(state["projection"], dtype=float)
            if projection.ndim != 2 or projection.shape[0] != 2 or projection.shape[1] < 2:
                raise ValueError("A two-dimensional episodic projection is required")
            if not np.isfinite(projection).all():
                raise ValueError("Projector contains nonfinite parameters")
            active_modes = np.asarray(state["active_modes"], dtype=bool)
            effective_rank = int(state["effective_rank"])
            if (active_modes.shape != (projection.shape[1],)
                    or int(active_modes.sum()) != effective_rank or effective_rank < 2):
                raise ValueError("Projector active modes do not match its dimension/effective rank")
            initial_projection = np.eye(2, projection.shape[1])
            parameter_distance = float(np.linalg.norm(projection - initial_projection))
            # P = W.T W describes the selected subspace independently of a
            # harmless rotation of the two output coordinates.
            subspace_distance = float(np.linalg.norm(
                projection.T @ projection - initial_projection.T @ initial_projection))
            for field, expected in (("projection_subspace_displacement", subspace_distance),
                                    ("projection_parameter_displacement", parameter_distance)):
                if field in state and not np.isclose(float(state[field]), expected, rtol=1e-6, atol=1e-6):
                    raise ValueError(f"Saved {field} differs from projection parameters")
            rows.append({
                "split_seed": run["split_seed"], "seed": run["seed"],
                "representation": representation, "best_epoch": best_epoch,
                "epochs_run": int(state["epochs_run"]), "initial_validation_mae": initial,
                "best_validation_mae": best, "validation_mae_change": best - initial,
                "projection_subspace_displacement": subspace_distance,
                "projection_parameter_displacement": parameter_distance,
                "projection_input_dimension": projection.shape[1],
                "active_modes": int(active_modes.sum()),
                "effective_rank": effective_rank,
                "eigenvalue_floor": float(state["eigenvalue_floor"]),
                "selection_role": "source_validation",
            })
    return pd.DataFrame(rows)


def write_report(out, curves, comparisons, training, choices, draws):
    lines = ["# DOC station adaptation: episodic representations and matched PCA", "",
             "The complete nine-run panel compares frozen environmental/fusion predictions with",
             "source-trained, two-dimensional episodic representations and their matched PCA controls.",
             "Partitions 142–144 have been seen during development; these are development results",
             "on same-cohort station holdouts, not independent external-basin confirmation.", "",
             "Negative paired MAE difference and positive relative reduction favor the candidate.",
             f"Intervals use {draws:,} joint whole-station bootstrap resamples, preserving repeated",
             "station identities across partitions. Seed losses are averaged within partitions,",
             "followed by equal partition weighting. The analysis makes no test-based selections.", ""]
    for title, selector in (
        ("Episodic training versus matched PCA", "episodic_vs_pca"),
        ("GRU versus tree representations", "gru_vs_tree_episodic"),
        ("Complete episodic adapters versus constant correction", "episodic_vs_constant"),
    ):
        lines += [f"## {title}", "",
                  "| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in comparisons[comparisons.comparison.str.contains(selector)].itertuples():
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
    lines += ["", "## Source-selected training and adaptation", "",
              "Projector initialization, best validation epoch/loss and displacement from PCA are",
              "recorded for each GRU/tree run in `projector_training.csv`. The table below averages",
              "the nine source-validation training summaries; these are not test-error estimates.", "",
              "| Representation | Mean best epoch | Initial validation MAE | Best validation MAE | Mean subspace displacement |",
              "|---|---:|---:|---:|---:|"]
    for name, group in training.groupby("representation"):
        lines.append(f"| {name.upper()} | {group.best_epoch.mean():.2f} | "
                     f"{group.initial_validation_mae.mean():.4f} | {group.best_validation_mae.mean():.4f} | "
                     f"{group.projection_subspace_displacement.mean():.4f} |")
    lines += ["", "Subspace displacement is the Frobenius distance between the final and initial",
              "projection matrices WᵀW, making it invariant to rotations within the two output dimensions.", "",
              "| Adapter | K | Finite shape enabled | Constant shape selected |",
              "|---|---:|---:|---:|"]
    selected = choices[~choices.model_name.str.endswith("_constant") & choices.k.isin((3, 5))]
    for (model, k), group in selected.groupby(["model_name", "k"]):
        lines.append(f"| {LABELS[model]} | {k} | {int(group.finite_shape_enabled.sum())}/9 | "
                     f"{int(group.constant_shape_selected.sum())}/9 |")
    lines += ["", "Finite regularization enables a shape term, but does not imply a nonzero correction",
              "at every station. Infinite regularization selects a constant correction; all alpha and",
              "ridge choices are retained in `adapter_choices.csv`.", "",
              "## Interpretation", "",
              "The primary comparison asks whether supervised episodic projection improves the complete",
              "support adapter relative to its within-run PCA representation. Each representation keeps",
              "its own source-selected level shrinkage and ridge strength, so this is not a fixed-alpha",
              "ablation of the shape term. The GRU-versus-tree comparison uses equally sized projections",
              "and the same support budget; it does not isolate river messages or establish causal effects.", "",
              "The five source context forests are refitted without held-station labels. GRU/tree feature",
              "extractors remain frozen from prior training and are evaluated under label-hidden source",
              "views. Their weights are not refitted as out-of-fold experts. Source validation selects the",
              "projection epoch and final adapter parameters; this is conditional development evaluation.", "",
              "K = 0 is exactly the frozen base. Support observations can postdate query months: this is",
              "retrospective reconstruction. Q90 thresholds come from each run's source training labels.",
              "Unique query counts and unstable-tail flags are in `run_metrics.csv`; seeds and repeated",
              "stations do not increase the ecological sample size. Complete K curves, all partition/seed",
              "directions and station responses remain available alongside the aggregate comparisons.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def analyze(root: Path, *, draws=5000):
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, states = load_panel(root)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    comparisons, directions, partition_directions, stations = compare(panel, draws)
    choices = adapter_choices(states)
    training = projector_training(states)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("run_metrics", runs), ("split_metrics", splits), ("k_curves", curves),
                        ("comparisons", comparisons), ("split_seed_directions", directions),
                        ("partition_directions", partition_directions), ("station_heterogeneity", stations),
                        ("adapter_choices", choices), ("projector_training", training)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, comparisons, training, choices, draws)
    (out / "status.json").write_text(json.dumps({
        "complete": True, "available_runs": 9, "expected_runs": 9,
        "bootstrap_draws": draws, "models": list(MODELS), "n_comparisons": len(comparisons),
        "scientific_role": "seen-partition development evaluation; not independent confirmation",
    }, indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps({
        "sources": sources, "analysis_script": str(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)), "bootstrap_seed": 42,
        "estimand": "individual-seed mean within partition; query-cell weighted; partition equal",
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
