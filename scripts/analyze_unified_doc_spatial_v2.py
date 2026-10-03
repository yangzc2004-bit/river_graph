"""Compare frozen robust fusion and support-conditioned DOC shape predictions.

This script reads completed prediction panels. It never selects a candidate,
fits an adapter, or changes the saved version-1 references.
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

DEFAULT_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v2")
BASELINE_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation")
SPLITS = (142, 143, 144)
SEEDS = (42, 43, 44)
KS = (0, 1, 3, 5)
MODELS = tuple(f"{base}_{adapter}" for base in ("context", "fusion")
               for adapter in ("constant", "gru_shape", "tree_shape"))
REFERENCES = ("v1_extra_trees_calibrated", "v1_hybrid_calibrated")
METRICS = ("mae", "rmse", "r2", "log_mae", "q90_mae")
COLUMNS = ("split_seed", "seed", "cell", "station", "month", "y_true", "y_pred",
           "ecological_novelty", "upstream_support", "model_name", "k")
LABELS = {
    "context_constant": "Environmental + constant correction",
    "context_gru_shape": "Environmental + GRU shape",
    "context_tree_shape": "Environmental + tree shape",
    "fusion_constant": "Robust fusion + constant correction",
    "fusion_gru_shape": "Robust fusion + GRU shape",
    "fusion_tree_shape": "Robust fusion + tree shape",
    "v1_extra_trees_calibrated": "Version-1 calibrated ExtraTrees",
    "v1_hybrid_calibrated": "Version-1 calibrated hybrid",
}


def comparison_definitions():
    comparisons = []
    for k in (0, 5):
        comparisons.extend([
            (f"fusion_vs_context_k{k}", "fusion_constant", k, "context_constant", k),
            (f"fusion_vs_v1_hybrid_k{k}", "fusion_constant", k, "v1_hybrid_calibrated", k),
        ])
    for base in ("context", "fusion"):
        for k in (3, 5):
            for shape in ("gru", "tree"):
                comparisons.append((f"{base}_{shape}_shape_vs_constant_k{k}",
                                    f"{base}_{shape}_shape", k, f"{base}_constant", k))
            comparisons.append((f"{base}_gru_vs_tree_shape_k{k}",
                                f"{base}_gru_shape", k, f"{base}_tree_shape", k))
    return comparisons


def canonical_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the runner's model/K names or the shared analysis convention."""
    frame = frame.copy()
    for alias, canonical in (("model", "model_name"), ("K", "k")):
        if alias in frame:
            if canonical in frame:
                np.testing.assert_array_equal(frame[alias].to_numpy(), frame[canonical].to_numpy())
                frame = frame.drop(columns=alias)
            else:
                frame = frame.rename(columns={alias: canonical})
    missing = set(COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return frame[list(COLUMNS)]


def validate_panel(frame: pd.DataFrame, *, expected_models: tuple[str, ...]) -> None:
    keys = ["split_seed", "seed", "model_name", "k", "cell"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Prediction panel is empty or has duplicate query rows")
    numeric = ["y_true", "y_pred", "ecological_novelty", "upstream_support"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Prediction panel contains nonfinite values")
    if (frame[["y_true", "y_pred"]] < 0).any().any() or not frame.upstream_support.between(0, 1).all():
        raise ValueError("Invalid DOC concentration or upstream-support fraction")
    if frame[["station", "month"]].isna().any().any():
        raise ValueError("Station/month identity is missing")
    if frame.cell.dtype.kind not in "iu" or (frame.cell < 0).any():
        raise ValueError("Cell identities must be nonnegative integers")
    identity = frame.groupby("cell")[["station", "month", "y_true"]].nunique()
    if (identity != 1).any().any():
        raise ValueError("Station/month/truth identity changes across runs")
    pairs = set(map(tuple, frame[["split_seed", "seed"]].drop_duplicates().to_numpy()))
    if pairs != {(s, r) for s in SPLITS for r in SEEDS}:
        raise ValueError("All nine partition-by-training-seed runs are required")
    expected_combinations = {(model, k) for model in expected_models for k in KS}
    for split, part in frame.groupby("split_seed"):
        reference = None
        for (_, _, _), group in part.groupby(["seed", "model_name", "k"]):
            query = group.sort_values("cell")[["cell", "station", "month", "y_true"]].reset_index(drop=True)
            if reference is None:
                reference = query
            else:
                pd.testing.assert_frame_equal(query, reference, check_dtype=False, check_exact=True,
                                              obj=f"fixed query in partition {split}")
        for seed, group in part.groupby("seed"):
            combinations = set(map(tuple, group[["model_name", "k"]].drop_duplicates().to_numpy()))
            if combinations != expected_combinations:
                raise ValueError(f"Incomplete model/K panel in split {split}, seed {seed}")
            if set(MODELS).issubset(expected_models):
                zero = group[group.k.eq(0)].pivot(index="cell", columns="model_name", values="y_pred")
                for base in ("context", "fusion"):
                    for adapter in ("gru_shape", "tree_shape"):
                        np.testing.assert_array_equal(zero[f"{base}_constant"], zero[f"{base}_{adapter}"])


def load_panels(root: Path, baseline_root: Path) -> tuple[pd.DataFrame, dict, list]:
    frames, references, thresholds, sources = [], [], {}, []
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            prediction_path, config_path = run / "predictions.parquet", run / "config.json"
            if not prediction_path.is_file() or not config_path.is_file():
                raise ValueError(f"Missing version-2 prediction/configuration: {run}")
            config = json.loads(config_path.read_text())
            completion_path = run / "complete.json"
            if not completion_path.is_file():
                raise ValueError(f"Version-2 run is not complete: {run}")
            completion = json.loads(completion_path.read_text())
            if completion["config_hash"] != config_digest(config):
                raise ValueError(f"Changed version-2 configuration identity: {run}")
            for path in (prediction_path, config_path):
                if completion["files"].get(path.name) != sha256(path):
                    raise ValueError(f"Unbound or changed version-2 analysis input: {path}")
            if (config["split_seed"], config["seed"]) != (split, seed):
                raise ValueError(f"Run/configuration identity differs: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid training Q90 threshold: {run}")
            thresholds[(split, seed)] = threshold
            frame = canonical_panel(pd.read_parquet(prediction_path))
            if not frame.split_seed.eq(split).all() or not frame.seed.eq(seed).all():
                raise ValueError(f"Run/prediction identity differs: {run}")
            frames.append(frame)
            sources.extend({"path": str(p), "sha256": sha256(p)} for p in (prediction_path, config_path))
            sources.append({"path": str(completion_path), "sha256": sha256(completion_path)})
            old_run = baseline_root / "runs" / run.name
            old_path, old_config_path = old_run / "predictions.parquet", old_run / "config.json"
            completion_path = old_run / "complete.json"
            completion = json.loads(completion_path.read_text())
            old_config = json.loads(old_config_path.read_text())
            if completion["config_hash"] != config_digest(old_config):
                raise ValueError(f"Changed version-1 configuration identity: {old_run}")
            for path in (old_path, old_config_path):
                content_hash = sha256(path)
                if completion["files"].get(path.name) != content_hash:
                    raise ValueError(f"Changed fixed version-1 reference: {path}")
                sources.append({"path": str(path), "sha256": content_hash})
            if float(old_config["q90_threshold_train"]) != threshold:
                raise ValueError("Version-1 and version-2 training Q90 thresholds differ")
            old = pd.read_parquet(old_path, columns=list(COLUMNS))
            old = old[old.model_name.isin(("extra_trees_calibrated", "hybrid_calibrated"))].copy()
            old["model_name"] = "v1_" + old.model_name
            references.append(old)
    new, old = pd.concat(frames, ignore_index=True), pd.concat(references, ignore_index=True)
    validate_panel(new, expected_models=MODELS)
    validate_panel(old, expected_models=REFERENCES)
    panel = pd.concat([new, old], ignore_index=True)
    validate_panel(panel, expected_models=MODELS + REFERENCES)
    return panel, thresholds, sources


def compare(panel: pd.DataFrame, draws: int):
    results, directions, station_results = [], [], []
    for name, candidate, ck, reference, rk in comparison_definitions():
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        effect = {"comparison": name, "candidate": candidate, "candidate_k": ck,
                  "reference": reference, "reference_k": rk,
                  **joint_station_bootstrap(pairs, draws=draws)}
        per_seed = pairs.groupby(["split_seed", "seed"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
        per_seed["delta_mae"] = per_seed.candidate_mae - per_seed.reference_mae
        per_seed["relative_gain_pct"] = 100 * (1 - per_seed.candidate_mae / per_seed.reference_mae)
        per_seed["comparison"] = name
        per_seed["direction"] = np.select([per_seed.delta_mae < 0, per_seed.delta_mae > 0],
                                          ["candidate_better", "reference_better"], default="equal")
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
        station["delta_mae"] = station.candidate_mae - station.reference_mae
        station["relative_gain_pct"] = 100 * (1 - station.candidate_mae / station.reference_mae)
        station["comparison"] = name
        station_results.append(station)
        results.append(effect)
        directions.append(per_seed)
    return pd.DataFrame(results), pd.concat(directions, ignore_index=True), pd.concat(station_results, ignore_index=True)


def read_run_choices(run: Path):
    """Export selected source-validation choices, keeping their saved identities."""
    config = json.loads((run / "config.json").read_text())
    completion = json.loads((run / "complete.json").read_text())
    records, sources = {}, []
    for name in ("fusion.json", "adapters.json"):
        path = run / name
        content_hash = sha256(path)
        if completion["files"].get(name) != content_hash:
            raise ValueError(f"Unbound or changed fitted choices: {path}")
        records[name] = json.loads(path.read_text())
        sources.append({"path": str(path), "sha256": content_hash})
    fusion = records["fusion.json"]
    selected = fusion["selected"]
    b0, bc, br = selected["coefficients"]
    base = {"split_seed": config["split_seed"], "seed": config["seed"]}
    fusion_row = {
        **base, "selected_name": selected["name"], "selected_kind": selected["kind"],
        "ridge_strength": selected["strength"], "convex_weight": selected["weight"],
        "station_cv_mae": selected["cv_mae"], "refit_validation_mae": selected["refit_training_mae"],
        "correction_intercept": b0, "correction_context_coefficient": bc,
        "correction_temporal_difference_coefficient": br,
        "effective_context_coefficient": 1 + bc - br, "effective_temporal_coefficient": br,
        "selection_role": fusion["selection_role"], "n_station_folds": fusion["n_splits"],
    }
    adapters = records["adapters.json"]
    if set(adapters) != set(MODELS):
        raise ValueError(f"The six adapter choices are incomplete: {run}")
    rows = []
    for model, state in adapters.items():
        if set(map(int, state["selection_by_k"])) != set(KS):
            raise ValueError(f"Adapter choices do not cover all K: {run}, {model}")
        for k, choice in state["selection_by_k"].items():
            k = int(k)
            ridge = choice["ridge_strength"]
            constant = ridge == "infinity" or np.isinf(float(ridge))
            rows.append({
                **base, "model_name": model, "k": k, "alpha": choice["alpha"],
                "ridge_strength": "infinity" if constant else float(ridge),
                "constant_shape_selected": constant,
                "finite_shape_enabled": bool(not constant and k > 1),
                "selection_role": state["selection_role"],
            })
    return fusion_row, rows, sources


def write_report(out: Path, curves: pd.DataFrame, comparisons: pd.DataFrame, draws: int,
                 fusion_choices: pd.DataFrame, adapter_choices: pd.DataFrame):
    lines = ["# DOC spatial adaptation: robust fusion and temporal-shape comparison", "",
             "All nine saved expert packages are evaluated on the original fixed station queries.",
             "The three partition outcomes were seen during development. This analysis evaluates the",
             "saved source-selected predictions; it does not choose configurations from outer-test errors.", "",
             "## Robust fusion", "",
             "Negative paired MAE difference and positive relative reduction favor the named candidate.",
             f"Intervals use {draws:,} joint whole-station bootstrap resamples. Seed losses are averaged",
             "within each partition, followed by equal partition weighting.", ""]
    for section, subset in (
        (None, comparisons[comparisons.comparison.str.startswith("fusion_vs_")]),
        ("## Shape adaptation", comparisons[~comparisons.comparison.str.startswith("fusion_vs_")]),
    ):
        if section:
            lines += ["", section, ""]
        lines += ["| Comparison | Reference MAE | Candidate MAE | Difference [95% CI] | Reduction [95% CI] | Improved partitions; fits |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in subset.itertuples():
            lines.append(f"| {row.comparison} | {row.reference_mae:.4f} | {row.candidate_mae:.4f} | "
                         f"{row.delta_mae:.4f} [{row.delta_ci_low:.4f}, {row.delta_ci_high:.4f}] | "
                         f"{row.relative_gain_pct:.2f}% [{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}] | "
                         f"{row.improved_splits}/3; {row.improved_split_seed_pairs}/9 |")
    lines += ["", "## Error profiles", "",
              "| Predictor | K | MAE | RMSE | R² | Log MAE | Training-Q90 MAE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in curves[curves.k.isin((0, 5))].itertuples():
        lines.append(f"| {LABELS[row.model_name]} | {row.k} | {row.mae:.4f} | {row.rmse:.4f} | "
                     f"{row.r2:.4f} | {row.log_mae:.4f} | {row.q90_mae:.4f} |")
    lines += ["", "## Source-selected mechanisms", ""]
    fusion_counts = fusion_choices.selected_kind.value_counts()
    lines.append("Fusion selections: " + "; ".join(f"{kind} {count}/9" for kind, count in fusion_counts.items()) + ".")
    lines += ["", "| Shape arm | K | Finite shape enabled | Constant shape selected |",
              "|---|---:|---:|---:|"]
    shape_choices = adapter_choices[adapter_choices.model_name.str.endswith("_shape") & adapter_choices.k.isin((3, 5))]
    for (model, k), group in shape_choices.groupby(["model_name", "k"]):
        lines.append(f"| {LABELS[model]} | {k} | {int(group.finite_shape_enabled.sum())}/9 | "
                     f"{int(group.constant_shape_selected.sum())}/9 |")
    lines += ["", "Finite regularization enables a shape term; it does not guarantee a nonzero correction",
              "at every station. Infinite regularization selects the constant-offset case. Full source",
              "choices are in `fusion_choices.csv` and `adapter_choices.csv`, including each alpha."]
    lines += ["", "## Reading the comparisons", "",
              "Fusion versus context measures the benefit of robust expert combination. Fusion versus",
              "the version-1 hybrid measures the change from the earlier unrestricted selection procedure.",
              "A shape-versus-constant comparison compares the complete temporal-shape adapter with the constant",
              "adapter, allowing each its own source-selected level shrinkage; it is not a fixed-alpha",
              "ablation of the shape term. GRU-versus-tree shape compares two equally sized",
              "source-trained representations under the matched support adapter; it does not establish",
              "a river-message or causal effect. A null result is retained alongside improvements.", "",
              "K = 0 shape predictions equal their own unadapted base. K > 0 support labels may postdate",
              "query months: this is retrospective reconstruction. Source-station meta-CV is conditional",
              "on the previously selected frozen experts, rather than an independent evaluation of",
              "their full fitting procedure.", "",
              "The complete K curves are in `k_curves.csv`; per-run Q90 sample counts and unstable-tail",
              "flags are in `run_metrics.csv`. Partition metrics retain unique query counts; seeds and",
              "repeated stations are not additional ecological observations. `split_seed_directions.csv`",
              "retains every result and `station_heterogeneity.csv` describes station responses.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def analyze(root: Path, *, draws: int = 5000, baseline_root: Path = BASELINE_ROOT) -> Path:
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources = load_panels(root, baseline_root)
    metrics = []
    for (split, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        metrics.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                        "n_stations": group.station.nunique(),
                        **metric_values(group.y_true, group.y_pred, thresholds[(int(split), int(seed))])})
    metrics = pd.DataFrame(metrics)
    splits = metrics.groupby(["split_seed", "model_name", "k"], as_index=False).agg(
        **{name: (name, "mean") for name in METRICS}, n_seeds=("seed", "size"),
        mae_sd_seed=("mae", "std"), n_stations=("n_stations", "first"),
        n_query_cells=("n_query_cells", "first"), q90_n=("q90_n", "first"),
        q90_unstable=("q90_unstable", "any"))
    curves = splits.groupby(["model_name", "k"], as_index=False).agg(
        **{name: (name, "mean") for name in METRICS}, n_splits=("split_seed", "size"),
        mae_sd_split=("mae", "std"))
    comparisons, consistency, stations = compare(panel, draws)
    fusion_rows, adapter_rows = [], []
    for split in SPLITS:
        for seed in SEEDS:
            fusion_row, choices, choice_sources = read_run_choices(root / "runs" / f"split{split}_seed{seed}")
            fusion_rows.append(fusion_row)
            adapter_rows.extend(choices)
            sources.extend(choice_sources)
    fusion_choices, adapter_choices = pd.DataFrame(fusion_rows), pd.DataFrame(adapter_rows)
    partition_directions = consistency.groupby(["comparison", "split_seed"], as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
        n_seeds=("seed", "size"))
    partition_directions["delta_mae"] = partition_directions.candidate_mae - partition_directions.reference_mae
    partition_directions["relative_gain_pct"] = 100 * (
        1 - partition_directions.candidate_mae / partition_directions.reference_mae)
    partition_directions["direction"] = np.select(
        [partition_directions.delta_mae < 0, partition_directions.delta_mae > 0],
        ["candidate_better", "reference_better"], default="equal")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("run_metrics", metrics), ("split_metrics", splits), ("k_curves", curves),
                        ("comparisons", comparisons), ("split_seed_directions", consistency),
                        ("partition_directions", partition_directions), ("station_heterogeneity", stations),
                        ("fusion_choices", fusion_choices), ("adapter_choices", adapter_choices)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, comparisons, draws, fusion_choices, adapter_choices)
    (out / "status.json").write_text(json.dumps({
        "complete": True, "available_runs": 9, "expected_runs": 9,
        "bootstrap_draws": draws, "models": list(MODELS), "fixed_references": list(REFERENCES),
        "scientific_role": "seen-partition development evaluation; not independent confirmation",
        "n_comparisons": len(comparisons),
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
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    print(analyze(args.root, draws=args.draws, baseline_root=args.baseline_root))


if __name__ == "__main__":
    main()
