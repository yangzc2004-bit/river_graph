"""Describe source-validation chemical-kernel calibration development only.

No outer target predictions or scores are read. Selected source-validation scores
are tuning evidence, not unbiased confirmation of station-transfer performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments/phase4_transfer/doc_chemical_kernel_v1")
PIPELINES = ("neural_chemistry", "neural_chemistry_integrated", "tree_chemistry")
MODES = ("parent", "masks", "chemistry")
MODELS = tuple(f"{pipeline}_{mode}" for pipeline in PIPELINES for mode in MODES)
KS = (0, 1, 3, 5)
SEEDS = (42, 43, 44)
SPLITS = (142, 143, 144)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bound_file(run, name, completion, sources):
    path = run / name
    value = sha256(path)
    if completion["files"].get(name) != value:
        raise ValueError(f"Changed or unbound source-validation artifact: {path}")
    sources.append({"path": str(path), "sha256": value})
    return path


def read_bound_json(run, name, completion, sources):
    return json.loads(bound_file(run, name, completion, sources).read_text())


def comparison_definitions():
    """All eighteen nontrivial source-validation contrasts, fixed before reading."""
    rows = []
    for pipeline in PIPELINES:
        for k in (3, 5):
            for candidate, reference in (("masks", "parent"), ("chemistry", "parent"), ("chemistry", "masks")):
                rows.append({"comparison": f"{pipeline}_{candidate}_vs_{reference}_k{k}", "pipeline": pipeline,
                    "candidate": f"{pipeline}_{candidate}", "reference": f"{pipeline}_{reference}", "k": k})
    return rows


def validate_panel(frame, *, require_nine=True):
    keys = ["split_seed", "seed", "model_name", "k", "cell"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Empty or duplicated source-validation predictions")
    if require_nine and set(map(tuple, frame[["split_seed", "seed"]].drop_duplicates().to_numpy())) != {(p, s) for p in SPLITS for s in SEEDS}:
        raise ValueError("Incomplete source-validation packages")
    numeric = ["y_true", "y_pred", "parent_y_pred", "active_support_count", "kernel_log_correction", "parent_preclip_z"]
    if (not np.isfinite(frame[numeric]).all().all() or (frame[["y_true", "y_pred", "parent_y_pred"]] < 0).any().any()
            or frame[["station", "month"]].isna().any().any() or frame.cell.dtype.kind not in "iu"):
        raise ValueError("Invalid source-validation prediction values or coordinates")
    if not np.isin(frame.aux_available, [False, True]).all():
        raise ValueError("Nonbinary auxiliary availability")
    if (frame.groupby("cell")[["station", "month", "y_true"]].nunique() != 1).any().any():
        raise ValueError("Validation cell identity changes across packages")
    for (_, _), run in frame.groupby(["split_seed", "seed"]):
        if set(map(tuple, run[["model_name", "k"]].drop_duplicates().to_numpy())) != {(m, k) for m in MODELS for k in KS}:
            raise ValueError("Incomplete validation model/K panel")
        reference = None
        for (_, _), group in run.groupby(["model_name", "k"]):
            view = group.sort_values("cell")[["cell", "station", "month", "y_true"]].reset_index(drop=True)
            if reference is None:
                reference = view
            else:
                pd.testing.assert_frame_equal(view, reference, check_exact=True)
        for pipeline in PIPELINES:
            for k in KS:
                parent = run[run.model_name.eq(f"{pipeline}_parent") & run.k.eq(k)].sort_values("cell")
                for mode in MODES:
                    group = run[run.model_name.eq(f"{pipeline}_{mode}") & run.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(group.parent_y_pred, parent.y_pred)
                    if k in (0, 1) or mode == "parent":
                        np.testing.assert_array_equal(group.y_pred, parent.y_pred)
                        np.testing.assert_array_equal(group.kernel_log_correction, np.zeros(len(group)))
                    inactive = ~group.aux_available.to_numpy(dtype=bool)
                    np.testing.assert_array_equal(group.y_pred.to_numpy()[inactive], parent.y_pred.to_numpy()[inactive])
                    no_donors = group.active_support_count.to_numpy() < 2
                    np.testing.assert_array_equal(group.kernel_log_correction.to_numpy()[no_donors], np.zeros(int(no_donors.sum())))
                    zero = group.kernel_log_correction.eq(0).to_numpy()
                    np.testing.assert_array_equal(group.y_pred.to_numpy()[zero], parent.y_pred.to_numpy()[zero])
    for split, part in frame.groupby("split_seed"):
        base = None
        for seed, group in part.groupby("seed"):
            view = group.drop_duplicates("cell").sort_values("cell")[["cell", "station", "month", "y_true"]].reset_index(drop=True)
            if base is None:
                base = view
            else:
                pd.testing.assert_frame_equal(view, base, check_exact=True, obj=f"source-validation queries split{split}/seed{seed}")


def summarize(panel, thresholds):
    rows = []
    for (split, seed, model, k), frame in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        y, pred = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
        error = pred - y
        active = frame.aux_available.to_numpy(dtype=bool)
        high = y >= thresholds[(int(split), int(seed))]
        row = {"split_seed": split, "seed": seed, "model_name": model, "pipeline": frame.pipeline.iloc[0],
            "kernel_mode": frame.kernel_mode.iloc[0], "k": k, "n": len(frame), "n_stations": frame.station.nunique(),
            "n_active": int(active.sum()), "n_q90": int(high.sum()), "n_ordinary": int((~high).sum()),
            "mae": float(np.abs(error).mean()), "active_mae": float(np.abs(error[active]).mean()) if active.any() else np.nan,
            "rmse": float(np.sqrt(np.square(error).mean())), "log_mae": float(np.abs(np.log1p(pred) - np.log1p(y)).mean()),
            "signed_bias": float(error.mean()), "q90_mae": float(np.abs(error[high]).mean()) if high.any() else np.nan,
            "q90_signed_bias": float(error[high].mean()) if high.any() else np.nan,
            "ordinary_mae": float(np.abs(error[~high]).mean()) if (~high).any() else np.nan,
            "ordinary_signed_bias": float(error[~high].mean()) if (~high).any() else np.nan,
            "q90_unstable": int(high.sum()) < 20}
        rows.append(row)
    runs = pd.DataFrame(rows)
    keys = ["model_name", "pipeline", "kernel_mode", "k"]
    metrics = ["mae", "active_mae", "rmse", "log_mae", "signed_bias", "q90_mae", "q90_signed_bias", "ordinary_mae", "ordinary_signed_bias"]
    parts = runs.groupby(["split_seed", *keys], as_index=False).agg(**{m: (m, "mean") for m in metrics},
        **{c: (c, "first") for c in ["n", "n_stations", "n_active", "n_q90", "n_ordinary"]}, q90_unstable=("q90_unstable", "any"))
    curves = parts.groupby(keys, as_index=False).agg(**{m: (m, "mean") for m in metrics},
        n_partitions=("split_seed", "nunique"), n_q90_nonempty_partitions=("q90_mae", "count"), q90_unstable_any=("q90_unstable", "any"))
    return runs, parts, curves


def compare(panel, thresholds):
    runs, station_rows = [], []
    for definition in comparison_definitions():
        a = panel[panel.model_name.eq(definition["candidate"]) & panel.k.eq(definition["k"])]
        b = panel[panel.model_name.eq(definition["reference"]) & panel.k.eq(definition["k"])]
        merged = a.merge(b[["split_seed", "seed", "cell", "y_pred"]].rename(columns={"y_pred": "reference_pred"}),
                         on=["split_seed", "seed", "cell"], how="inner", validate="one_to_one")
        if len(merged) != len(a) or len(a) != len(b):
            raise ValueError("Unmatched source-validation comparisons")
        for (split, seed), full in merged.groupby(["split_seed", "seed"]):
            high = full.y_true >= thresholds[(int(split), int(seed))]
            for region, mask in (("overall", np.ones(len(full), dtype=bool)), ("active", full.aux_available.to_numpy(dtype=bool)),
                                 ("q90", high.to_numpy()), ("ordinary", ~high.to_numpy())):
                frame = full.loc[mask]
                if frame.empty:
                    continue
                error_a = np.abs(frame.y_pred.to_numpy() - frame.y_true.to_numpy())
                error_b = np.abs(frame.reference_pred.to_numpy() - frame.y_true.to_numpy())
                delta = error_a - error_b
                runs.append({**definition, "split_seed": split, "seed": seed, "region": region, "n": len(frame),
                    "candidate_mae": float(error_a.mean()), "reference_mae": float(error_b.mean()),
                    "delta_mae": float(delta.mean()), "relative_gain_pct": 100 * float(-delta.mean() / error_b.mean()),
                    "direction": "candidate_better" if delta.mean() < 0 else "reference_better" if delta.mean() > 0 else "equal"})
                if region == "overall":
                    temporary = frame[["station"]].assign(mae_reduction=error_b - error_a)
                    for station, group in temporary.groupby("station"):
                        station_rows.append({**definition, "split_seed": split, "seed": seed, "station": station,
                            "n_station_cells": len(group), "n_partition_cells": len(frame),
                            "mae_reduction": float(group.mae_reduction.mean()),
                            "weighted_gain": float(group.mae_reduction.sum() / len(frame))})
    runs = pd.DataFrame(runs)
    keys = ["comparison", "pipeline", "candidate", "reference", "k", "region"]
    parts = runs.groupby([*keys, "split_seed"], as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"), delta_mae=("delta_mae", "mean"),
        n=("n", "first"), improved_seeds=("direction", lambda v: int(v.eq("candidate_better").sum())), n_seeds=("seed", "nunique"))
    parts["relative_gain_pct"] = -100 * parts.delta_mae / parts.reference_mae
    parts["direction"] = np.select([parts.delta_mae < 0, parts.delta_mae > 0], ["candidate_better", "reference_better"], default="equal")
    effects = parts.groupby(keys, as_index=False).agg(candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"),
        delta_mae=("delta_mae", "mean"), improved_partitions=("direction", lambda v: int(v.eq("candidate_better").sum())),
        equal_partitions=("direction", lambda v: int(v.eq("equal").sum())), n_partitions=("split_seed", "nunique"),
        improved_seed_fits=("improved_seeds", "sum"), n_seed_fits=("n_seeds", "sum"))
    effects["relative_gain_pct"] = -100 * effects.delta_mae / effects.reference_mae
    stations = pd.DataFrame(station_rows)
    weighted = stations.groupby(["comparison", "station", "split_seed"], as_index=False).agg(weighted_gain=("weighted_gain", "mean"))
    partition_counts = parts[parts.region.eq("overall")].groupby("comparison").split_seed.nunique().to_dict()
    weighted["weighted_gain"] /= weighted.comparison.map(partition_counts)
    global_stations = weighted.groupby(["comparison", "station"], as_index=False).weighted_gain.sum()
    concentration = []
    for name, group in global_stations.groupby("comparison"):
        positive = group.weighted_gain.clip(lower=0)
        negative = -group.weighted_gain.clip(upper=0)
        concentration.append({"comparison": name, "net_mae_reduction": float(group.weighted_gain.sum()),
            "positive_gain_mass": float(positive.sum()), "harm_mass": float(negative.sum()),
            "stations_improved": int((group.weighted_gain > 0).sum()), "stations_worsened": int((group.weighted_gain < 0).sum()),
            "stations_equal": int((group.weighted_gain == 0).sum()), "n_stations_unique": group.station.nunique(),
            "top5_positive_gain_share": float(positive.nlargest(5).sum() / positive.sum()) if positive.sum() else np.nan,
            "top5_harm_share": float(negative.nlargest(5).sum() / negative.sum()) if negative.sum() else np.nan})
    return effects, runs, parts, stations, global_stations, pd.DataFrame(concentration)


def correction_diagnostics(panel):
    rows = []
    for (split, seed, model, k), frame in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        for scope, sub in (("all", frame), ("auxiliary_active", frame[frame.aux_available])):
            if sub.empty:
                continue
            absolute_log = np.abs(sub.kernel_log_correction.to_numpy())
            absolute_native = np.abs(sub.y_pred.to_numpy() - sub.parent_y_pred.to_numpy())
            donors = sub.active_support_count.to_numpy()
            rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k, "scope": scope,
                "n_query_rows": len(sub), "active_support_count_mean": float(donors.mean()),
                "at_least_two_donor_fraction": float((donors >= 2).mean()),
                "changed_prediction_fraction": float((absolute_native != 0).mean()),
                "log_correction_mean_abs": float(absolute_log.mean()), "log_correction_q95_abs": float(np.quantile(absolute_log, .95)),
                "log_correction_max_abs": float(absolute_log.max()), "native_correction_mean_abs": float(absolute_native.mean()),
                "native_correction_q95_abs": float(np.quantile(absolute_native, .95)), "native_correction_max_abs": float(absolute_native.max())})
    return pd.DataFrame(rows)


def incremental_station_cv(candidate_stations):
    """Cross-validate only kernel strength/bandwidth, conditional on fixed parent.

    Parent experts, linear calibration and representations were already selected
    on source validation. This is not an OOF assessment of that complete parent.
    """
    scores = candidate_stations.copy()
    keys = ["split_seed", "seed", "pipeline", "mode", "k", "eta", "bandwidth_scale", "station"]
    if scores.duplicated(keys).any() or not np.isfinite(scores[["error_sum", "active_error_sum", "n", "n_active"]]).all().all():
        raise ValueError("Invalid candidate-by-station CV statistics")
    if ((scores[["error_sum", "active_error_sum", "n_active"]] < 0).any().any()
            or (scores.n <= 0).any() or (scores.n_active > scores.n).any()):
        raise ValueError("Invalid station error sums or denominators")
    folds, choices, evaluated = {}, [], []
    for split, split_scores in scores.groupby("split_seed"):
        station_ids = np.asarray(sorted(split_scores.station.unique()))
        if len(station_ids) < 5:
            raise ValueError("Five validation-station folds require at least five stations")
        permutation = np.random.default_rng(3100 + int(split)).permutation(station_ids)
        groups = np.array_split(permutation, 5)
        assignment = {str(station): fold for fold, group in enumerate(groups) for station in group}
        folds[str(int(split))] = {"seed": 3100 + int(split), "station_to_fold": assignment,
                                 "fold_station_counts": [len(group) for group in groups]}
        for (seed, pipeline, mode, k), task in split_scores.groupby(["seed", "pipeline", "mode", "k"]):
            if mode not in ("masks", "chemistry") or int(k) not in (3, 5):
                raise ValueError("Conditional CV requires the two kernels at K3/K5")
            candidates = set(map(tuple, task[["eta", "bandwidth_scale"]].drop_duplicates().to_numpy()))
            if candidates != {(eta, band) for eta in (0, .25, .5, 1) for band in (.5, 1, 2)}:
                raise ValueError("Incomplete kernel candidate grid in station CV")
            for _, group in task.groupby(["eta", "bandwidth_scale"]):
                if set(group.station) != set(station_ids):
                    raise ValueError("Candidate station populations differ")
            base = task[task.eta.eq(0) & task.bandwidth_scale.eq(1)].set_index("station")
            for fold in range(5):
                held_stations = set(groups[fold])
                training = task[~task.station.isin(held_stations)]
                training_scores = training.groupby(["eta", "bandwidth_scale"], as_index=False).agg(
                    n_active=("n_active", "sum"), active_error_sum=("active_error_sum", "sum"))
                if (training_scores.n_active <= 0).any():
                    raise ValueError("A CV training fold has no active validation query")
                training_scores["active_mae"] = training_scores.active_error_sum / training_scores.n_active
                chosen = min(training_scores.to_dict("records"), key=lambda row: (row["active_mae"], row["eta"],
                                 row["bandwidth_scale"] != 1, row["bandwidth_scale"]))
                held = task[task.station.isin(held_stations) & task.eta.eq(chosen["eta"])
                            & task.bandwidth_scale.eq(chosen["bandwidth_scale"])].copy()
                reference = base.loc[held.station]
                np.testing.assert_array_equal(held.n, reference.n)
                np.testing.assert_array_equal(held.n_active, reference.n_active)
                identity = {"split_seed": int(split), "seed": int(seed), "pipeline": pipeline, "mode": mode,
                            "k": int(k), "fold": fold}
                choices.append({**identity, **chosen, "selection_role": "other_four_source_validation_station_folds",
                    "n_training_stations": len(station_ids) - len(held_stations), "n_held_stations": len(held_stations),
                    "held_n": int(held.n.sum()), "held_n_active": int(held.n_active.sum()),
                    "held_mae": float(held.error_sum.sum() / held.n.sum()),
                    "held_active_mae": float(held.active_error_sum.sum() / held.n_active.sum()) if held.n_active.sum() else np.nan,
                    "parent_held_mae": float(reference.error_sum.sum() / reference.n.sum()),
                    "parent_held_active_mae": float(reference.active_error_sum.sum() / reference.n_active.sum()) if reference.n_active.sum() else np.nan})
                for row, ref in zip(held.itertuples(), reference.itertuples(), strict=True):
                    evaluated.append({**identity, "station": row.station, "eta": chosen["eta"],
                        "bandwidth_scale": chosen["bandwidth_scale"], "n": row.n, "n_active": row.n_active,
                        "error_sum": row.error_sum, "active_error_sum": row.active_error_sum,
                        "parent_error_sum": ref.error_sum, "parent_active_error_sum": ref.active_error_sum})
    stations = pd.DataFrame(evaluated)
    keys = ["split_seed", "seed", "pipeline", "mode", "k"]
    runs = stations.groupby(keys, as_index=False).agg(**{c: (c, "sum") for c in
        ("error_sum", "active_error_sum", "parent_error_sum", "parent_active_error_sum", "n", "n_active")},
        n_stations=("station", "nunique"))
    runs["mae"] = runs.error_sum / runs.n
    runs["active_mae"] = runs.active_error_sum / runs.n_active
    runs["parent_mae"] = runs.parent_error_sum / runs.n
    runs["parent_active_mae"] = runs.parent_active_error_sum / runs.n_active
    runs["delta_mae"] = runs.mae - runs.parent_mae
    runs["delta_active_mae"] = runs.active_mae - runs.parent_active_mae
    metrics = ("mae", "active_mae", "parent_mae", "parent_active_mae", "delta_mae", "delta_active_mae")
    parts = runs.groupby(["split_seed", "pipeline", "mode", "k"], as_index=False).agg(**{c: (c, "mean") for c in metrics},
        improved_seed_fits=("delta_mae", lambda v: int((v < 0).sum())), n_seed_fits=("seed", "nunique"), n=("n", "first"), n_active=("n_active", "first"))
    parts["direction"] = np.select([parts.delta_mae < 0, parts.delta_mae > 0], ["candidate_better", "parent_better"], default="equal")
    curves = parts.groupby(["pipeline", "mode", "k"], as_index=False).agg(**{c: (c, "mean") for c in metrics},
        improved_partitions=("delta_mae", lambda v: int((v < 0).sum())), n_partitions=("split_seed", "nunique"),
        improved_seed_fits=("improved_seed_fits", "sum"), n_seed_fits=("n_seed_fits", "sum"))
    curves["relative_gain_pct"] = -100 * curves.delta_mae / curves.parent_mae
    return folds, pd.DataFrame(choices), stations, runs, parts, curves


def cv_contrasts(cv_runs, cv_stations):
    """Summarize held-station choices with the same fixed comparison list."""
    rows, station_rows = [], []
    for definition in comparison_definitions():
        task = cv_runs[cv_runs.pipeline.eq(definition["pipeline"]) & cv_runs.k.eq(definition["k"])]
        mode = definition["candidate"].removeprefix(definition["pipeline"] + "_")
        reference_mode = definition["reference"].removeprefix(definition["pipeline"] + "_")
        a = task[task["mode"].eq(mode)].set_index(["split_seed", "seed"])
        if reference_mode == "parent":
            b = a.rename(columns={"mae": "candidate_unused", "active_mae": "active_unused", "parent_mae": "mae", "parent_active_mae": "active_mae"})
        else:
            b = task[task["mode"].eq(reference_mode)].set_index(["split_seed", "seed"]).loc[a.index]
        for region, field in (("overall", "mae"), ("active", "active_mae")):
            for index, candidate_value, reference_value in zip(a.index, a[field], b[field], strict=True):
                rows.append({**definition, "split_seed": index[0], "seed": index[1], "region": region,
                    "candidate_mae": candidate_value, "reference_mae": reference_value,
                    "delta_mae": candidate_value-reference_value})
        ss = cv_stations[cv_stations.pipeline.eq(definition["pipeline"]) & cv_stations.k.eq(definition["k"])]
        left = ss[ss["mode"].eq(mode)].set_index(["split_seed", "seed", "station"])
        right = left if reference_mode == "parent" else ss[ss["mode"].eq(reference_mode)].set_index(["split_seed", "seed", "station"]).loc[left.index]
        reference_error = right.parent_error_sum if reference_mode == "parent" else right.error_sum
        denominator = left.groupby(level=["split_seed", "seed"]).n.transform("sum")
        for index, gain in zip(left.index, (reference_error-left.error_sum)/denominator, strict=True):
            station_rows.append({"comparison": definition["comparison"], "split_seed": index[0], "seed": index[1],
                "station": index[2], "weighted_gain": gain})
    runs = pd.DataFrame(rows)
    keys = ["comparison", "pipeline", "candidate", "reference", "k", "region"]
    parts = runs.groupby([*keys, "split_seed"], as_index=False).agg(candidate_mae=("candidate_mae", "mean"),
        reference_mae=("reference_mae", "mean"), delta_mae=("delta_mae", "mean"),
        improved_seed_fits=("delta_mae", lambda v: int((v < 0).sum())), n_seed_fits=("seed", "nunique"))
    parts["direction"] = np.select([parts.delta_mae < 0, parts.delta_mae > 0], ["candidate_better", "reference_better"], default="equal")
    effects = parts.groupby(keys, as_index=False).agg(candidate_mae=("candidate_mae", "mean"),
        reference_mae=("reference_mae", "mean"), delta_mae=("delta_mae", "mean"),
        improved_partitions=("delta_mae", lambda v: int((v < 0).sum())), n_partitions=("split_seed", "nunique"),
        improved_seed_fits=("improved_seed_fits", "sum"), n_seed_fits=("n_seed_fits", "sum"))
    effects["relative_gain_pct"] = -100 * effects.delta_mae / effects.reference_mae
    weighted = pd.DataFrame(station_rows).groupby(["comparison", "split_seed", "station"], as_index=False).weighted_gain.mean()
    counts = parts[parts.region.eq("overall")].groupby("comparison").split_seed.nunique().to_dict()
    weighted["weighted_gain"] /= weighted.comparison.map(counts)
    global_stations = weighted.groupby(["comparison", "station"], as_index=False).weighted_gain.sum()
    concentration = []
    for name, group in global_stations.groupby("comparison"):
        pos, neg = group.weighted_gain.clip(lower=0), -group.weighted_gain.clip(upper=0)
        concentration.append({"comparison": name, "net_mae_reduction": group.weighted_gain.sum(),
            "stations_improved": int((group.weighted_gain > 0).sum()), "stations_worsened": int((group.weighted_gain < 0).sum()),
            "stations_equal": int((group.weighted_gain == 0).sum()), "n_stations_unique": group.station.nunique(),
            "positive_gain_mass": pos.sum(), "harm_mass": neg.sum(),
            "top5_positive_gain_share": pos.nlargest(5).sum()/pos.sum() if pos.sum() else np.nan,
            "top5_harm_share": neg.nlargest(5).sum()/neg.sum() if neg.sum() else np.nan})
    return runs, parts, effects, global_stations, pd.DataFrame(concentration)


def source_states(root, *, require_nine=True):
    """Load validation-only products and bind sources without target outcomes."""
    from river_graph.experiments.provenance import run_identity_sha256
    from river_graph.experiments.unified_spatial_protocol import support_query_cells

    sources, frames, candidate_frames, station_frames, selection_rows, distance_rows, thresholds = [], [], [], [], [], [], {}
    snapshot_path = root / "runtime_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    sources.append({"path": str(snapshot_path), "sha256": sha256(snapshot_path)})
    for name, value in snapshot.items():
        path = root / "code_snapshot" / name
        if sha256(path) != value:
            raise ValueError("Execution archive differs from its runtime snapshot")
        sources.append({"path": str(path), "sha256": value})
    runs = sorted((root / "runs").glob("split*_seed*"))
    if not runs or (require_nine and {run.name for run in runs} != {f"split{p}_seed{s}" for p in SPLITS for s in SEEDS}):
        raise ValueError("Expected completed source-only run packages are absent")
    for run in runs:
        completion_path = run / "complete.json"
        completion = json.loads(completion_path.read_text())
        sources.append({"path": str(completion_path), "sha256": sha256(completion_path)})
        config = read_bound_json(run, "config.json", completion, sources)
        identity = {"split_seed": config["split_seed"], "seed": config["seed"]}
        if (completion["config_hash"] != config_digest(config) or config["runtime_snapshot_hash"] != config_digest(snapshot)
                or config["experiment"] != "doc_chemical_kernel_v1" or tuple(config["models"]) != MODELS
                or tuple(config["pipelines"]) != PIPELINES or config["kernel_modes"] != ["masks", "chemistry"]
                or tuple(config["k_values"]) != KS or config["eta_values"] != [0, .25, .5, 1]
                or config["bandwidth_values"] != [.5, 1, 2] or config["max_source_months_per_station"] != 24
                or config["selection_role"] != "source_validation" or config["target_evaluation"] is not False
                or config["new_neural_fits"] != 0 or config["new_forest_fits"] != 0 or config["linear_calibrators_refitted"] is not False
                or run.name != f"split{identity['split_seed']}_seed{identity['seed']}"):
            raise ValueError("Changed source-only kernel protocol")
        prior = Path(config["prior_run"])
        for path, value in ((prior / "complete.json", config["prior_completion_hash"]),
                (Path(config["decoder_run"]) / "complete.json", config["decoder_completion_hash"]),
                (Path(config["dataset_path"]), config["dataset_hash"]), (Path(config["mask_path"]), config["mask_hash"])):
            if sha256(path) != value:
                raise ValueError("Changed frozen source artifact")
            sources.append({"path": str(path), "sha256": value})
        for name, value in config["parent_bindings"].items():
            if sha256(prior / name) != value:
                raise ValueError("Frozen parent component changed")
            sources.append({"path": str(prior / name), "sha256": value})
        old_config = json.loads((prior / "config.json").read_text())
        for field in ("split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train", "source_station_ids"):
            if config[field] != old_config[field]:
                raise ValueError(f"Parent/source identity differs: {field}")
        sources.append({"path": str(prior / "config.json"), "sha256": sha256(prior / "config.json")})
        distances = read_bound_json(run, "distances.json", completion, sources)
        if set(distances) != {"masks", "chemistry"}:
            raise ValueError("Missing source-only distance definition")
        for mode, state in distances.items():
            if (state["source_role"] != "source_training" or state["max_months_per_station"] != 24
                    or state["source_station_indices"] != config["source_station_ids"]
                    or not np.isfinite(state["distance_scale"]) or state["distance_scale"] <= 0
                    or state["fallback"] != (state["n_positive_pairs"] == 0)):
                raise ValueError("Invalid source-only bandwidth definition")
            distance_rows.append({**identity, "mode": mode, **{field: state[field] for field in
                ("distance_scale", "fallback", "n_source_stations", "n_active_source_stations", "n_sampled_months", "n_pairs", "n_positive_pairs")}})
        checks = read_bound_json(run, "linear_remainder_checks.json", completion, sources)
        if checks["label_scope"] != "split.val only" or len(checks["checks"]) != 12 or checks["parent_validation_groups_reproduced"] != 52:
            raise ValueError("Incomplete fixed-parent reconstruction checks")
        for row in checks["checks"]:
            if (row["query_candidate_max_abs_difference"] > 1e-12 or row["query_final_max_abs_difference"] > 1e-12
                    or row["support_preclip_used"] is not True or row["all_k_support_moments"] is not True):
                raise ValueError("Changed parent linear residual definition")
        selection = read_bound_json(run, "kernel_selection.json", completion, sources)
        if (selection["selection_role"] != "source_validation" or selection["score"] != "active_validation_native_mae"
                or selection["tie_rule"] != "smaller eta; bandwidth1; smaller bandwidth"
                or selection["eta_values"] != [0, .25, .5, 1] or selection["bandwidth_values"] != [.5, 1, 2]):
            raise ValueError("Changed kernel selection rule")
        candidates = pd.read_csv(bound_file(run, "all_candidate_scores.csv", completion, sources))
        if len(candidates) != 156:
            raise ValueError("Incomplete kernel selection grid")
        for pipeline in PIPELINES:
            for mode in ("masks", "chemistry"):
                for k in KS:
                    group = candidates[candidates.pipeline.eq(pipeline) & candidates.kernel_mode.eq(mode) & candidates.k.eq(k)]
                    expected = {(0, 1)} if k < 2 else {(eta, band) for eta in (0, .25, .5, 1) for band in (.5, 1, 2)}
                    if set(map(tuple, group[["eta", "bandwidth_scale"]].to_numpy())) != expected:
                        raise ValueError("Kernel candidate grid differs")
                    good = group[group.valid]
                    best = min(good.to_dict("records"), key=lambda row: (row["active_mae"], row["eta"], row["bandwidth_scale"] != 1, row["bandwidth_scale"]))
                    chosen = selection["pipelines"][pipeline][mode][str(k)]
                    for field in ("eta", "bandwidth_scale", "source_distance_scale", "mae", "active_mae", "n", "n_active"):
                        np.testing.assert_allclose(chosen[field], best[field], rtol=0, atol=1e-12)
                    if chosen["locked"] != (k < 2):
                        raise ValueError("K0/K1 selection must be locked")
                    selection_rows.append({**identity, "pipeline": pipeline, "mode": mode, "k": k,
                        **{field: chosen[field] for field in ("eta", "bandwidth_scale", "source_distance_scale", "mae", "active_mae", "locked")}})
        candidate_frames.append(candidates.assign(**identity))
        station_path = bound_file(run, "candidate_station_scores.csv", completion, sources)
        station_scores = pd.read_csv(station_path, dtype={"station": str}).rename(columns={"kernel_mode": "mode"})
        if "valid" in station_scores and not station_scores.valid.all():
            raise ValueError("A failed candidate needs explicit CV eligibility handling")
        station_frames.append(station_scores.assign(**identity))
        path = bound_file(run, "validation_predictions.parquet", completion, sources)
        meta = read_bound_json(run, "validation_predictions.meta.json", completion, sources)
        expected_meta = {"prediction_sha256": sha256(path), "config_hash": config_digest(config),
            "runtime_snapshot_hash": config["runtime_snapshot_hash"], "dataset_sha256": config["dataset_hash"],
            "mask_sha256": config["mask_hash"], "selection_role": "source_validation",
            "run_identity_sha256": run_identity_sha256(config_digest(config), config["started_at"], config["runtime_snapshot_hash"])}
        if any(meta.get(key) != value for key, value in expected_meta.items()) or config_digest(meta["config"]) != config_digest(config):
            raise ValueError("Source-validation prediction sidecar differs")
        for name, value in meta["model_files"].items():
            if sha256(run / name) != value or completion["files"].get(name) != value:
                raise ValueError("Changed source-validation model dependency")
        frame = pd.read_parquet(path)
        if (len(frame) != meta["rows"] or not frame.evaluation_role.eq("source_validation").all()
                or not frame.split_seed.eq(identity["split_seed"]).all() or not frame.seed.eq(identity["seed"]).all()):
            raise ValueError("Predictions are not exclusively source validation")
        with np.load(config["mask_path"], allow_pickle=False) as masks:
            if (not np.isin(frame.cell, masks["val"]).all() or np.intersect1d(frame.cell, masks["test"]).size):
                raise ValueError("Validation product includes nonvalidation cells")
            # Frozen feature archive determines timeline length without reading DOC labels.
            with np.load(prior / "representations.npz", allow_pickle=False) as reps:
                n_months = len(reps["active"]) // 357
            _, query = support_query_cells({name: masks[name] for name in masks.files}, target_role="val", k=0, n_months=n_months)
            np.testing.assert_array_equal(np.sort(frame.cell.unique()), query)
        threshold = float(config["q90_threshold_train"])
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid inherited source-training Q90")
        thresholds[(identity["split_seed"], identity["seed"])] = threshold
        metrics = pd.read_csv(bound_file(run, "source_validation.csv", completion, sources))
        old_metrics = pd.read_csv(prior / "source_validation.csv")
        if len(metrics) != 36 or metrics[["model_name", "k"]].duplicated().any():
            raise ValueError("Incomplete source-validation curve table")
        for (model, k), group in frame.groupby(["model_name", "k"]):
            row = metrics[metrics.model_name.eq(model) & metrics.k.eq(k)].iloc[0]
            error = np.abs(group.y_pred-group.y_true)
            np.testing.assert_allclose([row.mae, row.active_mae], [error.mean(), error[group.aux_available].mean()], rtol=0, atol=1e-12)
            if model.endswith("_parent"):
                old_name = model.removesuffix("_parent") + "_selected"
                old = old_metrics[old_metrics.model_name.eq(old_name) & old_metrics.k.eq(k)].iloc[0]
                np.testing.assert_allclose(row[["mae", "active_mae", "n", "n_active"]].to_numpy(dtype=float), old[["mae", "active_mae", "n", "n_active"]].to_numpy(dtype=float), rtol=0, atol=1e-12)
            else:
                pipe, mode = group.pipeline.iloc[0], group.kernel_mode.iloc[0]
                chosen = selection["pipelines"][pipe][mode][str(k)]
                np.testing.assert_allclose([row.mae, row.active_mae], [chosen["mae"], chosen["active_mae"]], rtol=0, atol=1e-12)
                if not group.eta.eq(chosen["eta"]).all() or not group.bandwidth_scale.eq(chosen["bandwidth_scale"]).all():
                    raise ValueError("Validation output differs from selected kernel parameters")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, require_nine=require_nine)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    station_candidates = pd.concat(station_frames, ignore_index=True)
    totals = station_candidates.groupby(["split_seed", "seed", "pipeline", "mode", "k", "eta", "bandwidth_scale"], as_index=False).agg(
        error_sum=("error_sum", "sum"), active_error_sum=("active_error_sum", "sum"), n=("n", "sum"), n_active=("n_active", "sum"))
    for row in totals.itertuples():
        expected = candidates[candidates.split_seed.eq(row.split_seed) & candidates.seed.eq(row.seed)
            & candidates.pipeline.eq(row.pipeline) & candidates.kernel_mode.eq(row.mode) & candidates.k.eq(row.k)
            & candidates.eta.eq(row.eta) & candidates.bandwidth_scale.eq(row.bandwidth_scale)].iloc[0]
        np.testing.assert_allclose([row.error_sum/row.n, row.active_error_sum/row.n_active, row.n, row.n_active],
                                  [expected.mae, expected.active_mae, expected.n, expected.n_active], rtol=0, atol=1e-12)
    return panel, thresholds, sources, candidates, station_candidates, pd.DataFrame(selection_rows), pd.DataFrame(distance_rows)


def write_report(out, curves, cv_effects, effects, selections, cv_choices, corrections, distances, population):
    lines = ["# Bounded chemical support interpolation: source-validation pilot", "",
        "This report uses source-validation observations only. No outer station-test predictions or labels are read.",
        "The native experts, selected linear support representations, calibrators and ecological mixtures remain frozen.",
        "The added operator transfers the centered remainder left after the complete linear support correction.",
        "Its RBF distance unit is fitted from source-station chemical coordinates, independently of DOC labels.", "",
        "## Incremental station cross-validation", "",
        "Five station folds select only eta and bandwidth on the other four source-validation folds, then score the held fold.",
        "A shared RNG(3100 + partition) station assignment is used across seeds, pipelines and coordinate modes.",
        "The parent was previously selected using source validation: this conditional CV reduces optimism for the added",
        "kernel but is not an out-of-fold evaluation of the complete model. The full-validation optimum below is tuning",
        "evidence; its eta0 fallback guarantees that it cannot be worse than the parent on its selection score.",
        "No confidence intervals or independent-confirmation claims are attached to this development pilot.", "",
        "| Pipeline | K | Comparison | CV candidate MAE | CV reference MAE | ΔMAE | Relative gain | Better partitions | Better seed fits |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in cv_effects[cv_effects.region.eq("overall")].itertuples():
        candidate = row.candidate.removeprefix(row.pipeline + "_")
        reference = row.reference.removeprefix(row.pipeline + "_")
        lines.append(f"| {row.pipeline} | {row.k} | {candidate} vs {reference} | {row.candidate_mae:.6f} | {row.reference_mae:.6f} | "
                     f"{row.delta_mae:+.6f} | {row.relative_gain_pct:+.3f}% | {row.improved_partitions}/{row.n_partitions} | {row.improved_seed_fits}/{row.n_seed_fits} |")
    lines += ["", "Metrics pool held-query cells within each fitted seed, average seeds within partition, then weight partitions equally.",
        "Full and auxiliary-active CV scores are both saved. The full score includes the unchanged inactive fallback.", "",
        "## Complete full-validation fitted curves", "",
        "These scores are evaluated on the same validation observations used to choose the kernel parameters.", "",
        "| Model | K | Full MAE | Active MAE | Q90 MAE | Ordinary MAE | Signed bias |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in curves.itertuples():
        lines.append(f"| {row.model_name} | {row.k} | {row.mae:.6f} | {row.active_mae:.6f} | {row.q90_mae:.6f} | {row.ordinary_mae:.6f} | {row.signed_bias:+.6f} |")
    lines += ["", "## Full-validation kernel settings", "",
        "| Pipeline | Mode | K | eta0 / .25 / .5 / 1 | bandwidth .5 / 1 / 2 |",
        "|---|---|---:|---|---|"]
    for (pipe, mode, k), group in selections.groupby(["pipeline", "mode", "k"], sort=False):
        etas = "/".join(str(int(group.eta.eq(v).sum())) for v in (0, .25, .5, 1))
        bands = "/".join(str(int(group.bandwidth_scale.eq(v).sum())) for v in (.5, 1, 2))
        lines.append(f"| {pipe} | {mode} | {k} | {etas} | {bands} |")
    lines += ["", "The separate CV-choice table records every held-fold selection and its training/held error.",
        "K0 and K1 are exact parent controls. Fewer than two chemically active support rows, identical donor coordinates,",
        "inactive queries or eta0 yield exactly zero kernel correction. Availability-only coordinates control the extra",
        "kernel information; the already-frozen parent can itself use measured chemistry.", "",
        "## Population and transfer diagnostics", "",
        f"Validation queries contain {population['unique_station_months']:,} unique station-months at {population['unique_stations']} unique stations;",
        f"the three partitions contribute {population['partition_cell_occurrences']:,} station-month occurrences. Seeds repeat the same ecological observations.",
        "Source-derived Q90 thresholds include ties. Tail groups below20 observations are marked unstable.",
        "Correction summaries distinguish all-query and chemically active populations; maxima are diagnostics, not averages.",
        "Station gain/harm contribution tables use exactly the partition-equal estimator and are included for both",
        "the full-validation optimum and the incremental CV, so a few influential stations remain visible.", "",
        "## Scientific scope", "",
        "The decision about further testing should rely on coherent incremental-CV behavior relative to the frozen parent",
        "and the availability-only kernel. A gain at the full-validation optimum alone cannot establish useful transfer.",
        "Neither chemical similarity nor RBF weight is interpreted as physical transport or a causal coefficient.",
        "A later frozen fresh-partition evaluation would be required for confirmation. This pilot never opens that evaluation.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true", help="Check a smoke package without writing or interpreting pilot results")
    args = parser.parse_args()
    panel, thresholds, sources, candidates, candidate_stations, choices, distances = source_states(args.root, require_nine=not args.check_only)
    folds, cv_choices, cv_stations, cv_runs, cv_parts, cv_curves = incremental_station_cv(candidate_stations)
    if args.check_only:
        print(f"Source-only checks passed: {len(panel[['split_seed', 'seed']].drop_duplicates())} packages, "
              f"{len(choices)} kernel choices, {len(cv_choices)} station-CV choices. No target outcome read and no report written.")
        return
    runs, parts, curves = summarize(panel, thresholds)
    effects, directions, partition_directions, station_gains, global_gains, concentration = compare(panel, thresholds)
    cv_directions, cv_partition_directions, cv_effects, cv_global_gains, cv_concentration = cv_contrasts(cv_runs, cv_stations)
    corrections = correction_diagnostics(panel)
    selection_counts = choices.groupby(["pipeline", "mode", "k", "eta", "bandwidth_scale"], as_index=False).size()
    cv_choice_counts = cv_choices.groupby(["pipeline", "mode", "k", "eta", "bandwidth_scale"], as_index=False).size()
    population = {"unique_station_months": panel.cell.nunique(), "unique_stations": panel.station.nunique(),
                  "partition_cell_occurrences": len(panel.drop_duplicates(["split_seed", "cell"]))}
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables = (("full_validation_metrics_by_run", runs), ("full_validation_metrics_by_partition", parts),
        ("full_validation_k_curves", curves), ("full_validation_comparisons", effects),
        ("full_validation_directions_by_seed", directions), ("full_validation_directions_by_partition", partition_directions),
        ("full_validation_station_gains", station_gains), ("full_validation_global_station_gains", global_gains),
        ("full_validation_gain_loss_concentration", concentration), ("full_validation_choices", choices),
        ("full_validation_choice_counts", selection_counts), ("full_validation_candidate_scores", candidates),
        ("correction_diagnostics", corrections), ("source_distance_diagnostics", distances),
        ("incremental_cv_choices", cv_choices), ("incremental_cv_choice_counts", cv_choice_counts),
        ("incremental_cv_station_scores", cv_stations), ("incremental_cv_metrics_by_run", cv_runs),
        ("incremental_cv_metrics_by_partition", cv_parts), ("incremental_cv_k_curves", cv_curves),
        ("incremental_cv_directions_by_seed", cv_directions), ("incremental_cv_directions_by_partition", cv_partition_directions),
        ("incremental_cv_comparisons", cv_effects), ("incremental_cv_global_station_gains", cv_global_gains),
        ("incremental_cv_gain_loss_concentration", cv_concentration))
    for name, table in tables:
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "incremental_cv_folds.json").write_text(json.dumps({"selection_role": "source_validation_incremental_station_cv",
        "parent_out_of_fold": False, "number_of_folds": 5, "folds": folds}, indent=2) + "\n")
    (out / "population.json").write_text(json.dumps(population, indent=2) + "\n")
    write_report(out, curves, cv_effects, effects, choices, cv_choices, corrections, distances, population)
    hashes = {f"{name}.csv": sha256(out / f"{name}.csv") for name, _ in tables}
    for name in ("findings.md", "incremental_cv_folds.json", "population.json"):
        hashes[name] = sha256(out / name)
    for name in ("src/river_graph/experiments/provenance.py", "src/river_graph/experiments/unified_spatial_protocol.py"):
        sources.append({"path": name, "sha256": sha256(name)})
    (out / "analysis_manifest.json").write_text(json.dumps({"analysis_script": str(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)), "sources": sources, "outputs": hashes,
        "source_only": True, "target_outcomes_read": False, "parent_out_of_fold": False,
        "comparison_count": len(comparison_definitions()), "bootstrap_draws": 0,
        "estimand": "cell pooled within each seed; mean seeds within partition; equal mean of partitions",
        "role": "incremental station CV conditional on a source-validation-selected frozen parent; calibration development only"}, indent=2) + "\n")
    print(cv_effects[cv_effects.region.eq("overall")][["comparison", "candidate_mae", "reference_mae", "delta_mae", "improved_partitions"]].to_string(index=False))
    print(f"Saved source-only conditional station-CV diagnostics to {out}")


if __name__ == "__main__":
    main()
