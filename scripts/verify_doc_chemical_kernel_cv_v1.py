"""Independently reproduce conditional kernel CV from source-validation sums.

This checks only the new kernel hyperparameter selection. The fixed parent was
already selected on all source validation stations, so it is not a whole-model
out-of-fold assessment. No dataset labels or target prediction files are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments/phase4_transfer/doc_chemical_kernel_v1")
SPLITS, SEEDS = (142, 143, 144), (42, 43, 44)
PIPELINES = ("neural_chemistry", "neural_chemistry_integrated", "tree_chemistry")
MODES, KS = ("masks", "chemistry"), (3, 5)
GRID = {(eta, band) for eta in (0., .25, .5, 1.) for band in (.5, 1., 2.)}
ATOL, RTOL = 1e-14, 1e-13


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def selected_row(frame, identity):
    keep = np.ones(len(frame), dtype=bool)
    for key, value in identity.items():
        keep &= frame[key].eq(value).to_numpy()
    result = frame.loc[keep]
    if len(result) != 1:
        raise ValueError(f"Expected exactly one row for {identity}")
    return result.iloc[0]


def verify(root):
    analysis = root / "analysis"
    sources = {}

    def bind(path):
        sources[str(path)] = sha256(path)
        return path

    report_path = bind(root / "verification/replay_checks.json")
    numerical = json.loads(report_path.read_text())
    if numerical["verified_runs"] != 9 or not numerical["all_nine_replayed"]:
        raise ValueError("A completed nine-package numerical replay is required")
    if numerical["target_labels_extracted"] or numerical["target_outcomes_evaluated"]:
        raise ValueError("The numerical replay is not source-validation-only")
    replay_runs = {row["run"]: row for row in numerical["results"]}
    bind(Path("scripts/analyze_doc_chemical_kernel_v1.py"))
    choices = pd.read_csv(bind(analysis / "incremental_cv_choices.csv"))
    stations = pd.read_csv(bind(analysis / "incremental_cv_station_scores.csv"),
                           dtype={"station": str})
    saved_runs = pd.read_csv(bind(analysis / "incremental_cv_metrics_by_run.csv"))
    effects = pd.read_csv(bind(analysis / "incremental_cv_comparisons.csv"))
    fold_definition = json.loads(bind(analysis / "incremental_cv_folds.json").read_text())
    if (fold_definition["parent_out_of_fold"]
            or fold_definition["number_of_folds"] != 5
            or fold_definition["selection_role"] != "source_validation_incremental_station_cv"):
        raise ValueError("Incorrect conditional CV scope")
    expected_tasks = {(pipe, mode, k) for pipe in PIPELINES for mode in MODES for k in KS}
    identities = ["split_seed", "seed", "pipeline", "mode", "k"]
    if len(choices) != 540 or choices.duplicated([*identities, "fold"]).any():
        raise ValueError("Expected 540 unique fold choices")
    if stations.duplicated([*identities, "station"]).any():
        raise ValueError("A validation station was evaluated more than once")
    run_records, station_records = [], []
    max_choice_difference = 0.
    checked_choices = 0
    for split in SPLITS:
        expected_folds = None
        for seed in SEEDS:
            name = f"split{split}_seed{seed}"
            run = root / "runs" / name
            completion_path = bind(run / "complete.json")
            if sha256(completion_path) != replay_runs[name]["completion_sha256"]:
                raise ValueError(f"Numerical replay no longer binds {name}")
            completion = json.loads(completion_path.read_text())
            candidate_path = bind(run / "candidate_station_scores.csv")
            if sha256(candidate_path) != completion["files"][candidate_path.name]:
                raise ValueError(f"Changed candidate station table: {name}")
            scores = pd.read_csv(candidate_path, dtype={"station": str}).rename(
                columns={"kernel_mode": "mode"})
            if not scores.split_seed.eq(split).all() or not scores.seed.eq(seed).all():
                raise ValueError("Run identity mismatch")
            if set(map(tuple, scores[["pipeline", "mode", "k"]].to_numpy())) != expected_tasks:
                raise ValueError("Incomplete station candidate task panel")
            station_ids = np.asarray(sorted(scores.station.unique()))
            groups = np.array_split(np.random.default_rng(3100 + split).permutation(station_ids), 5)
            assignment = {station: fold for fold, group in enumerate(groups) for station in group}
            metadata = fold_definition["folds"][str(split)]
            if (metadata["station_to_fold"] != assignment or metadata["seed"] != 3100 + split
                    or metadata["fold_station_counts"] != [len(group) for group in groups]):
                raise ValueError("Incorrect validation-station fold assignment")
            if expected_folds is not None and assignment != expected_folds:
                raise ValueError("Fold assignment differs across model seeds")
            expected_folds = assignment
            for (pipe, mode, k), task in scores.groupby(["pipeline", "mode", "k"]):
                identity = {"split_seed": split, "seed": seed, "pipeline": pipe, "mode": mode, "k": int(k)}
                if set(map(tuple, task[["eta", "bandwidth_scale"]].to_numpy())) != GRID:
                    raise ValueError("Incomplete kernel grid")
                for _, candidate in task.groupby(["eta", "bandwidth_scale"]):
                    if len(candidate) != len(station_ids) or set(candidate.station) != set(station_ids):
                        raise ValueError("Candidate station populations differ")
                base = task[(task.eta == 0) & (task.bandwidth_scale == 1)].set_index("station")
                held_all = []
                for fold, held_ids in enumerate(groups):
                    train = task[~task.station.isin(held_ids)]
                    ranked = []
                    for (eta, band), candidate in train.groupby(["eta", "bandwidth_scale"]):
                        error = float(np.sum(candidate.active_error_sum.to_numpy()))
                        n_active = int(np.sum(candidate.n_active.to_numpy()))
                        if n_active <= 0:
                            raise ValueError("Empty active training denominator")
                        ranked.append((error / n_active, eta, band != 1, band, error, n_active))
                    best = min(ranked)
                    saved = selected_row(choices, {**identity, "fold": fold})
                    if (saved.eta != best[1] or saved.bandwidth_scale != best[3]
                            or saved.selection_role != "other_four_source_validation_station_folds"):
                        raise ValueError("Wrong other-four-fold choice or tie order")
                    close([saved.active_mae, saved.active_error_sum, saved.n_active],
                          [best[0], best[4], best[5]])
                    max_choice_difference = max(max_choice_difference, abs(saved.active_mae - best[0]))
                    held = task[task.station.isin(held_ids) & (task.eta == best[1])
                                & (task.bandwidth_scale == best[3])].sort_values("station").copy()
                    ref = base.loc[held.station]
                    close([saved.n_training_stations, saved.n_held_stations, saved.held_n,
                           saved.held_n_active, saved.held_mae, saved.held_active_mae,
                           saved.parent_held_mae, saved.parent_held_active_mae],
                          [len(station_ids) - len(held_ids), len(held_ids), held.n.sum(),
                           held.n_active.sum(), held.error_sum.sum() / held.n.sum(),
                           held.active_error_sum.sum() / held.n_active.sum(),
                           ref.error_sum.sum() / ref.n.sum(), ref.active_error_sum.sum() / ref.n_active.sum()])
                    for row, parent in zip(held.itertuples(), ref.itertuples(), strict=True):
                        station_records.append({**identity, "fold": fold, "station": row.station,
                            "eta": best[1], "bandwidth_scale": best[3], "n": row.n,
                            "n_active": row.n_active, "error_sum": row.error_sum,
                            "active_error_sum": row.active_error_sum,
                            "parent_error_sum": parent.error_sum,
                            "parent_active_error_sum": parent.active_error_sum})
                    held_all.append(held)
                    checked_choices += 1
                joined = pd.concat(held_all)
                if len(joined) != len(station_ids) or joined.station.nunique() != len(station_ids):
                    raise ValueError("Not every validation station was held exactly once")
                reference = base.loc[joined.station]
                record = {**identity, "n_stations": len(station_ids)}
                for prefix, denominator in (("", "n"), ("active_", "n_active")):
                    record[denominator] = joined[denominator].sum()
                    record[prefix + "error_sum"] = joined[prefix + "error_sum"].sum()
                    record["parent_" + prefix + "error_sum"] = reference[prefix + "error_sum"].sum()
                    record[prefix + "mae"] = record[prefix + "error_sum"] / record[denominator]
                    record["parent_" + prefix + "mae"] = record["parent_" + prefix + "error_sum"] / record[denominator]
                    record["delta_" + prefix + "mae"] = record[prefix + "mae"] - record["parent_" + prefix + "mae"]
                saved = selected_row(saved_runs, identity)
                for field, value in record.items():
                    if field not in identity:
                        close(saved[field], value)
                run_records.append(record)
    expected_stations = pd.DataFrame(station_records)
    order = [*identities, "station"]
    pd.testing.assert_frame_equal(
        expected_stations.sort_values(order).reset_index(drop=True)[stations.columns],
        stations.sort_values(order).reset_index(drop=True), check_dtype=False, check_exact=True)
    runs = pd.DataFrame(run_records)
    expected_contrasts = {(pipe, k, candidate, reference, region) for pipe in PIPELINES for k in KS
                          for candidate, reference in (("masks", "parent"), ("chemistry", "parent"),
                                                       ("chemistry", "masks"))
                          for region in ("overall", "active")}
    actual_contrasts = {(r.pipeline, r.k, r.candidate.removeprefix(r.pipeline + "_"),
                        r.reference.removeprefix(r.pipeline + "_"), r.region) for r in effects.itertuples()}
    if len(effects) != 36 or actual_contrasts != expected_contrasts:
        raise ValueError("Incomplete contrast panel")
    max_contrast_difference = 0.
    for row in effects.itertuples():
        task = runs[(runs.pipeline == row.pipeline) & (runs.k == row.k)]
        candidate = row.candidate.removeprefix(row.pipeline + "_")
        reference = row.reference.removeprefix(row.pipeline + "_")
        a = task[task["mode"] == candidate].set_index(["split_seed", "seed"])
        field = "active_mae" if row.region == "active" else "mae"
        av = a[field]
        bv = (a["parent_" + field] if reference == "parent" else
              task[task["mode"] == reference].set_index(["split_seed", "seed"])[field].loc[a.index])
        delta = av - bv
        partition_delta = delta.groupby(level=0).mean()
        expected = [av.groupby(level=0).mean().mean(), bv.groupby(level=0).mean().mean(),
                    partition_delta.mean()]
        actual = [row.candidate_mae, row.reference_mae, row.delta_mae]
        close(actual, expected)
        max_contrast_difference = max(max_contrast_difference, float(np.max(np.abs(np.subtract(actual, expected)))))
        if (row.improved_partitions != int((partition_delta < 0).sum())
                or row.improved_seed_fits != int((delta < 0).sum())
                or row.n_partitions != 3 or row.n_seed_fits != 9):
            raise ValueError(f"Direction counts or partition weighting differ: {row.comparison}/{row.region}")
        close(row.relative_gain_pct, -100 * expected[2] / expected[1])
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "verified",
        "verifier_sha256": sha256(Path(__file__)), "verified_packages": 9,
        "conditional_cv_choices": checked_choices, "held_station_rows": len(stations),
        "run_metrics": len(runs), "overall_and_active_contrasts": len(effects),
        "station_rows_bitwise_exact": True, "numeric_rtol": RTOL, "numeric_atol": ATOL,
        "max_choice_mae_difference": max_choice_difference,
        "max_contrast_mae_difference": max_contrast_difference,
        "target_labels_read": False, "target_outcomes_read": False, "new_model_fits": 0,
        "scope": "Kernel-only conditional station CV; fixed parent selected on all source validation.",
        "aggregation": "Sum held-station errors within run; average seeds, then equal partitions.",
        "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = verify(args.root)
    output = args.root / "verification/conditional_cv_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {report['conditional_cv_choices']} fold choices and "
          f"{report['overall_and_active_contrasts']} contrasts: {output}")


if __name__ == "__main__":
    main()
