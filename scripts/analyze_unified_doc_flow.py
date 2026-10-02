"""Describe station-adaptation benefit by discharge variability and record availability.

Uses saved station responses and raw monthly hydro inputs only. No refitting,
selection of model variants, or outcome-derived strata are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.dataset import FEATURE_CHANNELS

DEFAULT_ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation")
COMPARISONS = ("hybrid_vs_et_k5_calibrated", "hybrid_calibration_k5_vs_k0")
STRATA = ("lower", "middle", "higher", "insufficient")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flow_descriptors(dataset: dict) -> pd.DataFrame:
    """Population CV of available monthly discharge, with missing months excluded."""
    channels = list(dataset.get("feature_channels", []))
    if channels != FEATURE_CHANNELS or channels[1] != "discharge":
        raise ValueError("Dataset channel order must be temperature, discharge")
    hydro = np.asarray(dataset["x"], dtype=float)
    observed = np.asarray(dataset["x_mask"], dtype=bool)
    if hydro.shape != observed.shape or hydro.shape[2] != 2:
        raise ValueError("Hydro values and masks must share station-month-channel shape")
    if not np.isfinite(hydro[..., 1][observed[..., 1]]).all():
        raise ValueError("Observed discharge contains nonfinite values")
    rows = []
    for station_index, station in enumerate(dataset["site_no"]):
        values = hydro[station_index, :, 1][observed[station_index, :, 1]]
        count = len(values)
        mean = float(values.mean()) if count else np.nan
        sd = float(values.std(ddof=0)) if count else np.nan
        enough = count >= 12 and mean > 0
        rows.append({
            "station": str(station), "station_index": station_index,
            "hydro_count": count, "flow_mean": mean, "flow_sd": sd,
            "negative_flow_months": int((values < 0).sum()),
            "flow_cv": sd / mean if enough else np.nan,
            "flow_status": ("sufficient" if enough else
                            "fewer_than_12_months" if count < 12 else "nonpositive_mean"),
            "doc_observed_months": int(np.asarray(dataset["y_mask"])[station_index].sum()),
            "doc_record_coverage": float(np.asarray(dataset["y_mask"])[station_index].mean()),
        })
    frame = pd.DataFrame(rows)
    if frame.station.duplicated().any():
        raise ValueError("Station identifiers are not unique")
    return frame


def analyze(root: Path, *, allow_partial: bool = False) -> Path:
    analysis = root / "analysis"
    if allow_partial and not (analysis / "status.json").exists():
        analysis = root / "analysis_partial"
    status_path = analysis / "status.json"
    station_path = analysis / "station_heterogeneity.csv"
    status = json.loads(status_path.read_text())
    complete = (status.get("complete") is True and status.get("confirmation_budget") is True
                and status.get("available_runs") == 9 and status.get("expected_runs") == 9)
    if not complete and not allow_partial:
        raise ValueError("Flow analysis requires all nine confirmation runs; use --allow-partial for checks")
    station = pd.read_csv(station_path, dtype={"station": str})
    station = station[station.comparison.isin(COMPARISONS)].copy()
    if station.empty or set(station.comparison) != set(COMPARISONS):
        raise ValueError("Both planned station-level comparisons must be present")
    keys = ["comparison", "split_seed", "station"]
    if station.duplicated(keys).any():
        raise ValueError("Duplicate station response rows")
    required = ["candidate_mae", "reference_mae", "n_query_cells"]
    if not np.isfinite(station[required].to_numpy()).all() or (station.n_query_cells < 1).any():
        raise ValueError("Station errors and fixed-query counts must be finite and nonempty")
    sources = {str(status_path): sha256(status_path), str(station_path): sha256(station_path)}
    thresholds, enriched, stratum_rows = [], [], []
    descriptions = None
    dataset_hash = None
    for split_seed, part in station.groupby("split_seed"):
        seeds = status["seeds_per_split"].get(str(split_seed), [])
        if not seeds:
            raise ValueError("Station responses have no corresponding completed training seeds")
        config_path = root / "runs" / f"split{split_seed}_seed{seeds[0]}" / "config.json"
        config = json.loads(config_path.read_text())
        dataset_path, mask_path = Path(config["dataset_path"]), Path(config["mask_path"])
        for name, path in (("dataset", dataset_path), ("mask", mask_path)):
            content_hash = sha256(path)
            if content_hash != config[f"{name}_hash"]:
                raise ValueError(f"Changed {name} input: {path}")
            sources[str(path)] = content_hash
        sources[str(config_path)] = sha256(config_path)
        if descriptions is None:
            dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
            descriptions = flow_descriptors(dataset)
            dataset_hash = config["dataset_hash"]
            n_months = dataset["y"].shape[1]
        elif config["dataset_hash"] != dataset_hash:
            raise ValueError("All station partitions must use the same dataset")
        with np.load(mask_path, allow_pickle=False) as mask:
            train_stations = np.unique(mask["train"] // n_months)
            test_cells = mask["test"].copy()
        expected_counts = pd.Series(test_cells // n_months).value_counts().sort_index() - 5
        expected = descriptions.set_index("station_index").loc[expected_counts.index, "station"]
        expected = pd.Series(expected_counts.to_numpy(), index=expected.to_numpy())
        for _, comparison in part.groupby("comparison"):
            got = comparison.set_index("station").n_query_cells.sort_index()
            pd.testing.assert_series_equal(got, expected.sort_index(), check_names=False,
                                           check_dtype=False, check_exact=True)
        source_cv = descriptions.loc[descriptions.station_index.isin(train_stations), "flow_cv"].dropna()
        q1, q2 = source_cv.quantile([1 / 3, 2 / 3]).to_numpy() if len(source_cv) else (np.nan, np.nan)
        source_record = descriptions.loc[descriptions.station_index.isin(train_stations), "doc_observed_months"]
        record_q1, record_q2 = source_record.quantile([1 / 3, 2 / 3]).to_numpy()
        thresholds.append({
            "split_seed": int(split_seed), "training_stations": len(train_stations),
            "training_stations_with_sufficient_flow": len(source_cv),
            "training_cv_q33": q1, "training_cv_q67": q2,
            "bins_degenerate": bool(q1 == q2) if len(source_cv) else True,
            "training_doc_count_q33": record_q1, "training_doc_count_q67": record_q2,
            "record_bins_degenerate": bool(record_q1 == record_q2),
        })
        merged = part.merge(descriptions, on="station", how="left", validate="many_to_one")
        if merged.station_index.isna().any():
            raise ValueError("Station response has no hydro descriptor")
        merged["flow_stratum"] = "insufficient"
        if len(source_cv):
            valid = merged.flow_cv.notna()
            merged.loc[valid, "flow_stratum"] = np.select(
                [merged.loc[valid, "flow_cv"] <= q1, merged.loc[valid, "flow_cv"] <= q2],
                ["lower", "middle"], default="higher")
        merged["training_cv_q33"], merged["training_cv_q67"] = q1, q2
        merged["record_availability_stratum"] = np.select(
            [merged.doc_observed_months <= record_q1, merged.doc_observed_months <= record_q2],
            ["lower", "middle"], default="higher")
        merged["training_doc_count_q33"], merged["training_doc_count_q67"] = record_q1, record_q2
        enriched.append(merged)
        for comparison in COMPARISONS:
            subset = merged[merged.comparison.eq(comparison)]
            for variable, column, levels in (
                ("flow_variability", "flow_stratum", STRATA),
                ("record_availability", "record_availability_stratum", STRATA[:3]),
            ):
                for level in levels:
                    group = subset[subset[column].eq(level)]
                    count = int(group.n_query_cells.sum())
                    reference = float(np.average(group.reference_mae, weights=group.n_query_cells)) if count else np.nan
                    candidate = float(np.average(group.candidate_mae, weights=group.n_query_cells)) if count else np.nan
                    stratum_rows.append({
                        "comparison": comparison, "split_seed": int(split_seed),
                        "variable": variable, "stratum": level,
                        "reference_mae": reference, "candidate_mae": candidate,
                        "delta_mae": candidate - reference,
                        "relative_gain_pct": 100 * (1 - candidate / reference) if reference > 0 else np.nan,
                        "n_query_cells": count, "n_stations": len(group), "n_training_seeds": len(seeds),
                    })
    station_rows = pd.concat(enriched, ignore_index=True)
    by_split = pd.DataFrame(stratum_rows)
    summary = []
    for (comparison, variable, level), group in by_split.groupby(["comparison", "variable", "stratum"]):
        nonempty = group[group.n_query_cells.gt(0)]
        reference, candidate = nonempty.reference_mae.mean(), nonempty.candidate_mae.mean()
        column = "flow_stratum" if variable == "flow_variability" else "record_availability_stratum"
        identities = station_rows[station_rows.comparison.eq(comparison) & station_rows[column].eq(level)]
        summary.append({
            "comparison": comparison, "variable": variable, "stratum": level,
            "reference_mae": reference, "candidate_mae": candidate,
            "delta_mae": candidate - reference,
            "relative_gain_pct": 100 * (1 - candidate / reference) if reference > 0 else np.nan,
            "n_splits": len(nonempty), "n_split_cell_occurrences": int(nonempty.n_query_cells.sum()),
            "n_station_split_occurrences": int(nonempty.n_stations.sum()),
            "n_stations_unique": identities.station.nunique(),
        })
    summary = pd.DataFrame(summary)
    out = root / ("flow_analysis" if complete else "flow_analysis_partial")
    out.mkdir(parents=True, exist_ok=True)
    descriptions.to_csv(out / "station_covariate_descriptors.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "training_stratum_thresholds.csv", index=False)
    station_rows.to_csv(out / "station_covariate_responses.csv", index=False)
    by_split.to_csv(out / "strata_by_split.csv", index=False)
    summary.to_csv(out / "strata_summary.csv", index=False)
    lines = ["# DOC reconstruction benefit, hydrology and record availability", "",
             "Descriptive analysis of saved predictions; no model is refitted.", ""]
    if not complete:
        lines += ["**Implementation/interim check only:** the full nine-fit confirmation batch is incomplete.", ""]
    lines += [
        "Monthly discharge is input channel 1, verified against dataset metadata and `dataset.FEATURE_CHANNELS`.",
        "The descriptor is population standard deviation divided by mean discharge over observed hydro months.",
        "It is invariant to multiplicative flow-unit conversion. A station needs at least 12 observed months",
        "and positive mean flow; other stations remain in an explicit insufficient-data group.",
        "Signed discharge is retained: the source quality rules allow reverse flow in tidal/backwater reaches.",
        "Each partition's training-station CV tertiles define lower, middle and higher variability.",
        "No DOC labels or prediction errors determine these strata. The full historical hydro record is used",
        "as a station descriptor, consistently with retrospective reconstruction; this is not event timing.", "",
        "Record availability is the total number of observed DOC months from y_mask, before reserving support.",
        "Training-station observation-count tertiles define lower, middle and higher record availability.",
        "This descriptor uses availability indicators only, never concentration values. The model still",
        "hides all target-station DOC from base inputs and gives every station exactly the same K support",
        "budget. Lower record availability therefore describes the monitored record, not fewer input",
        "support labels or a directly observed target temporal history.", "",
        "Station responses first average the existing seed losses. Stratum MAEs weight each station by",
        "its fixed query count within a partition, then average nonempty partitions equally. Empty strata",
        "retain zero counts and undefined error, rather than being assigned zero error. Repeated stations",
        "are counted once per partition and their unique count is also reported. Associations are descriptive;",
        "they do not establish that flow variability causes the model's gain.", "",
        "| Comparison | Variable | Stratum | Reference MAE | Candidate MAE | Reduction (%) | Partitions | Station cases / unique | Query cases |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.comparison} | {row.variable} | {row.stratum} | {row.reference_mae:.4f} | "
                     f"{row.candidate_mae:.4f} | {row.relative_gain_pct:.2f} | {row.n_splits} | "
                     f"{row.n_station_split_occurrences} / {row.n_stations_unique} | {row.n_split_cell_occurrences} |")
    (out / "findings.md").write_text("\n".join(lines) + "\n")
    (out / "sources.json").write_text(json.dumps({
        "complete_confirmation": complete, "available_runs": status["available_runs"],
        "sources": sources, "analysis_script_sha256": sha256(Path(__file__)),
        "flow_channel": 1, "flow_cv_ddof": 0, "minimum_observed_hydro_months": 12,
        "record_availability": "total y_mask observed months; concentration values unused",
        "threshold_population": "training stations within each partition",
        "estimand": "seed-mean station losses; query-cell weighting within partition; partition equal",
    }, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(analyze(args.root, allow_partial=args.allow_partial))


if __name__ == "__main__":
    main()
