"""Describe source-only DOC residual synchrony before fitting a transfer model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_joint_source_states_v1 import ROOT as INPUTS
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_source_synchrony_v1")


def source_residuals(truth, oof_log, train):
    """Read labels and OOF values only at explicitly allowed source cells."""
    truth, oof_log, train = np.asarray(truth), np.asarray(oof_log), np.asarray(train)
    if (truth.ndim != 2 or oof_log.shape != truth.shape or train.ndim != 1
            or train.dtype.kind not in "iu" or not len(train)
            or (train < 0).any() or (train >= truth.size).any()
            or len(np.unique(train)) != len(train)):
        raise ValueError("aligned grids and unique valid source cell IDs are required")
    y, base = truth.ravel()[train], oof_log.ravel()[train]
    if not np.isfinite(y).all() or not np.isfinite(base).all() or (y < 0).any():
        raise ValueError("allowed source labels and OOF predictions must be finite")
    months = truth.shape[1]
    stations = np.unique(train//months)
    residual = np.full((len(stations), months), np.nan)
    values = y-np.maximum(0., np.expm1(base))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite source residual")
    residual[np.searchsorted(stations, train//months), train % months] = values
    return stations, residual


def seasonal_shuffle(values, calendar_month, seed):
    """Keep each station/calendar month's values and missingness unchanged."""
    out = values.copy()
    rng = np.random.default_rng(seed)
    for row in range(len(values)):
        for month in range(1, 13):
            slots = np.flatnonzero((calendar_month == month) & np.isfinite(values[row]))
            out[row, slots] = rng.permutation(values[row, slots])
    return out


def pair_correlation(recipient, donor, lag=0):
    if lag < 0 or np.shape(recipient) != np.shape(donor):
        raise ValueError("nonnegative lag and aligned station series are required")
    a, b = (recipient[lag:], donor[:-lag]) if lag else (recipient, donor)
    valid = np.isfinite(a) & np.isfinite(b)
    count = int(valid.sum())
    if count < 12:
        return np.nan, count, "insufficient_overlap"
    a, b = a[valid]-a[valid].mean(), b[valid]-b[valid].mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    if denominator == 0:
        return np.nan, count, "constant_series"
    return float(np.clip(a @ b/denominator, -1., 1.)), count, "available"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for partition in (142, 143, 144):
        parent = PARENT/"runs"/f"split{partition}_seed42"
        source = INPUTS/"runs"/parent.name
        config = json.loads((parent/"config.json").read_text())
        verify_files(parent, "complete.json", config)
        verify_files(source, "complete.json", json.loads((source/"config.json").read_text()))
        for kind in ("dataset", "mask"):
            if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
                raise ValueError("source data or roles changed")
        data = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        with np.load(config["mask_path"], allow_pickle=False) as masks:
            train = masks["train"].copy()
            receiving = np.unique(np.r_[masks["val"], masks["test"]]//data["y"].shape[1])
        with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
            ids, residual = source_residuals(data["y"], saved["pred_z"], train)
        if np.intersect1d(ids, receiving).size:
            raise ValueError("receiving stations appeared in source synchrony")
        with np.load(source/"joint_source_inputs.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["source_station_ids"], ids)
            ecology = saved["env"].astype(float)
        names = np.asarray(data["site_no"], str)[ids]
        residual = residual-np.nanmean(residual, axis=1, keepdims=True)
        calendar = pd.PeriodIndex(np.asarray(data["months"], str), freq="M").month.to_numpy()
        shuffled = seasonal_shuffle(residual, calendar, np.random.SeedSequence([42, partition]))
        distances = np.sqrt(((ecology[:, None]-ecology[None])**2).sum(-1))
        order = np.argsort(distances, axis=1, kind="stable")
        for recipient in range(len(ids)):
            candidates = order[recipient][order[recipient] != recipient]
            for group, donors in (("nearest5", candidates[:5]), ("farthest5", candidates[-5:][::-1])):
                for donor in donors:
                    for lag in (0, 1, 3):
                        real, count, status = pair_correlation(residual[recipient], residual[donor], lag)
                        control, count_control, control_status = pair_correlation(residual[recipient], shuffled[donor], lag)
                        if count != count_control:
                            raise ValueError("shuffle changed source availability")
                        rows.append({"partition": partition, "station": names[recipient],
                            "donor": names[donor], "group": group, "lag": lag,
                            "ecological_distance": distances[recipient, donor], "n_common_months": count,
                            "real_corr": real, "shuffle_corr": control,
                            "real_status": status, "shuffle_status": control_status})
        sources[str(partition)] = {"source_stations": len(ids), "source_doc_cells": len(train),
            "receiving_labels_inspected": False, "source_station_ids": ids.tolist(),
            "parent_completion_hash": sha256_file(parent/"complete.json"),
            "dataset_hash": config["dataset_hash"], "mask_hash": config["mask_hash"],
            "source_inputs_hash": sha256_file(source/"joint_source_inputs.npz")}
    pairs = pd.DataFrame(rows)
    pairs["eligible"] = np.isfinite(pairs.real_corr) & np.isfinite(pairs.shuffle_corr)
    pairs["real_corr"] = pairs.real_corr.where(pairs.eligible)
    pairs["shuffle_corr"] = pairs.shuffle_corr.where(pairs.eligible)
    station = pairs.groupby(["partition", "station", "group", "lag"], as_index=False).agg(
        real_corr=("real_corr", "mean"), shuffle_corr=("shuffle_corr", "mean"),
        eligible_pairs=("eligible", "sum"), candidate_pairs=("donor", "size"))
    station["real_minus_shuffle"] = station.real_corr-station.shuffle_corr
    parts = station.groupby(["partition", "group", "lag"], as_index=False).agg(
        real_corr=("real_corr", "mean"), shuffle_corr=("shuffle_corr", "mean"),
        real_minus_shuffle=("real_minus_shuffle", "mean"), source_stations=("station", "size"),
        eligible_stations=("real_corr", "count"), eligible_pairs=("eligible_pairs", "sum"))
    summary = parts.groupby(["group", "lag"], as_index=False).agg(
        real_corr=("real_corr", "mean"), shuffle_corr=("shuffle_corr", "mean"),
        real_minus_shuffle=("real_minus_shuffle", "mean"),
        eligible_stations_mean=("eligible_stations", "mean"), eligible_pairs_mean=("eligible_pairs", "mean"))
    output = ROOT/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("pair_diagnostics", pairs), ("station_diagnostics", station),
                        ("partition_diagnostics", parts), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    write_json(output/"sources.json", {"audit_sha256": sha256_file(__file__),
        "study_plan_sha256": sha256_file(ROOT/"study_plan.md"), "partitions": sources,
        "scope": "source-only descriptive associations; no receiving labels, fitting or target evaluation",
        "pair_minimum_common_months": 12, "shuffle_seed": 42,
        "weighting": "eligible donors within recipient; recipients within partition; equal partitions",
        "dependence": "overlapping stations/pairs/partitions; no independent-replication inference",
        "outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
