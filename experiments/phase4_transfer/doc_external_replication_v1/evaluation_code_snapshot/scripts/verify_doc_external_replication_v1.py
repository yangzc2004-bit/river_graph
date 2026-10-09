"""Verify source choices, external cell identities and saved ensemble products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_external_replication_v1 import (
    COMPONENTS,
    EXTERNAL_PROTOCOL,
    ROOT,
    empirical_interval,
    source_ensemble_policy,
)
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.models.station_adapted_hybrid import station_residual_correction


def verify(root=ROOT):
    root = Path(root)
    points = json.loads((root/"point_generation_complete.json").read_text())
    config = points["config"]
    meta = json.loads((root/"predictions.meta.json").read_text())
    if (points["external_DOC_read"] or points["config_hash"] != digest(config)
            or meta["config"] != config or meta["config_hash"] != digest(config)):
        raise ValueError("external point generation and scoring definitions disagree")
    identities = {**points["files"], "predictions.parquet": meta["prediction_sha256"],
        "point_generation_complete.json": meta["point_generation_sha256"],
        "source_calibration.json": meta["source_calibration_sha256"]}
    for name, expected in identities.items():
        if sha256_file(root/name) != expected:
            raise ValueError(f"changed external product: {name}")
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    if (sha256_file(EXTERNAL_PROTOCOL) != config["external_protocol_hash"]
            or sha256_file(external["dataset_path"]) != config["external_scoring_hash"]
            or sha256_file(external["label_free_inputs"]) != config["inputs_hash"]):
        raise ValueError("external dataset or input identities changed")
    policies = json.loads((root/"source_calibration.json").read_text())
    if policies["external_DOC_used"] or policies["external_support_used"]:
        raise ValueError("external outcomes entered source calibration")
    source_panels = []
    source_stations = set()
    for completion, expected in config["source_completions"].items():
        if sha256_file(completion) != expected:
            raise ValueError("fitted source completion changed")
        run = Path(completion).parent
        fit = json.loads((run/"config.json").read_text())
        for marker in ("backbone_complete.json", "trees_complete.json", "current_complete.json", "complete.json"):
            verify_files(run, marker, fit)
        source_panels.append(pd.read_parquet(run/"predictions.parquet"))
        dataset = torch.load(fit["dataset_path"], weights_only=False, map_location="cpu")
        source_stations.update(np.asarray(dataset["site_no"], str))
    panel_source = pd.concat(source_panels, ignore_index=True)
    for name in config["procedures"]:
        group = panel_source[panel_source.model_name.eq(name)]
        if not group.visibility_role.eq("source_validation").all():
            raise ValueError("support and interval choices require source validation")
        mean = group.groupby("cell", sort=True).agg(pred=("y_pred", "mean"), truth=("y_true", "first"),
            n=("seed", "nunique"), truths=("y_true", "nunique"))
        if not mean.n.eq(5).all() or not mean.truths.eq(1).all():
            raise ValueError("source ensemble has incomplete seeds or inconsistent labels")
        expected = source_ensemble_policy(mean.pred.to_numpy(), mean.index.to_numpy(),
            mean.truth.to_numpy(), n_months=dataset["y"].shape[1])
        # Portable replay differs only at floating-point rounding from saved predictions.
        saved = policies["procedures"][name]
        for k in ("0", "1", "3", "5"):
            np.testing.assert_equal(saved["curve"][k]["selected"]["alpha"], expected["curve"][k]["selected"]["alpha"])
            np.testing.assert_allclose(saved["curve"][k]["log_half_width"], expected["curve"][k]["log_half_width"], atol=1e-12)
        np.testing.assert_allclose(saved["primary_k0_log_half_width"], expected["primary_k0_log_half_width"], atol=1e-12)
    dataset = torch.load(external["dataset_path"], weights_only=False, map_location="cpu")
    names, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    if source_stations.intersection(names):
        raise ValueError("external stations overlap the ST source cohort")
    months = len(dates)
    truth = np.asarray(dataset["y"]).ravel()
    cells = np.flatnonzero(np.asarray(dataset["y_mask"], bool))
    schedule, query = support_schedule(cells, months)
    scored, full = pd.read_parquet(root/"predictions.parquet"), pd.read_parquet(root/"full_grid.parquet")
    if scored.duplicated(["model_name", "population", "k", "cell"]).any():
        raise ValueError("duplicate scoring identities")
    if len(scored) != len(config["procedures"])*(len(cells)+4*len(query)):
        raise ValueError("missing or additional external scoring rows")
    with np.load(root/"seed_components.npz", allow_pickle=False) as seed:
        np.testing.assert_array_equal(seed["station"], names)
        np.testing.assert_array_equal(seed["month"], dates)
        for name in config["procedures"]:
            grid = full[full.model_name.eq(name)]
            np.testing.assert_array_equal(grid.station, np.repeat(names, months))
            np.testing.assert_array_equal(grid.month, np.tile(dates, len(names)))
            for key in COMPONENTS:
                members = seed[f"{name}__{key}"]
                assert members.shape == (5, len(names), months)
                np.testing.assert_array_equal(grid[key], members.mean(0).ravel())
            center = grid.final_pred.to_numpy()
            policy = policies["procedures"][name]
            for population, k, selected in [("all_observed_k0", 0, cells),
                    *[("fixed_query_curve", k, query) for k in (0, 1, 3, 5)]]:
                rows = scored[scored.model_name.eq(name) & scored.population.eq(population) & scored.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(rows.cell, selected)
                np.testing.assert_array_equal(rows.station, names[selected//months])
                np.testing.assert_array_equal(rows.month, dates[selected % months])
                np.testing.assert_array_equal(rows.y_true, truth[selected])
                support = np.sort(np.concatenate([value[:k] for value in schedule.values()]))
                expected = center[selected] if k == 0 else station_residual_correction(
                    center[selected], selected, center[support], support, truth[support], n_months=months,
                    alpha=policy["curve"][str(k)]["selected"]["alpha"])
                np.testing.assert_array_equal(rows.y_pred, expected)
                width = (policy["primary_k0_log_half_width"] if population == "all_observed_k0"
                         else policy["curve"][str(k)]["log_half_width"])
                lower, upper = empirical_interval(expected, width)
                np.testing.assert_array_equal(rows.pi_lower, lower)
                np.testing.assert_array_equal(rows.pi_upper, upper)
    if not np.isfinite(scored[["y_true", "y_pred", "pi_lower", "pi_upper"]]).all().all():
        raise ValueError("nonfinite external product")
    report = {"status": "passed", "source_packages": 5, "procedures": config["procedures"],
        "external_stations": len(names), "k0_cells": len(cells), "fixed_query_cells": len(query),
        "source_external_station_overlap": 0, "source_policy_recomputed": True,
        "config_hash": digest(config), "verifier_sha256": sha256_file(__file__)}
    write_json(root/"verification/report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
