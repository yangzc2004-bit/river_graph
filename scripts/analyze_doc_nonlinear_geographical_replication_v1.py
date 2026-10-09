"""Replay the fixed nonlinear geographical model and its support products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_geographical_confirmation_v1 import (
    additional_metrics,
    improving_region_count,
    strata_for_run,
    summaries,
)
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_geographical_confirmation_v1 import TASKS, support_curves
from run_doc_nonlinear_geographical_replication_v1 import ARMS, MODELS, ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.nonlinear_native_residual import NonlinearNativeResidual


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("nonlinear geographical runtime differs")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("geographical input changed")
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched geographical parent changed")
    if sha256_file(config["source_decision"]) != config["source_decision_hash"]:
        raise ValueError("source-fixed candidate decision changed")
    verify_files(run, "complete.json", config)
    verify_files(run, "native_complete.json", config)
    verify_files(run, "point_complete.json", config)
    source = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    truth = np.asarray(source["y"], float).copy()
    months = truth.shape[1]
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    cells = split["test"]
    model = NonlinearNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    with np.load(run/"test_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "age", "support", "env", "extra")}
        context = saved["context"].copy()
        np.testing.assert_array_equal(saved["cell"], cells)
    prediction = model.predict(inputs, context)
    ids = np.unique(cells//months)
    with np.load(run/"components.npz", allow_pickle=False) as saved:
        native, integrated, environment = (saved[key].copy() for key in ("native", "integrated", "environment"))
    np.testing.assert_allclose(prediction, native[ids], rtol=1e-6, atol=1e-6)
    memory = EcologicalResidualTransfer.from_dict(json.loads((run/"memory.json").read_text()))
    np.testing.assert_array_equal(memory.predict(environment, native), integrated)
    old_model = torch.load(parent/"native.pt", weights_only=False, map_location="cpu")
    new_model = model.to_payload()
    for name in ("initial_spatial", "initial_temporal", "initial_decay"):
        for key, value in old_model[name].items():
            torch.testing.assert_close(new_model[name][key], value, rtol=0, atol=0)
    for filename in ("predictions.parquet", "support_curves.parquet"):
        frame = pd.read_parquet(run/filename)
        prior = pd.read_parquet(parent/filename)
        original = frame[frame.model_name.isin(prior.model_name.unique())].reset_index(drop=True)
        pd.testing.assert_frame_equal(original, prior.reset_index(drop=True), check_exact=True)
        meta = json.loads((run/filename.replace(".parquet", ".meta.json")).read_text())
        if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run/filename)
                or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], runtime)):
            raise ValueError("nonlinear geographical sidecar differs")
        if (set(frame.model_name) != set(MODELS) or not frame.visibility_role.eq("test").all()
                or frame.duplicated(["model_name", "k", "cell"]).any() or not np.isfinite(frame.y_pred).all()):
            raise ValueError("incomplete nonlinear geographical panel")
        reference = None
        for _, group in frame.groupby(["model_name", "k"]):
            query = group.cell.to_numpy()
            if reference is None:
                reference = query
            np.testing.assert_array_equal(query, reference)
            np.testing.assert_array_equal(group.y_true, truth.ravel()[query])
            np.testing.assert_array_equal(group.station, np.asarray(source["site_no"], str)[query//months])
            np.testing.assert_array_equal(group.month, np.asarray(source["months"], str)[query % months])
        if filename == "predictions.parquet":
            np.testing.assert_array_equal(reference, cells)
            np.testing.assert_array_equal(frame[frame.model_name.eq(ARMS[0])].y_pred, native.ravel()[cells])
            np.testing.assert_array_equal(frame[frame.model_name.eq(ARMS[1])].y_pred, integrated.ravel()[cells])
    dataset = strip_auxiliary_water(source)
    dataset["y"] = development_labels(dataset, split)
    replay_curves, adapters = support_curves(dict(zip(ARMS, (native, integrated), strict=True)), dataset, split, truth)
    replay_curves["split_seed"], replay_curves["seed"], replay_curves["target_huc4"] = (
        config["split_seed"], config["seed"], config["target_huc4"])
    saved_curves = pd.read_parquet(run/"support_curves.parquet")
    pd.testing.assert_frame_equal(saved_curves[saved_curves.model_name.isin(ARMS)].reset_index(drop=True),
                                  replay_curves.reset_index(drop=True), check_exact=True)
    if adapters != json.loads((run/"support_adapters.json").read_text()):
        raise ValueError("validation-selected support policy differs")
    return {"run": run.name, "identity": True, "unchanged_parent_products": True,
            "matched_initial_backbone": "bitwise", "integrated_and_support_replay": "bitwise",
            "native_subset_replay": "rtol/atol1e-6: different inference batch composition",
            "test_cells": len(cells), "scope": "retrospective ST357 replication"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    regions = json.loads((TASKS/"task_protocol.json").read_text())["geographical_targets"]
    paths = [args.root/"runs"/f"huc4_{region}_seed{seed}" for region in regions for seed in args.seeds]
    if any(not (path/"complete.json").exists() for path in paths):
        raise ValueError("complete every fixed region/seed package before analysis")
    write_json(args.root/"verification/replay.json", [verify(path, runtime) for path in paths])
    if args.verify_only:
        return
    frames, curves, thresholds, stratification = [], [], {}, []
    for path in paths:
        config = json.loads((path/"config.json").read_text())
        primary = pd.read_parquet(path/"predictions.parquet")
        frames.append(primary)
        curves.append(pd.read_parquet(path/"support_curves.parquet"))
        thresholds[(config["split_seed"], config["seed"])] = config["q90_threshold_train"]
        stratification.extend(strata_for_run(path, primary))
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    primary, curves = pd.concat(frames, ignore_index=True), pd.concat(curves, ignore_index=True)
    for label, panel in (("primary", primary), ("curves", curves)):
        runs, parts, summary, station_runs = summaries(panel, thresholds)
        for name, table in (("run_metrics", runs), ("region_metrics", parts), ("summary", summary),
                            ("station_metrics", station_runs), ("q90_diagnostics", additional_metrics(panel, thresholds))):
            table.to_csv(output/f"{label}_{name}.csv", index=False)
    effects, stations = [], []
    comparisons = [(ARMS[1], reference) for reference in
                   ("unmonitored_integrated", "current_model", "station_hidden_trees", ARMS[0])]
    comparisons += [(ARMS[0], "unmonitored_residual")]
    for population, panel in (("all_observed_k0", primary), ("fixed_query_curve", curves)):
        for k in sorted(panel.k.unique()):
            for candidate, reference in comparisons:
                comparison = paired(panel[panel.k.eq(k)], candidate, reference)
                for zone in ("overall", "q90"):
                    selected = comparison
                    if zone == "q90":
                        limit = np.array([thresholds[(s, t)] for s, t in zip(
                            comparison.split_seed, comparison.seed, strict=True)])
                        selected = comparison[comparison.y_true_candidate.to_numpy() >= limit]
                    record = {"candidate": candidate, "reference": reference, "population": population, "k": k, "zone": zone,
                              "n_regions_usable": selected.split_seed.nunique()}
                    if selected.split_seed.nunique() != len(regions):
                        effects.append({**record, "relative_gain_pct": np.nan, "gain_ci_low_pct": np.nan,
                            "gain_ci_high_pct": np.nan, "improving_regions": np.nan,
                            "status": "not estimable for five-region mean: empty tail"})
                        continue
                    effects.append({**record, **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                        "improving_regions": improving_region_count(selected), "status": "estimated"})
                group = comparison.groupby(["split_seed", "station"], as_index=False).agg(
                    candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"), n_cells=("cell", "nunique"))
                group["delta_mae"] = group.candidate_mae-group.reference_mae
                group["candidate"], group["reference"], group["k"], group["population"] = candidate, reference, k, population
                stations.append(group)
    pd.DataFrame(effects).to_csv(output/"paired_effects.csv", index=False)
    pd.concat(stations, ignore_index=True).to_csv(output/"station_effects.csv", index=False)
    strata = pd.DataFrame(stratification)
    strata.to_csv(output/"strata_by_run.csv", index=False)
    region_strata = strata.groupby(["split_seed", "model_name", "k", "stratum", "group"], as_index=False).mae.mean()
    region_strata.to_csv(output/"strata_by_region.csv", index=False)
    region_strata.groupby(["model_name", "k", "stratum", "group"], as_index=False).agg(
        mae=("mae", "mean"), n_regions=("split_seed", "nunique")).to_csv(output/"strata_summary.csv", index=False)
    write_json(output/"sources.json", {"bootstrap_draws": args.bootstrap_draws, "seeds": args.seeds,
        "regions": regions, "estimand": "seed mean within region, equal regions", "runtime": runtime,
        "scope": "retrospective ST357 replication of source-fixed nonlinear head",
        "analyzer_sha256": sha256_file(__file__), "completed_runs": {str(path/"complete.json"): sha256_file(path/"complete.json") for path in paths}})
    print(pd.read_csv(output/"primary_summary.csv")[["model_name", "mae", "station_equal_mae", "q90_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
