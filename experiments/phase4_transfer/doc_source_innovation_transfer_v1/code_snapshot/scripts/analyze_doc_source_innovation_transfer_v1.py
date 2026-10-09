"""Replay source innovations and quantify their DOC development value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_source_innovation_transfer_v1 import ARMS, ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    corrected_prediction,
    select_correction,
    source_residual_grid,
)


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("innovation execution changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("source data or roles changed")
    parent, source = Path(config["parent_run"]), Path(config["source_inputs_run"])
    verify_files(parent, "complete.json", json.loads((parent/"config.json").read_text()))
    if (sha256_file(parent/"complete.json") != config["parent_completion_hash"]
            or sha256_file(source/"joint_source_inputs.npz") != config["source_inputs_hash"]):
        raise ValueError("source cache or parent changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], runtime)
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("source innovation product identity changed")
    data = torch.load(config["dataset_path"], map_location="cpu", weights_only=False)
    with np.load(config["mask_path"], allow_pickle=False) as masks:
        train, val = masks["train"].copy(), masks["val"].copy()
        receiving_ids = np.unique(np.r_[val, masks["test"]]//data["y"].shape[1])
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        ids, residuals = source_residual_grid(data["y"], saved["pred_z"], train)
    if np.intersect1d(ids, receiving_ids).size:
        raise ValueError("receiving station entered source library")
    folds = json.loads((parent/"oof_records.json").read_text())
    hidden = []
    for fold in folds:
        if set(fold["hidden_stations"]) & set(fold["fitted_stations"]):
            raise ValueError("OOF reference includes held source station")
        hidden.extend(fold["hidden_stations"])
    np.testing.assert_array_equal(np.sort(hidden), ids)
    with np.load(source/"joint_source_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_station_ids"], ids)
        ecology = saved["env"].copy()
    rebuilt = SourceDOCInnovationLibrary().fit(np.asarray(data["site_no"], str)[ids],
        np.asarray(data["months"], str), ecology, residuals)
    library = SourceDOCInnovationLibrary.load(run/"source_library.npz")
    for name in ("source_names_", "months_", "ecology_", "seasonal_mean_", "innovations_"):
        np.testing.assert_array_equal(getattr(library, name), getattr(rebuilt, name))
    if library.to_dict() != rebuilt.to_dict():
        raise ValueError("library statistics failed replay")
    with np.load(run/"receiver_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    np.testing.assert_array_equal(inputs["cells"], val)
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(inputs["ecology"], saved["env"])
    np.testing.assert_array_equal(inputs["station"], np.asarray(data["site_no"], str)[inputs["receiver_ids"]])
    np.testing.assert_array_equal(inputs["months"], np.asarray(data["months"], str))
    parts = library.predict_components(inputs["station"], inputs["ecology"], inputs["months"])
    if parts["donors"] != json.loads((run/"donors.json").read_text()):
        raise ValueError("source candidates or ecological weights changed")
    panel, previous = pd.read_parquet(run/"predictions.parquet"), pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(previous.model_name.unique())][previous.columns]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), previous.reset_index(drop=True), check_exact=True)
    template = previous[previous.model_name.eq("unmonitored_integrated")]
    index = np.searchsorted(inputs["receiver_ids"], val//len(inputs["months"])), val % len(inputs["months"])
    selections = json.loads((run/"selection.json").read_text())
    for arm, key in zip(ARMS, ("real_innovation", "historical_innovation"), strict=True):
        frame = panel[panel.model_name.eq(arm)]
        np.testing.assert_array_equal(frame.cell, val)
        innovation = parts[key][index]
        selected = select_correction(template.y_pred, innovation, template.y_true)
        if selections[arm] != selected:
            raise ValueError("source-validation scalar selection changed")
        np.testing.assert_array_equal(frame.y_pred,
            corrected_prediction(template.y_pred, innovation, selected["alpha"]))
        np.testing.assert_array_equal(frame.source_innovation, innovation)
        for field in ("support_count", "all_current_support_count", "weight_mass"):
            np.testing.assert_array_equal(frame[f"source_{field}"], parts[field][index])
    if not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid source-validation product")
    return {"run": run.name, "identity": True, "library_and_prediction_replay": "bitwise",
        "unchanged_parent_predictions": True, "source_station_exclusion": True,
        "old_target_evaluation": False, "no_neural_or_forest_refit": True,
        "alpha_real": selections[ARMS[0]]["alpha"], "alpha_history": selections[ARMS[1]]["alpha"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {p.parent.name for p in complete} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("complete all nine source packages")
    verification = [verify(p.parent, runtime) for p in complete]
    write_json(args.root/"verification/replay.json", verification)
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    pd.DataFrame(verification).to_csv(output/"selection_diagnostics.csv", index=False)
    comparisons = [(arm, reference) for arm in ARMS for reference in
                   ("unmonitored_integrated", "station_hidden_trees", "current_model")]
    comparisons.append((ARMS[0], ARMS[1]))
    effects = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                cutoff = np.array([thresholds[(p, s)] for p, s in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= cutoff]
            direction = selected.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
            part, seed = direction.groupby("split_seed").mean(), direction.groupby("seed").mean()
            effects.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
                "improved_partitions": int((part.candidate_error < part.reference_error).sum()),
                "improved_training_seeds": int((seed.candidate_error < seed.reference_error).sum())})
    effects = pd.DataFrame(effects)
    effects.to_csv(output/"paired_effects.csv", index=False)
    panel["absolute_error"], panel["signed_error"] = np.abs(panel.y_pred-panel.y_true), panel.y_pred-panel.y_true
    station = panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size"))
    station.to_csv(output/"station_metrics.csv", index=False)
    candidate_panel = panel[panel.model_name.eq(ARMS[0])].copy()
    candidate_panel["has_current_source"] = candidate_panel.source_all_current_support_count.gt(0)
    candidate_panel["has_matched_source"] = candidate_panel.source_support_count.gt(0)
    support = candidate_panel.groupby(["split_seed", "seed"], as_index=False).agg(
        query_cells=("cell", "size"), current_coverage=("has_current_source", "mean"),
        matched_coverage=("has_matched_source", "mean"), count_current=("source_all_current_support_count", "mean"),
        count_matched=("source_support_count", "mean"), weight_mass=("source_weight_mass", "mean"),
        mean_abs_innovation=("source_innovation", lambda value: np.abs(value).mean()))
    support.to_csv(output/"source_support.csv", index=False)
    strata = panel[panel.model_name.isin(("unmonitored_integrated", *ARMS))].copy()
    support_by_cell = candidate_panel[["split_seed", "seed", "cell", "has_matched_source"]]
    strata = strata.merge(support_by_cell, on=["split_seed", "seed", "cell"], validate="many_to_one")
    strata.groupby(["split_seed", "seed", "model_name", "has_matched_source"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size"))\
        .to_csv(output/"support_strata.csv", index=False)
    write_json(output/"sources.json", {"evaluation_role": "source_validation_selected_development",
        "bootstrap_draws": args.bootstrap_draws, "selection_optimism": "alpha includes0 and is selected on this panel",
        "analyzer_sha256": sha256_file(__file__), "runs": {p.parent.name: sha256_file(p) for p in complete},
        "numerical_outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
