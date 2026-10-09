"""Replay region-hidden DOC source fits and report matched development effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_regional_source_training_v1 import ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.regional_source_views import huc4_source_folds

ARMS = ("regional_trees", "regional_residual", "regional_integrated")


def verify(run):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    for kind in ("dataset", "mask", "nodes"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError(f"regional source {kind} changed")
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("retained comparator changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["runtime_snapshot_hash"] != config["runtime_snapshot_hash"]
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("regional source prediction identity changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    model = EncoderNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
        context, cells = saved["context"].copy(), saved["cell"].copy()
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(cells, saved["val"])
        train = saved["train"].copy()
    n_months = context.shape[1]
    ids = np.unique(cells//n_months)
    prediction = model.predict(inputs, context)
    selected = prediction[np.searchsorted(ids, cells//n_months), cells % n_months]
    np.testing.assert_array_equal(selected, panel[panel.model_name.eq(ARMS[1])].y_pred)
    forest = joblib.load(run/"regional_trees.joblib")
    with np.load(run/"tree_inputs.npz", allow_pickle=False) as saved:
        tree = np.maximum(0., np.expm1(forest.predict(saved["query_features"])))
        full_context = saved["context"].copy()
    np.testing.assert_allclose(tree, panel[panel.model_name.eq(ARMS[0])].y_pred, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(full_context[ids], context)
    with np.load(run/"source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    source_mask = np.zeros(oof.size, dtype=bool)
    source_mask[train] = True
    if not np.isfinite(oof.ravel()[source_mask]).all() or not np.isnan(oof.ravel()[~source_mask]).all():
        raise ValueError("regional OOF covers cells outside source fitting labels")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    nodes = pd.read_csv(config["nodes_path"], dtype=str).set_index("site_no")
    huc = nodes.loc[np.asarray(dataset["site_no"], str), "huc_cd"].tolist()
    expected = huc4_source_folds(train, n_months, huc, config["seed"])
    records = json.loads((run/"oof_records.json").read_text())
    if len(records) != len(expected):
        raise ValueError("regional OOF fold count changed")
    source_stations = set((train//n_months).tolist())
    for record, fold in zip(records, expected, strict=True):
        np.testing.assert_array_equal(record["hidden_stations"], fold)
        held, fitted = set(record["hidden_stations"]), set(record["fitted_stations"])
        if held & fitted or held | fitted != source_stations:
            raise ValueError("regional OOF station exclusion failed")
        held_huc = {str(huc[i]).split(".")[0].zfill(8)[:4] for i in held}
        fitted_huc = {str(huc[i]).split(".")[0].zfill(8)[:4] for i in fitted}
        if held_huc & fitted_huc or held_huc != set(record["hidden_huc4"]) or fitted_huc != set(record["fitted_huc4"]):
            raise ValueError("held HUC4 region entered its OOF reference")
        remaining = train[np.isin(train//n_months, sorted(fitted))]
        inner = huc4_source_folds(remaining, n_months, huc, config["seed"])
        for actual, group in zip(record["inner_folds"], inner, strict=True):
            np.testing.assert_array_equal(actual, group)
    memory = EcologicalResidualTransfer.from_dict(json.loads((run/"memory.json").read_text()))
    c = np.zeros((memory.n_nodes_, memory.n_months_))
    p = c.copy()
    c[ids], p[ids] = context, prediction
    np.testing.assert_array_equal(memory.predict(c, p).ravel()[cells], panel[panel.model_name.eq(ARMS[2])].y_pred)
    reference_truth = prior[prior.model_name.eq("unmonitored_integrated")].y_true.to_numpy()
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
        np.testing.assert_array_equal(group.y_true, reference_truth)
    if (not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all()
            or set(panel.model_name) != set(config["models"])):
        raise ValueError("invalid region-hidden source validation panel")
    return {"run": run.name, "identity": True, "neural_and_integrated_replay": "bitwise",
            "tree_max_abs_replay": float(np.max(np.abs(tree-panel[panel.model_name.eq(ARMS[0])].y_pred))),
            "regional_oof_exclusion": True, "unchanged_parent_predictions": True,
            "old_target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {p.parent.name for p in complete} != {f"split{p}_seed{s}" for p in (142, 143, 144) for s in (42, 43, 44)}:
        raise ValueError("complete the nine fixed source packages")
    runtime = verify_runtime_snapshot(args.root)
    if any(json.loads((p.parent/"config.json").read_text())["runtime_snapshot_hash"] != runtime for p in complete):
        raise ValueError("source package runtime differs from its saved execution")
    write_json(args.root/"verification/replay.json", [verify(p.parent) for p in complete])
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(candidate, ref) for candidate in ARMS for ref in
                   ("unmonitored_integrated", "station_hidden_trees", "current_model")]
    comparisons += [("regional_residual", "unmonitored_residual"),
                    ("regional_residual", "regional_trees"), ("regional_integrated", "regional_trees")]
    effects = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                cutoff = np.array([thresholds[(p, s)] for p, s in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= cutoff]
            direction = selected.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
            part = direction.groupby("split_seed").mean()
            effects.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
                "improved_partitions": int((part.candidate_error < part.reference_error).sum()),
                "evaluation_role": "source_validation_development"})
    effects = pd.DataFrame(effects)
    effects.to_csv(output/"paired_effects.csv", index=False)
    panel["absolute_error"] = np.abs(panel.y_pred-panel.y_true)
    panel["signed_error"] = panel.y_pred-panel.y_true
    panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size")
    ).to_csv(output/"station_metrics.csv", index=False)
    availability = []
    for path in complete:
        config = json.loads((path.parent/"config.json").read_text())
        dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        x_mask = np.asarray(dataset["x_mask"], dtype=bool).sum(axis=-1).ravel()
        package = panel[(panel.split_seed == config["split_seed"]) & (panel.seed == config["seed"])].copy()
        package["hydro_availability"] = x_mask[package.cell.to_numpy()]
        availability.append(package.groupby(["split_seed", "seed", "model_name", "hydro_availability"], as_index=False).agg(
            mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size")))
    pd.concat(availability, ignore_index=True).to_csv(output/"hydro_strata.csv", index=False)
    write_json(output/"sources.json", {"evaluation_role": "source_validation_development", "bootstrap_draws": args.bootstrap_draws,
        "analyzer_sha256": sha256_file(__file__), "runs": {p.parent.name: sha256_file(p) for p in complete},
        "numerical_outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
