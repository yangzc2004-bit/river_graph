"""Replay the station-hidden tree base and quantify the source-input change."""
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
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_trees_v1")


def verify(run):
    config = json.loads((run / "config.json").read_text())
    verify_files(run, "complete.json", config)
    meta = json.loads((run / "predictions.meta.json").read_text())
    if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run / "predictions.parquet")
            or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])):
        raise ValueError("changed tree product identity")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    ancestor = json.loads((Path(config["parent_run"]) / "config.json").read_text())
    daily, _, _ = load_daily_pack(Path(ancestor["parent_run"]).parent.parent, config["dataset_hash"], dataset["y"].shape)
    features = np.column_stack([build_rf_features(dataset, split, FIT_ROLES,
        target_transform="log1p", include_network=True), daily.reshape(-1, 8)])
    forest = joblib.load(run / "station_hidden_trees.joblib")
    prediction = np.maximum(0, np.expm1(forest.predict(features[split["val"]])))
    panel = pd.read_parquet(run / "predictions.parquet")
    for _, rows in panel.groupby("model_name"):
        np.testing.assert_array_equal(np.sort(rows.cell), split["val"])
        if not rows.visibility_role.eq("val").all():
            raise ValueError("nonvalidation evaluation")
    np.testing.assert_allclose(prediction, panel[panel.model_name.eq("station_hidden_trees")].y_pred,
                               rtol=1e-12, atol=1e-12)
    return {"run": run.name, "tree_replay": True, "target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    runs = sorted((args.root / "runs").glob("*/complete.json"))
    if len(runs) != 9:
        raise ValueError("need the complete nine-package source-validation experiment")
    write_json(args.root / "verification" / "replay.json", [verify(path.parent) for path in runs])
    panel, thresholds = load_panel(args.root)
    by_run, parts, summary = metric_summary(panel, thresholds)
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    for name, table in (("run_metrics", by_run), ("partition_metrics", parts), ("summary", summary)):
        table.to_csv(output / f"{name}.csv", index=False)
    results = []
    for reference in ("current_model", "matched_daily_trees"):
        pair = paired(panel, "station_hidden_trees", reference)
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                limit = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= limit]
            results.append({"candidate": "station_hidden_trees", "reference": reference, "region": region,
                            **joint_station_bootstrap(selected, draws=args.bootstrap_draws)})
    effects = pd.DataFrame(results)
    effects.to_csv(output / "paired_effects.csv", index=False)
    write_json(output / "sources.json", {"packages": len(runs), "bootstrap_draws": args.bootstrap_draws,
        "evaluation_role": "selected source-validation development",
        "runs": {str(path): sha256_file(path) for path in runs}})
    print(summary[["model_name", "mae", "q90_mae"]].to_string(index=False))
    print(effects[["reference", "region", "relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
