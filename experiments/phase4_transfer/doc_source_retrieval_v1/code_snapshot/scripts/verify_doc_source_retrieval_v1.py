"""Replay learned donor corrections and verify source-only prediction products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_source_retrieval_v1 import ROOT, inference_retrieval
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.doc_source_retrieval import SourceRetrievalAttention


def verify_run(run):
    config = json.loads((run / "config.json").read_text())
    verify_files(run, "complete.json", config)
    meta = json.loads((run / "predictions.meta.json").read_text())
    if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run / "predictions.parquet")
            or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])):
        raise ValueError("prediction sidecar identity differs")
    panel = pd.read_parquet(run / "predictions.parquet")
    if not panel.visibility_role.eq("val").all() or not panel.k.eq(0).all() or set(panel.model_name) != set(config["models"]):
        raise ValueError("source-validation-only model set differs")
    with np.load(config["mask_path"], allow_pickle=False) as masks:
        expected = np.sort(masks["val"])
        if np.isin(panel.cell, masks["test"]).any():
            raise ValueError("target cells entered development evaluation")
    for name, part in panel.groupby("model_name"):
        np.testing.assert_array_equal(np.sort(part.cell), expected)
    records = json.loads((run / "nested_fold_records.json").read_text())
    for outer in records:
        excluded = set(outer["excluded_stations"])
        if excluded & set(outer["donor_stations"]):
            raise ValueError("pseudo-target station entered its donor bank")
        for inner in outer["inner_records"]:
            if (excluded & set(inner["held_stations"]) or excluded & set(inner["fitted_stations"])
                    or set(inner["held_stations"]) & set(inner["fitted_stations"])):
                raise ValueError("nested donor forest station leakage")
    replay = []
    for arm in ("retrieval", "hydro_pretrained_retrieval"):
        model = SourceRetrievalAttention.from_payload(torch.load(run / f"{arm}.pt", weights_only=False, map_location="cpu"))
        selection = json.loads((run / f"{arm}.json").read_text())
        if selection["selection_role"] != "source_validation":
            raise ValueError("invalid retrieval selection role")
        with np.load(run / f"{arm}_validation_inputs.npz", allow_pickle=False) as cached:
            prepared = {key: cached[key].copy() for key in cached.files}
        g = selection["gamma"]
        for ablation in (None, "uniform", "zero_source_values"):
            delta, _, _, _ = inference_retrieval(model, prepared, scale=selection["residual_scale"], ablation=ablation)
            p = np.maximum(0, prepared["context"]+(1-g)*(prepared["temporal"]-prepared["context"])
                           +g*(prepared["static"]+delta))
            name = arm if ablation is None else f"{arm}_{ablation}"
            selected = panel[panel.model_name.eq(name)].sort_values("cell")
            np.testing.assert_array_equal(p, selected.y_pred.to_numpy())
            replay.append({"model_name": name, "query_rows": len(p), "bitwise_replay": True})
    return {"run": run.name, "target_evaluation": False, "n_models": panel.model_name.nunique(),
            "nested_station_exclusion": True, "replay": replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    runs = sorted((args.root / "runs").glob("*/complete.json"))
    if not runs:
        raise ValueError("no complete runs")
    results = [verify_run(path.parent) for path in runs]
    write_json(args.root / "verification" / "replay.json", results)
    print(json.dumps({"verified_runs": len(results), "target_evaluation": False}), flush=True)


if __name__ == "__main__":
    main()
