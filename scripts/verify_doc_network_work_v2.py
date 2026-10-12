"""Independently audit query truth, coverage, source selection and checkpoint replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import MASKS, context, verify
from run_doc_network_work_v2 import (
    MODELS,
    REFERENCE,
    REGIONS,
    ROOT,
    SEEDS,
    SOURCE,
    config_for,
    neural_cells,
)
from run_doc_river_architecture_comparison_v1 import digest, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_domain_work import features
from river_graph.experiments.doc_fusion_benchmark import (
    concentration_prediction,
    select_recipe,
)
from river_graph.experiments.doc_network_work import PhySTArchitecture
from river_graph.experiments.provenance import sha256_file


def independent_metrics(part):
    actual, predicted = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
    residual = predicted-actual
    sst = float(np.sum((actual-actual.mean())**2))
    return {"mae": float(np.mean(np.abs(residual))),
        "log1p_mae": float(np.mean(np.abs(np.log1p(predicted)-np.log1p(actual)))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1-float(np.sum(residual**2))/sst if sst > 0 else np.nan,
        "bias": float(np.mean(residual))}


def main():
    protocol = json.loads((ROOT/"protocol.json").read_text())
    manifest = json.loads((ROOT/"manifest.json").read_text())
    batch = json.loads((ROOT/"batch_complete.json").read_text())
    assert manifest["protocol_hash"] == batch["protocol_hash"] == digest(protocol)
    assert sha256_file("scripts/run_ladder.py") == protocol["original_dispatcher_sha256"]
    assert sha256_file("data/processed/mississippi_graph_graphfix_st357.pt") == protocol["dataset_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT/"code_snapshot"/name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected
    for name, expected in protocol["preparation"]["files"].items():
        assert sha256_file(name) == expected
    assert sha256_file(REFERENCE/"all_selections.json") == protocol["reference_gate_sha256"]
    workspace = Path(json.loads((ROOT/"launch.json").read_text())["cwd"])
    # Replay the precise official source and frozen implementation used in fitting.
    torch.set_num_threads(protocol["threads"])
    if str(workspace/"src") not in sys.path:
        sys.path.insert(0, str(workspace/"src"))
    data = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    months, fits, replay_max, frames = data["y"].shape[1], 0, 0., []
    flat = np.asarray(data["y"]).ravel()
    gate = json.loads((ROOT/"all_selections.json").read_text())
    with threadpool_limits(limits=protocol["threads"]):
        for region in REGIONS:
            run = ROOT/"regions"/f"huc4_{region}"
            assert sha256_file(run/"selection_complete.json") == gate["regional_selections"][region]
            verify(run, "selection_complete.json", config_for(protocol, region))
            verify(run, "test_complete.json")
            ctx = context(data, region)
            with np.load(MASKS/f"huc4_{region}.npz", allow_pickle=False) as mask:
                test = mask["test"]
                roles = {role: set(mask[role]//months) for role in ("train", "val", "test")}
            assert not roles["train"] & roles["val"] and not roles["train"] & roles["test"] and not roles["val"] & roles["test"]
            assert set(ctx["inputs"]["normalization"]["source_rows"]) == roles["train"]
            output = pd.read_parquet(run/"predictions.parquet")
            assert not output.duplicated(["model_name", "seed", "cell"]).any()
            for name in MODELS:
                for seed in SEEDS:
                    part = output[(output.model_name == name) & (output.seed == seed)]
                    np.testing.assert_array_equal(part.cell, test)
                    np.testing.assert_array_equal(part.y_true, flat[test])
                    assert np.isfinite(part.y_pred).all() and (part.y_pred >= 0).all()
            selection = json.loads((run/"selection.json").read_text())
            assert selection["test_used"] is False
            for family, choice in selection["selections"].items():
                assert choice["recipe_id"] == select_recipe(choice["validation_scores"])
                for path, expected in selection["fit_receipts"].items():
                    if path.split("/")[1] != family:
                        continue
                    directory = run/path
                    assert sha256_file(directory/"fit_complete.json") == expected
                    config = json.loads((directory/"config.json").read_text())
                    verify(directory, "fit_complete.json", config)
                    saved = pd.read_parquet(directory/"validation_predictions.parquet")
                    np.testing.assert_array_equal(saved.cell, ctx["split"]["val"])
                    np.testing.assert_array_equal(saved.y_true, flat[ctx["split"]["val"]])
                    info = json.loads((directory/"info.json").read_text())
                    if family.startswith("ssn"):
                        model = joblib.load(directory/"model.joblib")
                        assert set(model.nodes) == roles["train"]
                        predicted, _ = concentration_prediction(model.predict(features(ctx["inputs"], saved.cell.to_numpy(), months, "doc_ecological_rf"), saved.cell.to_numpy()//months), config["recipe"]["target"])
                    else:
                        model = PhySTArchitecture(SOURCE, config["recipe"])
                        model.load_state_dict(torch.load(directory/"model.pt", weights_only=True, map_location="cpu"))
                        predicted = neural_cells(model, ctx, data, saved.cell.to_numpy(), "val", info["target_normalization"], protocol["neural_settings"])
                    difference = float(np.max(np.abs(predicted-saved.y_pred.to_numpy())))
                    assert difference < 1e-6
                    replay_max = max(replay_max, difference)
                    if config["recipe_id"] == choice["recipe_id"]:
                        part = output[(output.model_name == family) & (output.seed == (42 if family.startswith("ssn") else config["seed"]))]
                        if family.startswith("ssn"):
                            predicted, _ = concentration_prediction(model.predict(features(ctx["inputs"], test, months, "doc_ecological_rf"), test//months), config["recipe"]["target"])
                        else:
                            predicted = neural_cells(model, ctx, data, test, "test", info["target_normalization"], protocol["neural_settings"])
                        np.testing.assert_allclose(predicted, part.y_pred, atol=1e-6, rtol=0)
                    fits += 1
                    del model
            frames.append(output)
    assert fits == protocol["unique_new_fits"] == 45
    joined = pd.concat(frames, ignore_index=True)
    summary = pd.read_csv(ROOT/"analysis/summary.csv").set_index("model_name")
    maximum = 0.
    for name in MODELS:
        parts = [independent_metrics(part) for _, part in joined[joined.model_name == name].groupby("seed")]
        means = pd.DataFrame(parts).mean()
        for metric, actual in means.items():
            maximum = max(maximum, abs(float(summary.loc[name, metric])-actual))
        assert summary.loc[name, "unique_query_cells"] == 7262
        assert summary.loc[name, "coverage"] == 1.
    assert maximum < 1e-10
    verify(ROOT/"analysis", "complete.json", {"protocol_hash": digest(protocol)})
    write_json(ROOT/"analysis/independent_audit.json", {"status": "passed", "candidate_checkpoint_replays": fits,
        "maximum_validation_replay_difference": replay_max, "maximum_independent_metric_difference": maximum,
        "unique_query_cells": 7262, "models": len(MODELS), "coverage": 1., "source_only_selection": True,
        "reference_protocol_sha256": protocol["reference_protocol_sha256"], "protocol_hash": digest(protocol)})
    print(json.dumps({"stage": "audit_passed", "fits": fits, "maximum_metric_difference": maximum, "maximum_replay_difference": replay_max}))


if __name__ == "__main__":
    main()
