"""Replay fixed geographical attention components, support and diagnostics."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from run_doc_current_source_attention_geographical_v1 import (
    ARMS,
    MODES,
    ROOT,
    model_views,
    prepare,
)
from run_doc_geographical_confirmation_v1 import support_curves
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import HUC4_BLOCKS
from river_graph.models.current_source_attention import CurrentSourceAttentionResidual
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("geographical attention runtime changed")
    prepared = prepare(config["target_huc4"], config["seed"])
    for name in ("parent", "previous"):
        if (str(prepared[name]) != config[f"{name}_run"]
                or sha256_file(prepared[name]/"complete.json") != config[f"{name}_completion_hash"]):
            raise ValueError("saved geographical parents changed")
    if sha256_file(config["source_decision"]) != config["source_decision_hash"]:
        raise ValueError("source-selected research decision changed")
    if prepared["source_roles"] != json.loads((run/"source_library_roles.json").read_text()):
        raise ValueError("query/donor library exclusion changed")
    with np.load(run/"attention_candidates.npz", allow_pickle=False) as saved:
        for role in ("source", "full"):
            for name, values in prepared[f"{role}_candidates"].items():
                np.testing.assert_array_equal(saved[f"{role}_{name}"], values)
    panel = pd.read_parquet(run/"predictions.parquet")
    previous = pd.read_parquet(prepared["previous"]/"predictions.parquet")
    original = panel[panel.model_name.isin(previous.model_name.unique())][previous.columns]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), previous.reset_index(drop=True), check_exact=True)
    curves = pd.read_parquet(run/"support_curves.parquet")
    previous_curves = pd.read_parquet(prepared["previous"]/"support_curves.parquet")
    original_curves = curves[curves.model_name.isin(previous_curves.model_name.unique())][previous_curves.columns]
    pd.testing.assert_frame_equal(original_curves.reset_index(drop=True), previous_curves.reset_index(drop=True), check_exact=True)
    for name in ("predictions.parquet", "support_curves.parquet"):
        meta = json.loads((run/name).with_suffix(".meta.json").read_text())
        if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run/name)
                or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], runtime)):
            raise ValueError("geographical prediction sidecar failed")
    bases = {}
    with np.load(run/"components.npz", allow_pickle=False) as components:
        np.testing.assert_array_equal(components["environment"], prepared["context"])
        for prefix, (source_mode, attention_mode) in MODES.items():
            model = CurrentSourceAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt",
                weights_only=False, map_location="cpu"))
            if digest(model._config()) != digest(prepared["settings"]):
                raise ValueError("geographical training budget/readout changed")
            expected = {"attention_mode": attention_mode, "attention_heads": 2,
                        "attention_dimensions": 32, "daily_start": 30}
            if model.attention_config != expected:
                raise ValueError("geographical attention operator changed")
            for name in ("spatial", "temporal", "decay"):
                for key, value in getattr(prepared["retained"], f"_initial_{name}_state").items():
                    torch.testing.assert_close(getattr(model, f"_initial_{name}_state")[key], value, rtol=0, atol=0)
            _, view = model_views(prepared, source_mode)
            native = model.predict(view, prepared["context"])
            memory = EcologicalResidualTransfer.from_dict(json.loads((run/f"{prefix}_memory.json").read_text()))
            integrated = memory.predict(prepared["context"], native)
            for kind, grid in (("native", native), ("integrated", integrated)):
                name = f"{prefix}_{kind}"
                np.testing.assert_array_equal(grid, components[name])
                selected = panel[panel.model_name.eq(name)]
                np.testing.assert_array_equal(selected.cell, prepared["split"]["test"])
                np.testing.assert_array_equal(selected.y_pred, grid.ravel()[prepared["split"]["test"]])
                bases[name] = grid
            diagnostic = model.diagnostics(view, prepared["split"]["test"])
            with np.load(run/f"{prefix}_diagnostics.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["cells"], prepared["split"]["test"])
                for name, values in diagnostic.items():
                    np.testing.assert_array_equal(saved[name], values)
            if attention_mode == "fixed_prior" and any(model.to_dict()[name] != 0
                    for name in ("query_parameter_change", "key_parameter_change")):
                raise ValueError("fixed-prior allocation changed")
    truth = np.asarray(torch.load(config["dataset_path"], weights_only=False, map_location="cpu")["y"], float)
    expected_curves, adapters = support_curves(bases, prepared["data"], prepared["split"], truth)
    expected_curves["split_seed"], expected_curves["seed"], expected_curves["target_huc4"] = (
        int(config["target_huc4"]), config["seed"], config["target_huc4"])
    selected = curves[curves.model_name.isin(ARMS)]
    pd.testing.assert_frame_equal(selected.reset_index(drop=True), expected_curves.reset_index(drop=True), check_exact=True)
    if adapters != json.loads((run/"support_adapters.json").read_text()):
        raise ValueError("saved support policy replay failed")
    if not np.isfinite(panel.y_pred).all() or not np.isfinite(curves.y_pred).all():
        raise ValueError("nonfinite geographical products")
    return {"run": run.name, "identity": True, "nested_source_exclusion": True,
            "candidates_initial_weights_components_support_diagnostics": "bitwise",
            "new_forest_fits": 0, "new_neural_fits": 3,
            "scope": "retrospective ST357 roles; not external-basin validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    parser.add_argument("--huc4", nargs="+", choices=HUC4_BLOCKS, default=list(HUC4_BLOCKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--report", default="replay.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    results = []
    for huc4 in args.huc4:
        for seed in args.seeds:
            results.append(verify(args.root/"runs"/f"huc4_{huc4}_seed{seed}", runtime))
            print(json.dumps(results[-1]), flush=True)
    write_json(args.root/"verification"/args.report, results)


if __name__ == "__main__":
    main()
