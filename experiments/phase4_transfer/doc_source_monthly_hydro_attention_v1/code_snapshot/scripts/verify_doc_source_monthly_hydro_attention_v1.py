"""Replay source monthly hydro keys, source exclusion and unchanged reference states."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from run_doc_source_monthly_hydro_attention_v1 import MODES, ROOT, model_views, prepare
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.source_monthly_hydro_attention import (
    SourceMonthlyHydroAttentionResidual,
)


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("attention runtime identity changed")
    prepared = prepare(config["split_seed"], config["seed"])
    for name in ("parent", "learning", "source", "probe", "native_attention", "preceding_attention"):
        if (str(prepared[name]) != config[f"{name}_run"]
                or sha256_file(prepared[name]/"complete.json") != config[f"{name}_completion_hash"]):
            raise ValueError("attention parent version changed")
    if prepared["source_roles"] != json.loads((run/"source_library_roles.json").read_text()):
        raise ValueError("query/source exclusion roles changed")
    with np.load(run/"attention_candidates.npz", allow_pickle=False) as saved:
        for role in ("source", "validation"):
            for name, values in prepared[f"{role}_candidates"].items():
                np.testing.assert_array_equal(saved[f"{role}_{name}"], values)
    with np.load(run/"relative_source_library.npz", allow_pickle=False) as saved:
        library = prepared["relative_library"]
        for key, attribute in (("innovations", "innovations_"), ("seasonal_mean", "seasonal_mean_"),
                               ("source_names", "source_names_"), ("ecology", "ecology_")):
            np.testing.assert_array_equal(saved[key], getattr(library, attribute))
    with np.load(run/"source_monthly_hydro_bank.npz", allow_pickle=False) as saved:
        for mode, bank in prepared["monthly_banks"].items():
            np.testing.assert_array_equal(saved[mode], bank)
    units = json.loads((run/"relative_source_library_units.json").read_text())
    if units["source_value_units"] != "dimensionless OOF log1p residual including seasonal mean":
        raise ValueError("relative source library units changed")
    if config["source_reference_tertiles"] != prepared["source_reference_tertiles"]:
        raise ValueError("receiving reference groups must come from source OOF predictions")
    meta = json.loads((run/"predictions.meta.json").read_text())
    if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")
            or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], runtime)):
        raise ValueError("attention product identity failed")
    previous = pd.read_parquet(prepared["preceding_attention"]/"predictions.parquet")
    panel = pd.read_parquet(run/"predictions.parquet")
    selected = panel[panel.model_name.isin(previous.model_name.unique())][previous.columns]
    pd.testing.assert_frame_equal(selected.reset_index(drop=True), previous.reset_index(drop=True), check_exact=True)
    data, split = prepared["data"], prepared["split"]
    ids, val = prepared["validation_ids"], split["val"]
    t = data["y"].shape[1]
    local = np.searchsorted(ids, val//t), val % t
    base = np.zeros(data["y"].shape)
    base[ids] = prepared["validation_base"]
    summaries = []
    for prefix, (source_mode, attention_mode) in MODES.items():
        model = SourceMonthlyHydroAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt",
            weights_only=False, map_location="cpu"))
        if model.to_dict()["protocol"]["source_value_units"] != "dimensionless":
            raise ValueError("relative attention did not retain its source value conversion")
        if "including source seasonal mean" not in model.to_dict()["protocol"]["source_values"]:
            raise ValueError("source branch must preserve the permitted donor seasonal mean")
        if model.to_dict()["protocol"]["new_monthly_key_initialization"] != "zero coefficients; original query and key coefficients retained":
            raise ValueError("monthly key initialization protocol changed")
        if torch.count_nonzero(model._initial_attention_head["key.weight"][:, -4:]):
            raise ValueError("new monthly key coefficients were not zero initialized")
        if digest(model._config()) != digest(prepared["settings"]):
            raise ValueError("attention readout or training budget changed")
        if model.attention_config != {"attention_mode": attention_mode, "attention_heads": 2,
                                     "attention_dimensions": 32, "daily_start": 30}:
            raise ValueError("attention operator configuration changed")
        for name in ("spatial", "temporal", "decay"):
            expected = getattr(prepared["retained"], f"_initial_{name}_state")
            actual = getattr(model, f"_initial_{name}_state")
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        _, view = model_views(prepared, source_mode)
        prediction = model.predict(view, prepared["validation_base"])
        np.testing.assert_array_equal(prediction[local], panel[panel.model_name.eq(f"{prefix}_native")].y_pred)
        memory = EcologicalResidualTransfer.from_dict(json.loads((run/f"{prefix}_memory.json").read_text()))
        native = base.copy()
        native[ids] = prediction
        np.testing.assert_array_equal(memory.predict(base, native).ravel()[val],
            panel[panel.model_name.eq(f"{prefix}_integrated")].y_pred)
        diagnostics = model.diagnostics(view, np.ravel_multi_index(local, (len(ids), t)))
        with np.load(run/f"{prefix}_diagnostics.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["cells"], val)
            for name in diagnostics:
                np.testing.assert_array_equal(saved[name], diagnostics[name])
        if (not np.isfinite(prediction).all() or not all(np.isfinite(v).all() for v in diagnostics.values())
                or (diagnostics["prior_mass"] < 0).any() or (diagnostics["prior_mass"] > 1).any()):
            raise ValueError("nonfinite or invalid attention product")
        if attention_mode == "fixed_prior" and any(model.to_dict()[name] != 0.
                for name in ("query_parameter_change", "key_parameter_change")):
            raise ValueError("fixed-prior control learned donor allocation")
        summaries.append({"model": prefix, "epochs": model.epochs_run_, "selected_epoch": model.best_epoch_,
                          "selected_scale": model.selected_scale_, "parameters": model.trainable_parameter_count_})
    return {"run": run.name, "identity": True, "candidates_roles_initial_weights": "exact",
            "neural_memory_diagnostics_replay": "bitwise", "new_forest_fits": 0,
            "evaluation_role": "source_validation", "fits": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--report", default="replay.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    rows = []
    for partition in args.splits:
        for seed in args.seeds:
            rows.append(verify(args.root/"runs"/f"split{partition}_seed{seed}", runtime))
            print(json.dumps(rows[-1]), flush=True)
    write_json(args.root/"verification"/args.report, rows)


if __name__ == "__main__":
    main()
