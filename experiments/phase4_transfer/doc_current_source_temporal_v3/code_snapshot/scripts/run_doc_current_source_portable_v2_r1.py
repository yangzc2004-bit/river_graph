"""Refit the confirmed current-source DOC architecture on saved deployment roles."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fit_portable_doc_sources_v1 import source_support_policies, validate_source_roles
from portable_current_source_doc_v2 import PortableCurrentSourceDOC
from portable_doc_reconstructor_v1 import PortableDOCReconstructor
from run_doc_geographical_confirmation_v1 import fit_memory
from run_doc_source_innovation_learning_v1 import pair_references
from run_doc_source_retrieval_v1 import safe_training_arrays
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
    nested_current_available_candidates,
)
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)
from river_graph.models.source_level_attention import source_level_input_view
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_current_source_portable_v2_r1")
PARENT = Path("experiments/phase4_transfer/doc_portable_source_fit_v1")
SOURCE_DECISION = Path("experiments/phase4_transfer/doc_current_availability_attention_v1/research_decision.md")
MODELS = ("available_real_native", "available_real_integrated")


def source_attention_reference(base, observed_loss):
    """Retain observed OOF references; absent source cells carry zero support."""
    base, mask = np.asarray(base), np.asarray(observed_loss)
    if (base.ndim != 2 or mask.shape != base.shape or mask.dtype != np.bool_
            or not np.isfinite(base[mask]).all() or (base[mask] < 0).any()):
        raise ValueError("finite nonnegative observed OOF reference and aligned loss mask required")
    return np.where(mask, base, 0.)


def deployment_pair_references(run, data, split, prepared, folds, config, progress):
    """Reuse the ten completed V2 references without changing their bindings."""
    previous = Path("experiments/phase4_transfer/doc_current_source_portable_v2/runs/seed42")
    if config["seed"] != 42:
        return pair_references(run, data, split, prepared["daily"],
            joblib.load(Path(config["parent_run"])/"station_hidden_trees.joblib"), folds, config, progress)
    old = json.loads((previous/"config.json").read_text())
    if any(old[key] != config[key] for key in ("seed", "dataset_hash", "mask_hash", "daily_hash", "parent_completion_hash")):
        raise ValueError("cached complementary references belong to different source roles")
    references, identities = {}, {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = f"pair_reference_{a}_{b}"
        verify_files(previous, f"{prefix}_complete.json", old)
        with np.load(previous/f"{prefix}.npz", allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads((previous/f"{prefix}.json").read_text()),
                "cells": saved["cells"].copy(), "pred_z": saved["pred_z"].copy()}
        identities[prefix] = sha256_file(previous/f"{prefix}_complete.json")
        progress("reused_pair_reference", {"query_donor_folds": [a, b]})
    write_json(run/"reference_reuse.json", {"previous_run": str(previous),
        "previous_config_hash": sha256_file(previous/"config.json"), "stage_hashes": identities,
        "reference_role": "ten verified V2 pair references; no refit or overwrite"})
    return references, ["reference_reuse.json"]


def prepare(parent):
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    helper = PortableDOCReconstructor.from_fitted_run(parent)
    data = strip_auxiliary_water(torch.load(old["dataset_path"], weights_only=False, map_location="cpu"))
    with np.load(old["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    validate_source_roles(data, split)
    data["y"] = development_labels(data, split)
    with np.load(parent/"oof.npz", allow_pickle=False) as saved:
        oof = saved["new"].copy()
    if sha256_file(old["daily_path"]) != old["daily_hash"]:
        raise ValueError("saved deployment daily input changed")
    with np.load(old["daily_path"], allow_pickle=False) as saved:
        daily = saved["full"].copy()
    with np.load(parent/"components.npz", allow_pickle=False) as saved:
        context = saved["environment"].copy()
    expert = UnifiedDOCReconstructor.load(parent/"backbone", data, split)
    features = extract_raw_temporal_inputs(expert, data, split)
    extra = build_regime_head_features(data["regime"], split["train"],
        np.maximum(0., np.expm1(oof.ravel()[split["train"]])), context,
        build_causal_flow_features(data)["full"], n_months=data["y"].shape[1])
    if digest(extra["normalization"]) != digest(helper.readout_normalization):
        raise ValueError("deployment base preprocessing must remain unchanged")
    source, full, arrays = safe_training_arrays(features, extra, daily, data, split, oof, context)
    return {"old": old, "helper": helper, "data": data, "split": split, "oof": oof,
            "daily": daily, "context": context, "source": source, "full": full, "arrays": arrays}


def run_one(root, seed, runtime):
    parent, run = PARENT/"runs"/f"seed{seed}", root/"runs"/f"seed{seed}"
    if (run/"complete.json").exists():
        config = json.loads((run/"config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("preserve the saved current-source deployment fitting version")
        verify_files(run, "complete.json", config)
        return
    started = time.monotonic()
    prepared = prepare(parent)
    old, data, split = (prepared[name] for name in ("old", "data", "split"))
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
        "daily_path", "daily_hash", "q90_threshold_train", "seed")}, "experiment": ROOT.name,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "parent_run": str(parent), "parent_completion_hash": sha256_file(parent/"complete.json"),
        "source_decision": str(SOURCE_DECISION), "source_decision_hash": sha256_file(SOURCE_DECISION),
        "models": [*old["models"], *MODELS], "candidate_count": 20, "lookback": 12,
        "attention_heads": 2, "attention_dimensions": 32, "epochs": 30, "patience": 5,
        "new_pair_reference_fits": 0 if seed == 42 else 10, "reused_pair_reference_fits": 10 if seed == 42 else 0,
        "new_neural_fits": 1,
        "execution_repair": "zero reference at unobserved source cells; preserve observed OOF values",
        "selection_role": "source_validation", "evaluation_role": "source_validation_deployment_refit",
        "receiving_water_quality": "none", "current_source_requires_previous_observation": False,
        "external_result_exposure": "02040104 inspected for earlier release; no labels used by this fit"}
    if (run/"config.json").exists():
        existing = json.loads((run/"config.json").read_text())
        if digest({k: v for k, v in config.items() if k != "started_at"}) != digest(
                {k: v for k, v in existing.items() if k != "started_at"}):
            raise ValueError("saved source deployment configuration changed")
        config = existing
    else:
        write_json(run/"config.json", config)

    def progress(stage, row):
        record = {"run": run.name, "stage": stage, **row, "elapsed_seconds": time.monotonic()-started}
        write_json(root/"progress.json", record)
        print(json.dumps(record), flush=True)

    t = data["y"].shape[1]
    ids, val_ids = np.unique(split["train"]//t), np.unique(split["val"]//t)
    folds = station_folds(split["train"], t, seed)
    references, files = deployment_pair_references(run, data, split, prepared, folds, config, progress)
    ecology, names, months = prepared["full"]["env"], np.asarray(data["site_no"], str), data["months"]
    parts, aggregate_roles = nested_innovation_inputs(data["y"], names, months, ecology,
        split["train"], folds, references)
    hydro = prepared["source"]["extra"][..., 30:38]
    candidates, roles = nested_current_available_candidates(data["y"], names, months, ecology,
        split["train"], folds, references, hydro)
    aggregate_ids, residual = source_residual_grid(data["y"], prepared["oof"], split["train"])
    relative_ids, relative_residual = relative_source_residual_grid(data["y"], prepared["oof"], split["train"])
    np.testing.assert_array_equal(aggregate_ids, ids)
    np.testing.assert_array_equal(relative_ids, ids)
    aggregate = SourceDOCInnovationLibrary().fit(names[ids], months, ecology[ids], residual)
    relative = SourceDOCInnovationLibrary().fit(names[ids], months, ecology[ids], relative_residual)
    full_parts = aggregate.predict_components(names, ecology, months)
    full_candidates = current_available_candidates(relative, names, ecology, months, names[ids], hydro)
    source = source_level_input_view({**prepared["source"],
        "attention_reference": source_attention_reference(prepared["arrays"][1], prepared["arrays"][3]),
        "extra": np.concatenate([prepared["source"]["extra"], innovation_readout_features(parts, "real", 1.)], -1)},
        candidates, "real")
    full = source_level_input_view({**prepared["full"], "attention_reference": prepared["context"],
        "extra": np.concatenate([prepared["full"]["extra"], innovation_readout_features(full_parts, "real", 1.)], -1)},
        full_candidates, "real")
    validation = {key: value if key == "donor_hydro_bank" else value[val_ids] for key, value in full.items()}
    aggregate.save(run/"aggregate_source_library.npz")
    relative.save(run/"relative_source_library.npz")
    write_json(run/"source_library_roles.json", {"aggregate": aggregate_roles, "individual": roles})
    np.savez_compressed(run/"source_hydro.npz", hydro=hydro, station=names[ids], months=np.asarray(months, str))
    files += ["config.json", "aggregate_source_library.npz", "relative_source_library.npz",
              "source_library_roles.json", "source_hydro.npz"]
    if (run/"neural_complete.json").exists():
        verify_files(run, "neural_complete.json", config)
        model = AvailableSourceAttentionResidual.from_payload(torch.load(run/"native.pt",
            weights_only=False, map_location="cpu"))
    else:
        retained = copy.deepcopy(prepared["helper"].native)
        for name in ("spatial", "temporal", "decay"):
            getattr(retained, name).load_state_dict(getattr(retained, f"_initial_{name}_state"))
        settings = {**retained._config(), "extra_dim": 41,
            "interaction_indices": (*retained.interaction_indices, 38), "epochs": 30, "patience": 5}
        model = AvailableSourceAttentionResidual(retained.spatial, retained.temporal, retained.decay,
            **settings).fit(source, *prepared["arrays"][1:4], validation, *prepared["arrays"][5:],
                tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                progress=lambda row: progress("current_source_attention", row))
        torch.save(model.to_payload(), run/"native.pt")
        write_json(run/"native.json", model.to_dict())
        bind_files(run, "neural_complete.json", [run/"native.pt", run/"native.json"], config)
    native = model.predict(full, prepared["context"])
    memory = fit_memory(data, split, prepared["oof"], prepared["context"], native, t)
    integrated = memory.predict(prepared["context"], native)
    bases = dict(zip(MODELS, (native, integrated), strict=True))
    adapters = source_support_policies(bases, data, split)
    write_json(run/"memory.json", memory.to_dict())
    write_json(run/"support_adapters.json", adapters)
    np.savez_compressed(run/"components.npz", environment=prepared["context"], native=native, integrated=integrated)
    files += ["native.pt", "native.json", "neural_complete.json", "memory.json", "support_adapters.json", "components.npz"]
    bind_files(run, "point_complete.json", [run/name for name in files], config)
    metadata = {"source_run": str(run), "source_fit_state_sha256": sha256_file(run/"point_complete.json"),
        "source_seed": seed, "recipe": "available_real_integrated", "support_adapters": adapters["available_real_integrated"],
        "parent_completion_sha256": config["parent_completion_hash"], "fit_role": "source_training",
        "target_information": "no receiving DOC/pH/conductance; fixed current-source architecture"}
    portable = PortableCurrentSourceDOC(local_helper=prepared["helper"], native=model,
        aggregate_library=aggregate, relative_library=relative, source_hydro=hydro,
        memory_state=memory.to_dict(), metadata=metadata)
    inputs = {key: np.asarray(data[key])[val_ids] for key in ("site_no", "x", "x_mask", "static", "regime")}
    inputs.update(months=months, daily_features=prepared["daily"][val_ids])
    actual = portable.predict(inputs)
    np.testing.assert_allclose(actual, integrated[val_ids], rtol=1e-6, atol=1e-6)
    export = run/"export"
    if not export.exists():
        portable.save(export)
    restored = PortableCurrentSourceDOC.load(export)
    if restored.metadata != portable.metadata:
        raise ValueError("cached portable export has different fitted source states")
    loaded = restored.predict(inputs)
    np.testing.assert_allclose(loaded, actual, rtol=0, atol=1e-12)
    write_json(run/"portable_replay.json", {"save_load": "within1e-12 mg/L; parallel forest reduction",
        "max_abs_save_load_error": float(np.abs(loaded-actual).max()), "validation_cells": len(split["val"]),
        "max_abs_training_facade_error": float(np.abs(actual-integrated[val_ids]).max()),
        "new_pair_forests": config["new_pair_reference_fits"],
        "reused_pair_forests": config["reused_pair_reference_fits"], "new_neural_fits": 1, "external_labels_used": False})
    previous = pd.read_parquet(parent/"predictions.parquet")
    template = previous[previous.model_name.eq("unmonitored_integrated")].copy()
    np.testing.assert_array_equal(template.cell, split["val"])
    frames = [previous]
    for name, grid in bases.items():
        frame = template.copy()
        frame["model_name"], frame["y_pred"] = name, grid.ravel()[split["val"]]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/"predictions.parquet", index=False)
    files += ["point_complete.json", "portable_replay.json", "export/manifest.json"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run/name for name in (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress("complete", {"new_pair_reference_fits": config["new_pair_reference_fits"],
        "reused_pair_reference_fits": config["reused_pair_reference_fits"], "new_neural_fits": 1,
        "validation_cells": len(split["val"]), "portable_save_load": "within1e-12mg/L"})
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/"training.pid").write_text(str(os.getpid())+"\n")
    torch.set_num_threads(2)
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_current_source_portable_v2_r1.py",
        "scripts/portable_current_source_doc_v2.py", "scripts/portable_doc_reconstructor_v1.py",
        "scripts/verify_portable_current_source_doc_v2.py", "scripts/fit_portable_doc_sources_v1.py",
        "scripts/run_doc_source_innovation_learning_v1.py", "scripts/run_doc_source_retrieval_v1.py",
        "scripts/run_doc_geographical_confirmation_v1.py", "scripts/run_unified_doc_spatial.py",
        "tests/test_portable_current_source_doc.py", "tests/test_current_source_portable_refit.py",
        str(SOURCE_DECISION), str(ROOT/"study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = args.root/"runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("changed deployment fitting code requires a new version")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for seed in args.seeds:
        run_one(args.root, seed, digest(snapshot))


if __name__ == "__main__":
    main()
