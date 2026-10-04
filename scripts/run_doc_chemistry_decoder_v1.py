"""Fit a small nonlinear chemistry decoder on the retained DOC backbone."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_auxiliary_chemistry_v1 import (
    ARMS,
    AUX_PATHS,
    BASIS,
    INTEGRATED,
    MODES,
    NEURAL,
    REFERENCES,
    TREES,
    final_frame,
    fit_choices,
    reference_checks,
)
from run_doc_daily_hydro_memory_v1 import KS, load_daily_pack
from run_doc_daily_hydro_readout_v1 import array_digest
from run_doc_distribution_head_v1 import head_feature_blocks, observed_features
from run_doc_ecological_transfer_v1 import load_inputs
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.auxiliary_chemistry_features import (
    apply_auxiliary_mode,
    build_auxiliary_chemistry_features,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.nonlinear_chemistry_head import NonlinearChemistryHead
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_chemistry_decoder_v1")
CHEMISTRY = Path("experiments/phase4_transfer/doc_auxiliary_chemistry_v1")
FEATURE_DIM = 554
LINEAR_NAMES = (f"linear_chemistry_{BASIS}", f"linear_chemistry_integrated_{BASIS}")
MODELS = (tuple(f"{arm}_{BASIS}" for arm in ARMS)
          + tuple(f"{arm}_integrated_{BASIS}" for arm in INTEGRATED) + LINEAR_NAMES)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_daily_hydro_memory_v1.py", "scripts/run_doc_daily_hydro_readout_v1.py",
                 "scripts/run_doc_daily_hydro_support_basis_v1.py", "scripts/run_doc_distribution_head_v1.py",
                 "scripts/run_doc_auxiliary_chemistry_v1.py", "scripts/run_doc_chemistry_decoder_v1.py",
                 str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution changed; preserve this version and use a fresh root")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def augment(vectors, auxiliary):
    result = np.concatenate((vectors, np.asarray(auxiliary, dtype=np.float32)), axis=1)
    if result.shape != (len(vectors), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid nonlinear chemistry input layout")
    return result


def linear_reference_frame(frame):
    names = {f"neural_chemistry_{BASIS}": LINEAR_NAMES[0],
             f"neural_chemistry_integrated_{BASIS}": LINEAR_NAMES[1]}
    result = frame[frame.model_name.isin(names)].copy()
    result["model_name"] = result.model_name.map(names)
    return result


def run_one(root, chemistry_root, partition, seed, runtime, *, epochs, patience):
    started = time.monotonic()
    old_run = chemistry_root / "runs" / f"split{partition}_seed{seed}"
    old_config = json.loads((old_run / "config.json").read_text())
    verify_files(old_run, "complete.json", old_config)
    prior = Path(old_config["prior_run"])
    pc, sources, _, dataset, split, full, parent, oof = load_inputs(prior)
    run = root / "runs" / old_run.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["chemistry_completion_hash"] != sha256_file(old_run / "complete.json")):
            raise ValueError("Saved nonlinear chemistry package differs from frozen settings")
        verify_files(run, "complete.json", config)
        return
    run.mkdir(parents=True, exist_ok=True)
    previous_full = pd.read_parquet(old_run / "full_grid.parquet")
    previous_frame = pd.read_parquet(old_run / "predictions.parquet")
    aux_data = {key: torch.load(path, weights_only=False) for key, path in AUX_PATHS.items()}
    chemistry = build_auxiliary_chemistry_features(dataset, aux_data["ph"], aux_data["ec"])
    auxiliary, active = chemistry["full"].reshape(-1, 4), chemistry["active"].ravel()
    if (array_digest(auxiliary) != old_config["auxiliary_feature_hash"]
            or array_digest(active) != old_config["auxiliary_active_hash"]):
        raise ValueError("Auxiliary inputs changed since the linear study")
    truth = np.asarray(dataset["y"], dtype=np.float64)
    shape, months = truth.shape, truth.shape[1]
    context, native, memory = full.context_pred.to_numpy(), parent.off_pred.to_numpy(), parent.ecological_memory.to_numpy()
    daily, _, _ = load_daily_pack(Path(pc["daily_features_path"]).parent, pc["dataset_hash"], shape)
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        basis = saved[BASIS].copy()
    config = {**old_config, "experiment": "doc_chemistry_decoder_v1",
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "chemistry_run": str(old_run), "chemistry_completion_hash": sha256_file(old_run / "complete.json"),
        "chemistry_full_grid_hash": sha256_file(old_run / "full_grid.parquet"),
        "chemistry_prediction_hash": sha256_file(old_run / "predictions.parquet"),
        "availability_audit_path": str(chemistry_root / "availability_audit.json"),
        "models": MODELS, "linear_reference_models": LINEAR_NAMES,
        "epochs": epochs, "patience": patience, "torch_threads": torch.get_num_threads(),
        "feature_dim": FEATURE_DIM, "head_parameters": 1111, "chemistry_embedding_dim": 8,
        "feature_layout": "source-standardized original550+aux4; phi8; hidden64xphi8; output1070",
        "head_class": "NonlinearChemistryHead", "activation": "SiLU",
        "new_head_fits": 3, "new_tree_fits": 0, "trees_reused_fixed_controls": True,
        "linear_reference_role": "previously seen linear-study development results",
        "study_role": "nonlinear chemistry response on existing DOC development partitions"}
    write_json(run / "config.json", config)
    model = EncoderNativeResidual.from_payload(torch.load(prior / "off.pt", weights_only=True))
    for module in (model.spatial, model.temporal, model.decay, model.head):
        module.requires_grad_(False)
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    raw, flow = extract_raw_temporal_inputs(expert, dataset, split), build_causal_flow_features(dataset)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context.reshape(shape), flow["full"], n_months=months)
    source_ids = raw["source_station_ids"]
    np.testing.assert_array_equal(source_ids, extra["source_station_ids"])
    source_inputs = {key: raw[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    source_inputs["extra"] = np.concatenate((extra["source_extra"], daily[source_ids]), axis=-1)
    full_inputs = {key: raw[f"full_{key}"] for key in ("raw", "age", "support")}
    full_inputs["env"], full_inputs["extra"] = raw["env"], np.concatenate((extra["full_extra"], daily), axis=-1)
    train_mask = np.zeros(shape, dtype=bool)
    train_mask.ravel()[split["train"]] = True
    local_cells = np.flatnonzero(train_mask[source_ids])
    source_cells = source_ids[local_cells//months]*months+local_cells%months
    _, val_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    source_features, source_delta = observed_features(model, source_inputs, local_cells)
    validation_features, _ = observed_features(model, full_inputs, val_cells)
    source_base = np.maximum(0, np.expm1(oof.ravel()[source_cells])+model.selected_scale_*source_delta)
    old_inputs = json.loads((old_run / "input_definition.json").read_text())
    identities = {"source_original_feature_hash": array_digest(source_features),
        "validation_original_feature_hash": array_digest(validation_features), "modes": {}}
    for key in ("source_original_feature_hash", "validation_original_feature_hash"):
        if identities[key] != old_inputs[key]:
            raise ValueError("Frozen parent features differ from the linear experiment")
    np.savez_compressed(run / "source_training.npz", source_cells=source_cells, source_local_cells=local_cells,
        source_station_ids=source_ids, source_base=source_base, source_active=active[source_cells],
        validation_cells=val_cells, validation_base=native[val_cells], validation_active=active[val_cells])
    with np.load(old_run / "source_training.npz", allow_pickle=False) as previous:
        for key, value in {"source_cells": source_cells, "source_base": source_base,
                           "validation_cells": val_cells, "validation_base": native[val_cells]}.items():
            np.testing.assert_array_equal(value, previous[key])
    heads = {}
    for mode in MODES:
        block = apply_auxiliary_mode(chemistry["full"], mode).reshape(-1, 4)
        sx, vx = augment(source_features, block[source_cells]), augment(validation_features, block[val_cells])
        identities["modes"][mode] = {"source_feature_hash": array_digest(sx), "validation_feature_hash": array_digest(vx)}
        head = NonlinearChemistryHead(n_features=FEATURE_DIM, epochs=epochs, patience=patience,
            batch_size=512, seed=seed, learning_rate=.001).fit(sx, source_base, truth.ravel()[source_cells],
                vx, native[val_cells], truth.ravel()[val_cells], source_active=active[source_cells],
                validation_active=active[val_cells], tail_threshold=config["q90_threshold_train"],
                selection_role="source_validation", progress=lambda row, name=mode: print(
                    f"{run.name}/neural_{name}: {json.dumps(row)}", flush=True))
        if head.trainable_parameter_count_ != 1111:
            raise ValueError("Unexpected nonlinear decoder parameter count")
        heads[mode] = head
        torch.save(head.to_payload(), run / f"neural_{mode}.pt")
        write_json(run / f"neural_{mode}.json", head.to_dict())
        pd.DataFrame(head.to_dict()["trace"]).to_csv(run / f"neural_{mode}_trace.csv", index=False)
    write_json(run / "input_definition.json", identities)
    bases = {arm: previous_full[f"{arm}_pred"].to_numpy().copy() for arm in (*REFERENCES, *TREES)}
    del expert, raw, flow, extra, source_inputs, source_features, validation_features, sx, vx, oof, daily
    gc.collect()
    components = previous_full.copy()
    components["linear_chemistry_pred"] = previous_full.neural_chemistry_pred
    components["linear_chemistry_integrated_k0_pred"] = previous_full.neural_chemistry_integrated_k0_pred
    for mode in MODES:
        bases[f"neural_{mode}"] = np.empty(truth.size)
    for cells, vectors, _ in head_feature_blocks(model, full_inputs):
        for mode in MODES:
            block = apply_auxiliary_mode(auxiliary[cells].reshape(1, -1, 4), mode).reshape(-1, 4)
            bases[f"neural_{mode}"][cells] = heads[mode].predict(augment(vectors, block), native[cells], active=active[cells])
    del model, full_inputs
    gc.collect()
    for arm in NEURAL:
        components[f"{arm}_pred"] = bases[arm]
        np.testing.assert_array_equal(bases[arm][~active], native[~active])
    adapters, mixers = fit_choices(bases, context, memory, basis, truth.ravel(), split, months, active)
    write_json(run / "adapters.json", {name: value.to_dict() for name, value in adapters.items()})
    write_json(run / "mixers.json", {name: value.to_dict() for name, value in mixers.items()})
    for arm in INTEGRATED:
        candidate = mixers[f"{arm}_integrated_{BASIS}"].selected_base(context, bases[arm], memory, k=0)
        fallback = mixers[f"point_integrated_{BASIS}"].selected_base(context, native, memory, k=0)
        components[f"{arm}_integrated_k0_pred"] = candidate if arm == "point" else np.where(active, candidate, fallback)
    validation = final_frame(full, bases, context, memory, basis, adapters, mixers, truth.ravel(), split, months, active, role="val")
    validation["error"] = np.abs(validation.y_pred-truth.ravel()[validation.cell.to_numpy()])
    rows = []
    for (name, k), group in validation.groupby(["model_name", "k"], sort=True):
        selected = active[group.cell.to_numpy()]
        rows.append({"model_name": name, "k": int(k), "n": len(group), "n_active": int(selected.sum()),
            "mae": float(group.error.mean()), "active_mae": float(group.loc[selected, "error"].mean())})
    old_validation = pd.read_csv(old_run / "source_validation.csv")
    mapping = {f"neural_chemistry_{BASIS}": LINEAR_NAMES[0], f"neural_chemistry_integrated_{BASIS}": LINEAR_NAMES[1]}
    copied = old_validation[old_validation.model_name.isin(mapping)].copy()
    copied["model_name"] = copied.model_name.map(mapping)
    pd.concat([pd.DataFrame(rows), copied], ignore_index=True).to_csv(run / "source_validation.csv", index=False)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = final_frame(full, bases, context, memory, basis, adapters, mixers, labels, split, months, active, role="test")
    checks = reference_checks(frame, prior)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    for column in ("ph_available", "ec_available", "aux_available", "doc_observed"):
        frame[column] = components[column].to_numpy()[frame.cell.to_numpy()]
    frame = pd.concat([frame, linear_reference_frame(previous_frame)], ignore_index=True)
    for name in (*[f"{arm}_{BASIS}" for arm in TREES], *LINEAR_NAMES):
        old_name = name.replace("linear_chemistry", "neural_chemistry")
        for k in KS:
            left = frame[frame.model_name.eq(name) & frame.k.eq(k)].sort_values("cell")
            right = previous_frame[previous_frame.model_name.eq(old_name) & previous_frame.k.eq(k)].sort_values("cell")
            for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                np.testing.assert_array_equal(left[column], right[column])
            checks.append({"model_name": name, "k": k, "rows": len(left), "bitwise_exact": True})
    write_json(run / "reference_checks.json", checks)
    if (set(frame.model_name) != set(MODELS) or not np.isfinite(frame.select_dtypes(include="number")).all().all()
            or not np.isfinite(components.select_dtypes(include="number")).all().all()):
        raise ValueError("Invalid nonlinear chemistry product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    files = ["input_definition.json", "source_training.npz", "adapters.json", "mixers.json"]
    files += [f"neural_{mode}.{suffix}" for mode in MODES for suffix in ("pt", "json")]
    for name in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, name, config, runtime, files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-started})
    completed = ["config.json", *files, "source_validation.csv", "reference_checks.json", "timing.json",
        "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json",
        *[f"neural_{mode}_trace.csv" for mode in MODES]]
    bind_files(run, "complete.json", [run / name for name in completed], config)
    print(f"{run.name}: complete in{time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--chemistry-root", type=Path, default=CHEMISTRY)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.chemistry_root, partition, seed, runtime, epochs=args.epochs, patience=args.patience)
            gc.collect()


if __name__ == "__main__":
    main()
