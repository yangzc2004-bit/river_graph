"""Learn a small station-support readout from frozen current DOC hidden states."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_memory_v1 import KS, load_daily_pack
from run_doc_daily_hydro_support_basis_v1 import (
    fit_arm_mixers,
    integrated_frame,
    validation_rows,
)
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_support_readout import EpisodicSupportReadout
from river_graph.models.episodic_temporal_adapter import (
    EpisodicTemporalAdapter,
    anchor_normalize,
)
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_readout_v1")
PRIOR = Path("experiments/phase4_transfer/doc_daily_hydro_support_basis_v1")
SHAPES = ("constant", "legacy", "refreshed_fixed", "refreshed_learned")
MODELS = tuple(f"off_{basis}_{stage}" for basis in SHAPES
               for stage in ("direct", "integrated"))


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v2.py", "scripts/run_doc_daily_hydro_memory_v1.py",
                 "scripts/run_doc_daily_hydro_support_basis_v1.py",
                 "scripts/run_doc_daily_hydro_readout_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution changed; preserve this version and use another root")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def array_digest(values):
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def export_hidden(model, inputs, batch_size=2048):
    """Export station-major states without changing any selected expert weight."""
    prepared = model._prepare_inputs(inputs)
    shape = tuple(prepared["age"].shape)
    roots = (model.spatial, model.temporal, model.decay, model.head)
    states = [(module, module.training) for root in roots for module in root.modules()]
    try:
        for root in roots:
            root.eval()
        with torch.inference_mode():
            hidden = torch.empty((np.prod(shape), model.hidden_size), dtype=prepared["raw"].dtype)
            for start in range(0, len(hidden), batch_size):
                cells = torch.arange(start, min(start+batch_size, len(hidden)))
                hidden[start:start+len(cells)] = model._hidden_cells(prepared, cells)
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("Nonfinite frozen hidden state")
        return hidden.reshape(*shape, model.hidden_size).numpy().copy()
    finally:
        for module, state in states:
            module.training = state


def basis_statistics(hidden, readout, anchors, floor, batch_size=2048):
    """Use the exact parent projection batching and anchor reductions."""
    shape = hidden.shape[:2]
    flat = torch.as_tensor(hidden.reshape(-1, hidden.shape[-1]))
    projection = torch.as_tensor(readout, dtype=torch.float64)
    raw = torch.empty((len(flat), 2), dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            end = min(start+batch_size, len(flat))
            raw[start:end] = flat[start:end].double() @ projection
        raw = raw.reshape(*shape, 2)
        basis, stats = anchor_normalize(raw, torch.as_tensor(anchors),
                                       scale_floor=floor, return_stats=True)
        anchor_raw = raw[:, anchors]
        mean = anchor_raw.mean(1, keepdim=True)
        scale = (anchor_raw-mean).square().mean((1, 2)).clamp_min(floor**2).sqrt()
    return {"basis": basis.numpy().copy(), "raw_basis": raw.numpy().copy(),
            "station_anchor_mean": mean[:, 0].numpy().copy(),
            "station_rms": stats["station_rms"], "station_scale": scale.numpy().copy(),
            "floor_hit": stats["floor_hit"]}


def external_name(name):
    if name.startswith("off_integrated_"):
        return f"off_{name.removeprefix('off_integrated_')}_integrated"
    return f"{name}_direct"


def check_controls(frame, previous):
    checks = []
    for basis, parent_basis in (("constant", "constant"), ("legacy", "legacy"),
                                ("refreshed_fixed", "refreshed")):
        for stage in ("direct", "integrated"):
            name = f"off_{basis}_{stage}"
            parent_name = f"off{'_integrated' if stage == 'integrated' else ''}_{parent_basis}"
            for k in KS:
                current = frame[frame.model_name.eq(name) & frame.k.eq(k)].sort_values("cell")
                parent = previous[previous.model_name.eq(parent_name) & previous.k.eq(k)].sort_values("cell")
                for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    np.testing.assert_array_equal(current[column], parent[column])
                checks.append({"model_name": name, "parent_model": parent_name, "k": k,
                               "rows": len(current), "bitwise_exact": True})
    for stage in ("direct", "integrated"):
        for k in (0, 1):
            reference = frame[frame.model_name.eq(f"off_legacy_{stage}") & frame.k.eq(k)].sort_values("cell")
            for basis in SHAPES:
                current = frame[frame.model_name.eq(f"off_{basis}_{stage}") & frame.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(current.y_pred, reference.y_pred)
                np.testing.assert_array_equal(current.cell, reference.cell)
    return checks


def run_one(root, prior_root, partition, seed, runtime, *, epochs, patience, batch_size):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, _, dataset, split, parent_full = read_source(prior)
    expert_run = Path(pc["prior_run"])
    mc = json.loads((expert_run / "config.json").read_text())
    verify_files(expert_run, "complete.json", mc)
    if sha256_file(expert_run / "complete.json") != pc["prior_completion_hash"]:
        raise ValueError("Selected native expert parent changed")
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["batch_size"] != batch_size
                or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved readout differs from frozen inputs/settings")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    shape, months = truth.shape, truth.shape[1]
    context, native = parent_full.context_pred.to_numpy(), parent_full.off_pred.to_numpy()
    memory = parent_full.ecological_memory.to_numpy()
    basis_run, oof_run = Path(pc["basis_run"]), Path(mc["oof_run"])
    verify_files(oof_run, "complete.json", json.loads((oof_run / "config.json").read_text()))
    if sha256_file(oof_run / "complete.json") != mc["oof_completion_hash"]:
        raise ValueError("Source forest OOF parent changed")
    temporal = EpisodicTemporalAdapter.from_payload(torch.load(basis_run / "memory.pt", weights_only=False))
    daily, _, daily_bindings = load_daily_pack(Path(pc["daily_features_path"]).parent,
                                              pc["dataset_hash"], shape)
    model = EncoderNativeResidual.from_payload(torch.load(expert_run / "off.pt", weights_only=True))
    config = {"experiment": "doc_daily_hydro_readout_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "expert_run": str(expert_run), "expert_completion_hash": sha256_file(expert_run / "complete.json"),
        **{key: pc[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "source_run",
            "source_completion_hash", "basis_run", "basis_completion_hash", "q90_threshold_train", "query_cells")},
        "oof_run": str(oof_run), "oof_completion_hash": mc["oof_completion_hash"], **daily_bindings,
        "parent_checkpoint_hash": sha256_file(expert_run / "off.pt"),
        "basis_checkpoint_hash": sha256_file(basis_run / "memory.pt"),
        "parent_representation_hash": sha256_file(prior / "representations.npz"),
        "parent_full_grid_hash": sha256_file(prior / "full_grid.parquet"),
        "target_analyte": "doc", "target_transform": "log1p", "inference_roles": ["train"],
        "arms": ["off"], "basis_names": SHAPES, "models": MODELS, "k_values": KS,
        "anchor_count": temporal.anchor_count, "scale_floor": temporal.scale_floor,
        "batch_size": batch_size, "hidden_batch_size": 2048, "projection_batch_size": 2048,
        "epochs": epochs, "patience": patience, "learning_rate": .001, "gradient_clip_norm": 1,
        "trainable_parameter_count": 128, "readout_parameterization": "P0+norm(P0)*D; D initialized zero",
        "torch_threads": torch.get_num_threads(), "backbone_retraining": False, "forest_retraining": False,
        "readout_fitting": True, "gamma_k0": "frozen parent source-validation choice",
        "selection_role": "source_validation", "train_k": [3, 5], "train_ridge": [1, 10], "train_alpha": 1,
        "source_objective": "equal-station native query MAE; four K/ridge combinations equal",
        "validation_objective": "pooled query native MAE; four K/ridge combinations equal; epoch0 included",
        "source_base": "OOF forest plus fixed source-trained neural native residual; not fully OOF neural",
        "source_visibility": "receiving station-fold DOC hidden in all source input months",
        "normalization": "same 32 fixed record-wide calendar anchors and station scalar RMS",
        "study_role": "retrospective support readout development on existing partitions"}
    write_json(run / "config.json", config)
    with np.load(oof_run / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    train_mask = np.zeros(shape, dtype=bool)
    train_mask.ravel()[split["train"]] = True
    val_mask = np.zeros(shape, dtype=bool)
    val_mask.ravel()[split["val"]] = True
    if not np.isfinite(oof[train_mask]).all() or not np.isnan(oof[~train_mask]).all():
        raise ValueError("Source OOF grid has incorrect coverage")
    flow = build_causal_flow_features(dataset)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context.reshape(shape),
        flow["full"], n_months=months)
    expert = UnifiedDOCReconstructor.load(Path(pc["source_run"]), dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    source_ids, val_ids = features["source_station_ids"], np.unique(split["val"] // months)
    np.testing.assert_array_equal(source_ids, extra["source_station_ids"])
    source_inputs = {key: features[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    source_inputs["extra"] = np.concatenate((extra["source_extra"], daily[source_ids]), axis=-1)
    full_inputs = {key: features[f"full_{key}"] for key in ("raw", "age", "support")}
    full_inputs["env"], full_inputs["extra"] = features["env"], np.concatenate((extra["full_extra"], daily), axis=-1)
    if np.count_nonzero(source_inputs["raw"][..., 8:10]) or np.count_nonzero(full_inputs["raw"][val_ids, :, 8:10]):
        raise ValueError("Receiving source/validation DOC is visible")
    source_native = np.maximum(0, np.expm1(oof[source_ids]) + model.selected_scale_*model.predict_delta(source_inputs))
    np.savez_compressed(run / "source_training.npz", source_native=source_native,
        source_station_ids=source_ids, validation_station_ids=val_ids,
        source_mask=train_mask[source_ids], validation_mask=val_mask[val_ids])
    print(f"{run.name}: exporting frozen source/full hidden states", flush=True)
    source_hidden = export_hidden(model, source_inputs)
    full_hidden = export_hidden(model, full_inputs)
    write_json(run / "input_definition.json", {
        "source_hidden_sha256": array_digest(source_hidden), "full_hidden_sha256": array_digest(full_hidden),
        "source_hidden_shape": source_hidden.shape, "full_hidden_shape": full_hidden.shape,
        "source_extra_sha256": array_digest(source_inputs["extra"]),
        "source_station_ids": source_ids.tolist(), "validation_station_ids": val_ids.tolist(),
        "forest_only_oof": True, "frozen_neural_source_trained": True})
    readout = EpisodicSupportReadout(temporal.readout, anchor_count=temporal.anchor_count,
        scale_floor=temporal.scale_floor, seed=seed, max_epochs=epochs, patience=patience,
        lr=.001, batch_size=batch_size)
    readout.fit(source_hidden, source_native, truth[source_ids], train_mask[source_ids],
        full_hidden[val_ids], native.reshape(shape)[val_ids], truth[val_ids], val_mask[val_ids],
        selection_role="source_validation",
        progress=lambda row: print(f"{run.name}/readout: {json.dumps(row)}", flush=True))
    torch.save(readout.to_payload(), run / "readout.pt")
    summary = readout.to_dict()
    write_json(run / "readout.json", summary)
    pd.DataFrame(summary["trace"]).to_csv(run / "readout_trace.csv", index=False)
    anchors = temporal.anchor_months(months)
    initial = basis_statistics(full_hidden, temporal.readout, anchors, temporal.scale_floor)
    learned = readout.transform(full_hidden).reshape(*shape, 2)
    payload = readout.to_payload()
    selected_projection = readout.readout_.detach().cpu().numpy()
    selected = basis_statistics(full_hidden, selected_projection, anchors, temporal.scale_floor)
    np.testing.assert_array_equal(learned, selected["basis"])
    np.testing.assert_array_equal(readout.transform_initial(full_hidden).reshape(*shape, 2), initial["basis"])
    with np.load(prior / "representations.npz", allow_pickle=False) as saved:
        shapes = {"constant": saved["constant"].copy(), "legacy": saved["legacy"].copy(),
                  "refreshed_fixed": saved["off_refreshed"].reshape(-1, 2).copy(),
                  "refreshed_learned": learned.reshape(-1, 2)}
        np.testing.assert_array_equal(initial["basis"], saved["off_refreshed"])
    archive = {**shapes, "anchor_months": anchors}
    for name, block in (("refreshed_fixed", initial), ("refreshed_learned", selected)):
        archive.update({f"{name}_{key}": value for key, value in block.items() if key != "basis"})
    np.savez_compressed(run / "representations.npz", **archive)
    write_json(run / "basis_definition.json", {"anchor_count": temporal.anchor_count,
        "scale_floor": temporal.scale_floor, "initial_readout": temporal.readout.numpy().tolist(),
        "selected_readout": selected_projection.tolist(), "readout_fitting": "source support episodes",
        "initialization": "saved v4 readout exactly; zero delta; no PCA/whitening/QR",
        "normalization_role": "retrospective feature-only record normalization",
        "parent_checkpoint_hash": config["parent_checkpoint_hash"], "checkpoint_keys": list(payload)})
    del source_hidden, full_hidden, source_inputs, full_inputs, expert, features, model, extra, flow, daily, oof, readout
    gc.collect()
    direct = fit_adapters({"off": native}, shapes, truth.ravel(), split, months)
    parent_mixers = json.loads((prior / "mixers.json").read_text())
    gamma = parent_mixers["off_integrated_constant"]["gamma_k0"]
    mixers = fit_arm_mixers("off", context, native, memory, shapes, truth.ravel(), split, months, gamma)
    write_json(run / "adapters.json", {external_name(name): model.to_dict() for name, model in direct.items()})
    write_json(run / "mixers.json", {external_name(name): model.to_dict() for name, model in mixers.items()})
    rows = validation_rows("off", context, native, memory, shapes, direct, mixers, truth.ravel(), split, months)
    for row in rows:
        row["model_name"] = external_name(row["model_name"])
    pd.DataFrame(rows).to_csv(run / "source_validation.csv", index=False)
    print(f"{run.name}: readout and source-validation selection finished", flush=True)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = make_predictions(parent_full, {"off": native}, shapes, direct, labels, split, months)
    frame["regional_gamma"] = 0.0
    frame = pd.concat((frame, integrated_frame(parent_full, "off", context, native, memory,
        shapes, mixers, labels, split, months)), ignore_index=True)
    frame["model_name"] = frame.model_name.map(external_name)
    checks = check_controls(frame, pd.read_parquet(prior / "predictions.parquet"))
    write_json(run / "control_checks.json", checks)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite learned-readout product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    (run / "full_grid.parquet").write_bytes((prior / "full_grid.parquet").read_bytes())
    model_files = ["readout.pt", "readout.json", "input_definition.json", "source_training.npz",
                   "representations.npz", "basis_definition.json", "adapters.json", "mixers.json"]
    for filename in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, filename, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-start})
    files = ["config.json", *model_files, "readout_trace.csv", "source_validation.csv", "control_checks.json",
             "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json", "timing.json"]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: complete in {time.monotonic()-start:.1f}s; only128 learned parameters", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime,
                    epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
            gc.collect()


if __name__ == "__main__":
    main()
