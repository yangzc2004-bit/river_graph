"""Develop donor-to-local residual contrast in the existing DOC predictor.

This version changes retrieval fusion: a zero-initialized donor contrast is
trained against the current local prediction. V1 remains an independent result.
Only source-training and source-validation labels are supplied to fitting.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_source_retrieval_v1 import (
    hidden_rows,
    inference_retrieval,
    safe_training_arrays,
    selected_query,
)
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.doc_source_retrieval import (
    SourceResidualBank,
    SourceRetrievalAttention,
)
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_source_retrieval_v2")
PRIOR = Path("experiments/phase4_transfer/doc_source_retrieval_v1")


def donor_contrast(prepared, context, base, scale):
    """A donor and local prediction estimate the same forest residual.

    Contrast replaces part of that estimate; it cannot double-add the whole
    forest residual. Shared query design columns are not prediction values.
    """
    result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in prepared.items()}
    result["values"][..., 1:] = 0
    result["values"][..., 0] -= ((np.asarray(base)-context)/scale)[:, None]
    return result


def predict_contrast(model, prepared, *, scale, ablation=None):
    if ablation == "zero_source_values":
        prepared = {**prepared, "values": prepared["values"].copy()}
        prepared["values"][..., 0] = -((prepared["base"]-prepared["context"])/scale)[:, None]
        ablation = None
    return inference_retrieval(model, prepared, scale=scale, ablation=ablation)


def fit_contrast(episodes, validation, *, scale, seed=42, epochs=30, patience=5, progress=None):
    model = SourceRetrievalAttention(episodes[0]["query"].shape[1], seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensors = [{key: torch.as_tensor(item[key], dtype=torch.bool if key == "valid" else torch.float32)
                for key in ("query", "keys", "values", "valid", "base", "truth", "weights")}
               for item in episodes]
    vy, vb = np.asarray(validation["truth"]), np.asarray(validation["base"])
    best, saved, stale, trace, best_epoch = np.inf, None, 0, [], 0

    def evaluate(epoch, training_loss):
        nonlocal best, saved, stale, best_epoch
        model.eval()
        delta, _, _, _ = inference_retrieval(model, validation, scale=scale)
        score = float(np.abs(np.maximum(0, vb+delta)-vy).mean())
        if score < best:
            best, saved, stale, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        row = {"epoch": epoch, "training_loss": training_loss, "validation_mae": score,
               "best_epoch": best_epoch, "head_norm": float(model.projection.weight.detach().norm())}
        trace.append(row)
        if progress:
            progress(row)

    evaluate(0, None)
    for epoch in range(1, epochs+1):
        model.train()
        rng, losses = np.random.default_rng(np.random.SeedSequence([seed, epoch])), []
        for episode in tensors:
            order = rng.permutation(len(episode["truth"]))
            for start in range(0, len(order), 512):
                rows = order[start:start+512]
                optimizer.zero_grad()
                delta = model(*(episode[key][rows] for key in ("query", "keys", "values", "valid")))
                pred = (episode["base"][rows]+scale*delta).clamp_min(0)
                loss = (episode["weights"][rows]*(pred-episode["truth"][rows]).abs()).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
        evaluate(epoch, float(np.mean(losses)))
        if stale >= patience:
            break
    model.load_state_dict(saved)
    return model.eval(), {"selection_role": "source_validation", "best_epoch": best_epoch,
                          "validation_mae": best, "trace": trace, "residual_scale": scale}


def freeze(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_retrieval_v1.py",
                 "scripts/run_doc_source_retrieval_v2.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("saved execution changed")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def run_one(root, partition, seed, runtime, *, epochs=30, bank_run=None,
            experiment_name="doc_source_retrieval_v2"):
    start = time.monotonic()
    parent = PRIOR / "runs" / f"split{partition}_seed{seed}"
    old = json.loads((parent / "config.json").read_text())
    verify_files(parent, "complete.json", old)
    run = root / "runs" / parent.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("cannot reuse changed code")
        verify_files(run, "complete.json", config)
        return
    config = {**{key: old[key] for key in ("split_seed", "seed", "dataset_path", "dataset_hash",
               "mask_path", "mask_hash", "q90_threshold_train", "information_condition", "source_neural_status")},
        "experiment": experiment_name, "parent_run": str(parent),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parent_completion_hash": sha256_file(parent / "complete.json"), "epochs": epochs, "patience": 5,
        "runtime_snapshot_hash": runtime, "evaluation_role": "source_validation_only",
        "models": ["current_model", "matched_daily_trees", "static_memory", "retrieval_contrast",
                   "retrieval_contrast_uniform", "retrieval_contrast_zero_source_values"],
        "main_change": "train a donor-minus-current residual contrast; no query-design value shortcut",
        "donor_bank": "unchanged v1 nested station OOF profiles", "zero_initialization": "current complete prediction"}
    if bank_run is not None:
        bank_config = json.loads((bank_run / "config.json").read_text())
        verify_files(bank_run, "complete.json", bank_config)
        config.update(bank_preparation_run=str(bank_run),
                      bank_preparation_hash=sha256_file(bank_run / "complete.json"),
                      donor_bank=bank_config["donor_bank"])
    write_json(run / "config.json", config)

    def progress(row):
        entry = {"run": run.name, "stage": "contrast_retrieval", **row, "elapsed_seconds": time.monotonic()-start}
        write_json(root / "progress.json", entry)
        print(json.dumps(entry), flush=True)

    dataset = torch.load(config["dataset_path"], map_location="cpu", weights_only=False)
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    original = Path(old["parent_run"])
    original_config = json.loads((original / "config.json").read_text())
    expert = UnifiedDOCReconstructor.load(Path(original_config["source_run"]), dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    full = pd.read_parquet(original / "full_grid.parquet")
    shape, train, val = tuple(dataset["y"].shape), split["train"], split["val"]
    context, base = full.context_pred.to_numpy().reshape(shape), full.daily_integrated_k0_pred.to_numpy()
    months = shape[1]
    with np.load(Path(original_config["oof_run"]) / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    daily, _, _ = load_daily_pack(original.parent.parent, config["dataset_hash"], shape)
    extra = build_regime_head_features(dataset["regime"], train, np.maximum(0, np.expm1(oof.ravel()[train])),
        context, build_causal_flow_features(dataset)["full"], n_months=months)
    source_inputs, full_inputs, _ = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    retained = EncoderNativeResidual.from_payload(torch.load(original / "daily.pt", weights_only=False, map_location="cpu"))
    source_ids = features["source_station_ids"]
    source_cells = np.searchsorted(source_ids, train//months)*months+train % months
    hidden = hidden_rows(retained, source_inputs, source_cells)
    with torch.no_grad():
        prepared_source = retained._prepare_inputs(source_inputs)
        delta = torch.cat([retained._delta_cells(prepared_source, source_cells[i:i+512])
                           for i in range(0, len(source_cells), 512)]).numpy()*retained.selected_scale_
    mixer = json.loads((original / "mixers.json").read_text())["daily_integrated_gru_tuned_anchor"]
    gamma = mixer["gamma_k0"]
    bank_run = parent if bank_run is None else bank_run
    bank = SourceResidualBank.from_dict(json.loads((bank_run / "source_bank.json").read_text()))
    states = json.loads((bank_run / "nested_banks.json").read_text())
    episodes = []
    with np.load(bank_run / "nested_episodes.npz", allow_pickle=False) as saved:
        for i, state in enumerate(states):
            local_bank = SourceResidualBank.from_dict(state)
            cells, local_context = saved[f"query_cells{i}"], saved[f"context{i}"]
            indexes = np.searchsorted(train, cells)
            query = selected_query(local_bank, dataset, cells, local_context, hidden[indexes], residual_scale=bank.residual_scale_)
            static = query["values"][..., 0].mean(1)*bank.residual_scale_
            local = np.maximum(0, local_context+(1-gamma)*delta[indexes]+gamma*static)
            query = donor_contrast(query, local_context, local, bank.residual_scale_)
            _, inverse, counts = np.unique(cells//months, return_inverse=True, return_counts=True)
            episodes.append({**query, "base": local, "truth": np.asarray(dataset["y"]).ravel()[cells],
                             "weights": len(cells)/(len(counts)*counts[inverse])})
    validation = selected_query(bank, dataset, val, context.ravel()[val], hidden_rows(retained, full_inputs, val),
                                residual_scale=bank.residual_scale_)
    validation = donor_contrast(validation, context.ravel()[val], base[val], bank.residual_scale_)
    validation.update(base=base[val], context=context.ravel()[val], truth=np.asarray(dataset["y"]).ravel()[val])
    attention, selection = fit_contrast(episodes, validation, scale=bank.residual_scale_, seed=seed,
        epochs=epochs, progress=progress)
    torch.save(attention.to_payload(), run / "retrieval_contrast.pt")
    write_json(run / "retrieval_contrast.json", selection)
    np.savez_compressed(run / "validation_inputs.npz", **{key: validation[key] for key in
        ("query", "keys", "values", "valid", "base", "context")})
    previous = pd.read_parquet(parent / "predictions.parquet")
    frames = [previous[previous.model_name.isin(config["models"])].copy()]
    template = previous[previous.model_name.eq("current_model")].copy()
    for ablation in (None, "uniform", "zero_source_values"):
        correction, entropy, count, maximum = predict_contrast(attention, validation,
            scale=bank.residual_scale_, ablation=ablation)
        frame = template.copy()
        frame["model_name"] = "retrieval_contrast"+("_"+ablation if ablation else "")
        frame["y_pred"] = np.maximum(0, base[val]+correction)
        frame["retrieval_delta"], frame["retrieval_entropy"] = correction, entropy
        frame["effective_donors"], frame["max_donor_weight"] = count, maximum
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
    files = ["config.json", "retrieval_contrast.pt", "retrieval_contrast.json", "validation_inputs.npz"]
    bind_product(run, "predictions.parquet", config, runtime, files)
    bind_files(run, "complete.json", [run / name for name in
        (*files, "predictions.parquet", "predictions.meta.json")], config)
    progress({"status": "complete", "best_epoch": selection["best_epoch"]})
    del expert, features, retained, episodes, source_inputs, full_inputs
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.root.mkdir(parents=True, exist_ok=True)
    runtime = freeze(args.root)
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, runtime, epochs=args.epochs)


if __name__ == "__main__":
    main()
