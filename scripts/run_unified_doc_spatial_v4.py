"""Train the existing DOC recurrent state for support-to-query reconstruction."""
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
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source
from run_unified_doc_spatial_v3 import load_fusion

from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.episodic_station_adapter import EpisodicStationProjector
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v4")
PRIOR = Path("experiments/phase4_transfer/unified_doc_spatial_v3")
KS = (0, 1, 3, 5)
HEADS = ("constant", "gru_episodic", "tree_episodic", "gru_frozen_anchor", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{head}" for base in ("context", "fusion") for head in HEADS)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_unified_doc_spatial_v3.py",
                 "scripts/run_unified_doc_spatial_v4.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution code changed: preserve this batch and use a new directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def readout_matrix(state):
    projector = EpisodicStationProjector.from_dict(state)
    basis = projector.basis_
    weights = basis.components_.T / np.sqrt(np.maximum(basis.eigenvalues_, basis.eigenvalue_floor))
    weights[:, ~projector.active_modes_] = 0
    return weights @ projector.projection_.T


def fit_memory(expert, features, oof_z, truth, split, context, readout, *, seed,
               epochs, patience, progress=None):
    """Only source and validation arrays reach recurrent fitting."""
    months = truth.shape[1]
    source_ids = features["source_station_ids"]
    validation_ids = np.unique(split["val"] // months)
    masks = {}
    for role in ("train", "val"):
        mask = np.zeros(truth.size, dtype=bool)
        mask[split[role]] = True
        masks[role] = mask.reshape(truth.shape)
    source = {key: features[f"source_{key}"] for key in ("encoded", "age", "support")}
    validation = {key: features[f"full_{key}"][validation_ids] for key in source}
    model = EpisodicTemporalAdapter(
        expert.temporal, expert.decay, readout, lookback=12, epochs=epochs,
        patience=patience, learning_rate=1e-4, batch_size=8, seed=seed,
        anchor_count=32, scale_floor=1e-4,
    )
    model.fit(source, oof_z[source_ids], truth[source_ids], masks["train"][source_ids],
              validation, context.reshape(truth.shape)[validation_ids],
              truth[validation_ids], masks["val"][validation_ids],
              selection_role="source_validation", progress=progress)
    return model


def run_one(root, prior_root, split_seed, seed, runtime, *, epochs, patience):
    started = time.monotonic()
    prior = prior_root / "runs" / f"split{split_seed}_seed{seed}"
    previous_config = json.loads((prior / "config.json").read_text())
    verify_files(prior, "complete.json", previous_config)
    source = Path(previous_config["source_run"])
    old_config, _, dataset, split, full = read_source(source)
    if sha256_file(source / "complete.json") != previous_config["source_completion_hash"]:
        raise ValueError("Source expert package changed")
    fusion_run = Path(previous_config["fusion_run"])
    if sha256_file(fusion_run / "complete.json") != previous_config["fusion_completion_hash"]:
        raise ValueError("Frozen fusion package changed")
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved run does not match current settings or sources")
        verify_files(run, "complete.json", config)
        print(f"{prior.name}: verified completed experiment", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "unified_doc_spatial_v4", "split_seed": split_seed, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "source_run": str(source), "source_completion_hash": sha256_file(source / "complete.json"),
        "fusion_run": str(fusion_run), "fusion_completion_hash": sha256_file(fusion_run / "complete.json"),
        **{key: old_config[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                          "q90_threshold_train", "query_cells")},
        "target_analyte": "doc", "target_transform": "log1p", "inference_roles": ["train"],
        "models": MODELS, "k_values": KS, "lookback": 12,
        "epochs": epochs, "patience": patience, "learning_rate": 1e-4, "batch_size": 8,
        "gradient_clip_norm": 1, "anchor_count": 32, "scale_floor": 1e-4,
        "trainable": ["GRUCell", "observation_decay"], "readout": "fixed v3 GRU supervised projection",
        "training_K": [3, 5], "training_ridge": [1, 10], "training_alpha": 1,
        "training_loss": "raw query MAE, equal source station, K and ridge",
        "epoch_selection": "source-validation pooled query MAE; epoch0 included",
        "final_alpha_grid": [0, .25, .5, .75, 1], "final_ridge_grid": [.1, 1, 10, "infinity"],
        "normalization": "station scalar RMS over fixed calendar anchors; retrospective",
        "torch_threads": torch.get_num_threads(),
        "study_role": "development on previously evaluated station partitions",
    }
    write_json(run / "config.json", config)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months = truth.shape[1]
    context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
    fusion, combined, fusion_val = load_fusion(fusion_run, context, temporal, split, months)
    write_json(run / "fusion.json", fusion.to_dict())
    with np.load(prior / "source_oof.npz", allow_pickle=False) as cache:
        oof_z = cache["pred_z"].copy()
    train_mask = np.zeros(truth.size, dtype=bool)
    train_mask[split["train"]] = True
    if not np.isfinite(oof_z.ravel()[train_mask]).all() or not np.isnan(oof_z.ravel()[~train_mask]).all():
        raise ValueError("OOF labels do not cover exactly source training cells")
    with np.load(prior / "representations.npz", allow_pickle=False) as cache:
        shapes = {key: cache[key].copy() for key in HEADS[:3]}
    readout = readout_matrix(json.loads((prior / "gru_projector.json").read_text()))
    experts = UnifiedDOCReconstructor.load(source, dataset, split)
    features = extract_temporal_inputs(experts, dataset, split,
        progress=lambda row: print(f"{prior.name}: {row}", flush=True))

    def progress(row):
        print(f"{prior.name}: {json.dumps(row, allow_nan=False)}", flush=True)

    memory = fit_memory(experts.residual.model, features, oof_z, truth, split, context,
                        readout, seed=seed, epochs=epochs, patience=patience, progress=progress)
    torch.save(memory.to_payload(), run / "memory.pt")
    write_json(run / "memory.json", memory.to_dict())
    pd.DataFrame(memory.to_dict()["trace"]).to_csv(run / "training_trace.csv", index=False)
    full_inputs = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
    shapes["gru_frozen_anchor"] = memory.transform_initial(full_inputs).reshape(-1, 2)
    shapes["gru_tuned_anchor"] = memory.transform(full_inputs).reshape(-1, 2)
    write_json(run / "normalization.json", {
        "initial": memory.normalization_stats(full_inputs, initial=True),
        "selected": memory.normalization_stats(full_inputs),
    })
    reload_model = EpisodicTemporalAdapter.from_payload(torch.load(run / "memory.pt", weights_only=False))
    small_inputs = {key: value[:2] for key, value in full_inputs.items()}
    np.testing.assert_array_equal(memory.transform(small_inputs), reload_model.transform(small_inputs))
    del experts, features, full_inputs, reload_model, oof_z
    gc.collect()
    adapters = fit_adapters({"context": context, "fusion": fusion_val}, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    np.savez_compressed(run / "representations.npz", **shapes)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = make_predictions(full, {"context": context, "fusion": combined}, shapes,
                             adapters, labels, split, months)
    frame["split_seed"], frame["seed"] = split_seed, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "adaptation_delta"]].to_numpy()).all():
        raise ValueError("Nonfinite prediction product")
    previous = pd.read_parquet(prior / "predictions.parquet")
    for base in ("context", "fusion"):
        for head in HEADS[:3]:
            name = f"{base}_{head}"
            for k in KS:
                a = previous[previous.model_name.eq(name) & previous.k.eq(k)].sort_values("cell")
                b = frame[frame.model_name.eq(name) & frame.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(a.cell, b.cell)
                np.testing.assert_array_equal(a.y_pred, b.y_pred)
    frame.to_parquet(run / "predictions.parquet", index=False)
    state_names = ("fusion.json", "adapters.json", "memory.pt", "memory.json",
                   "normalization.json", "representations.npz")
    write_json(run / "predictions.meta.json", {
        "config": config, "config_hash": digest(config), "runtime_snapshot_hash": runtime,
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime),
        "prediction_sha256": sha256_file(run / "predictions.parquet"),
        "model_files": {name: sha256_file(run / name) for name in state_names},
        "rows": len(frame), "selection_role": "source_validation",
    })
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic() - started})
    bind_files(run, "complete.json", [run / name for name in (*state_names, "config.json",
               "predictions.parquet", "predictions.meta.json", "training_trace.csv", "timing.json")], config)
    print(f"{prior.name}: complete, {time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for split_seed in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, split_seed, seed, runtime,
                    epochs=args.epochs, patience=args.patience)
            gc.collect()


if __name__ == "__main__":
    main()
