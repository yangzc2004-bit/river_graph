"""Train a scalar native-DOC correction on the existing GRU and frozen context."""
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

from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_tail_residual_v1")
PRIOR = Path("experiments/phase4_transfer/unified_doc_spatial_v4")
ARMS = {"native_mae": 1.0, "native_tail": 2.0}
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Code changed; preserve the batch and use a new directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def training_arrays(features, split, truth, oof_z, context):
    """Pass only source/validation labels into the trainable residual."""
    months = truth.shape[1]
    source_ids = features["source_station_ids"]
    val_ids = np.unique(split["val"] // months)
    source = {key: features[f"source_{key}"] for key in ("encoded", "age", "support")}
    validation = {key: features[f"full_{key}"][val_ids] for key in source}
    train_mask = np.zeros(truth.size, dtype=bool)
    train_mask[split["train"]] = True
    _, val_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    val_mask = np.zeros(truth.size, dtype=bool)
    val_mask[val_query] = True
    return (source, np.maximum(0, np.expm1(oof_z[source_ids])), truth[source_ids],
            train_mask.reshape(truth.shape)[source_ids], validation,
            context.reshape(truth.shape)[val_ids], truth[val_ids],
            val_mask.reshape(truth.shape)[val_ids])


def bind_product(run, filename, config, runtime, model_files):
    path = run / filename
    write_json(path.with_suffix(".meta.json"), {
        "config": config, "config_hash": digest(config), "runtime_snapshot_hash": runtime,
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime),
        "prediction_sha256": sha256_file(path),
        "model_files": {name: sha256_file(run / name) for name in model_files},
        "rows": len(pd.read_parquet(path)), "selection_role": "source_validation",
    })


def run_one(root, prior_root, split_seed, seed, runtime, *, epochs, patience):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{split_seed}_seed{seed}"
    prior_config = json.loads((prior / "config.json").read_text())
    verify_files(prior, "complete.json", prior_config)
    source = Path(prior_config["source_run"])
    old_config, _, dataset, split, full = read_source(source)
    v3 = Path(prior_config["prior_run"])
    v3_config = json.loads((v3 / "config.json").read_text())
    verify_files(v3, "complete.json", v3_config)
    if (sha256_file(source / "complete.json") != prior_config["source_completion_hash"]
            or sha256_file(v3 / "complete.json") != prior_config["prior_completion_hash"]):
        raise ValueError("Prior source package changed")
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Cached run settings or sources changed")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "doc_tail_residual_v1", "split_seed": split_seed, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "source_run": str(source), "source_completion_hash": sha256_file(source / "complete.json"),
        "oof_run": str(v3), "oof_completion_hash": sha256_file(v3 / "complete.json"),
        **{key: old_config[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                          "q90_threshold_train", "query_cells")},
        "target_analyte": "doc", "residual_scale": "native mg/L", "inference_roles": ["train"],
        "models": MODELS, "k_values": [0, 1, 3, 5], "lookback": 12,
        "epochs": epochs, "patience": patience, "learning_rate": 1e-4,
        "head_learning_rate": 1e-3, "batch_size": 512, "gradient_clip_norm": 1,
        "arms": ARMS, "residual_scales": [0, .25, .5, 1],
        "epoch_selection": "source-validation K0 pooled query MAE; epoch0 included",
        "trainable": ["original GRUCell", "observation_decay", "zero-initialized scalar head"],
        "support_basis": "frozen v4 GRU; same basis for every arm",
        "alpha_grid": [0, .25, .5, .75, 1], "ridge_grid": [.1, 1, 10, "infinity"],
        "torch_threads": torch.get_num_threads(),
        "study_role": "development on previously evaluated station partitions",
    }
    write_json(run / "config.json", config)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months = truth.shape[1]
    context = full.context_pred.to_numpy()
    with np.load(v3 / "source_oof.npz", allow_pickle=False) as saved:
        oof_z = saved["pred_z"].copy()
    train_mask = np.zeros(truth.size, dtype=bool)
    train_mask[split["train"]] = True
    if (not np.isfinite(oof_z.ravel()[train_mask]).all()
            or not np.isnan(oof_z.ravel()[~train_mask]).all()):
        raise ValueError("OOF predictions must cover exactly the source-training cells")
    with np.load(prior / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    expert = UnifiedDOCReconstructor.load(source, dataset, split)
    features = extract_temporal_inputs(expert, dataset, split,
        progress=lambda row: print(f"{run.name}: {row}", flush=True))
    inputs = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
    arrays = training_arrays(features, split, truth, oof_z, context)
    bases = {"context": context}
    components = full[["cell", "station", "month", "analyte", "visibility_role"]].copy()
    components["context_pred"] = context
    model_files = ["adapters.json"]
    for name, tail_weight in ARMS.items():
        model = NativeTemporalResidual(
            expert.residual.model.temporal, expert.residual.model.decay, seed=seed,
            epochs=epochs, patience=patience, tail_weight=tail_weight,
        ).fit(*arrays, tail_threshold=config["q90_threshold_train"],
              selection_role="source_validation",
              progress=lambda row, arm=name: print(f"{run.name}/{arm}: {json.dumps(row)}", flush=True))
        torch.save(model.to_payload(), run / f"{name}.pt")
        write_json(run / f"{name}.json", model.to_dict())
        pd.DataFrame(model.to_dict()["trace"]).to_csv(run / f"{name}_trace.csv", index=False)
        delta = model.predict_delta(inputs).ravel()
        scale = model.to_dict()["selected_scale"]
        bases[name] = context.copy() if scale == 0 else np.maximum(0, context + scale * delta)
        components[f"{name}_delta"] = delta
        components[f"{name}_pred"] = bases[name]
        model_files.extend([f"{name}.pt", f"{name}.json"])
    del expert, features, inputs, arrays, oof_z
    gc.collect()
    adapters = fit_adapters(bases, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = make_predictions(full, bases, shapes, adapters, labels, split, months)
    frame["split_seed"], frame["seed"] = split_seed, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_true", "y_pred", "base_pred", "adaptation_delta"]]).all().all():
        raise ValueError("Nonfinite query product")
    previous = pd.read_parquet(prior / "predictions.parquet")
    for name in (f"context_{shape}" for shape in SHAPES):
        a = frame[frame.model_name.eq(name)].sort_values(["k", "cell"])
        b = previous[previous.model_name.eq(name)].sort_values(["k", "cell"])
        np.testing.assert_array_equal(a.y_pred, b.y_pred)
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    for filename in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, filename, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic() - start})
    files = [*model_files, "config.json", "predictions.parquet", "predictions.meta.json",
             "full_grid.parquet", "full_grid.meta.json", "timing.json",
             *(f"{name}_trace.csv" for name in ARMS)]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: complete, {time.monotonic()-start:.1f}s", flush=True)


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
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime,
                    epochs=args.epochs, patience=args.patience)
            gc.collect()


if __name__ == "__main__":
    main()
