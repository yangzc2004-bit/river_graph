"""Test recurrent-state by flow interactions in the existing DOC residual head."""
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
from run_doc_tail_residual_v1 import bind_product, training_arrays
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_flow_interaction_v1")
PRIOR = Path("experiments/phase4_transfer/doc_flow_residual_v1")
ARMS = ("interaction_frozen", "interaction_tuned")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)


def arm_extra(flow, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown flow arm: {arm}")
    values = flow["full"].copy()
    return values


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_flow_interaction_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Code changed; preserve this batch and use a new directory")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def run_one(root, prior_root, partition, seed, runtime, *, epochs, patience):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc = json.loads((prior / "config.json").read_text())
    verify_files(prior, "complete.json", pc)
    sources = {"source": Path(pc["source_run"]), "basis": Path(pc["basis_run"]),
               "oof": Path(pc["oof_run"])}
    for key, path in sources.items():
        expected = pc[f"{key}_completion_hash"]
        if sha256_file(path / "complete.json") != expected:
            raise ValueError(f"Changed {key} package")
        verify_files(path, "complete.json", json.loads((path / "config.json").read_text()))
    old_config, _, dataset, split, full = read_source(sources["source"])
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved configuration or input changed")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    flow = build_causal_flow_features(dataset)
    flow_definition = {key: value for key, value in flow.items() if key != "full"}
    config = {
        "experiment": "doc_flow_interaction_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{f"{key}_run": str(path) for key, path in sources.items()},
        **{f"{key}_completion_hash": sha256_file(path / "complete.json") for key, path in sources.items()},
        **{key: old_config[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                          "q90_threshold_train", "query_cells")},
        "target_analyte": "doc", "residual_scale": "native mg/L", "inference_roles": ["train"],
        "models": MODELS, "k_values": [0, 1, 3, 5], "lookback": 12,
        "epochs": epochs, "patience": patience, "learning_rate": 1e-4,
        "head_learning_rate": 1e-3, "batch_size": 512, "gradient_clip_norm": 1,
        "arms": ARMS, "tail_weight": 2.0, "residual_scales": [0, .25, .5, 1],
        "extra_dim": flow["full"].shape[-1], "flow_definition": flow_definition,
        "interaction_indices": flow["value_feature_indices"],
        "train_memory_by_arm": {"interaction_frozen": False, "interaction_tuned": True},
        "interaction_order": "hidden-major outer product",
        "epoch_selection": "source-validation K0 pooled query MAE; epoch0 included",
        "support_basis": "frozen v4 GRU; same for all arms", "alpha_grid": [0, .25, .5, .75, 1],
        "ridge_grid": [.1, 1, 10, "infinity"], "torch_threads": torch.get_num_threads(),
        "study_role": "development on previously evaluated station partitions",
    }
    write_json(run / "config.json", config)
    write_json(run / "flow_features.json", flow_definition)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months = truth.shape[1]
    context = full.context_pred.to_numpy()
    with np.load(sources["oof"] / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    train_mask = np.zeros(truth.size, dtype=bool)
    train_mask[split["train"]] = True
    if (not np.isfinite(oof.ravel()[train_mask]).all()
            or not np.isnan(oof.ravel()[~train_mask]).all()):
        raise ValueError("OOF predictions do not match exactly the training cells")
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    features = extract_temporal_inputs(expert, dataset, split)
    arrays = training_arrays(features, split, truth, oof, context)
    val_ids = np.unique(split["val"] // months)
    base_inputs = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
    bases = {"context": context}
    components = full[["cell", "station", "month", "analyte", "visibility_role"]].copy()
    components["context_pred"] = context
    for j, name in enumerate(flow["feature_names"]):
        components[f"flow_{name}"] = flow["full"][..., j].ravel()
    model_files = ["adapters.json", "flow_features.json"]
    for arm in ARMS:
        extra = arm_extra(flow, arm)
        source_inputs = {**arrays[0], "extra": extra[features["source_station_ids"]]}
        val_inputs = {**arrays[4], "extra": extra[val_ids]}
        inputs = {**base_inputs, "extra": extra}
        model = NativeTemporalResidual(
            expert.residual.model.temporal, expert.residual.model.decay, seed=seed,
            epochs=epochs, patience=patience, tail_weight=2, extra_dim=extra.shape[-1],
            interaction_indices=flow["value_feature_indices"], train_memory=arm == "interaction_tuned",
        ).fit(source_inputs, *arrays[1:4], val_inputs, *arrays[5:],
              tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
              progress=lambda row, name=arm: print(f"{run.name}/{name}: {json.dumps(row)}", flush=True))
        torch.save(model.to_payload(), run / f"{arm}.pt")
        write_json(run / f"{arm}.json", model.to_dict())
        pd.DataFrame(model.to_dict()["trace"]).to_csv(run / f"{arm}_trace.csv", index=False)
        delta = model.predict_delta(inputs).ravel()
        scale = model.to_dict()["selected_scale"]
        bases[arm] = context.copy() if scale == 0 else np.maximum(0, context + scale * delta)
        components[f"{arm}_delta"] = delta
        components[f"{arm}_pred"] = bases[arm]
        model_files.extend((f"{arm}.pt", f"{arm}.json"))
    del expert, features, inputs, base_inputs, arrays, oof, flow, source_inputs, val_inputs, extra
    gc.collect()
    adapters = fit_adapters(bases, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = make_predictions(full, bases, shapes, adapters, labels, split, months)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_true", "y_pred", "base_pred", "adaptation_delta"]]).all().all():
        raise ValueError("Nonfinite adapted prediction")
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
