"""Tune the existing DOC encoder together with its native residual memory."""
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
from run_doc_ecological_transfer_v1 import load_inputs
from run_doc_ecological_transfer_v2 import validation_episodes
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_encoder_residual_v1")
PRIOR = Path("experiments/phase4_transfer/doc_regime_residual_v1")
ARMS = ("frozen", "last_self", "last_self_ecology")
SHAPES = ("constant", "gru_tuned_anchor")
KS = (0, 1, 3, 5)
REFERENCE_MODELS = tuple(f"prior_{base}_{shape}" for base in ("concentration", "ecological_affine") for shape in SHAPES)
MODELS = (tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
          + tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
          + REFERENCE_MODELS)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_encoder_residual_v1.py", str(ROOT / "study_plan.md")):
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


def training_arrays(features, split, truth, oof, context):
    months = truth.shape[1]
    source_ids = features["source_station_ids"]
    val_ids = np.unique(split["val"] // months)
    source = {key: features[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    validation = {key: features[f"full_{key}"][val_ids] for key in ("raw", "age", "support")}
    validation["env"] = features["env"][val_ids]
    train_mask = np.zeros(truth.size, dtype=bool)
    train_mask[split["train"]] = True
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    val_mask = np.zeros(truth.size, dtype=bool)
    val_mask[query] = True
    return (source, np.maximum(0, np.expm1(oof[source_ids])), truth[source_ids],
            train_mask.reshape(truth.shape)[source_ids], validation,
            context.reshape(truth.shape)[val_ids], truth[val_ids],
            val_mask.reshape(truth.shape)[val_ids])


def fit_mixers(context, bases, memory, shapes, truth, split, months):
    """Select K0 mixture, then joint support/mixing choices on validation only."""
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    mixers = {}
    for arm in ARMS:
        scores = []
        for gamma in (0, .25, .5, 1):
            pred = (bases[arm][query] if gamma == 0 else np.maximum(
                0, context[query] + (1-gamma)*(bases[arm][query]-context[query]) + gamma*memory[query]))
            scores.append({"gamma": gamma, "mae": float(np.abs(pred-truth[query]).mean())})
        gamma_k0 = min(scores, key=lambda row: (row["mae"], row["gamma"]))["gamma"]
        for shape_name, basis in shapes.items():
            episodes = validation_episodes(context, bases[arm], memory, basis, truth, split, months)
            kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
            model = SupportAwareResidualTransfer(months, **kwargs).fit(
                episodes, gamma_k0=gamma_k0, selection_role="source_validation")
            name = f"{arm}_integrated_{shape_name}"
            mixers[name] = model
    return mixers


def integrated_queries(full, context, bases, memory, shapes, mixers, support_truth, split, months):
    frames = []
    for k in KS:
        support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
        for arm in ARMS:
            for shape_name, basis in shapes.items():
                name = f"{arm}_integrated_{shape_name}"
                model = mixers[name]
                pred = model.adapt(context[query], bases[arm][query], memory[query], query,
                    context[support], bases[arm][support], memory[support], support,
                    support_truth[support], basis[query], basis[support], k=k)
                base = model.selected_base(context[query], bases[arm][query], memory[query], k=k)
                frame = full.iloc[query][["cell", "station", "month", "analyte", "visibility_role",
                                         "ecological_novelty", "upstream_support"]].copy()
                frame["model_name"], frame["k"] = name, k
                frame["y_pred"], frame["base_pred"] = pred, base
                frame["adaptation_delta"] = np.log1p(pred)-np.log1p(base)
                frame["support_count"], frame["regional_gamma"] = k, model.selected_gamma(k)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_one(root, prior_root, partition, seed, runtime, *, epochs, patience):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, sources, old, dataset, split, full, prior_full, oof = load_inputs(prior)
    memory_run = Path(pc["memory_run"])
    memory_config = json.loads((memory_run / "config.json").read_text())
    verify_files(memory_run, "complete.json", memory_config)
    if sha256_file(memory_run / "complete.json") != pc["memory_completion_hash"]:
        raise ValueError("Frozen ecological profile package changed")
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved run settings or inputs changed")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months = truth.shape[1]
    context = full.context_pred.to_numpy()
    flow = build_causal_flow_features(dataset)
    train_cells = np.asarray(split["train"])
    selected_oof = np.maximum(0, np.expm1(oof.ravel()[train_cells]))
    extra = build_regime_head_features(dataset["regime"], train_cells, selected_oof,
        context.reshape(truth.shape), flow["full"], n_months=months)
    feature_definition = {key: value for key, value in extra.items()
                          if key not in ("source_extra", "full_extra", "source_station_ids")}
    feature_definition["source_station_ids"] = extra["source_station_ids"].tolist()
    config = {
        "experiment": "doc_encoder_residual_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "memory_run": str(memory_run), "memory_completion_hash": sha256_file(memory_run / "complete.json"),
        "memory_file_hash": sha256_file(memory_run / "ecological_affine.json"),
        **{f"{key}_run": str(path) for key, path in sources.items()},
        **{f"{key}_completion_hash": sha256_file(path / "complete.json") for key, path in sources.items()},
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                    "q90_threshold_train", "query_cells")},
        "target_analyte": "doc", "residual_scale": "native mg/L", "inference_roles": ["train"],
        "models": MODELS, "arms": ARMS, "k_values": KS, "lookback": 12,
        "epochs": epochs, "patience": patience, "learning_rate": 1e-4,
        "head_learning_rate": 1e-3, "batch_size": 512, "gradient_clip_norm": 1,
        "tail_weight": 2, "residual_scales": [0, .25, .5, 1], "extra_dim": 30,
        "feature_definition": feature_definition, "interaction_indices": extra["interaction_indices"]["concentration"],
        "encoder_learning_rate": 1e-5, "encoder_modes": ARMS, "encoder_dropout": "off in every arm",
        "encoder_scope": "no-message self path; empty edges retained",
        "encoder_normalization": "unchanged historical expert input normalization",
        "interaction_order": "hidden-major outer product", "train_memory": True,
        "epoch_selection": "source-validation K0 pooled query MAE; epoch0 included",
        "support_basis": "unchanged frozen v4 GRU; same for all arms",
        "ecological_integration": "frozen v1 ecological_affine profile; source-validation gamma/alpha/ridge",
        "gamma_grid": [0, .25, .5, 1], "alpha_grid": [0, .25, .5, .75, 1],
        "ridge_grid": [.1, 1, 10, "infinity"], "torch_threads": torch.get_num_threads(),
        "full_grid_role": "native neural components and integrated K0 bases; K-dependent queries in predictions.parquet",
        "study_role": "model development after seeing prior spatial-transfer results",
    }
    write_json(run / "config.json", config)
    write_json(run / "feature_definition.json", feature_definition)
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    np.testing.assert_array_equal(features["source_station_ids"], extra["source_station_ids"])
    arrays = training_arrays(features, split, truth, oof, context)
    val_ids = np.unique(split["val"] // months)
    source_inputs = {**arrays[0], "extra": extra["source_extra"]}
    val_inputs = {**arrays[4], "extra": extra["full_extra"][val_ids]}
    full_inputs = {**{key: features[f"full_{key}"] for key in ("raw", "age", "support")},
                   "env": features["env"], "extra": extra["full_extra"]}
    bases = {"context": context}
    components = full[["cell", "station", "month", "analyte", "visibility_role"]].copy()
    components["context_pred"] = context
    profile = EcologicalResidualTransfer.from_dict(json.loads((memory_run / "ecological_affine.json").read_text()))
    memory = profile.predict_delta(context.reshape(truth.shape)).ravel()
    np.testing.assert_array_equal(memory, prior_full.ecological_memory)
    components["ecological_memory"] = memory
    model_files = ["adapters.json", "mixers.json", "feature_definition.json"]
    for arm in ARMS:
        model = EncoderNativeResidual(expert.residual.model.spatial, expert.residual.model.temporal,
            expert.residual.model.decay, encoder_mode=arm, encoder_learning_rate=1e-5,
            seed=seed, epochs=epochs, patience=patience, tail_weight=2, extra_dim=30,
            interaction_indices=extra["interaction_indices"]["concentration"]).fit(
                source_inputs, *arrays[1:4], val_inputs, *arrays[5:],
                tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                progress=lambda row, name=arm: print(f"{run.name}/{name}: {json.dumps(row)}", flush=True))
        torch.save(model.to_payload(), run / f"{arm}.pt")
        write_json(run / f"{arm}.json", model.to_dict())
        pd.DataFrame(model.to_dict()["trace"]).to_csv(run / f"{arm}_trace.csv", index=False)
        delta = model.predict_delta(full_inputs).ravel()
        scale = model.to_dict()["selected_scale"]
        bases[arm] = context.copy() if scale == 0 else np.maximum(0, context + scale*delta)
        components[f"{arm}_delta"], components[f"{arm}_pred"] = delta, bases[arm]
        model_files.extend((f"{arm}.pt", f"{arm}.json"))
    del expert, features, arrays, source_inputs, val_inputs, full_inputs, extra, flow, oof
    gc.collect()
    adapters = fit_adapters(bases, shapes, truth.ravel(), split, months)
    mixers = fit_mixers(context, bases, memory, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    write_json(run / "mixers.json", {name: model.to_dict() for name, model in mixers.items()})
    for arm in ARMS:
        components[f"{arm}_integrated_k0_pred"] = mixers[f"{arm}_integrated_constant"].selected_base(
            context, bases[arm], memory, k=0)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    direct = make_predictions(full, bases, shapes, adapters, labels, split, months)
    direct["regional_gamma"] = 0.0
    integrated = integrated_queries(full, context, bases, memory, shapes, mixers, labels, split, months)
    previous = pd.read_parquet(prior / "predictions.parquet")
    reference_mapping = {f"concentration_{shape}": f"prior_concentration_{shape}" for shape in SHAPES}
    reference_mapping.update({f"prior_ecological_affine_{shape}": f"prior_ecological_affine_{shape}" for shape in SHAPES})
    reference = previous[previous.model_name.isin(reference_mapping)].copy()
    reference["model_name"] = reference.model_name.map(reference_mapping)
    reference = reference.drop(columns=["split_seed", "seed", "y_true"])
    frame = pd.concat([direct, integrated, reference], ignore_index=True)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite regime residual query product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    for filename in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, filename, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-start})
    files = ["config.json", *model_files, "predictions.parquet", "predictions.meta.json",
             "full_grid.parquet", "full_grid.meta.json", "timing.json", *(f"{arm}_trace.csv" for arm in ARMS)]
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
            run_one(args.root, args.prior_root, partition, seed, runtime, epochs=args.epochs, patience=args.patience)
            gc.collect()


if __name__ == "__main__":
    main()
