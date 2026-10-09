"""Fit conditional density heads on the frozen selected DOC expert features."""
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
from run_doc_daily_hydro_memory_v1 import KS, load_daily_pack
from run_doc_daily_hydro_readout_v1 import array_digest
from run_doc_daily_hydro_support_basis_v1 import (
    fit_arm_mixers,
    integrated_frame,
    validation_rows,
)
from run_doc_ecological_transfer_v1 import load_inputs
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.distributional_residual_head import DistributionalResidualHead
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_distribution_head_v1")
PRIOR = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
ARMS = ("point", "single", "mixture")
SHAPES = ("constant", "gru_tuned_anchor")
INTERACTION_INDICES = (0, 2, 4, 28, 30, 31, 32)
FEATURE_DIM = 550
FEATURE_BATCH_SIZE = 512
MODELS = (tuple(f"{arm}_{basis}" for arm in ("context", *ARMS) for basis in SHAPES)
          + tuple(f"{arm}_integrated_{basis}" for arm in ARMS for basis in SHAPES))


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_daily_hydro_memory_v1.py", "scripts/run_doc_daily_hydro_readout_v1.py",
                 "scripts/run_doc_daily_hydro_support_basis_v1.py", "scripts/run_doc_distribution_head_v1.py",
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


def head_feature_blocks(model, inputs, selected_cells=None):
    """Reproduce the actual native head vectors in its original512-cell batches.

    Selected cells are flat identities in this input view. Evaluate their full
    parent batches, then filter, preserving parent hidden/head arithmetic.
    """
    prepared = model._prepare_inputs(inputs)
    n, months = prepared["age"].shape
    selected = None
    if selected_cells is not None:
        selected = np.zeros(n*months, dtype=bool)
        selected[np.asarray(selected_cells, dtype=np.int64)] = True
    for module in (model.spatial, model.temporal, model.decay, model.head):
        module.eval()
    with torch.inference_mode():
        for start in range(0, n*months, FEATURE_BATCH_SIZE):
            end = min(start+FEATURE_BATCH_SIZE, n*months)
            if selected is not None and not selected[start:end].any():
                continue
            cells = torch.arange(start, end)
            hidden = model._hidden_cells(prepared, cells)
            extra = prepared["extra"][cells//months, cells%months]
            interaction = (hidden[:, :, None]*extra[:, INTERACTION_INDICES][:, None, :]).flatten(1)
            vectors = torch.cat((hidden, extra, interaction), dim=1)
            delta = model.head(vectors).squeeze(-1).double()
            if vectors.shape[1] != FEATURE_DIM or not torch.isfinite(vectors).all() or not torch.isfinite(delta).all():
                raise FloatingPointError("Invalid native-head feature block")
            chosen = np.ones(len(cells), dtype=bool) if selected is None else selected[start:end]
            yield cells.numpy()[chosen], vectors.numpy()[chosen].copy(), delta.numpy()[chosen].copy()


def observed_features(model, inputs, cells):
    blocks = list(head_feature_blocks(model, inputs, cells))
    identities = np.concatenate([block[0] for block in blocks])
    np.testing.assert_array_equal(identities, np.sort(cells))
    return np.concatenate([block[1] for block in blocks]), np.concatenate([block[2] for block in blocks])


def fit_density_mixers(context, bases, memory, shapes, truth, split, months):
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    mixers = {}
    for arm in ARMS:
        scores = []
        for gamma in (0, .25, .5, 1):
            predicted = bases[arm][query] if gamma == 0 else np.maximum(0,
                context[query]+(1-gamma)*(bases[arm][query]-context[query])+gamma*memory[query])
            scores.append({"gamma": gamma, "mae": float(np.abs(predicted-truth[query]).mean())})
        gamma = min(scores, key=lambda row: (row["mae"], row["gamma"]))["gamma"]
        mixers.update(fit_arm_mixers(arm, context, bases[arm], memory, shapes, truth, split, months, gamma))
    return mixers


def point_checks(frame, previous, adapters, mixers, prior):
    old_adapters, old_mixers = json.loads((prior / "adapters.json").read_text()), json.loads((prior / "mixers.json").read_text())
    checks = []
    for basis in SHAPES:
        if adapters[f"point_{basis}"].to_dict() != old_adapters[f"off_{basis}"]:
            raise ValueError("Unchanged point adapter differs")
        if mixers[f"point_integrated_{basis}"].to_dict() != old_mixers[f"off_integrated_{basis}"]:
            raise ValueError("Unchanged point mixture differs")
        for suffix in ("", "_integrated"):
            for k in KS:
                a = frame[frame.model_name.eq(f"point{suffix}_{basis}") & frame.k.eq(k)].sort_values("cell")
                b = previous[previous.model_name.eq(f"off{suffix}_{basis}") & previous.k.eq(k)].sort_values("cell")
                for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    np.testing.assert_array_equal(a[column], b[column])
                checks.append({"model_name": f"point{suffix}_{basis}", "k": k,
                               "rows": len(a), "bitwise_exact": True})
    return checks


def run_one(root, prior_root, partition, seed, runtime, *, epochs, patience):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, sources, old, dataset, split, full, parent_full, oof = load_inputs(prior)
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved density package differs from frozen settings")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    shape, months = truth.shape, truth.shape[1]
    context, native = full.context_pred.to_numpy(), parent_full.off_pred.to_numpy()
    memory = parent_full.ecological_memory.to_numpy()
    daily, _, daily_identity = load_daily_pack(Path(pc["daily_features_path"]).parent, pc["dataset_hash"], shape)
    config = {"experiment": "doc_distribution_head_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{f"{key}_run": str(path) for key, path in sources.items()},
        **{f"{key}_completion_hash": sha256_file(path / "complete.json") for key, path in sources.items()},
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train", "query_cells")},
        **daily_identity, "parent_checkpoint_hash": sha256_file(prior / "off.pt"),
        "parent_full_grid_hash": sha256_file(prior / "full_grid.parquet"),
        "target_analyte": "doc", "target_transform": "log1p", "inference_roles": ["train"],
        "selection_role": "source_validation", "models": MODELS, "arms": ARMS, "basis_names": SHAPES,
        "k_values": KS, "feature_dim": FEATURE_DIM, "feature_batch_size": FEATURE_BATCH_SIZE,
        "extra_dim": 38, "interaction_indices": INTERACTION_INDICES,
        "feature_definition": pc["feature_definition"], "epochs": epochs, "patience": patience,
        "learning_rate": .001, "batch_size": 512, "gradient_clip_norm": 1,
        "torch_threads": torch.get_num_threads(), "sigma_floor": .03, "sigma_initial_floor": .05,
        "feature_std_floor": 1e-6, "median_iterations": 64, "median_bracket_sigma": 12,
        "correction_scales": [0, .25, .5, 1], "head_parameters": {"single": 1102, "mixture": 2755},
        "backbone_retraining": False, "forest_retraining": False, "readout_fitting": False,
        "source_training": "unweighted log-residual Gaussian/mixture NLL; no new tail weights",
        "point_selection": "held source-validation K0 query native MAE; epoch0 and exact point fallback",
        "source_base": "station OOF forest plus frozen source-trained neural correction; forest only OOF",
        "point_estimate": "density median followed by nonnegative native inverse; not mixture mean",
        "support_basis": "unchanged v4 GRU and constant", "mixture_role": "conditional statistical components",
        "study_role": "conditional density-head DOC development on existing spatial partitions"}
    write_json(run / "config.json", config)
    (run / "feature_definition.json").write_bytes((prior / "feature_definition.json").read_bytes())
    model = EncoderNativeResidual.from_payload(torch.load(prior / "off.pt", weights_only=True))
    for module in (model.spatial, model.temporal, model.decay, model.head):
        module.requires_grad_(False)
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    raw = extract_raw_temporal_inputs(expert, dataset, split)
    flow = build_causal_flow_features(dataset)
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
    print(f"{run.name}: extracting frozen550-dimensional observed head features", flush=True)
    source_features, source_delta = observed_features(model, source_inputs, local_cells)
    val_features, _ = observed_features(model, full_inputs, val_cells)
    source_base = np.maximum(0, np.expm1(oof.ravel()[source_cells])+model.selected_scale_*source_delta)
    np.savez_compressed(run / "source_training.npz", source_cells=source_cells,
        source_local_cells=local_cells, source_station_ids=source_ids, source_base=source_base,
        validation_cells=val_cells, validation_base=native[val_cells])
    write_json(run / "input_definition.json", {"feature_dim": FEATURE_DIM,
        "feature_order": "hidden64,extra38,hidden-major64x7 interactions",
        "source_feature_sha256": array_digest(source_features), "validation_feature_sha256": array_digest(val_features),
        "source_base_sha256": array_digest(source_base), "source_cells_sha256": array_digest(source_cells),
        "source_feature_shape": source_features.shape, "validation_feature_shape": val_features.shape,
        "source_station_ids": source_ids.tolist(), "source_neural_is_oof": False,
        "source_visibility": "receiving station-fold hidden", "full_visibility": "train DOC only"})
    densities = {}
    for arm, count in (("single", 1), ("mixture", 2)):
        density = DistributionalResidualHead(n_features=FEATURE_DIM, components=count,
            epochs=epochs, patience=patience, batch_size=512, seed=seed, learning_rate=.001).fit(
                source_features, source_base, truth.ravel()[source_cells], val_features,
                native[val_cells], truth.ravel()[val_cells], selection_role="source_validation",
                progress=lambda row, name=arm: print(f"{run.name}/{name}: {json.dumps(row)}", flush=True))
        densities[arm] = density
        torch.save(density.to_payload(), run / f"{arm}.pt")
        write_json(run / f"{arm}.json", density.to_dict())
        pd.DataFrame(density.to_dict()["trace"]).to_csv(run / f"{arm}_trace.csv", index=False)
    del expert, raw, source_inputs, source_features, val_features, extra, flow, oof, daily
    gc.collect()
    bases = {"context": context, "point": native}
    components = full[["cell", "station", "month", "analyte", "visibility_role",
                        "ecological_novelty", "upstream_support"]].copy()
    components["context_pred"], components["ecological_memory"], components["point_pred"] = context, memory, native
    distributions = {arm: {"weights": np.empty((truth.size, density.components)),
        "means": np.empty((truth.size, density.components)), "scales": np.empty((truth.size, density.components)),
        "median_residual": np.empty(truth.size), "y_pred": np.empty(truth.size)} for arm, density in densities.items()}
    for cells, vectors, _ in head_feature_blocks(model, full_inputs):
        for arm, density in densities.items():
            block = density.predict_distribution(vectors)
            for name in ("weights", "means", "scales", "median_residual"):
                distributions[arm][name][cells] = block[name]
            distributions[arm]["y_pred"][cells] = density.predict(vectors, native[cells])
    del model, full_inputs
    gc.collect()
    for arm, block in distributions.items():
        bases[arm] = block["y_pred"]
        components[f"{arm}_pred"], components[f"{arm}_median_residual"] = block["y_pred"], block["median_residual"]
        for index in range(densities[arm].components):
            for key in ("weights", "means", "scales"):
                components[f"{arm}_{key}_{index}"] = block[key][:, index]
    shapes = shapes_for(prior, sources["basis"])
    adapters = fit_adapters(bases, shapes, truth.ravel(), split, months)
    mixers = fit_density_mixers(context, bases, memory, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: value.to_dict() for name, value in adapters.items()})
    write_json(run / "mixers.json", {name: value.to_dict() for name, value in mixers.items()})
    rows = []
    for arm in ARMS:
        rows.extend(validation_rows(arm, context, bases[arm], memory, shapes, adapters,
                                    mixers, truth.ravel(), split, months))
        components[f"{arm}_integrated_k0_pred"] = mixers[f"{arm}_integrated_constant"].selected_base(
            context, bases[arm], memory, k=0)
    pd.DataFrame(rows).to_csv(run / "source_validation.csv", index=False)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = make_predictions(full, bases, shapes, adapters, labels, split, months)
    frame["regional_gamma"] = 0.0
    blocks = [frame]
    for arm in ARMS:
        blocks.append(integrated_frame(full, arm, context, bases[arm], memory, shapes,
                                       mixers, labels, split, months))
    frame = pd.concat(blocks, ignore_index=True)
    write_json(run / "point_checks.json", point_checks(frame,
        pd.read_parquet(prior / "predictions.parquet"), adapters, mixers, prior))
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite density-head query product")
    if not np.isfinite(components.select_dtypes(include="number")).all().all():
        raise ValueError("Nonfinite density-head full-grid product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    model_files = ["single.pt", "mixture.pt", "single.json", "mixture.json", "adapters.json",
        "mixers.json", "feature_definition.json", "source_training.npz", "input_definition.json"]
    for name in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, name, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-start})
    files = ["config.json", *model_files, "single_trace.csv", "mixture_trace.csv", "source_validation.csv",
        "point_checks.json", "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json", "timing.json"]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: complete in {time.monotonic()-start:.1f}s; two new density heads", flush=True)


def shapes_for(prior, basis_run):
    with np.load(basis_run / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    if not (prior / "complete.json").exists():
        raise ValueError("Missing parent identity")
    return shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
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
