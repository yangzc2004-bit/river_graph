"""Develop zero-observation DOC source retrieval using source/validation roles only."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unmonitored_doc import (
    development_labels,
    strip_auxiliary_water,
    validate_zero_observation_view,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.doc_source_data import nested_source_episodes
from river_graph.models.doc_source_retrieval import (
    SourceResidualBank,
    fit_retrieval,
)
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.hydro_pretraining import pretrain_hydro
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_source_retrieval_v1")
PARENT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_source_retrieval_v1.py",
                 "scripts/verify_doc_source_retrieval_v1.py", "scripts/analyze_doc_source_retrieval_v1.py",
                 "scripts/run_doc_daily_hydro_residual_v1.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_unified_doc_spatial.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("preserve this execution and use a new version after code changes")
    else:
        write_json(path, snapshot)
        for name in snapshot:
            destination = root / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def hidden_rows(model, inputs, cells, *, batch=512):
    prepared = model._prepare_inputs(inputs)
    with torch.no_grad():
        return torch.cat([model._hidden_cells(prepared, cells[start:start+batch])
                          for start in range(0, len(cells), batch)]).numpy()


def selected_query(bank, dataset, cells, context, hidden, *, residual_scale):
    months = dataset["y"].shape[1]
    rows, dates = cells//months, cells % months
    prepared = bank.query(np.asarray(dataset["regime"])[rows, 4:13],
        np.asarray(dataset["x"])[rows, dates], np.asarray(dataset["x_mask"])[rows, dates],
        np.asarray(dataset["months"])[dates], context, hidden,
        station_names=np.asarray(dataset["site_no"], str)[rows])
    # Each nested bank has its own source scale; express values in final-bank units.
    ratio = bank.residual_scale_/residual_scale
    prepared["values"][..., 0] *= ratio
    prepared["keys"][..., -6:] *= ratio
    return prepared


def safe_training_arrays(features, extra, daily, dataset, split, oof, context):
    source_ids = features["source_station_ids"]
    months = dataset["y"].shape[1]
    val_ids = np.unique(split["val"]//months)
    source = {key: features[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    source["extra"] = np.concatenate([extra["source_extra"], daily[source_ids]], axis=-1)
    full = {key: features[f"full_{key}"] for key in ("raw", "age", "support")}
    full["env"] = features["env"]
    full["extra"] = np.concatenate([extra["full_extra"], daily], axis=-1)
    validate_zero_observation_view(full, np.unique(np.r_[split["val"], split["test"]]//months))
    validate_zero_observation_view(source, np.arange(len(source_ids)))
    validation = {key: values[val_ids] for key, values in full.items()}
    masks = {}
    for role in ("train", "val"):
        masks[role] = np.zeros(dataset["y"].shape, dtype=bool)
        masks[role].ravel()[split[role]] = True
    return source, full, (source, np.maximum(0, np.expm1(oof[source_ids])), np.asarray(dataset["y"])[source_ids],
        masks["train"][source_ids], validation, context[val_ids], np.asarray(dataset["y"])[val_ids], masks["val"][val_ids])


def inference_retrieval(model, prepared, *, scale, batch=512, ablation=None):
    output, entropy, counts, max_weight = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(prepared["query"]), batch):
            rows = slice(start, start+batch)
            inputs = [torch.as_tensor(prepared[key][rows], dtype=torch.bool if key == "valid" else torch.float32)
                      for key in ("query", "keys", "values", "valid")]
            if ablation == "zero_source_values":
                inputs[2] = inputs[2].clone()
                inputs[2][..., 0] = 0
            delta, diagnostic = model(*inputs, diagnostics=True, uniform=ablation == "uniform")
            output.append(delta.numpy()*scale)
            entropy.append(diagnostic["entropy"].numpy())
            counts.append(diagnostic["effective_donors"].numpy())
            max_weight.append(diagnostic["max_weight"].numpy())
    return np.concatenate(output), np.concatenate(entropy), np.concatenate(counts), np.concatenate(max_weight)


def run_one(root, parent, partition, seed, runtime, *, epochs, hydro_epochs, smoke=False):
    started = time.monotonic()
    previous = parent / "runs" / f"split{partition}_seed{seed}"
    old = json.loads((previous / "config.json").read_text())
    verify_files(previous, "complete.json", old)
    run = root / "runs" / previous.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime:
            raise ValueError("saved execution changed")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: reused complete source-validation study", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    config = {"experiment": "doc_source_retrieval_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "parent_run": str(previous), "parent_completion_hash": sha256_file(previous / "complete.json"),
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train")},
        "selection_role": "source_validation", "evaluation_role": "source_validation_only",
        "query_policy": "all observed validation cells, no support reserved in primary K0",
        "information_condition": "target station has no DOC/pH/conductance inputs",
        "models": ["current_model", "matched_daily_trees", "static_memory", "hydro_pretrained",
                   "retrieval", "retrieval_uniform", "retrieval_zero_source_values",
                   "hydro_pretrained_retrieval", "hydro_pretrained_retrieval_uniform",
                   "hydro_pretrained_retrieval_zero_source_values"],
        "epochs": epochs, "hydro_epochs": hydro_epochs, "patience": 5, "max_donors": 20,
        "retrieval_heads": 2, "retrieval_head_dim": 32, "lookback": 12, "smoke": smoke,
        "bootstrap_unit": "station", "source_neural_status": "source-trained weights; station-fold-hidden inputs",
        "donor_residuals": "inner station cross-fit inside each outer pseudo-target fold",
        "tree_covariates": "same monthly inputs and8daily descriptors; all chemistry absent"}
    if (run / "config.json").exists():
        existing = json.loads((run / "config.json").read_text())
        for key in ("runtime_snapshot_hash", "parent_completion_hash", "epochs", "hydro_epochs", "smoke"):
            if existing[key] != config[key]:
                raise ValueError(f"partial run differs in {key}")
        config = existing
    else:
        write_json(run / "config.json", config)

    def progress(stage, row):
        row = {key: (len(value) if key in ("excluded_stations", "donor_stations") else value)
               for key, value in row.items()}
        record = {"run": run.name, "stage": stage, **row,
                  "elapsed_seconds": time.monotonic()-started}
        write_json(root / "progress.json", record)
        print(json.dumps(record), flush=True)

    dataset = torch.load(config["dataset_path"], map_location="cpu", weights_only=False)
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError(f"changed {kind} input")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(dataset)
    dataset["y"] = development_labels(dataset, split)
    source_run = Path(old["source_run"])
    expert = UnifiedDOCReconstructor.load(source_run, dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    day = pd.read_parquet(previous / "full_grid.parquet")
    shape = tuple(dataset["y"].shape)
    n, months = shape
    context = day.context_pred.to_numpy().reshape(shape)
    temporal = day.daily_pred.to_numpy().reshape(shape)
    current = day.daily_integrated_k0_pred.to_numpy().reshape(shape)
    static = day.ecological_memory.to_numpy().reshape(shape)
    daily, _, _ = load_daily_pack(parent, config["dataset_hash"], shape)
    with np.load(Path(old["oof_run"]) / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    train = np.asarray(split["train"], dtype=np.int64)
    val = np.asarray(split["val"], dtype=np.int64)
    extra = build_regime_head_features(dataset["regime"], train,
        np.maximum(0, np.expm1(oof.ravel()[train])), context,
        build_causal_flow_features(dataset)["full"], n_months=months)
    source_inputs, full_inputs, arrays = safe_training_arrays(features, extra, daily, dataset, split, oof, context)
    retained = EncoderNativeResidual.from_payload(torch.load(previous / "daily.pt", weights_only=False, map_location="cpu"))
    # Retained full predictions replay before fitting any new component.
    val_delta = retained.predict_delta({key: values[np.unique(val//months)] for key, values in full_inputs.items()})
    val_ids = np.unique(val//months)
    rows = np.searchsorted(val_ids, val//months)
    replay = np.maximum(0, context.ravel()[val]+retained.selected_scale_*val_delta[rows, val % months])
    np.testing.assert_allclose(replay, temporal.ravel()[val], rtol=1e-6, atol=1e-5)
    progress("inputs", {"n_source": len(train), "n_validation": len(val), "target_labels_read": False})
    model_files = ["config.json"]

    # The strong tree comparator receives the same available daily-flow inputs.
    tree_features = np.column_stack([build_rf_features(dataset, split, FIT_ROLES,
        target_transform="log1p", include_network=True), daily.reshape(n*months, -1)])
    scores, best_tree, best_score = [], None, np.inf
    for name, leaf, maximum in (("leaf4", 4, 1.), ("leaf2", 2, 1.), ("leaf1", 1, 1.), ("sqrt", 1, "sqrt")):
        tree = ExtraTreesRegressor(n_estimators=12 if smoke else 300, min_samples_leaf=leaf,
                                  max_features=maximum, n_jobs=2, random_state=seed)
        tree.fit(tree_features[train], np.log1p(np.asarray(dataset["y"]).ravel()[train]))
        prediction = np.maximum(0, np.expm1(tree.predict(tree_features[val])))
        score = float(np.abs(prediction-np.asarray(dataset["y"]).ravel()[val]).mean())
        scores.append({"model": name, "validation_mae": score})
        if score < best_score:
            best_tree, best_score, tree_prediction = tree, score, prediction
    joblib.dump(best_tree, run / "matched_daily_trees.joblib", compress=3)
    write_json(run / "tree_selection.json", scores)
    model_files += ["tree_selection.json", "matched_daily_trees.joblib"]
    del tree_features, best_tree, tree

    # Cache independent nested episodes; later operators reuse exactly the same bank.
    if (run / "nested_complete.json").exists():
        verify_files(run, "nested_complete.json", config)
        bank_entries = []
        with np.load(run / "nested_episodes.npz", allow_pickle=False) as saved:
            bank_states = json.loads((run / "nested_banks.json").read_text())
            for i, state in enumerate(bank_states):
                bank_entries.append({"bank": SourceResidualBank.from_dict(state),
                                     "query_cells": saved[f"cells{i}"].copy(), "context": saved[f"context{i}"].copy()})
    else:
        forest = expert.context_forest
        if smoke:
            from sklearn.base import clone
            forest = clone(forest).set_params(n_estimators=12)
        bank_entries, records = nested_source_episodes(forest, dataset, split, expert.rf.folds,
            progress=lambda row: progress("nested_donors", row))
        np.savez_compressed(run / "nested_episodes.npz", **{f"{key}{i}": entry[key]
            for i, entry in enumerate(bank_entries) for key in ("query_cells", "context")})
        write_json(run / "nested_banks.json", [entry["bank"].to_dict() for entry in bank_entries])
        write_json(run / "nested_fold_records.json", records)
        bind_files(run, "nested_complete.json", [run / name for name in
            ("nested_episodes.npz", "nested_banks.json", "nested_fold_records.json")], config)
    bank = SourceResidualBank().fit(np.asarray(dataset["regime"])[:, 4:13], dataset["x"], dataset["x_mask"],
        dataset["months"], train, np.maximum(0, np.expm1(oof.ravel()[train])),
        np.asarray(dataset["y"]).ravel()[train], station_names=dataset["site_no"])
    write_json(run / "source_bank.json", bank.to_dict())
    model_files += ["source_bank.json", "nested_banks.json", "nested_fold_records.json"]
    old_profile = json.loads((Path(old["memory_run"]) / "ecological_affine.json").read_text())
    # A safe nested static reference, with the parent's fixed profile hyperparameters.
    nested_static = []
    for entry in bank_entries:
        donors = np.isin(np.asarray(dataset["site_no"], str), entry["bank"].station_names_)
        donor_cells = train[donors[train//months]]
        # The inner cross-fitted bank profile supplies the episode's source-only reference.
        qcells = entry["query_cells"]
        prepared = selected_query(entry["bank"], dataset, qcells, entry["context"],
            np.zeros((len(qcells), retained.hidden_size)), residual_scale=bank.residual_scale_)
        value = prepared["values"][..., 0]
        nested_static.append(value.mean(1)*bank.residual_scale_)
        assert not np.isin(qcells//months, donor_cells//months).any()
    write_json(run / "static_reference.json", {"parent": old_profile["selected"],
        "episode_reference": "uniform nearest donor profiles using inner OOF residuals",
        "validation_reference": "unchanged parent ecological-affine memory",
        "note": "new attention projection preserves parent memory at zero; episode/reference distribution difference is reported"})
    model_files.append("static_reference.json")

    predictions = {"current_model": current.ravel()[val], "matched_daily_trees": tree_prediction}
    gamma_scores = []
    for gamma in (0., .25, .5, 1.):
        p = np.maximum(0, context.ravel()[val]+(1-gamma)*(temporal.ravel()[val]-context.ravel()[val])+gamma*static.ravel()[val])
        gamma_scores.append({"gamma": gamma, "mae": float(np.abs(p-np.asarray(dataset["y"]).ravel()[val]).mean())})
    gamma = min(gamma_scores, key=lambda row: (row["mae"], row["gamma"]))["gamma"]
    predictions["static_memory"] = np.maximum(0, context.ravel()[val]+(1-gamma)*(temporal.ravel()[val]-context.ravel()[val])+gamma*static.ravel()[val])
    write_json(run / "static_selection.json", gamma_scores)
    model_files.append("static_selection.json")
    source_ids = features["source_station_ids"]
    source_cells = np.searchsorted(source_ids, train//months)*months+train % months
    current_hidden = hidden_rows(retained, source_inputs, source_cells)

    hydro_path = run / "hydro_initialization.pt"
    if (run / "hydro_initialization_complete.json").exists():
        verify_files(run, "hydro_initialization_complete.json", config)
        hydro_model = EncoderNativeResidual.from_payload(torch.load(hydro_path, weights_only=False, map_location="cpu"))
    else:
        hydro_model, pretraining = pretrain_hydro(retained, source_inputs,
            np.asarray(dataset["x"])[source_ids], np.asarray(dataset["x_mask"])[source_ids], seed=seed,
            epochs=hydro_epochs, max_training_cells=512 if smoke else 8192,
            progress=lambda row: progress("hydro_pretraining", row))
        # Checkpoints store the pretrained initialization, not old DOC-fit distances.
        hydro_model._initial_spatial_state = copy_state(hydro_model.spatial)
        hydro_model._initial_temporal_state = copy_state(hydro_model.temporal)
        hydro_model._initial_decay_state = copy_state(hydro_model.decay)
        hydro_model._initial_last_self_state = copy_state(hydro_model.spatial.convs[-1].self_lin)
        hydro_model._initial_ecology_state = copy_state(hydro_model.spatial.env_encoder)
        torch.save(hydro_model.to_payload(), hydro_path)
        write_json(run / "hydro_pretraining.json", pretraining)
        bind_files(run, "hydro_initialization_complete.json", [hydro_path, run / "hydro_pretraining.json"], config)
    model_files += ["hydro_initialization.pt", "hydro_pretraining.json", "hydro_initialization_complete.json"]
    doc_path = run / "hydro_doc.pt"
    if (run / "hydro_doc_complete.json").exists():
        verify_files(run, "hydro_doc_complete.json", config)
        hydro_doc = EncoderNativeResidual.from_payload(torch.load(doc_path, weights_only=False, map_location="cpu"))
    else:
        hydro_doc = EncoderNativeResidual(hydro_model.spatial, hydro_model.temporal, hydro_model.decay,
            encoder_mode="last_self_ecology", encoder_learning_rate=1e-5, extra_dim=38,
            interaction_indices=(0, 2, 4, 28, 30, 31, 32), epochs=epochs, patience=5, seed=seed,
            tail_weight=2).fit(*arrays, tail_threshold=config["q90_threshold_train"],
                selection_role="source_validation", progress=lambda row: progress("hydro_doc", row))
        torch.save(hydro_doc.to_payload(), doc_path)
        write_json(run / "hydro_doc.json", hydro_doc.to_dict())
        bind_files(run, "hydro_doc_complete.json", [doc_path, run / "hydro_doc.json"], config)
    model_files += ["hydro_doc.pt", "hydro_doc.json", "hydro_doc_complete.json"]
    hydro_val_delta = hydro_doc.predict_delta({key: value[val_ids] for key, value in full_inputs.items()})
    hydro_temporal = np.maximum(0, context.ravel()[val]+hydro_doc.selected_scale_*hydro_val_delta[rows, val % months])
    candidates = [{"gamma": g, "mae": float(np.abs(np.maximum(0, context.ravel()[val]
        +(1-g)*(hydro_temporal-context.ravel()[val])+g*static.ravel()[val])-np.asarray(dataset["y"]).ravel()[val]).mean())}
                  for g in (0., .25, .5, 1.)]
    chosen = min(candidates, key=lambda row: (row["mae"], row["gamma"]))
    predictions["hydro_pretrained"] = np.maximum(0, context.ravel()[val]+(1-chosen["gamma"])*(hydro_temporal-context.ravel()[val])
                                                 +chosen["gamma"]*static.ravel()[val])
    write_json(run / "hydro_selection.json", candidates)
    model_files.append("hydro_selection.json")

    diagnostics = {}
    for name, model, train_hidden, validation_temporal in (
        ("retrieval", retained, current_hidden, temporal.ravel()[val]),
        ("hydro_pretrained_retrieval", hydro_doc, hidden_rows(hydro_doc, source_inputs, source_cells), hydro_temporal)):
        episodes = []
        for index, entry in enumerate(bank_entries):
            cells = entry["query_cells"]
            indexes = np.searchsorted(train, cells)
            prepared = selected_query(entry["bank"], dataset, cells, entry["context"], train_hidden[indexes],
                                      residual_scale=bank.residual_scale_)
            unique, inverse, counts = np.unique(cells//months, return_inverse=True, return_counts=True)
            weights = len(cells)/(len(unique)*counts[inverse])
            episodes.append({**prepared, "static": nested_static[index], "context": entry["context"],
                "temporal": entry["context"], "truth": np.asarray(dataset["y"]).ravel()[cells], "weights": weights})
        validation = selected_query(bank, dataset, val, context.ravel()[val], hidden_rows(model, full_inputs, val),
                                    residual_scale=bank.residual_scale_)
        validation.update({"static": static.ravel()[val], "context": context.ravel()[val],
                           "temporal": validation_temporal, "truth": np.asarray(dataset["y"]).ravel()[val]})
        attention, selection = fit_retrieval(episodes, validation, residual_scale=bank.residual_scale_, seed=seed,
            epochs=epochs, patience=5, progress=lambda row, arm=name: progress(arm, row))
        torch.save(attention.to_payload(), run / f"{name}.pt")
        write_json(run / f"{name}.json", selection)
        delta, entropy, effective, maximum = inference_retrieval(attention, validation, scale=bank.residual_scale_)
        g = selection["gamma"]
        predictions[name] = np.maximum(0, context.ravel()[val]+(1-g)*(validation_temporal-context.ravel()[val])
                                       +g*(static.ravel()[val]+delta))
        for ablation in ("uniform", "zero_source_values"):
            ablated, _, _, _ = inference_retrieval(attention, validation, scale=bank.residual_scale_, ablation=ablation)
            predictions[f"{name}_{ablation}"] = np.maximum(0, context.ravel()[val]
                +(1-g)*(validation_temporal-context.ravel()[val])+g*(static.ravel()[val]+ablated))
        diagnostics[name] = {"retrieval_delta": delta, "retrieval_entropy": entropy,
                             "effective_donors": effective, "max_donor_weight": maximum,
                             "nearest_distance": validation["nearest_distance"]}
        model_files += [f"{name}.pt", f"{name}.json"]
        # Retain the exact label-free query input for independent checkpoint replay.
        np.savez_compressed(run / f"{name}_validation_inputs.npz", **{key: validation[key]
            for key in ("query", "keys", "values", "valid", "static", "context", "temporal")})
        model_files.append(f"{name}_validation_inputs.npz")
        del episodes, validation, attention
    frames = []
    for name, values in predictions.items():
        frame = pd.DataFrame({"cell": val, "station": np.asarray(dataset["site_no"], str)[val//months],
            "month": np.asarray(dataset["months"], str)[val % months], "analyte": "doc",
            "visibility_role": "val", "model_name": name, "k": 0, "support_count": 0,
            "y_pred": values, "y_true": np.asarray(dataset["y"]).ravel()[val],
            "split_seed": partition, "seed": seed,
            "hydro_completeness": np.asarray(dataset["x_mask"])[val//months, val % months].mean(-1)})
        for key, values in diagnostics.get(name, {}).items():
            frame[key] = values
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    if not np.isfinite(panel[["y_pred", "y_true"]]).all().all():
        raise FloatingPointError("nonfinite validation product")
    panel.to_parquet(run / "predictions.parquet", index=False)
    bind_product(run, "predictions.parquet", config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-started})
    bind_files(run, "complete.json", [run / name for name in
        (*model_files, "predictions.parquet", "predictions.meta.json", "timing.json", "nested_complete.json")], config)
    progress("complete", {"models": len(predictions), "validation_rows": len(panel)})
    del expert, features, retained, hydro_model, hydro_doc, source_inputs, full_inputs, arrays
    gc.collect()


def copy_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hydro-epochs", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.root == ROOT:
        args.root = ROOT / "_smoke"
    torch.set_num_threads(args.torch_threads)
    args.root.mkdir(parents=True, exist_ok=True)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.parent, partition, seed, runtime,
                    epochs=1 if args.smoke else args.epochs,
                    hydro_epochs=1 if args.smoke else args.hydro_epochs, smoke=args.smoke)


if __name__ == "__main__":
    main()
