"""Add measured auxiliary chemistry to the retained DOC reconstruction model."""
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
from run_doc_daily_hydro_memory_v1 import KS, load_daily_pack
from run_doc_daily_hydro_readout_v1 import array_digest
from run_doc_daily_hydro_support_basis_v1 import fit_arm_mixers, integrated_frame
from run_doc_distribution_head_v1 import head_feature_blocks, observed_features
from run_doc_ecological_transfer_v1 import load_inputs
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.auxiliary_chemistry_features import (
    apply_auxiliary_mode,
    build_auxiliary_chemistry_features,
)
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.daily_hydro_tree import build_daily_tree_features
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.frozen_native_feature_head import FrozenNativeFeatureHead
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import (
    SupportShapeAdapter,
    SupportShapeEpisode,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_auxiliary_chemistry_v1")
PRIOR = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
AUX_PATHS = {"ph": Path("data/processed/mississippi_graph_ph_st357.pt"),
             "ec": Path("data/processed/mississippi_graph_spec_conductance_st357.pt")}
MODES = ("no_aux", "masks", "chemistry")
NEURAL = tuple(f"neural_{mode}" for mode in MODES)
TREES = tuple(f"tree_{mode}" for mode in MODES)
REFERENCES = ("point", "context", "tree_prior")
ARMS = (*REFERENCES, *NEURAL, *TREES)
INTEGRATED = ("point", *NEURAL)
BASIS = "gru_tuned_anchor"
FEATURE_DIM = 682
MODELS = (tuple(f"{arm}_{BASIS}" for arm in ARMS)
          + tuple(f"{arm}_integrated_{BASIS}" for arm in INTEGRATED))


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_daily_hydro_memory_v1.py", "scripts/run_doc_daily_hydro_readout_v1.py",
                 "scripts/run_doc_daily_hydro_support_basis_v1.py", "scripts/run_doc_distribution_head_v1.py",
                 "scripts/run_doc_auxiliary_chemistry_v1.py", str(ROOT / "study_plan.md")):
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
    auxiliary = np.asarray(auxiliary, dtype=np.float32)
    interaction = (vectors[:, :64, None]*auxiliary[:, None, :2]).reshape(len(vectors), -1)
    result = np.concatenate((vectors, auxiliary, interaction), axis=1)
    if result.shape != (len(vectors), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("Invalid auxiliary native-head feature layout")
    return result


def validation_tasks(split, months, active):
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        query = query[active[query]]
        support = support[np.isin(support//months, np.unique(query//months))]
        if not len(query):
            raise ValueError("Auxiliary calibration needs active validation queries")
        yield k, support, query


def fit_choices(bases, context, memory, basis, truth, split, months, active):
    shapes = {BASIS: basis}
    adapters = fit_adapters({arm: bases[arm] for arm in REFERENCES}, shapes, truth, split, months)
    mixers = {}
    _, full_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    for arm in (*NEURAL, *TREES):
        episodes = [SupportShapeEpisode(k, query, truth[query], bases[arm][query], support,
                    truth[support], bases[arm][support], basis[query], basis[support])
                    for k, support, query in validation_tasks(split, months, active)]
        adapters[f"{arm}_{BASIS}"] = SupportShapeAdapter(n_months=months).fit(
            episodes, selection_role="source_validation")
    for arm in INTEGRATED:
        query = full_query if arm == "point" else full_query[active[full_query]]
        scores = []
        for gamma in (0, .25, .5, 1):
            predicted = bases[arm][query] if gamma == 0 else np.maximum(0,
                context[query]+(1-gamma)*(bases[arm][query]-context[query])+gamma*memory[query])
            scores.append((float(np.abs(predicted-truth[query]).mean()), gamma))
        _, gamma = min(scores)
        if arm == "point":
            mixers.update(fit_arm_mixers(arm, context, bases[arm], memory, shapes, truth, split, months, gamma))
        else:
            episodes = [SupportAwareTransferEpisode(k, query, truth[query], support, truth[support],
                context[query], bases[arm][query], memory[query], context[support], bases[arm][support],
                memory[support], basis[query], basis[support])
                for k, support, query in validation_tasks(split, months, active)]
            mixers[f"{arm}_integrated_{BASIS}"] = SupportAwareResidualTransfer(months).fit(
                episodes, gamma_k0=gamma, selection_role="source_validation")
    return adapters, mixers


def direct_frame(full, bases, basis, adapters, labels, split, months, role):
    rows = []
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        for arm, base in bases.items():
            name = f"{arm}_{BASIS}"
            predicted = adapters[name].adapt(base[query], query, base[support], support, labels[support],
                query_basis=basis[query], support_basis=basis[support], k=k)
            row = full.iloc[query][["cell", "station", "month", "analyte", "visibility_role",
                                  "ecological_novelty", "upstream_support"]].copy()
            row["model_name"], row["k"], row["y_pred"], row["base_pred"] = name, k, predicted, base[query]
            row["adaptation_delta"] = np.log1p(predicted)-np.log1p(base[query])
            row["support_count"], row["regional_gamma"] = k, 0.0
            rows.append(row)
    return pd.concat(rows, ignore_index=True)


def final_frame(full, bases, context, memory, basis, adapters, mixers, labels, split, months, active, *, role):
    frame = direct_frame(full, bases, basis, adapters, labels, split, months, role)
    frame = pd.concat([frame, *[integrated_frame(full, arm, context, bases[arm], memory, {BASIS: basis},
        mixers, labels, split, months, role=role) for arm in INTEGRATED]], ignore_index=True)
    frame["candidate_y_pred"], frame["aux_fallback"] = frame.y_pred.copy(), False
    columns = ("y_pred", "base_pred", "adaptation_delta", "regional_gamma")
    for arm in (*NEURAL, *TREES):
        stages = ("", "_integrated") if arm in NEURAL else ("",)
        for stage in stages:
            name = f"{arm}{stage}_{BASIS}"
            parent = f"{'point' if arm in NEURAL else 'tree_prior'}{stage}_{BASIS}"
            for k in KS:
                selected = frame.model_name.eq(name) & frame.k.eq(k) & ~active[frame.cell.to_numpy()]
                cells = frame.loc[selected, "cell"].to_numpy()
                fallback = frame[frame.model_name.eq(parent) & frame.k.eq(k)].set_index("cell").loc[cells]
                for column in columns:
                    frame.loc[selected, column] = fallback[column].to_numpy()
                frame.loc[selected, "aux_fallback"] = True
    return frame


def reference_checks(frame, prior):
    previous = pd.read_parquet(prior / "predictions.parquet")
    checks = []
    for new, old in (("point", "off"), ("point_integrated", "off_integrated"),
                     ("context", "context"), ("tree_prior", "tree_current")):
        for k in KS:
            left = frame[frame.model_name.eq(f"{new}_{BASIS}") & frame.k.eq(k)].sort_values("cell")
            right = previous[previous.model_name.eq(f"{old}_{BASIS}") & previous.k.eq(k)].sort_values("cell")
            for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                np.testing.assert_array_equal(left[column], right[column])
            checks.append({"model_name": f"{new}_{BASIS}", "k": k, "rows": len(left), "bitwise_exact": True})
    return checks


def run_one(root, prior_root, partition, seed, runtime, *, epochs, patience):
    started = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, sources, old, dataset, split, full, parent, oof = load_inputs(prior)
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["epochs"] != epochs
                or config["patience"] != patience or config["torch_threads"] != torch.get_num_threads()
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved auxiliary package differs from frozen settings")
        verify_files(run, "complete.json", config)
        return
    run.mkdir(parents=True, exist_ok=True)
    aux_data = {key: torch.load(path, weights_only=False) for key, path in AUX_PATHS.items()}
    chemistry = build_auxiliary_chemistry_features(dataset, aux_data["ph"], aux_data["ec"])
    auxiliary = chemistry["full"].reshape(-1, 4)
    active = chemistry["active"].ravel()
    truth = np.asarray(dataset["y"], dtype=np.float64)
    shape, months = truth.shape, truth.shape[1]
    context, native, memory = full.context_pred.to_numpy(), parent.off_pred.to_numpy(), parent.ecological_memory.to_numpy()
    daily, _, daily_identity = load_daily_pack(Path(pc["daily_features_path"]).parent, pc["dataset_hash"], shape)
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        basis = saved[BASIS].copy()
    config = {"experiment": "doc_auxiliary_chemistry_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{f"{key}_run": str(path) for key, path in sources.items()},
        **{f"{key}_completion_hash": sha256_file(path / "complete.json") for key, path in sources.items()},
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train", "query_cells")},
        **daily_identity, "auxiliary_paths": {key: str(path) for key, path in AUX_PATHS.items()},
        "auxiliary_hashes": {key: sha256_file(path) for key, path in AUX_PATHS.items()},
        "auxiliary_provenance_hashes": {key: sha256_file(path.with_suffix(".provenance.json")) for key, path in AUX_PATHS.items()},
        "availability_audit_hash": sha256_file(ROOT / "availability_audit.json"),
        "auxiliary_policy": chemistry["policy"], "auxiliary_feature_names": chemistry["feature_names"],
        "auxiliary_feature_hash": array_digest(auxiliary), "auxiliary_active_hash": array_digest(active),
        "models": MODELS, "arms": ARMS, "neural_arms": NEURAL, "tree_arms": TREES,
        "modes": MODES, "basis_names": [BASIS], "k_values": KS,
        "target_analyte": "doc", "target_transform": "native residual; tree log1p", "inference_roles": ["train"],
        "selection_role": "source_validation", "epochs": epochs, "patience": patience,
        "learning_rate": .001, "batch_size": 512, "tail_weight": 2,
        "torch_threads": torch.get_num_threads(), "gradient_clip_norm": 1, "correction_scales": [0,.25,.5,1],
        "feature_dim": FEATURE_DIM, "feature_layout": "original550,aux4,hidden-major64x2 value interactions",
        "feature_std_floor": 1e-6, "head_parameters": FEATURE_DIM+1,
        "tree_feature_dim": 151, "tree_n_jobs": 2, "tree_hyperparameter_search": False,
        "neural_parent_hash": sha256_file(prior / "off.pt"), "tree_parent_hash": sha256_file(prior / "tree_current.joblib"),
        "backbone_retraining": False, "source_neural_is_oof": False,
        "support_selection_query": "aux-active val only; constant parent fallback elsewhere",
        "full_grid_role": "conditional chemistry components; final K0; no held DOC truth",
        "study_role": "known-auxiliary-chemistry DOC reconstruction on existing development partitions"}
    write_json(run / "config.json", config)
    model = EncoderNativeResidual.from_payload(torch.load(prior / "off.pt", weights_only=True))
    for module in (model.spatial, model.temporal, model.decay, model.head):
        module.requires_grad_(False)
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    raw, flow = extract_raw_temporal_inputs(expert, dataset, split), build_causal_flow_features(dataset)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0,np.expm1(oof.ravel()[split["train"]])), context.reshape(shape), flow["full"], n_months=months)
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
    source_base = np.maximum(0,np.expm1(oof.ravel()[source_cells])+model.selected_scale_*source_delta)
    np.savez_compressed(run / "source_training.npz", source_cells=source_cells, source_local_cells=local_cells,
        source_station_ids=source_ids, source_base=source_base, source_active=active[source_cells],
        validation_cells=val_cells, validation_base=native[val_cells], validation_active=active[val_cells])
    identities = {"source_original_feature_hash": array_digest(source_features),
        "validation_original_feature_hash": array_digest(validation_features), "modes": {}}
    heads = {}
    for mode in MODES:
        block = apply_auxiliary_mode(chemistry["full"], mode).reshape(-1, 4)
        sx, vx = augment(source_features, block[source_cells]), augment(validation_features, block[val_cells])
        identities["modes"][mode] = {"source_feature_hash": array_digest(sx), "validation_feature_hash": array_digest(vx)}
        head = FrozenNativeFeatureHead(n_features=FEATURE_DIM, epochs=epochs, patience=patience,
            batch_size=512,seed=seed,learning_rate=.001).fit(sx, source_base, truth.ravel()[source_cells],
                vx, native[val_cells], truth.ravel()[val_cells], source_active=active[source_cells],
                validation_active=active[val_cells],tail_threshold=config["q90_threshold_train"],
                selection_role="source_validation", progress=lambda row,name=mode: print(
                    f"{run.name}/neural_{name}: {json.dumps(row)}",flush=True))
        heads[mode] = head
        torch.save(head.to_payload(), run / f"neural_{mode}.pt")
        write_json(run / f"neural_{mode}.json", head.to_dict())
        pd.DataFrame(head.to_dict()["trace"]).to_csv(run / f"neural_{mode}_trace.csv", index=False)
    write_json(run / "input_definition.json", identities)
    tree_features = build_daily_tree_features(dataset, split, daily, mode="current")
    parent_tree = joblib.load(prior / "tree_current.joblib")
    tree_prior = np.maximum(0,np.expm1(parent_tree.predict(tree_features)))
    np.testing.assert_array_equal(tree_prior, parent.tree_current_pred)
    bases = {"point": native, "context": context, "tree_prior": tree_prior}
    for mode in MODES:
        block = apply_auxiliary_mode(chemistry["full"], mode).reshape(-1,4)
        features = np.concatenate((tree_features, block),axis=1)
        tree = clone(parent_tree).set_params(n_jobs=2)
        tree.fit(features[split["train"]], np.log1p(truth.ravel()[split["train"]]))
        candidate = np.maximum(0,np.expm1(tree.predict(features)))
        prediction = np.where(active, candidate, tree_prior)
        bases[f"tree_{mode}"] = prediction
        joblib.dump(tree,run / f"tree_{mode}.joblib",compress=3)
        write_json(run / f"tree_{mode}.json", {"mode": mode, "n_features": features.shape[1],
            "source_cells": len(split["train"]), "forest_parameters": tree.get_params(deep=False),
            "parent_parameters": parent_tree.get_params(deep=False), "target_transform": "log1p",
            "inactive_fallback": "tree_current", "hyperparameter_search": False,
            "validation_mae": float(np.abs(prediction[val_cells]-truth.ravel()[val_cells]).mean())})
        print(f"{run.name}/tree_{mode}: fitted",flush=True)
    del expert, raw, flow, extra, source_inputs, source_features, validation_features, sx, vx, features, tree_features, oof, daily, parent_tree
    gc.collect()
    components = full[["cell","station","month","analyte","visibility_role","ecological_novelty","upstream_support"]].copy()
    for key, value in bases.items():
        components[f"{key}_pred"] = value
    components["ecological_memory"] = memory
    components["ph_available"],components["ec_available"] = auxiliary[:,2].astype(bool),auxiliary[:,3].astype(bool)
    components["aux_available"],components["doc_observed"] = active,np.asarray(dataset["y_mask"]).ravel().astype(bool)
    for mode in MODES:
        bases[f"neural_{mode}"] = np.empty(truth.size)
    for cells, vectors, _ in head_feature_blocks(model, full_inputs):
        for mode in MODES:
            block = apply_auxiliary_mode(auxiliary[cells].reshape(1,-1,4),mode).reshape(-1,4)
            features = augment(vectors,block)
            bases[f"neural_{mode}"][cells] = heads[mode].predict(features,native[cells],active=active[cells])
    del model,full_inputs
    gc.collect()
    for arm in NEURAL:
        components[f"{arm}_pred"] = bases[arm]
        np.testing.assert_array_equal(bases[arm][~active],native[~active])
    adapters,mixers = fit_choices(bases,context,memory,basis,truth.ravel(),split,months,active)
    write_json(run / "adapters.json",{name:value.to_dict() for name,value in adapters.items()})
    write_json(run / "mixers.json",{name:value.to_dict() for name,value in mixers.items()})
    for arm in INTEGRATED:
        candidate = mixers[f"{arm}_integrated_{BASIS}"].selected_base(context,bases[arm],memory,k=0)
        fallback = mixers[f"point_integrated_{BASIS}"].selected_base(context,native,memory,k=0)
        components[f"{arm}_integrated_k0_pred"] = candidate if arm == "point" else np.where(active,candidate,fallback)
    validation = final_frame(full,bases,context,memory,basis,adapters,mixers,truth.ravel(),split,months,active,role="val")
    validation["error"] = np.abs(validation.y_pred-truth.ravel()[validation.cell.to_numpy()])
    rows = []
    for (name,k),group in validation.groupby(["model_name","k"],sort=True):
        selected = active[group.cell.to_numpy()]
        rows.append({"model_name":name,"k":int(k),"n":len(group),"n_active":int(selected.sum()),
                     "mae":float(group.error.mean()),"active_mae":float(group.loc[selected,"error"].mean())})
    pd.DataFrame(rows).to_csv(run / "source_validation.csv",index=False)
    support,_ = support_query_cells(split,target_role="test",k=5,n_months=months)
    labels = np.full(truth.size,np.nan)
    labels[support] = truth.ravel()[support]
    frame = final_frame(full,bases,context,memory,basis,adapters,mixers,labels,split,months,active,role="test")
    write_json(run / "reference_checks.json",reference_checks(frame,prior))
    frame["split_seed"],frame["seed"] = partition,seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    for column in ("ph_available","ec_available","aux_available","doc_observed"):
        frame[column] = components[column].to_numpy()[frame.cell.to_numpy()]
    if not np.isfinite(frame.select_dtypes(include="number")).all().all() or not np.isfinite(components.select_dtypes(include="number")).all().all():
        raise ValueError("Nonfinite auxiliary-chemistry product")
    frame.to_parquet(run / "predictions.parquet",index=False)
    components.to_parquet(run / "full_grid.parquet",index=False)
    files = ["input_definition.json","source_training.npz","adapters.json","mixers.json"]
    files += [f"neural_{mode}.{suffix}" for mode in MODES for suffix in ("pt","json")]
    files += [f"tree_{mode}.{suffix}" for mode in MODES for suffix in ("joblib","json")]
    for name in ("predictions.parquet","full_grid.parquet"):
        bind_product(run,name,config,runtime,files)
    write_json(run / "timing.json",{"elapsed_seconds":time.monotonic()-started})
    completed = ["config.json",*files,"source_validation.csv","reference_checks.json","timing.json",
        "predictions.parquet","predictions.meta.json","full_grid.parquet","full_grid.meta.json",
        *[f"neural_{mode}_trace.csv" for mode in MODES]]
    bind_files(run,"complete.json",[run / name for name in completed],config)
    print(f"{run.name}: complete in{time.monotonic()-started:.1f}s",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--prior-root",type=Path,default=PRIOR)
    parser.add_argument("--split-seeds",type=int,nargs="+",default=[142,143,144])
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--epochs",type=int,default=120)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--torch-threads",type=int,default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root,args.prior_root,partition,seed,runtime,epochs=args.epochs,patience=args.patience)
            gc.collect()


if __name__ == "__main__":
    main()
