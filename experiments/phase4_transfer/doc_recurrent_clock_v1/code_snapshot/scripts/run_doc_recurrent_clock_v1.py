"""Compare DOC-age, neutral-unseen and within-window flow recurrent clocks."""
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
from run_doc_daily_hydro_memory_v1 import (
    EXTRA_DIM,
    INTERACTION_INDICES,
    KS,
    grid_metadata,
    load_daily_pack,
    training_arrays,
)
from run_doc_daily_hydro_support_basis_v1 import fit_arm_mixers, integrated_frame, validation_rows
from run_doc_ecological_transfer_v1 import load_inputs
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.recurrent_clock_residual import ClockNativeResidual
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_recurrent_clock_v1")
PRIOR = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
ARMS = ("legacy", "unseen_neutral", "flow_window")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = (tuple(f"{arm}_{basis}" for arm in ("context", *ARMS) for basis in SHAPES)
          + tuple(f"{arm}_integrated_{basis}" for arm in ARMS for basis in SHAPES))


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_daily_hydro_memory_v1.py", "scripts/run_doc_daily_hydro_support_basis_v1.py",
                 "scripts/run_doc_recurrent_clock_v1.py", str(ROOT / "study_plan.md")):
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


def fit_clock_mixers(context, bases, memory, shapes, truth, split, months):
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    mixers = {}
    for arm in ARMS:
        scores = []
        for gamma in (0, .25, .5, 1):
            pred = bases[arm][query] if gamma == 0 else np.maximum(0,
                context[query]+(1-gamma)*(bases[arm][query]-context[query])+gamma*memory[query])
            scores.append({"gamma": gamma, "mae": float(np.abs(pred-truth[query]).mean())})
        gamma = min(scores, key=lambda row: (row["mae"], row["gamma"]))["gamma"]
        mixers.update(fit_arm_mixers(arm, context, bases[arm], memory, shapes, truth,
                                    split, months, gamma))
    return mixers


def legacy_checks(frame, previous, direct, mixers, prior):
    old_direct, old_mixers = json.loads((prior / "adapters.json").read_text()), json.loads((prior / "mixers.json").read_text())
    checks = []
    for basis in SHAPES:
        if direct[f"legacy_{basis}"].to_dict() != old_direct[f"off_{basis}"]:
            raise ValueError("Matched legacy direct adaptation changed")
        if mixers[f"legacy_integrated_{basis}"].to_dict() != old_mixers[f"off_integrated_{basis}"]:
            raise ValueError("Matched legacy ecological mixing changed")
        for suffix in ("", "_integrated"):
            for k in KS:
                current = frame[frame.model_name.eq(f"legacy{suffix}_{basis}") & frame.k.eq(k)].sort_values("cell")
                parent = previous[previous.model_name.eq(f"off{suffix}_{basis}") & previous.k.eq(k)].sort_values("cell")
                for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    np.testing.assert_array_equal(current[column], parent[column])
                checks.append({"model_name": f"legacy{suffix}_{basis}", "k": k,
                               "rows": len(current), "bitwise_exact": True})
    return checks


def clock_diagnostics(model, inputs, query_mask):
    candidates = np.flatnonzero(np.asarray(query_mask).ravel())
    cells = candidates[np.unique(np.linspace(0, len(candidates)-1, min(2048, len(candidates)), dtype=int))]
    block = model.clock_window(inputs, cells, include_gamma=True)
    valid, clock, gamma = block["valid"], block["clock"], block["gamma"]
    result = {"sample_cells": cells.tolist(), "valid_steps": int(valid.sum()),
              "clock_mean": float(clock[valid].mean()),
              "clock_quantiles": np.quantile(clock[valid], [.1, .5, .9]).tolist(),
              "mean_gamma": float(gamma[valid].mean()), "retention": {}}
    for lag in (3, 6, 11):
        eligible = valid[:, -lag:].all(1)
        result["retention"][str(lag)] = float(gamma[eligible, -lag:].prod(1).mean()) if eligible.any() else None
    return result


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
            raise ValueError("Saved clock package differs from the frozen study")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing package", flush=True)
        return
    if root == ROOT and (epochs != pc["epochs"] or patience != pc["patience"]):
        raise ValueError("Production clock controls require the parent's identical training budget")
    run.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months, context = truth.shape[1], full.context_pred.to_numpy()
    memory = parent_full.ecological_memory.to_numpy()
    daily, _, daily_identity = load_daily_pack(Path(pc["daily_features_path"]).parent,
                                              pc["dataset_hash"], truth.shape)
    flow = build_causal_flow_features(dataset)
    extra = build_regime_head_features(dataset["regime"], split["train"],
        np.maximum(0, np.expm1(oof.ravel()[split["train"]])), context.reshape(truth.shape),
        flow["full"], n_months=months)
    config = {"experiment": "doc_recurrent_clock_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{f"{key}_run": str(path) for key, path in sources.items()},
        **{f"{key}_completion_hash": sha256_file(path / "complete.json") for key, path in sources.items()},
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                    "q90_threshold_train", "query_cells")}, **daily_identity,
        "parent_checkpoint_hash": sha256_file(prior / "off.pt"),
        "parent_full_grid_hash": sha256_file(prior / "full_grid.parquet"),
        "target_analyte": "doc", "target_transform": "log1p", "residual_scale": "native mg/L",
        "inference_roles": ["train"], "selection_role": "source_validation",
        "models": MODELS, "arms": ARMS, "basis_names": SHAPES, "k_values": KS,
        "lookback": 12, "epochs": epochs, "patience": patience, "torch_threads": torch.get_num_threads(),
        "batch_size": 512, "gradient_clip_norm": 1, "learning_rate": 1e-4,
        "head_learning_rate": 1e-3, "encoder_learning_rate": 1e-5,
        "encoder_mode": "last_self_ecology", "encoder_dropout": "off",
        "hydro_sequence_mode": "off", "extra_dim": EXTRA_DIM,
        "feature_definition": pc["feature_definition"],
        "interaction_indices": INTERACTION_INDICES, "tail_weight": 2,
        "residual_scales": [0, .25, .5, 1], "train_memory": True,
        "epoch_selection": "unweighted source-validation K0 query MAE; epoch0 included",
        "source_sampling": "same uniform observed-cell shuffle as legacy",
        "support_basis": "unchanged frozen v4 GRU in all arms",
        "clock_scope": "only decay scalar changes; raw M1 age stays unchanged; shared-state intervention",
        "clock_definitions": {"legacy": "existing DOC age; receiving-unseen calendar index",
            "unseen_neutral": "age zero if last-observation-valid is zero",
            "flow_window": "log1p(capped12 monthly discharge age) / log13; current rolling window only"},
        "neutral_capacity": "same allocated parameters;64 decay age weights inactive at unseen receiving stations",
        "new_neural_fits": 2, "legacy_reuse": "verified parent off, same source inputs/loss/budget",
        "forest_retraining": False, "readout_fitting": False,
        "study_role": "DOC spatial-transfer recurrent-clock development on existing partitions"}
    write_json(run / "config.json", config)
    (run / "feature_definition.json").write_bytes((prior / "feature_definition.json").read_bytes())
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    np.testing.assert_array_equal(features["source_station_ids"], extra["source_station_ids"])
    arrays = training_arrays(features, split, truth, oof, context)
    val_ids = np.unique(split["val"] // months)
    all_extra = np.concatenate((extra["full_extra"], daily), axis=-1)
    source_extra = np.concatenate((extra["source_extra"], daily[features["source_station_ids"]]), axis=-1)
    source_inputs = {**arrays[0], "extra": source_extra}
    validation_inputs = {**arrays[4], "extra": all_extra[val_ids]}
    full_inputs = {**{key: features[f"full_{key}"] for key in ("raw", "age", "support")},
                   "env": features["env"], "extra": all_extra}
    bases = {"context": context, "legacy": parent_full.off_pred.to_numpy()}
    components = full[["cell", "station", "month", "analyte", "visibility_role",
                        "ecological_novelty", "upstream_support"]].copy()
    components["context_pred"], components["ecological_memory"] = context, memory
    components["legacy_delta"], components["legacy_pred"] = parent_full.off_delta, bases["legacy"]
    for name, values in grid_metadata(split, truth.shape, daily).items():
        components[name] = values.ravel()
    (run / "legacy.pt").write_bytes((prior / "off.pt").read_bytes())
    write_json(run / "legacy.json", json.loads((prior / "off.json").read_text()))
    legacy = ClockNativeResidual.from_payload(torch.load(prior / "off.pt", weights_only=True))
    diagnostics = {"legacy": clock_diagnostics(legacy, validation_inputs, arrays[7])}
    del legacy
    model_files = ["legacy.pt", "legacy.json", "adapters.json", "mixers.json", "feature_definition.json", "clock_diagnostics.json"]
    for clock in ARMS[1:]:
        model = ClockNativeResidual(expert.residual.model.spatial, expert.residual.model.temporal,
            expert.residual.model.decay, decay_clock=clock, encoder_mode="last_self_ecology",
            encoder_learning_rate=1e-5, hydro_sequence_mode="off", seed=seed,
            epochs=epochs, patience=patience, extra_dim=EXTRA_DIM, tail_weight=2,
            interaction_indices=INTERACTION_INDICES).fit(
                source_inputs, *arrays[1:4], validation_inputs, *arrays[5:],
                tail_threshold=config["q90_threshold_train"], selection_role="source_validation",
                progress=lambda row, name=clock: print(f"{run.name}/{name}: {json.dumps(row)}", flush=True))
        torch.save(model.to_payload(), run / f"{clock}.pt")
        write_json(run / f"{clock}.json", model.to_dict())
        pd.DataFrame(model.to_dict()["trace"]).to_csv(run / f"{clock}_trace.csv", index=False)
        delta = model.predict_delta(full_inputs).ravel()
        scale = model.selected_scale_
        bases[clock] = context.copy() if scale == 0 else np.maximum(0, context+scale*delta)
        diagnostics[clock] = clock_diagnostics(model, validation_inputs, arrays[7])
        components[f"{clock}_delta"], components[f"{clock}_pred"] = delta, bases[clock]
        model_files.extend((f"{clock}.pt", f"{clock}.json"))
    write_json(run / "clock_diagnostics.json", diagnostics)
    del expert, features, arrays, source_inputs, validation_inputs, full_inputs, extra, flow, oof, daily
    gc.collect()
    direct = fit_adapters(bases, shapes, truth.ravel(), split, months)
    mixers = fit_clock_mixers(context, bases, memory, shapes, truth.ravel(), split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in direct.items()})
    write_json(run / "mixers.json", {name: model.to_dict() for name, model in mixers.items()})
    validation = []
    for clock in ARMS:
        validation.extend(validation_rows(clock, context, bases[clock], memory, shapes,
            direct, mixers, truth.ravel(), split, months))
    pd.DataFrame(validation).to_csv(run / "source_validation.csv", index=False)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth.ravel()[support]
    frame = make_predictions(full, bases, shapes, direct, labels, split, months)
    frame["regional_gamma"] = 0.0
    blocks = [frame]
    for clock in ARMS:
        blocks.append(integrated_frame(full, clock, context, bases[clock], memory, shapes,
                                       mixers, labels, split, months))
        components[f"{clock}_integrated_k0_pred"] = mixers[f"{clock}_integrated_constant"].selected_base(
            context, bases[clock], memory, k=0)
    frame = pd.concat(blocks, ignore_index=True)
    write_json(run / "legacy_checks.json", legacy_checks(frame,
        pd.read_parquet(prior / "predictions.parquet"), direct, mixers, prior))
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth.ravel()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite recurrent-clock query product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    for name in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, name, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-start})
    files = ["config.json", *model_files, "source_validation.csv", "legacy_checks.json",
             "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json", "timing.json",
             *(f"{clock}_trace.csv" for clock in ARMS[1:])]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: complete in {time.monotonic()-start:.1f}s; two new neural fits", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=120)
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
