"""Refresh station-support representations from selected DOC recurrent models."""
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
from run_doc_daily_hydro_memory_v1 import ARMS, KS, load_daily_pack
from run_doc_ecological_transfer_v2 import validation_episodes
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_basis_refresh import refresh_support_basis
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_support_basis_v1")
PRIOR = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
SHAPES = ("constant", "legacy", "refreshed")
MODELS = tuple(f"{arm}{suffix}_{basis}" for arm in ARMS
               for suffix in ("", "_integrated") for basis in SHAPES)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v2.py", "scripts/run_doc_daily_hydro_memory_v1.py",
                 "scripts/run_doc_daily_hydro_support_basis_v1.py", str(ROOT / "study_plan.md")):
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


def full_hidden_inputs(config, dataset, split, daily):
    """Reproduce train-visible raw inputs; scalar-head extras are not evaluated."""
    expert = UnifiedDOCReconstructor.load(Path(config["source_run"]), dataset, split)
    features = extract_raw_temporal_inputs(expert, dataset, split)
    inputs = {key: features[f"full_{key}"] for key in ("raw", "age", "support")}
    inputs["env"], inputs["daily_history"] = features["env"], daily
    # The hidden method never reads this block; preparation checks its shape.
    inputs["extra"] = np.zeros((*dataset["y"].shape, config["extra_dim"]), dtype=np.float32)
    del expert, features
    gc.collect()
    return inputs


def shapes_for_arm(archive, arm):
    return {"constant": archive["constant"], "legacy": archive["legacy"],
            "refreshed": archive[f"{arm}_refreshed"].reshape(-1, 2)}


def fit_arm_mixers(arm, context, base, memory, shapes, truth, split, months, gamma_k0):
    mixers = {}
    for basis_name, basis in shapes.items():
        episodes = validation_episodes(context, base, memory, basis, truth, split, months)
        kwargs = {"ridge_strengths": (float("inf"),)} if basis_name == "constant" else {}
        mixers[f"{arm}_integrated_{basis_name}"] = SupportAwareResidualTransfer(months, **kwargs).fit(
            episodes, gamma_k0=gamma_k0, selection_role="source_validation")
    return mixers


def integrated_frame(full, arm, context, base, memory, shapes, mixers, labels, split, months,
                     *, role="test"):
    frames = []
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        for basis_name, basis in shapes.items():
            name = f"{arm}_integrated_{basis_name}"
            model = mixers[name]
            predicted = model.adapt(context[query], base[query], memory[query], query,
                context[support], base[support], memory[support], support,
                labels[support], basis[query], basis[support], k=k)
            selected_base = model.selected_base(context[query], base[query], memory[query], k=k)
            frame = full.iloc[query][["cell", "station", "month", "analyte", "visibility_role",
                                     "ecological_novelty", "upstream_support"]].copy()
            frame["model_name"], frame["k"] = name, k
            frame["y_pred"], frame["base_pred"] = predicted, selected_base
            frame["adaptation_delta"] = np.log1p(predicted)-np.log1p(selected_base)
            frame["support_count"], frame["regional_gamma"] = k, model.selected_gamma(k)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validation_rows(arm, context, base, memory, shapes, adapters, mixers, truth, split, months):
    rows = []
    for basis_name, basis in shapes.items():
        for k in KS:
            support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
            direct = adapters[f"{arm}_{basis_name}"]
            predicted = direct.adapt(base[query], query, base[support], support, truth[support],
                query_basis=basis[query], support_basis=basis[support], k=k)
            common = {"arm": arm, "basis": basis_name, "k": k, "n": len(query)}
            rows.append({**common, "model_name": f"{arm}_{basis_name}", "stage": "direct",
                         "mae": float(np.abs(predicted-truth[query]).mean())})
            integrated = mixers[f"{arm}_integrated_{basis_name}"]
            predicted = integrated.adapt(context[query], base[query], memory[query], query,
                context[support], base[support], memory[support], support, truth[support],
                basis[query], basis[support], k=k)
            rows.append({**common, "model_name": f"{arm}_integrated_{basis_name}", "stage": "integrated",
                         "mae": float(np.abs(predicted-truth[query]).mean())})
    return rows


def check_legacy(frame, previous):
    checks = []
    for arm in ARMS:
        for suffix in ("", "_integrated"):
            for basis_name, old_basis in (("constant", "constant"), ("legacy", "gru_tuned_anchor")):
                name, old_name = f"{arm}{suffix}_{basis_name}", f"{arm}{suffix}_{old_basis}"
                for k in KS:
                    a = frame[frame.model_name.eq(name) & frame.k.eq(k)].sort_values("cell")
                    b = previous[previous.model_name.eq(old_name) & previous.k.eq(k)].sort_values("cell")
                    for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                        np.testing.assert_array_equal(a[column], b[column])
                    checks.append({"model_name": name, "parent_model": old_name, "k": k,
                                   "rows": len(a), "bitwise_exact": True})
            panels = [frame[frame.model_name.eq(f"{arm}{suffix}_{shape}") & frame.k.eq(0)]
                      .sort_values("cell") for shape in SHAPES]
            for panel in panels[1:]:
                np.testing.assert_array_equal(panels[0].cell, panel.cell)
                np.testing.assert_array_equal(panels[0].y_pred, panel.y_pred)
    return checks


def run_one(root, prior_root, partition, seed, runtime, *, batch_size):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, _, dataset, split, parent_full = read_source(prior)
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime or config["batch_size"] != batch_size
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved basis experiment differs from frozen inputs")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    _, _, _, _, full = read_source(Path(pc["source_run"]))
    shape = tuple(dataset["y"].shape)
    truth, months = np.asarray(dataset["y"], dtype=np.float64).ravel(), shape[1]
    np.testing.assert_array_equal(full.cell, parent_full.cell)
    basis_run = Path(pc["basis_run"])
    adapter = EpisodicTemporalAdapter.from_payload(torch.load(basis_run / "memory.pt", weights_only=False))
    daily, _, daily_bindings = load_daily_pack(Path(pc["daily_features_path"]).parent, pc["dataset_hash"], shape)
    config = {"experiment": "doc_daily_hydro_support_basis_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{key: pc[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
            "source_run", "source_completion_hash", "basis_run", "basis_completion_hash", "q90_threshold_train",
            "query_cells", "extra_dim")}, **daily_bindings,
        "parent_checkpoint_hashes": {arm: sha256_file(prior / f"{arm}.pt") for arm in ARMS},
        "basis_checkpoint_hash": sha256_file(basis_run / "memory.pt"),
        "legacy_basis_hash": sha256_file(basis_run / "representations.npz"),
        "parent_full_grid_hash": sha256_file(prior / "full_grid.parquet"),
        "target_analyte": "doc", "target_transform": "log1p", "inference_roles": ["train"],
        "arms": ARMS, "basis_names": SHAPES, "models": MODELS, "k_values": KS,
        "anchor_count": adapter.anchor_count, "scale_floor": adapter.scale_floor,
        "batch_size": batch_size, "torch_threads": torch.get_num_threads(),
        "backbone_retraining": False, "forest_retraining": False, "readout_fitting": False,
        "gamma_k0": "frozen parent source-validation choice", "selection_role": "source_validation",
        "normalization": "same 32 fixed record-wide calendar anchors and station scalar RMS",
        "study_role": "retrospective support representation development on existing partitions"}
    write_json(run / "config.json", config)
    inputs = full_hidden_inputs(pc, dataset, split, daily)
    with np.load(basis_run / "representations.npz", allow_pickle=False) as saved:
        archive = {"constant": saved["constant"].copy(), "legacy": saved["gru_tuned_anchor"].copy()}
    definition = {"readout": adapter.readout.numpy().tolist(), "anchor_count": adapter.anchor_count,
                  "scale_floor": adapter.scale_floor, "readout_fitting": "none",
                  "raw_state": "selected current encoder/GRU; no scalar head or extra-head channels",
                  "normalization_role": "retrospective feature-only record normalization",
                  "parent_checkpoint_hashes": config["parent_checkpoint_hashes"]}
    for arm in ARMS:
        model = EncoderNativeResidual.from_payload(torch.load(prior / f"{arm}.pt", weights_only=True))
        print(f"{run.name}/{arm}: refreshing hidden support basis", flush=True)
        refreshed = refresh_support_basis(model, inputs, adapter, batch_size=batch_size)
        archive[f"{arm}_refreshed"] = refreshed["basis"]
        for name in ("raw_basis", "station_anchor_mean", "station_rms", "station_scale", "floor_hit"):
            archive[f"{arm}_{name}"] = refreshed[name]
        if "anchor_months" in archive:
            np.testing.assert_array_equal(archive["anchor_months"], refreshed["anchor_months"])
        archive["anchor_months"] = refreshed["anchor_months"]
        del model, refreshed
    del inputs, adapter, daily
    gc.collect()
    np.savez_compressed(run / "representations.npz", **archive)
    write_json(run / "basis_definition.json", definition)
    context, memory = parent_full.context_pred.to_numpy(), parent_full.ecological_memory.to_numpy()
    bases = {arm: parent_full[f"{arm}_pred"].to_numpy() for arm in ARMS}
    old_adapters = json.loads((prior / "adapters.json").read_text())
    old_mixers = json.loads((prior / "mixers.json").read_text())
    adapters, mixers, rows = {}, {}, []
    for arm in ARMS:
        shapes = shapes_for_arm(archive, arm)
        direct = fit_adapters({arm: bases[arm]}, shapes, truth, split, months)
        gamma_k0 = old_mixers[f"{arm}_integrated_constant"]["gamma_k0"]
        integrated = fit_arm_mixers(arm, context, bases[arm], memory, shapes, truth, split, months, gamma_k0)
        for basis_name, old_basis in (("constant", "constant"), ("legacy", "gru_tuned_anchor")):
            if direct[f"{arm}_{basis_name}"].to_dict() != old_adapters[f"{arm}_{old_basis}"]:
                raise ValueError("Old direct support fit changed")
            if integrated[f"{arm}_integrated_{basis_name}"].to_dict() != old_mixers[f"{arm}_integrated_{old_basis}"]:
                raise ValueError("Old integrated support fit changed")
        adapters.update(direct)
        mixers.update(integrated)
        rows.extend(validation_rows(arm, context, bases[arm], memory, shapes, direct, integrated,
                                    truth, split, months))
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    write_json(run / "mixers.json", {name: model.to_dict() for name, model in mixers.items()})
    pd.DataFrame(rows).to_csv(run / "source_validation.csv", index=False)
    print(f"{run.name}: source-validation adaptation finished", flush=True)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.shape, np.nan)
    labels[support] = truth[support]
    frames = []
    for arm in ARMS:
        shapes = shapes_for_arm(archive, arm)
        direct = make_predictions(full, {arm: bases[arm]}, shapes, adapters, labels, split, months)
        direct["regional_gamma"] = 0.0
        frames.extend((direct, integrated_frame(full, arm, context, bases[arm], memory, shapes,
                       mixers, labels, split, months)))
    frame = pd.concat(frames, ignore_index=True)
    checks = check_legacy(frame, pd.read_parquet(prior / "predictions.parquet"))
    write_json(run / "legacy_checks.json", checks)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite refreshed-basis product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    (run / "full_grid.parquet").write_bytes((prior / "full_grid.parquet").read_bytes())
    model_files = ["representations.npz", "basis_definition.json", "adapters.json", "mixers.json"]
    for filename in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, filename, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-start})
    files = ["config.json", *model_files, "source_validation.csv", "legacy_checks.json",
             "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json", "timing.json"]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: complete in {time.monotonic()-start:.1f}s; no new neural or forest fit", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime, batch_size=args.batch_size)
            gc.collect()


if __name__ == "__main__":
    main()
