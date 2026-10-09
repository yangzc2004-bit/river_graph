"""Fit ecological residual transfer on the current DOC interaction model."""
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
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer

ROOT = Path("experiments/phase4_transfer/doc_ecological_transfer_v1")
PRIOR = Path("experiments/phase4_transfer/doc_flow_interaction_v1")
ARMS = ("global_bias", "ecological_bias", "global_affine", "ecological_affine")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("interaction", *ARMS) for shape in SHAPES)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Execution changed; preserve outputs and use a new version")
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def load_inputs(prior):
    """Load frozen predictors and the source-only OOF context library."""
    pc = json.loads((prior / "config.json").read_text())
    verify_files(prior, "complete.json", pc)
    sources = {name: Path(pc[f"{name}_run"]) for name in ("source", "basis", "oof")}
    for name, path in sources.items():
        if sha256_file(path / "complete.json") != pc[f"{name}_completion_hash"]:
            raise ValueError(f"Changed {name} package")
        verify_files(path, "complete.json", json.loads((path / "config.json").read_text()))
    old_config, _, dataset, split, full = read_source(sources["source"])
    previous_full = pd.read_parquet(prior / "full_grid.parquet")
    np.testing.assert_array_equal(previous_full.cell, full.cell)
    np.testing.assert_array_equal(previous_full.context_pred, full.context_pred)
    shape = tuple(dataset["y"].shape)
    with np.load(sources["oof"] / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    mask = np.zeros(np.prod(shape), dtype=bool)
    mask[split["train"]] = True
    if (not np.isfinite(oof.ravel()[mask]).all()
            or not np.isnan(oof.ravel()[~mask]).all()):
        raise ValueError("OOF coverage must match exactly the source training cells")
    return pc, sources, old_config, dataset, split, full, previous_full, oof


def fit_memory(mode, dataset, split, context, interaction, oof):
    """The fit interface receives only explicitly selected source/val labels."""
    months = dataset["y"].shape[1]
    source_cells = np.asarray(split["train"], dtype=np.int64)
    _, validation_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
    return EcologicalResidualTransfer(mode=mode).fit(
        np.asarray(dataset["regime"]), source_cells,
        np.maximum(0, np.expm1(oof.ravel()[source_cells])), truth[source_cells],
        n_months=months, validation_cells=validation_cells,
        validation_y=truth[validation_cells], validation_context=context.ravel()[validation_cells],
        validation_temporal=interaction.ravel()[validation_cells],
        selection_role="source_validation",
    )


def run_one(root, prior_root, partition, seed, runtime):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    _pc, sources, old, dataset, split, full, previous_full, oof = load_inputs(prior)
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved run differs from the frozen sources")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "doc_ecological_transfer_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{f"{name}_run": str(path) for name, path in sources.items()},
        **{f"{name}_completion_hash": sha256_file(path / "complete.json") for name, path in sources.items()},
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                    "q90_threshold_train", "query_cells")},
        "target_analyte": "doc", "inference_roles": ["train"], "models": MODELS,
        "arms": ARMS, "k_values": [0, 1, 3, 5], "ecological_columns": list(range(4, 13)),
        "neighbor_grid": [20, 40, 80], "memory_ridge_grid": [.1, 1], "gamma_grid": [0, .25, .5, 1],
        "memory_loss": "station-balanced smooth native MAE plus identity-centered ridge",
        "smooth_epsilon_scaled": .05, "memory_max_iter": 200,
        "conditioning": "source-station-balanced standardized log1p(context prediction)",
        "residual_library": "source labels minus station OOF context prediction",
        "mixing": "context + (1-gamma)*(interaction-context) + gamma*memory",
        "selection": "source-validation K0 fixed-query pooled native MAE",
        "ecological_scaling": "source unique-station median/IQR; -1/nonfinite missing",
        "minimum_valid_ecology": 5, "donor_weighting": "uniform stations; uniform cells within station",
        "exclude_receiving_station": True, "support_basis": "frozen v4 GRU; matched across arms",
        "alpha_grid": [0, .25, .5, .75, 1], "ridge_grid": [.1, 1, 10, "infinity"],
        "torch_threads": torch.get_num_threads(),
        "study_role": "development on previously evaluated station partitions",
    }
    write_json(run / "config.json", config)
    shape = tuple(dataset["y"].shape)
    context = full.context_pred.to_numpy().reshape(shape)
    interaction = previous_full.interaction_tuned_pred.to_numpy().reshape(shape)
    bases = {"interaction": interaction.ravel()}
    identity = ["cell", "station", "month", "analyte", "visibility_role",
                "ecological_novelty", "upstream_support"]
    components = full[identity].copy()
    components["context_pred"] = context.ravel()
    components["interaction_pred"] = interaction.ravel()
    model_files = ["adapters.json"]
    for mode in ARMS:
        memory = fit_memory(mode, dataset, split, context, interaction, oof)
        state = memory.to_dict()
        write_json(run / f"{mode}.json", state)
        delta = memory.predict_delta(context)
        prediction = memory.predict(context, interaction)
        components[f"{mode}_memory"] = delta.ravel()
        components[f"{mode}_pred"] = prediction.ravel()
        bases[mode] = prediction.ravel()
        model_files.append(f"{mode}.json")
        print(f"{run.name}/{mode}: {json.dumps(state['selected'])}", flush=True)
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
    months = shape[1]
    adapters = fit_adapters(bases, shapes, truth, split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    support_truth = np.full(truth.shape, np.nan)
    support_truth[support] = truth[support]
    frame = make_predictions(full, bases, shapes, adapters, support_truth, split, months)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_true", "y_pred", "base_pred", "adaptation_delta"]]).all().all():
        raise ValueError("Nonfinite support-adapted product")
    previous_queries = pd.read_parquet(prior / "predictions.parquet")
    for shape_name in SHAPES:
        a = frame[frame.model_name.eq(f"interaction_{shape_name}")].sort_values(["k", "cell"])
        b = previous_queries[previous_queries.model_name.eq(f"interaction_tuned_{shape_name}")].sort_values(["k", "cell"])
        np.testing.assert_array_equal(a.y_pred, b.y_pred)
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    for filename in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, filename, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic() - start})
    files = ["config.json", *model_files, "predictions.parquet", "predictions.meta.json",
             "full_grid.parquet", "full_grid.meta.json", "timing.json"]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: complete, {time.monotonic()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime)
            gc.collect()


if __name__ == "__main__":
    main()
