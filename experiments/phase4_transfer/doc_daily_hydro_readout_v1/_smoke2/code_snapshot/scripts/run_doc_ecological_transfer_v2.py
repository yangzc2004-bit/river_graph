"""Update frozen ecological residual priors with target-station support."""
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
from run_doc_ecological_transfer_v1 import ARMS, MODELS, SHAPES, load_inputs
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)

ROOT = Path("experiments/phase4_transfer/doc_ecological_transfer_v2")
PRIOR = Path("experiments/phase4_transfer/doc_ecological_transfer_v1")
KS = (0, 1, 3, 5)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 str(ROOT / "study_plan.md")):
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


def validation_episodes(context, interaction, memory, basis, truth, split, months):
    episodes = []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        episodes.append(SupportAwareTransferEpisode(
            k=k, query_cells=query, query_values=truth[query],
            support_cells=support, support_values=truth[support],
            query_context=context[query], query_temporal=interaction[query], query_memory=memory[query],
            support_context=context[support], support_temporal=interaction[support], support_memory=memory[support],
            query_basis=basis[query], support_basis=basis[support],
        ))
    return episodes


def fit_mixers(context, interaction, memories, shapes, truth, split, months, profiles):
    mixers = {}
    for mode in ARMS:
        for shape_name, basis in shapes.items():
            episodes = validation_episodes(context, interaction, memories[mode], basis, truth, split, months)
            kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
            model = SupportAwareResidualTransfer(months, **kwargs).fit(
                episodes, gamma_k0=profiles[mode].to_dict()["selected"]["gamma"],
                selection_role="source_validation",
            )
            name = f"{mode}_{shape_name}"
            mixers[name] = model
            print(f"{name}: {json.dumps(model.to_dict()['selection_by_k'])}", flush=True)
    return mixers


def create_queries(full, context, interaction, memories, shapes, mixers, support_truth, split, months):
    """Prediction receives only the reserved target-support labels."""
    frames = []
    for k in KS:
        support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
        for mode in ARMS:
            for shape_name, basis in shapes.items():
                name = f"{mode}_{shape_name}"
                model = mixers[name]
                predicted = model.adapt(
                    context[query], interaction[query], memories[mode][query], query,
                    context[support], interaction[support], memories[mode][support], support,
                    support_truth[support], basis[query], basis[support], k=k,
                )
                base = model.selected_base(context[query], interaction[query], memories[mode][query], k=k)
                frame = full.iloc[query][["cell", "station", "month", "analyte", "visibility_role",
                                         "ecological_novelty", "upstream_support"]].copy()
                frame["model_name"], frame["k"] = name, k
                frame["y_pred"], frame["base_pred"] = predicted, base
                frame["adaptation_delta"] = np.log1p(predicted) - np.log1p(base)
                frame["support_count"], frame["regional_gamma"] = k, model.selected_gamma(k)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_one(root, prior_root, partition, seed, runtime):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, sources, old, dataset, split, full, previous_full, _oof = load_inputs(prior)
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")):
            raise ValueError("Saved wrapper differs from the frozen inputs")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    profile_hashes = {mode: sha256_file(prior / f"{mode}.json") for mode in ARMS}
    config = {
        "experiment": "doc_ecological_transfer_v2", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        **{f"{name}_run": str(path) for name, path in sources.items()},
        **{f"{name}_completion_hash": sha256_file(path / "complete.json") for name, path in sources.items()},
        **{key: old[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
                                    "q90_threshold_train", "query_cells")},
        "target_analyte": "doc", "inference_roles": ["train"], "models": MODELS, "arms": ARMS,
        "k_values": KS, "profile_files": profile_hashes, "profile_fitting": "none; frozen v1 memories",
        "gamma_grid": [0, .25, .5, 1], "alpha_grid": [0, .25, .5, .75, 1],
        "ridge_grid": [.1, 1, 10, "infinity"],
        "mixing": pc["mixing"], "gamma_k0": "locked to frozen v1 source-validation choice",
        "gamma_kpositive": "joint gamma/alpha/ridge by source-validation adapted native query MAE",
        "selection_role": "source_validation", "support_basis": "unchanged frozen v4 GRU",
        "torch_threads": torch.get_num_threads(),
        "study_role": "development after seeing v1 support-level results",
    }
    write_json(run / "config.json", config)
    write_json(run / "profile_bindings.json", {"prior_run": str(prior),
               "prior_completion_hash": config["prior_completion_hash"], "profiles": profile_hashes})
    context = full.context_pred.to_numpy()
    interaction = previous_full.interaction_pred.to_numpy()
    shape = tuple(dataset["y"].shape)
    profiles = {mode: EcologicalResidualTransfer.from_dict(json.loads((prior / f"{mode}.json").read_text()))
                for mode in ARMS}
    memories = {mode: model.predict_delta(context.reshape(shape)).ravel() for mode, model in profiles.items()}
    for mode in ARMS:
        np.testing.assert_array_equal(memories[mode], previous_full[f"{mode}_memory"].to_numpy())
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
    mixers = fit_mixers(context, interaction, memories, shapes, truth, split, shape[1], profiles)
    write_json(run / "mixers.json", {name: model.to_dict() for name, model in mixers.items()})
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=shape[1])
    support_truth = np.full(truth.shape, np.nan)
    support_truth[support] = truth[support]
    frame = create_queries(full, context, interaction, memories, shapes, mixers,
                           support_truth, split, shape[1])
    previous_queries = pd.read_parquet(prior / "predictions.parquet")
    for name in frame.model_name.unique():
        a = frame[frame.model_name.eq(name) & frame.k.eq(0)].sort_values("cell")
        b = previous_queries[previous_queries.model_name.eq(name) & previous_queries.k.eq(0)].sort_values("cell")
        np.testing.assert_array_equal(a.y_pred, b.y_pred)
    unchanged = previous_queries[previous_queries.model_name.str.startswith("interaction_")].copy()
    unchanged["regional_gamma"] = 0.0
    frame = pd.concat([frame, unchanged.drop(columns=["split_seed", "seed", "y_true"])], ignore_index=True)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite support-aware query product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    previous_full.to_parquet(run / "full_grid.parquet", index=False)
    model_files = ["mixers.json", "profile_bindings.json"]
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
