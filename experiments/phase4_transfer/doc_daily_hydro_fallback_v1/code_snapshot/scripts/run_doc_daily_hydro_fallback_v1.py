"""Retain the monthly DOC expert where numeric daily discharge inputs are absent."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_ecological_transfer_v1 import load_inputs
from run_doc_ecological_transfer_v2 import validation_episodes
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.daily_hydro_router import DailyHydroRouter
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_fallback_v1")
PRIOR = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
ARMS = ("monthly", "daily", "hybrid")
SHAPES = ("constant", "gru_tuned_anchor")
KS = (0, 1, 3, 5)
REFERENCE_MODELS = tuple(f"prior_{base}_{shape}" for base in
                        ("encoder", "encoder_integrated", "ecological_affine") for shape in SHAPES)
MODELS = (tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
          + tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
          + REFERENCE_MODELS)


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_unified_doc_spatial_v2.py", "scripts/run_doc_tail_residual_v1.py",
                 "scripts/run_doc_ecological_transfer_v1.py", "scripts/run_doc_ecological_transfer_v2.py",
                 "scripts/run_doc_daily_hydro_residual_v1.py",
                 "scripts/run_doc_daily_hydro_fallback_v1.py", str(ROOT / "study_plan.md")):
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


def fit_mixers(context, bases, memory, shapes, truth, split, months):
    """Fit downstream choices solely on the existing source-validation episodes."""
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    mixers = {}
    for arm in ARMS:
        scores = []
        for gamma in (0, .25, .5, 1):
            pred = bases[arm][query] if gamma == 0 else np.maximum(
                0, context[query] + (1-gamma)*(bases[arm][query]-context[query]) + gamma*memory[query])
            scores.append({"gamma": gamma, "mae": float(np.abs(pred-truth[query]).mean())})
        gamma_k0 = min(scores, key=lambda row: (row["mae"], row["gamma"]))["gamma"]
        for shape_name, basis in shapes.items():
            episodes = validation_episodes(context, bases[arm], memory, basis, truth, split, months)
            kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
            mixers[f"{arm}_integrated_{shape_name}"] = SupportAwareResidualTransfer(months, **kwargs).fit(
                episodes, gamma_k0=gamma_k0, selection_role="source_validation")
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


def source_validation_rows(bases, context, memory, mixers, truth, split, months, counts):
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    rows = []
    groups = {"all": np.ones(len(query), dtype=bool), "none_valid": counts[query] == 0,
              "some_valid": counts[query] > 0, "all_valid": counts[query] == 3,
              "partially_valid": (counts[query] > 0) & (counts[query] < 3)}
    for arm in ARMS:
        for stage in ("direct", "integrated"):
            model = mixers[f"{arm}_integrated_constant"]
            pred = bases[arm][query] if stage == "direct" else model.selected_base(
                context[query], bases[arm][query], memory[query], k=0)
            for group, selected in groups.items():
                n = int(selected.sum())
                rows.append({"arm": arm, "stage": stage, "group": group, "n": n,
                             "mae": float(np.abs(pred[selected]-truth[query][selected]).mean()) if n else None,
                             "gamma_k0": 0 if stage == "direct" else model.selected_gamma(0)})
    return rows


def run_one(root, prior_root, partition, seed, runtime):
    start = time.monotonic()
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    pc, sources, old, dataset, split, full, parent_grid, _oof = load_inputs(prior)
    verification = prior_root / "verification/replay_checks.json"
    daily, daily_meta, daily_identity = load_daily_pack(prior_root, old["dataset_hash"], tuple(dataset["y"].shape))
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if (config["runtime_snapshot_hash"] != runtime
                or config["prior_completion_hash"] != sha256_file(prior / "complete.json")
                or config["parent_verification_hash"] != sha256_file(verification)
                or any(config.get(key) != value for key, value in daily_identity.items())):
            raise ValueError("Saved run inputs or execution changed")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified existing product", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    router = DailyHydroRouter()
    config = {
        "experiment": "doc_daily_hydro_fallback_v1", "split_seed": partition, "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat(), "runtime_snapshot_hash": runtime,
        "prior_run": str(prior), "prior_completion_hash": sha256_file(prior / "complete.json"),
        "parent_verification_path": str(verification), "parent_verification_hash": sha256_file(verification),
        **{f"{key}_run": str(path) for key, path in sources.items()},
        **{f"{key}_completion_hash": sha256_file(path / "complete.json") for key, path in sources.items()},
        **{key: pc[key] for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash",
            "q90_threshold_train", "query_cells", "memory_run", "memory_completion_hash", "memory_file_hash")},
        **daily_identity, "target_analyte": "doc", "target_transform": "log1p",
        "residual_scale": "native mg/L", "inference_roles": ["train"], "models": MODELS,
        "arms": ARMS, "k_values": KS, "expert_retraining": False, "route_definition": router.to_dict(),
        "support_basis": "unchanged frozen v4 GRU; identical across arms",
        "ecological_integration": "frozen v1 ecological_affine profile; source-validation gamma/alpha/ridge",
        "gamma_grid": [0, .25, .5, 1], "alpha_grid": [0, .25, .5, .75, 1],
        "ridge_grid": [.1, 1, 10, "infinity"], "full_grid_role": "native bases and integrated K0 bases",
        "study_role": "development after seeing daily hydrology results; fixed availability route",
    }
    write_json(run / "config.json", config)
    write_json(run / "router.json", router.to_dict())
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
    months = dataset["y"].shape[1]
    context = parent_grid.context_pred.to_numpy()
    np.testing.assert_array_equal(context, full.context_pred)
    memory = parent_grid.ecological_memory.to_numpy()
    routed = router.predict_components(parent_grid.monthly_pred.to_numpy(), parent_grid.daily_pred.to_numpy(),
                                       daily.reshape(-1, 8))
    bases = {"context": context, "monthly": routed["monthly_pred"], "daily": routed["daily_pred"],
             "hybrid": routed["routed_pred"]}
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    adapters = fit_adapters(bases, shapes, truth, split, months)
    mixers = fit_mixers(context, bases, memory, shapes, truth, split, months)
    write_json(run / "adapters.json", {name: model.to_dict() for name, model in adapters.items()})
    write_json(run / "mixers.json", {name: model.to_dict() for name, model in mixers.items()})
    pd.DataFrame(source_validation_rows(bases, context, memory, mixers, truth, split, months,
                                      routed["daily_numeric_valid_count"])).to_csv(run / "source_validation.csv", index=False)
    components = full[["cell", "station", "month", "analyte", "visibility_role",
                       "ecological_novelty", "upstream_support"]].copy()
    components["context_pred"], components["ecological_memory"] = context, memory
    components["uses_daily"], components["daily_numeric_valid_count"] = routed["uses_daily"], routed["daily_numeric_valid_count"]
    for column, name in enumerate(daily_meta["feature_names"]):
        components[name] = daily[..., column].ravel()
    for arm in ARMS:
        components[f"{arm}_pred"] = bases[arm]
        components[f"{arm}_integrated_k0_pred"] = mixers[f"{arm}_integrated_constant"].selected_base(
            context, bases[arm], memory, k=0)
    support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(truth.size, np.nan)
    labels[support] = truth[support]
    direct = make_predictions(full, bases, shapes, adapters, labels, split, months)
    direct["regional_gamma"] = 0.0
    integrated = integrated_queries(full, context, bases, memory, shapes, mixers, labels, split, months)
    parent_predictions = pd.read_parquet(prior / "predictions.parquet")
    reference = parent_predictions[parent_predictions.model_name.isin(REFERENCE_MODELS)].copy()
    reference = reference.drop(columns=["split_seed", "seed", "y_true", *daily_meta["feature_names"]])
    frame = pd.concat([direct, integrated, reference], ignore_index=True)
    frame["split_seed"], frame["seed"] = partition, seed
    frame["y_true"] = truth[frame.cell.to_numpy()]
    for name in (*daily_meta["feature_names"], "uses_daily", "daily_numeric_valid_count"):
        frame[name] = components[name].to_numpy()[frame.cell.to_numpy()]
    if not np.isfinite(frame[["y_pred", "y_true", "base_pred", "adaptation_delta", "regional_gamma"]]).all().all():
        raise ValueError("Nonfinite routed query product")
    frame.to_parquet(run / "predictions.parquet", index=False)
    components.to_parquet(run / "full_grid.parquet", index=False)
    model_files = ["router.json", "adapters.json", "mixers.json"]
    for filename in ("predictions.parquet", "full_grid.parquet"):
        bind_product(run, filename, config, runtime, model_files)
    write_json(run / "timing.json", {"elapsed_seconds": time.monotonic()-start})
    files = ["config.json", *model_files, "source_validation.csv", "predictions.parquet", "predictions.meta.json",
             "full_grid.parquet", "full_grid.meta.json", "timing.json"]
    bind_files(run, "complete.json", [run / name for name in files], config)
    print(f"{run.name}: completed routed products, {time.monotonic()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime)
            gc.collect()


if __name__ == "__main__":
    main()
