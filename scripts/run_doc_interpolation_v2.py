"""Frozen retrospective interpolation comparison; enter via isolated run_ladder."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import bind, save_predictions, verify
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    role_labels,
    train_one,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_fusion_benchmark import metrics
from river_graph.experiments.doc_interpolation import (
    RECIPES,
    Interpolator,
    concentration,
    temporal_masks,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)

ROOT = Path("experiments/phase4_transfer/doc_interpolation_v2")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
GEOMETRY = Path("cache/doc_network_work_v1/geometry.npz")
PARENT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")
REFERENCE = Path("experiments/phase4_transfer/doc_network_work_v2")
SPACE_MASKS = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks")
REGIONS = ("1013", "1019", "0708", "1030", "1101")
SEEDS = (42, 43, 44)
LENGTHS = (3, 6, 12)
CODE = ("scripts/run_ladder.py", "scripts/run_doc_interpolation_v2.py", "scripts/start_doc_interpolation_v2.py",
        "scripts/verify_doc_interpolation_v2.py", "src/river_graph/experiments/doc_interpolation.py",
        "tests/test_doc_interpolation_v1.py", "tests/test_doc_interpolation_v2.py", "scripts/run_doc_fusion_benchmark_development_v1.py",
        "scripts/run_doc_river_architecture_comparison_v1.py", "src/river_graph/experiments/doc_fusion_benchmark.py",
        "src/river_graph/models/river_architecture_comparison.py", "src/river_graph/experiments/provenance.py")


def tasks():
    return [f"time_g{length}_s{seed}" for seed in SEEDS for length in LENGTHS]+[f"space_{r}" for r in REGIONS]


def freeze(root, repository, workers):
    root.mkdir(parents=True, exist_ok=True)
    d = torch.load(DATASET, weights_only=False, map_location="cpu")
    observation = np.isfinite(np.asarray(d["y"])) & np.asarray(d["y_mask"])
    for seed in SEEDS:
        for length, mask in temporal_masks(observation, seed).items():
            path = root/"masks"/f"time_g{length}_s{seed}.npz"
            if path.exists():
                with np.load(path) as saved:
                    for key, value in mask.items():
                        np.testing.assert_array_equal(saved[key], value)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **mask)
    for region in REGIONS:
        target = root/"masks"/f"space_{region}.npz"
        original = SPACE_MASKS/f"huc4_{region}.npz"
        if target.exists() and target.read_bytes() != original.read_bytes():
            raise ValueError("spatial mask changed")
        target.write_bytes(original.read_bytes())
    dependencies = {}
    for r in REGIONS:
        directory = REFERENCE/"regions"/f"huc4_{r}"
        verify(directory, "test_complete.json")
        for name in ("predictions.parquet", "predictions.meta.json", "test_complete.json"):
            dependencies[str(directory/name)] = sha256_file(directory/name)
    settings = json.loads((PARENT/"protocol.json").read_text())["settings"]
    settings["torch_threads"] = 1
    protocol = {"experiment": ROOT.name, "tasks": tasks(), "lengths": LENGTHS, "seeds": SEEDS,
        "regions": REGIONS, "workers": workers, "threads": 1, "settings": settings,
        "dataset_sha256": sha256_file(DATASET), "nodes_sha256": sha256_file(NODES),
        "geometry_sha256": sha256_file(GEOMETRY), "dependencies": dependencies,
        "mask_sha256": {t: sha256_file(root/"masks"/f"{t}.npz") for t in tasks()},
        "runtime_snapshot": {name: sha256_file(name) for name in CODE},
        "recipes": RECIPES, "joint_primary": ["mae", "log1p_mae"],
        "selection": "minimum native validation MAE, then log1p MAE, then recipe id; no target tuning",
        "time_design": "3-month fixed query/validation cores; 3/6/12-month excluded DOC windows; distinct >=24-month centers; >=6 remaining observations; largest-gap brackets within 12 outside months; observation dates only",
        "time_information": "offline smoothing; visible DOC before AND after gaps permitted; same visible DOC used as fusion supervision; DOC never an explicit fusion input; 12 past/current driver months",
        "space_information": "unchanged whole-HUC4 source/val/test stations; interpolation support is source TRAIN cells only; validation DOC never neighbours; contemporaneous/future source records permitted for retrospective reconstruction",
        "normalization": "covariates scaled on training station rows; all dates of exogenous covariates permitted; target DOC normalization uses training labels only",
        "kriging": "ordinary kriging with positive-definite exponential covariance, validation-chosen range/nugget; local30 spatial donors or20 space-time donors; native concentration; four-dimensional ECEF/time exponential covariance for ST",
        "distance": "ECEF chord km; river IDW uses cached NHD same-downstream/mainstem station distances; no travel-time interpretation",
        "fallback": "all nonfinite raw interpolation predictions use training station-month median, then station median, then source calendar-month median, then global median; intrinsic interpolation coverage reported separately",
        "postprocessing": "native predictions clamp [0,expm1(15)]; raw predictions retained; common subset uses same clamp",
        "aggregation": "per seed/mask metrics then mean; deterministic spatial methods repeated across nominal seeds, not independent fits; time query population identical across gap lengths within seed",
        "target_gate": "all 14 task selections and 9 temporal fusion fits frozen before ANY new target scoring",
        "scope": "internal development Mississippi retrospective reconstruction; observed hidden truths only; not naturally missing full-grid accuracy or external-river confirmation",
        "original_dispatcher_sha256": sha256_file(repository/"scripts/run_ladder.py"),
        "prior_paper_endpoints_modified": False, "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    # JSON canonicalizes tuples; compare normalized structures.
    protocol = json.loads(json.dumps(protocol))
    path = root/"protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("frozen protocol changed; create a new version")
    else:
        write_json(path, protocol)
        for name in CODE:
            target = root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json", {"protocol_hash": digest(protocol),
            "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(CODE[1]), "dataset_sha256": protocol["dataset_sha256"],
            "nodes_sha256": protocol["nodes_sha256"], "mask_sha256": protocol["mask_sha256"],
            "runtime_snapshot_hash": digest(protocol["runtime_snapshot"])})
    return protocol


def task_context(root, task, data):
    with np.load(root/"masks"/f"{task}.npz") as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test")}
    n, months = data["y"].shape
    rows = np.unique(split["train"]//months)
    geometry = np.load(GEOMETRY)
    interp = Interpolator(split["train"], np.asarray(data["y"]).ravel()[split["train"]],
                          (n, months), geometry["euclidean"], geometry["distance"])
    return split, rows, interp


def configuration(protocol, task):
    return {"experiment": ROOT.name, "task": task, "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "dataset_sha256": protocol["dataset_sha256"], "mask_sha256": protocol["mask_sha256"][task]}


def frame(data, cells, predicted, intrinsic, raw, model, seed, recipe):
    return pd.DataFrame({"cell": cells, "y_true": np.asarray(data["y"]).ravel()[cells],
        "station": cells//data["y"].shape[1], "high_doc": False,
        "y_pred": predicted, "intrinsic": intrinsic, "raw_prediction": raw,
        "model_name": model, "seed": seed, "recipe_id": recipe})


def fit_task(root, task, protocol, data):
    directory, config = root/"tasks"/task, configuration(protocol, task)
    if (directory/"selection_complete.json").exists():
        verify(directory, "selection_complete.json", config)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"config.json", config)
    split, rows, interp = task_context(root, task, data)
    families = [f for f in RECIPES if task.startswith("time") or f not in ("linear_time", "pchip_time")]
    selections, records, predictions = {}, [], []
    seed = int(task.split("_s")[-1]) if task.startswith("time") else 42
    for family in families:
        choices = []
        for recipe in RECIPES[family]:
            predicted, intrinsic, raw = interp.predict(family, recipe, split["val"])
            part = frame(data, split["val"], predicted, intrinsic, raw, family, seed, recipe["id"])
            score = metrics(part)
            choices.append((score["mae"], score["log1p_mae"], recipe["id"]))
            records.append({"model": family, "recipe": recipe, "metrics": score,
                            "intrinsic_coverage": float(intrinsic.mean())})
            predictions.append(part)
        winner = min(choices)[2]
        selections[family] = next(recipe for recipe in RECIPES[family] if recipe["id"] == winner)
    names = ["config.json", "selection.json", "validation_predictions.parquet", "validation_predictions.meta.json"]
    if task.startswith("time"):
        inputs = covariate_inputs({k:v for k,v in data.items() if k not in ("y", "y_mask")}, rows)
        graph = graph_view(data["edge_index"], inputs["order"], rows)
        train_y, val_y = (role_labels(data, split, role, rows) for role in ("train", "val"))
        def progress(_arm, item):
            write_json(root/f"progress_{task}.json", {"stage": "fusion_training", "task": task, **item})
        model, info, trace = train_one("graph_transformer", seed, inputs, graph, graph,
                                      train_y, val_y, protocol["settings"], progress)
        torch.save(model.state_dict(), directory/"model.pt")
        write_json(directory/"model_info.json", info)
        write_json(directory/"normalization.json", inputs["normalization"])
        write_json(directory/"trace.json", trace)
        times = np.unique(split["val"] % interp.t)
        grid = native_prediction(predict(model, inputs, graph, times, protocol["settings"]["batch_months"]), info["target_normalization"])
        mapping = {int(s):i for i,s in enumerate(rows)}
        values = grid[np.searchsorted(times, split["val"] % interp.t), [mapping[int(c//interp.t)] for c in split["val"]]]
        predictions.append(frame(data, split["val"], values, np.ones(len(values), bool), values, "fixed_fusion", seed, "fixed"))
        names.extend(["model.pt", "model_info.json", "normalization.json", "trace.json"])
    write_json(directory/"selection.json", {"selections": selections, "validation_records": records,
        "test_used": False, "source_cells_sha256": digest(split["train"].tolist())})
    save_predictions(directory, "validation_predictions", pd.concat(predictions, ignore_index=True), config)
    bind(directory, "selection_complete.json", config, names)
    write_json(root/f"progress_{task}.json", {"stage": "selection_complete", "task": task})
    print(json.dumps({"stage": "selection_complete", "task": task}), flush=True)


def score_task(root, task, protocol, data):
    directory, config = root/"tasks"/task, configuration(protocol, task)
    if (directory/"test_complete.json").exists():
        verify(directory, "test_complete.json", config)
        return
    split, rows, interp = task_context(root, task, data)
    selections = json.loads((directory/"selection.json").read_text())["selections"]
    parts = []
    seeds = [int(task.split("_s")[-1])] if task.startswith("time") else list(SEEDS)
    for family, recipe in selections.items():
        values, intrinsic, raw = interp.predict(family, recipe, split["test"])
        parts.extend(frame(data, split["test"], values, intrinsic, raw, family, seed, recipe["id"]) for seed in seeds)
    if task.startswith("time"):
        inputs = covariate_inputs({k:v for k,v in data.items() if k not in ("y", "y_mask")}, rows)
        graph = graph_view(data["edge_index"], inputs["order"], rows)
        model = RiverArchitectureModel("graph_transformer", **{k:protocol["settings"][k]
            for k in ("hidden", "temporal_hidden", "heads", "layers", "dropout")})
        model.load_state_dict(torch.load(directory/"model.pt", weights_only=True, map_location="cpu"))
        info = json.loads((directory/"model_info.json").read_text())
        times = np.unique(split["test"] % interp.t)
        grid = native_prediction(predict(model, inputs, graph, times, protocol["settings"]["batch_months"]), info["target_normalization"])
        mapping = {int(s):i for i,s in enumerate(rows)}
        values = grid[np.searchsorted(times, split["test"] % interp.t), [mapping[int(c//interp.t)] for c in split["test"]]]
        parts.append(frame(data, split["test"], values, np.ones(len(values), bool), values, "fixed_fusion", seeds[0], "fixed"))
    else:
        existing = pd.read_parquet(REFERENCE/"regions"/f"huc4_{task[6:]}"/"predictions.parquet")
        part = existing[existing.model_name == "fixed_fusion"].copy()
        part["intrinsic"], part["raw_prediction"], part["recipe_id"] = True, part.y_pred, "frozen_parent"
        part["station"], part["high_doc"] = part.cell//data["y"].shape[1], False
        parts.append(part[["cell", "station", "high_doc", "y_true", "y_pred", "intrinsic", "raw_prediction", "model_name", "seed", "recipe_id"]])
    output = pd.concat(parts, ignore_index=True)
    save_predictions(directory, "predictions", output, config)
    bind(directory, "test_complete.json", config, ["predictions.parquet", "predictions.meta.json"])


def analyze(root, protocol):
    parts = []
    for task in tasks():
        part = pd.read_parquet(root/"tasks"/task/"predictions.parquet")
        part["task"], part["track"] = task, "time" if task.startswith("time") else "space"
        part["gap_months"] = int(task.split("_")[1][1:]) if task.startswith("time") else 0
        parts.append(part)
    all_predictions = pd.concat(parts, ignore_index=True)
    rows, common_rows = [], []
    for (track, gap, seed), group in all_predictions.groupby(["track", "gap_months", "seed"]):
        common = group.groupby(["task", "cell"]).intrinsic.all()
        common_keys = set(common[common].index)
        for model, frame0 in group.groupby("model_name"):
            row = {"track": track, "gap_months": int(gap), "seed": int(seed), "model_name": model,
                   **metrics(frame0),
                   "query_count": len(frame0), "intrinsic_coverage": float(frame0.intrinsic.mean()), "coverage": 1.}
            rows.append(row)
            matched = frame0[["task", "cell"]].apply(tuple, axis=1).isin(common_keys)
            p = frame0[matched]
            if len(p):
                common_rows.append({**{k:row[k] for k in ("track", "gap_months", "seed", "model_name")},
                    **metrics(p.assign(y_pred=concentration(p.raw_prediction.to_numpy()))), "query_count": len(p)})
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(analysis/"by_seed.csv", index=False)
    by_seed.groupby(["track", "gap_months", "model_name"], as_index=False).mean(numeric_only=True).to_csv(analysis/"summary.csv", index=False)
    pd.DataFrame(common_rows).to_csv(analysis/"common_subset_by_seed.csv", index=False)
    bind(analysis, "complete.json", {"protocol_hash": digest(protocol)}, ["summary.csv", "by_seed.csv", "common_subset_by_seed.csv"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        protocol = (json.loads((args.root/"protocol.json").read_text()) if args.task
                    else freeze(args.root, args.repository, args.workers))
        if args.freeze_only:
            print(json.dumps({"stage": "frozen", "protocol_hash": digest(protocol)}))
            return
        data = torch.load(DATASET, weights_only=False, map_location="cpu")
        if args.task:
            fit_task(args.root, args.task, protocol, data)
            return
        pending, running = tasks(), []
        logs = args.root/"worker_logs"
        logs.mkdir(exist_ok=True)
        while pending or running:
            while pending and len(running) < args.workers:
                task = pending.pop(0)
                log = (logs/f"{task}.log").open("a")
                process = subprocess.Popen([os.sys.executable, "scripts/run_ladder.py", "--root", str(args.root),
                    "--task", task], stdout=log, stderr=subprocess.STDOUT)
                running.append((task, process, log))
            for item in list(running):
                task, process, log = item
                if process.poll() is not None:
                    log.close()
                    if process.returncode:
                        raise RuntimeError(f"task {task} failed; see its log")
                    running.remove(item)
            if running:
                time.sleep(.5)
        gate = {"protocol_hash": digest(protocol), "selection_hashes": {
            task: sha256_file(args.root/"tasks"/task/"selection_complete.json") for task in tasks()}, "test_started": False}
        write_json(args.root/"all_selections.json", gate)
        for task in tasks():
            score_task(args.root, task, protocol, data)
        analyze(args.root, protocol)
        write_json(args.root/"batch_complete.json", {"protocol_hash": digest(protocol), "tasks": len(tasks()), "new_fusion_fits": 9})
        print(json.dumps({"stage": "complete", "tasks": len(tasks()), "new_fusion_fits": 9}), flush=True)


if __name__ == "__main__":
    main()
