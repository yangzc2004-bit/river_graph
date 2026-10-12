"""Frozen same-split DOC information ablations through isolated run_ladder."""
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
from run_doc_fusion_benchmark_development_v1 import (
    bind,
    save_predictions,
    utc_now,
    verify,
)
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    role_labels,
    train_one,
    verify_run,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_fusion_benchmark import metrics, paired_basin_interval
from river_graph.experiments.doc_information_ablation import (
    ARMS,
    DEFINITIONS,
    PRIMARY,
    ablate_graph,
    ablate_inputs,
    model_arm,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)

ROOT = Path("experiments/phase4_transfer/doc_information_ablation_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
SPACE_PARENT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")
TIME_PARENT = Path("experiments/phase4_transfer/doc_interpolation_v2")
BENCHMARK = Path("experiments/phase4_transfer/doc_network_work_v2")
REGIONS = ("1013", "1019", "0708", "1030", "1101")
SEEDS = (42, 43, 44)
LENGTHS = (3, 6, 12)
CODE = ("scripts/run_ladder.py", "scripts/run_doc_information_ablation_v1.py",
        "scripts/start_doc_information_ablation_v1.py", "scripts/verify_doc_information_ablation_v1.py",
        "src/river_graph/experiments/doc_information_ablation.py", "tests/test_doc_information_ablation_v1.py",
        "scripts/run_doc_fusion_benchmark_development_v1.py", "scripts/run_doc_river_architecture_comparison_v1.py",
        "src/river_graph/experiments/doc_fusion_benchmark.py", "src/river_graph/experiments/provenance.py",
        "src/river_graph/models/river_architecture_comparison.py")


def tasks():
    result = []
    for region in REGIONS:
        for seed in SEEDS:
            for arm in ARMS:
                result.append({"id": f"space_{region}_{arm}_s{seed}", "track": "space", "region": region,
                               "seed": seed, "arm": arm, "mask": f"space_{region}", "gap_months": 0})
    for seed in SEEDS:
        for gap in LENGTHS:
            for arm in ("fixed_fusion", *PRIMARY):
                result.append({"id": f"time_g{gap}_{arm}_s{seed}", "track": "time", "region": "temporal",
                               "seed": seed, "arm": arm, "mask": f"time_g{gap}_s{seed}", "gap_months": gap})
    return result


def parent(task):
    if task["track"] == "time" and task["arm"] == "fixed_fusion":
        return TIME_PARENT/"tasks"/task["mask"], "model.pt", "model_info.json", False
    if task["track"] == "space" and task["arm"] in ("fixed_fusion", "local_no_messages"):
        arm = model_arm(task["arm"])
        return SPACE_PARENT/"runs"/f"huc4_{task['region']}_seed{task['seed']}", f"{arm}.pt", f"{arm}.json", True
    return None


def freeze(root, repository, workers):
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads((BENCHMARK/"protocol.json").read_text())
    temporal = json.loads((TIME_PARENT/"protocol.json").read_text())
    original = json.loads((SPACE_PARENT/"protocol.json").read_text())
    for name in ("scripts/run_doc_river_architecture_comparison_v1.py",
                 "src/river_graph/models/river_architecture_comparison.py"):
        if sha256_file(name) != original["runtime_snapshot"][name]:
            raise ValueError("reused neural execution code changed")
    if previous["dataset_sha256"] != sha256_file(DATASET) or previous["nodes_sha256"] != sha256_file(NODES):
        raise ValueError("comparison dataset changed")
    for field in ("regions", "seeds"):
        if previous[field] != list(REGIONS if field == "regions" else SEEDS) or previous[field] != temporal[field]:
            raise ValueError("comparison task definition changed")
    dependencies = {str(p): sha256_file(p) for p in (
        BENCHMARK/"protocol.json", BENCHMARK/"analysis/independent_audit.json",
        TIME_PARENT/"protocol.json", TIME_PARENT/"analysis/independent_audit.json", SPACE_PARENT/"protocol.json")}
    for task in tasks():
        source = TIME_PARENT/"masks"/f"{task['mask']}.npz"
        destination = root/"masks"/source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("copied comparison mask changed")
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(source.read_bytes())
        package = parent(task)
        if package:
            directory, checkpoint, info, _ = package
            if task["track"] == "space":
                verify_run(directory, json.loads((directory/"config.json").read_text()))
            else:
                verify(directory, "selection_complete.json")
            for name in (checkpoint, info, "normalization.json"):
                dependencies[str(directory/name)] = sha256_file(directory/name)
    settings = {"space": original["settings"], "time": temporal["settings"]}
    protocol = {"experiment": ROOT.name, "tasks": tasks(), "regions": list(REGIONS), "seeds": list(SEEDS),
        "lengths": list(LENGTHS), "arms": list(ARMS), "primary_arms": list(PRIMARY), "definitions": DEFINITIONS,
        "settings": settings, "workers": workers, "dataset_sha256": sha256_file(DATASET),
        "nodes_sha256": sha256_file(NODES), "dependencies": dependencies,
        "mask_sha256": {name: sha256_file(root/"masks"/f"{name}.npz") for name in sorted({t["mask"] for t in tasks()})},
        "runtime_snapshot": {name: sha256_file(name) for name in CODE},
        "selection": "same fixed architecture/settings; best source-validation native MAE epoch; no recipe or target tuning",
        "joint_primary": ["mae", "log1p_mae"], "target_gate": "all 141 selections bound before any new target scoring",
        "matching": "same parameter tensors and seed initialization, optimizer, month permutations, label weights and stopping rule; zero masked inputs retain constant/bias paths; local controls use the same attention projections and FF blocks",
        "new_fits": 102, "reused_fits": 39, "space_new_fits": 75, "time_new_fits": 27,
        "river_removal": "remove dense cross-station exchange and static columns 0..3; ecological/climate descriptors and generic coordinates retained; no edge attributes are inputs to this model",
        "overlap": "environment package includes direct river descriptors, so removals are conditional and contributions need not add to 100%; network relation control retains static river descriptors",
        "time_control": "current-only retains GRU operator/window size, zeros all earlier steps and current age channels; dynamic-input removal additionally zeros current drivers and season",
        "information": "no DOC inputs or station embeddings; source-only training graph; source+validation/target exogenous covariates at inference; same retrospective query populations as parent comparisons",
        "aggregation": "pool query cells per seed, then arithmetic seed mean; no prediction ensemble; temporal core query cells fixed across gap lengths within seed",
        "uncertainty": "5000 paired five-region bootstrap draws after seed-error averaging for spatial joint endpoints; exploratory/unadjusted; no temporal population significance from three masking seeds",
        "scope": "internal Mississippi development, observed hidden labels only; not external-river, true missing-cell or virtual-site validation",
        "postprocessing": "source-standardized log1p DOC MSE; inherited native clamp [0,expm1(15)]",
        "original_dispatcher_sha256": sha256_file(repository/"scripts/run_ladder.py"), "prior_paper_endpoints_modified": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    for region in REGIONS:
        if protocol["mask_sha256"][f"space_{region}"] != previous["mask_sha256"][region]:
            raise ValueError("spatial comparison split changed")
    protocol = json.loads(json.dumps(protocol))
    if (root/"protocol.json").exists():
        if json.loads((root/"protocol.json").read_text()) != protocol:
            raise ValueError("frozen ablation changed; create a new version")
    else:
        write_json(root/"protocol.json", protocol)
        for name in CODE:
            destination = root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(CODE[1]), "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
            "dataset_sha256": protocol["dataset_sha256"], "nodes_sha256": protocol["nodes_sha256"],
            "mask_sha256": protocol["mask_sha256"], "prior_paper_endpoints_modified": False})
    return protocol


def configuration(protocol, task):
    return {"experiment": ROOT.name, **task, "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]), "dataset_sha256": protocol["dataset_sha256"],
        "mask_sha256": protocol["mask_sha256"][task["mask"]]}


def task_context(root, task, data):
    with np.load(root/"masks"/f"{task['mask']}.npz") as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test")}
    months = data["y"].shape[1]
    rows = {role: np.unique(cells//months) for role, cells in split.items()}
    if task["track"] == "space":
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if np.intersect1d(rows[a], rows[b]).size:
                raise ValueError("whole-station roles overlap")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(split[a], split[b]).size:
            raise ValueError("label roles overlap")
    original = covariate_inputs({k:v for k,v in data.items() if k not in ("y", "y_mask")}, rows["train"])
    inputs = ablate_inputs(original, task["arm"])
    graphs = {}
    for role in ("train", "val", "test"):
        nodes = rows["train"] if role == "train" else np.union1d(rows["train"], rows[role])
        graphs[role] = ablate_graph(graph_view(data["edge_index"], original["order"], nodes), task["arm"])
    return {"split": split, "rows": rows, "inputs": inputs, "original": original, "graphs": graphs, "months": months}


def load_model(root, task, protocol):
    directory = root/"fits"/task["id"]
    info = json.loads((directory/"info.json").read_text())
    settings = protocol["settings"][task["track"]]
    model = RiverArchitectureModel(model_arm(task["arm"]), **{key:settings[key] for key in
        ("hidden", "temporal_hidden", "heads", "layers", "dropout")})
    package = parent(task)
    if package:
        source, checkpoint, _, payload = package
        loaded = torch.load(source/checkpoint, weights_only=not payload, map_location="cpu")
        state = loaded["state"] if payload else loaded
    else:
        state = torch.load(directory/"model.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    if sum(p.numel() for p in model.parameters()) != info["parameters"]:
        raise ValueError("unmatched parameter tensors")
    return model, info


def cells_prediction(model, ctx, cells, role, norm, batch):
    times = np.unique(cells % ctx["months"])
    graph = ctx["graphs"][role]
    grid = native_prediction(predict(model, ctx["inputs"], graph, times, batch), norm)
    return grid[np.searchsorted(times, cells % ctx["months"]), np.searchsorted(graph["rows"].numpy(), cells//ctx["months"])]


def frame(data, task, cells, predicted):
    return pd.DataFrame({"cell": cells, "station": cells//data["y"].shape[1],
        "y_true": np.asarray(data["y"]).ravel()[cells], "y_pred": predicted, "high_doc": False,
        "target_huc4": task["region"], "model_name": task["arm"], "seed": task["seed"],
        "track": task["track"], "gap_months": task["gap_months"]})


def fit_task(root, task, protocol, data):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"selection_complete.json").exists():
        verify(directory, "selection_complete.json", config)
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"config.json", config)
    ctx = task_context(root, task, data)
    settings = protocol["settings"][task["track"]]
    package = parent(task)
    names = ["config.json", "info.json", "normalization.json", "selection.json", "validation_predictions.parquet", "validation_predictions.meta.json"]
    if package:
        source, _, info_name, _ = package
        if json.loads((source/"normalization.json").read_text()) != ctx["original"]["normalization"]:
            raise ValueError("reference normalization differs")
        info = {**json.loads((source/info_name).read_text()), "reused": True, "parent": str(source)}
        write_json(directory/"info.json", info)
        model, _ = load_model(root, task, protocol)
    else:
        train_y = role_labels(data, ctx["split"], "train", ctx["graphs"]["train"]["rows"].numpy())
        val_y = role_labels(data, ctx["split"], "val", ctx["graphs"]["val"]["rows"].numpy())
        def progress(_arm, item):
            write_json(root/f"progress_{task['id']}.json", {"stage": "training", "task": task["id"], **item})
        model, info, trace = train_one(model_arm(task["arm"]), task["seed"], ctx["inputs"],
            ctx["graphs"]["train"], ctx["graphs"]["val"], train_y, val_y, settings, progress)
        info["reused"] = False
        torch.save(model.state_dict(), directory/"model.pt")
        write_json(directory/"info.json", info)
        write_json(directory/"trace.json", trace)
        names.extend(["model.pt", "trace.json"])
    write_json(directory/"normalization.json", ctx["original"]["normalization"])
    cells = ctx["split"]["val"]
    predicted = cells_prediction(model, ctx, cells, "val", info["target_normalization"], settings["batch_months"])
    part = frame(data, task, cells, predicted)
    write_json(directory/"selection.json", {"test_used": False, "validation_metrics": metrics(part),
        "best_epoch": info["best_epoch"], "definition": DEFINITIONS[task["arm"]]})
    save_predictions(directory, "validation_predictions", part, config)
    bind(directory, "selection_complete.json", config, names)
    write_json(root/f"progress_{task['id']}.json", {"stage": "selection_complete", "task": task["id"]})
    print(json.dumps({"stage": "selection_complete", "task": task["id"]}), flush=True)


def score_task(root, task, protocol, data):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"test_complete.json").exists():
        verify(directory, "test_complete.json", config)
        return
    ctx = task_context(root, task, data)
    settings = protocol["settings"][task["track"]]
    model, info = load_model(root, task, protocol)
    cells = ctx["split"]["test"]
    predicted = cells_prediction(model, ctx, cells, "test", info["target_normalization"], settings["batch_months"])
    save_predictions(directory, "predictions", frame(data, task, cells, predicted), config)
    bind(directory, "test_complete.json", config, ["predictions.parquet", "predictions.meta.json"])


def analyze(root, protocol):
    all_predictions = pd.concat([pd.read_parquet(root/"fits"/t["id"]/"predictions.parquet") for t in protocol["tasks"]], ignore_index=True)
    rows = [{"track": track, "gap_months": int(gap), "seed": int(seed), "model_name": model,
             **metrics(part), "query_count": len(part), "coverage": float(np.isfinite(part.y_pred).mean())}
            for (track, gap, seed, model), part in all_predictions.groupby(["track", "gap_months", "seed", "model_name"])]
    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby(["track", "gap_months", "model_name"], as_index=False).mean(numeric_only=True)
    for index, item in summary.iterrows():
        base = summary[(summary.track == item.track) & (summary.gap_months == item.gap_months) & (summary.model_name == "fixed_fusion")].iloc[0]
        summary.loc[index, "mae_increase"] = item.mae-base.mae
        summary.loc[index, "mae_increase_percent"] = 100*(item.mae-base.mae)/base.mae
    spatial = all_predictions[all_predictions.track == "space"]
    intervals = [paired_basin_interval(spatial, arm, endpoint) for arm in ARMS if arm != "fixed_fusion" for endpoint in ("mae", "log1p_mae")]
    regional = [{"region": region, "seed": int(seed), "model_name": arm, **metrics(part)}
        for (region, seed, arm), part in spatial.groupby(["target_huc4", "seed", "model_name"])]
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    by_seed.to_csv(analysis/"by_seed.csv", index=False)
    summary.to_csv(analysis/"summary.csv", index=False)
    pd.DataFrame(regional).to_csv(analysis/"by_region_seed.csv", index=False)
    write_json(analysis/"paired_comparisons.json", intervals)
    bind(analysis, "complete.json", {"protocol_hash": digest(protocol)},
         ["by_seed.csv", "summary.csv", "by_region_seed.csv", "paired_comparisons.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()
    protocol = json.loads((args.root/"protocol.json").read_text()) if args.task else freeze(args.root, args.repository, args.workers)
    if args.freeze_only:
        print(json.dumps({"stage": "frozen", "protocol_hash": digest(protocol)}))
        return
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    if args.task:
        task = next(t for t in protocol["tasks"] if t["id"] == args.task)
        threads = protocol["settings"][task["track"]]["torch_threads"]
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            fit_task(args.root, task, protocol, data)
        return
    pending, running = list(protocol["tasks"]), []
    logs = args.root/"worker_logs"
    logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running) < args.workers:
            task = pending.pop(0)
            log = (logs/f"{task['id']}.log").open("a")
            process = subprocess.Popen([os.sys.executable, "scripts/run_ladder.py", "--root", str(args.root), "--task", task["id"]], stdout=log, stderr=subprocess.STDOUT)
            running.append((task, process, log))
        time.sleep(.5)
        remaining = []
        for task, process, log in running:
            status = process.poll()
            if status is None:
                remaining.append((task, process, log))
            else:
                log.close()
                if status:
                    raise RuntimeError(f"worker failed: {task['id']} ({status}); inspect its log")
        running = remaining
    gate = {t["id"]: sha256_file(args.root/"fits"/t["id"]/"selection_complete.json") for t in protocol["tasks"]}
    write_json(args.root/"all_selections.json", {"protocol_hash": digest(protocol), "fit_receipts": gate, "test_used": False, "frozen_at": utc_now()})
    for task in protocol["tasks"]:
        threads = protocol["settings"][task["track"]]["torch_threads"]
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            score_task(args.root, task, protocol, data)
    analyze(args.root, protocol)
    write_json(args.root/"batch_complete.json", {"protocol_hash": digest(protocol), "tasks": len(protocol["tasks"]), "new_fits": 102, "reused_fits": 39})
    print(json.dumps({"stage": "complete", "tasks": len(protocol["tasks"])}), flush=True)


if __name__ == "__main__":
    main()
