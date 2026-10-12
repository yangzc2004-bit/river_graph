"""Train the unchanged unified DOC model separately on each available river."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_doc_unified_transfer_v2 as previous
import torch
from run_doc_fusion_benchmark_development_v1 import (
    bind,
    save_predictions,
    utc_now,
    verify,
)
from run_doc_river_architecture_comparison_v1 import digest, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_fusion_benchmark import metrics
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_per_river_v2")
SPLITS, SEEDS = (142, 143, 144), (42, 43, 44)
RIVERS = {
    "delaware": {
        "name": "Delaware HUC02040104",
        "inputs": "experiments/phase4_transfer/doc_unified_transfer_v2/external/inputs.npz",
        "labels": "experiments/phase4_transfer/doc_unmonitored_tasks_v1/external_inputs/02040104/dataset_scoring_only.pt",
    },
    "hackensack_passaic": {
        "name": "Hackensack-Passaic HUC02030103",
        "inputs": "experiments/phase4_transfer/doc_unified_generalization_v1/confirmation/inputs.npz",
        "labels": "experiments/phase4_transfer/doc_unified_generalization_v1/external_build/external_inputs/02030103/dataset_scoring_only.pt",
    },
}


def load_data(river):
    spec = RIVERS[river]
    labels = torch.load(spec["labels"], weights_only=False, map_location="cpu")
    with np.load(spec["inputs"]) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    for key in ("site_no", "months"):
        if not np.array_equal(np.asarray(labels[key], str), np.asarray(data[key], str)):
            raise ValueError(f"unaligned {key}: {river}")
    data["y"] = np.asarray(labels["y"]).copy()
    data["y_mask"] = np.asarray(labels["y_mask"], bool).copy()
    data["feature_channels"] = ["temperature", "discharge"]
    if data["y"].shape != data["x"].shape[:2]:
        raise ValueError("DOC and hydro grid mismatch")
    if not np.all(np.isfinite(data["y"][data["y_mask"]]) & (data["y"][data["y_mask"]] >= 0)):
        raise ValueError("invalid observed DOC")
    return data


def station_split(data, seed):
    n, months = data["y"].shape
    observed = np.asarray(data["y_mask"], bool)
    eligible = np.flatnonzero(observed.any(1))
    if len(eligible) != n or n < 10:
        raise ValueError("expected a measured river-station cohort")
    permutation = np.random.default_rng(seed).permutation(eligible)
    n_test, n_val = round(.2*n), round(.15*n)
    roles = {"test": permutation[:n_test], "val": permutation[n_test:n_test+n_val],
             "train": permutation[n_test+n_val:]}
    cells = np.flatnonzero(observed.ravel())
    split = {role: cells[np.isin(cells//months, rows)] for role, rows in roles.items()}
    if not all(len(value) for value in split.values()):
        raise ValueError("empty observed-label role")
    if not np.array_equal(np.sort(np.concatenate(list(split.values()))), cells):
        raise ValueError("station partition does not cover observed labels")
    return split


def tasks():
    return [{"river": river, "split": split, "seed": seed, "arm": "direction_heads",
             "id": f"{river}_split{split}_s{seed}"}
            for river in RIVERS for split in SPLITS for seed in SEEDS]


def freeze(root, repository, workers):
    root.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path("experiments/phase4_transfer/doc_unified_transfer_v2/protocol.json").read_text())
    code = tuple(name for name in old["runtime_snapshot"] if name.endswith(".py")) + (
        "scripts/run_doc_per_river_v2.py", "scripts/start_doc_per_river_v2.py")
    for name, expected in old["runtime_snapshot"].items():
        if name.startswith("src/river_graph/models/") and sha256_file(name) != expected:
            raise ValueError("the established model implementation changed")
    datasets, mask_hashes, counts = {}, {}, []
    for river, spec in RIVERS.items():
        data = load_data(river)
        datasets[river] = {key: sha256_file(spec[key]) for key in ("inputs", "labels")}
        for split_seed in SPLITS:
            split = station_split(data, split_seed)
            path = root/river/"masks"/f"split{split_seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path) as existing:
                    if any(not np.array_equal(existing[role], cells) for role, cells in split.items()):
                        raise ValueError("existing station split changed")
            else:
                np.savez_compressed(path, **split)
            mask_hashes[f"{river}_{split_seed}"] = sha256_file(path)
            counts.append({"river": river, "split": split_seed, "stations": len(data["site_no"]),
                "months": len(data["months"]), "observed_cells": int(data["y_mask"].sum()),
                **{f"{role}_stations": len(np.unique(cells//len(data["months"]))) for role, cells in split.items()},
                **{f"{role}_cells": len(cells) for role, cells in split.items()}})
    protocol = {"experiment": ROOT.name, "tasks": tasks(), "rivers": RIVERS,
        "settings": old["settings"], "neural_parameters": 12565, "datasets": datasets,
        "mask_sha256": mask_hashes, "runtime_snapshot": {name: sha256_file(name) for name in code},
        "workers": workers, "model": old["model"],
        "training": "fresh random initialization on the current river only; unchanged model/settings; no transferred checkpoint and no added gates",
        "split": "whole-station train/validation/test approximately65/15/20%, seeds142/143/144",
        "evaluation": "same train-only normalization, source-only training graph and training+held-station covariate inference graphs as the Mississippi station experiment; no DOC inputs",
        "selection": "inherited validation-MAE early stopping; no architecture or hyperparameter search",
        "aggregation": "arithmetic means of9 fit metrics per river, not an ensemble; river populations reported separately",
        "scope": "same-framework within-river reconstruction on two previously inspected monthly station datasets; not zero-shot transfer, arbitrary virtual-site validation or a SOTA comparison",
        "prior_paper_endpoints_modified": False}
    path = root/"protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("run configuration changed")
    else:
        write_json(path, protocol)
        for name in code:
            destination = root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json", {"created_at": utc_now(), "protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(__file__), "datasets": datasets, "mask_sha256": mask_hashes})
    pd.DataFrame(counts).to_csv(root/"cohort.csv", index=False)
    return protocol


def configuration(protocol, task):
    return {"experiment": ROOT.name, **task, "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "dataset_sha256": protocol["datasets"][task["river"]]["labels"],
        "input_sha256": protocol["datasets"][task["river"]]["inputs"],
        "mask_sha256": protocol["mask_sha256"][f"{task['river']}_{task['split']}"]}


def score(root, task, protocol, data):
    river_root = root/task["river"]
    directory = river_root/"fits"/task["id"]
    config = configuration(protocol, task)
    verify(directory, "selection_complete.json", config)
    if (directory/"test_complete.json").exists():
        verify(directory, "test_complete.json", config)
        return
    model, info = previous.load_model(directory, task, protocol)
    ctx = previous.context(river_root, task, data)
    values = previous.cell_prediction(model, ctx, "test", info["target_normalization"], protocol["settings"]["batch_months"])
    part = previous.frame(data, ctx["split"]["test"], values, task, "within_river_test")
    part["river"] = task["river"]
    save_predictions(directory, "test_predictions", part, config)
    bind(directory, "test_complete.json", config, ["test_predictions.parquet", "test_predictions.meta.json"])


def audit_and_summarize(root, protocol):
    detail, replay_max, metric_max = [], 0., 0.
    previous.configuration = configuration
    for river, spec in RIVERS.items():
        for key in ("inputs", "labels"):
            if sha256_file(spec[key]) != protocol["datasets"][river][key]:
                raise ValueError("river data changed")
    for task in tasks():
        data = load_data(task["river"])
        directory = root/task["river"]/"fits"/task["id"]
        config = configuration(protocol, task)
        verify(directory, "selection_complete.json", config)
        verify(directory, "test_complete.json", config)
        ctx = previous.context(root/task["river"], task, data)
        if json.loads((directory/"normalization.json").read_text()) != ctx["inputs"]["normalization"]:
            raise ValueError("normalization was not fitted only on training stations")
        model, info = previous.load_model(directory, task, protocol)
        if sum(p.numel() for p in model.parameters()) != 12565:
            raise ValueError("architecture capacity changed")
        for role, prefix in (("val", "validation"), ("test", "test")):
            part = pd.read_parquet(directory/f"{prefix}_predictions.parquet")
            cells = ctx["split"][role]
            np.testing.assert_array_equal(part.cell, cells)
            np.testing.assert_array_equal(part.y_true, data["y"].ravel()[cells])
            replay = previous.cell_prediction(model, ctx, role, info["target_normalization"], protocol["settings"]["batch_months"])
            difference = float(np.max(np.abs(replay-part.y_pred.to_numpy())))
            replay_max = max(replay_max, difference)
            if difference > 1e-6:
                raise ValueError("prediction replay differs")
            if role == "test":
                y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
                independent = {"mae": float(np.abs(p-y).mean()), "log1p_mae": float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
                    "rmse": float(np.sqrt(np.square(p-y).mean())), "r2": float(1-np.square(p-y).sum()/np.square(y-y.mean()).sum())}
                calculated = metrics(part)
                metric_max = max(metric_max, *(abs(independent[key]-calculated[key]) for key in independent))
                detail.append({**task, **calculated, "test_stations": part.station.nunique(), "best_epoch": info["best_epoch"]})
    if metric_max > 1e-10:
        raise ValueError("metric recomputation differs")
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    detail = pd.DataFrame(detail)
    detail.to_csv(analysis/"by_fit.csv", index=False)
    columns = ["mae", "log1p_mae", "rmse", "r2", "bias", "station_macro_mae", "query_count"]
    summary = detail.groupby("river", as_index=False)[columns].mean()
    summary["fits"] = 9
    summary.to_csv(analysis/"summary.csv", index=False)
    write_json(analysis/"audit.json", {"passed": True, "fits": len(detail), "prediction_replay_max_abs": replay_max,
        "metric_recompute_max_abs": metric_max, "new_gates": False, "transferred_weights": False})
    write_json(root/"batch_complete.json", {"completed_at": utc_now(), "fits": len(detail), "protocol_hash": digest(protocol)})
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.inspect:
        for river in RIVERS:
            data = load_data(river)
            split = station_split(data, 142)
            print(json.dumps({"river": river, "stations": len(data["site_no"]), "months": len(data["months"]),
                "doc_cells": int(data["y_mask"].sum()), "split_cells": {key: len(value) for key, value in split.items()}}))
        return
    protocol = json.loads((args.root/"protocol.json").read_text()) if args.task or args.audit else freeze(args.root, args.repository, args.workers)
    if args.freeze_only:
        print(json.dumps({"stage": "ready", "fits": len(tasks())}))
        return
    threads = protocol["settings"]["torch_threads"]
    torch.set_num_threads(threads)
    previous.configuration = configuration
    if args.audit:
        with threadpool_limits(limits=threads):
            audit_and_summarize(args.root, protocol)
        return
    if args.task:
        task = next(task for task in tasks() if task["id"] == args.task)
        with threadpool_limits(limits=threads):
            previous.fit(args.root/task["river"], task, protocol, load_data(task["river"]))
        return
    pending, running = list(tasks()), []
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
                    for _, worker, _ in running:
                        if worker.poll() is None:
                            worker.terminate()
                    raise RuntimeError(f"training failed: {task['id']}")
        running = remaining
    with threadpool_limits(limits=threads):
        for task in tasks():
            score(args.root, task, protocol, load_data(task["river"]))
        audit_and_summarize(args.root, protocol)


if __name__ == "__main__":
    main()
