"""Matched environmental/GRU models with local, GNN, hierarchical, or global attention.

Invoke through run_ladder.py --experiment doc-river-architecture-comparison-v1.
The protocol and source snapshot are frozen before any training. Completed
products are immutable and verified rather than overwritten on resumption.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import HUC4_BLOCKS
from river_graph.models.river_architecture_comparison import (
    ARMS,
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)

ROOT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
MASKS = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks")
CODE = ("scripts/run_ladder.py", "scripts/run_doc_river_architecture_comparison_v1.py",
        "src/river_graph/models/river_architecture_comparison.py",
        "src/river_graph/experiments/provenance.py", "src/river_graph/experiments/unmonitored_doc.py",
        "tests/test_river_architecture_comparison.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(root, settings):
    snapshot = {name: sha256_file(name) for name in CODE}
    protocol = {"experiment": ROOT.name, "settings": settings, "arms": list(ARMS),
        "regions": list(HUC4_BLOCKS), "seeds": [42, 43, 44], "dataset": str(DATASET),
        "dataset_hash": sha256_file(DATASET), "nodes_hash": sha256_file(NODES),
        "mask_hashes": {h: sha256_file(MASKS / f"huc4_{h}.npz") for h in HUC4_BLOCKS},
        "runtime_snapshot": snapshot,
        "primary_endpoint": "cell-pooled native DOC MAE over identical whole-HUC4 K0 query cells",
        "secondary_endpoints": ["basin-macro MAE", "station-macro MAE", "RMSE", "R2",
                                "training-Q90 tail MAE", "stream-order strata", "training/inference seconds"],
        "loss": "source-standardized log1p DOC MSE, equal observed-cell weight per epoch",
        "selection": "whole-validation-HUC4 native MAE, best epoch, no target selection",
        "training_graph": "source-station-induced graph only; no validation/test covariates in gradients",
        "inference_graph": "source+validation or source+target induced graph; covariates only",
        "inputs": ["13 river/ecological features", "latitude/longitude", "temperature/discharge",
                   "hydrological masks and ages", "causal 12-month history", "season sin/cos",
                   "known directed station-river graph"],
        "excluded_inputs": ["DOC values/history", "pH", "conductance", "station identity embeddings"],
        "hierarchy": "two shared sparse-attention blocks, each scanned low-to-high stream order; "
                     "same-order edges retained and all cross-order direct edges retained",
        "transformer": "two dense spatial self-attention blocks with six directed connectivity relation biases",
        "matching": "identical parameter tensors, initialization, hidden widths, loss, optimiser, "
                    "month order, epoch budget and validation rule; effective attention degree differs",
        "scope": "standalone small-model architecture comparison, not the released DOC ensemble; "
                 "previously inspected internal geographical tasks, not fresh external confirmation; "
                 "station graph, not all physical reaches; no temporal forecasting claim"}
    destination = root / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen comparison changed; use a new version/directory")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["protocol_hash"] != digest(protocol):
            raise ValueError("changed protocol manifest")
        for name, expected in snapshot.items():
            if sha256_file(root / "code_snapshot" / name) != expected:
                raise ValueError("changed execution-time source copy")
    else:
        write_json(destination, protocol)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        write_json(root / "manifest.json", {
            "frozen_at": datetime.now(timezone.utc).isoformat(), "protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "generator_script_sha256": snapshot[CODE[1]], "runtime_snapshot_hash": digest(snapshot),
            "dataset_sha256": protocol["dataset_hash"], "nodes_sha256": protocol["nodes_hash"],
            "mask_sha256": protocol["mask_hashes"], "torch_version": torch.__version__,
            "numpy_version": np.__version__, "python_version": platform.python_version(),
            "device": "cpu", "prior_paper_protocol_changed": False})
    return protocol


def role_labels(dataset, split, role, rows):
    n, t = dataset["y"].shape
    values = np.full((t, n), np.nan, dtype=np.float32)
    cells = np.asarray(split[role])
    values[cells % t, cells // t] = np.asarray(dataset["y"]).ravel()[cells]
    return torch.from_numpy(values[:, rows].copy())


def predict(model, inputs, graph, months, batch):
    model.eval()
    rows = graph["rows"]
    result = []
    with torch.no_grad():
        for start in range(0, len(months), batch):
            selected = months[start:start + batch]
            result.append(model(inputs["windows"][selected][:, rows], inputs["environment"][rows],
                                inputs["season"][selected], graph).numpy())
    return np.concatenate(result)


def native_prediction(z, normalization):
    return np.expm1(np.clip(z * normalization["std"] + normalization["mean"], 0., 15.))


def train_one(arm, seed, inputs, train_graph, val_graph, train_y, val_y, settings, progress):
    torch.manual_seed(seed)
    model = RiverArchitectureModel(arm, hidden=settings["hidden"],
        temporal_hidden=settings["temporal_hidden"], heads=settings["heads"],
        layers=settings["layers"], dropout=settings["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    source = torch.log1p(train_y[torch.isfinite(train_y)])
    norm = {"mean": float(source.mean()), "std": max(float(source.std()), .1)}
    target = (torch.log1p(train_y) - norm["mean"]) / norm["std"]
    active_months = torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    val_months = torch.where(torch.isfinite(val_y).any(1))[0].numpy()
    val_truth = val_y[val_months].numpy()
    visible = np.isfinite(val_truth)
    rng = np.random.default_rng(seed)
    rows = train_graph["rows"]
    expected_batch_cells = int(torch.isfinite(train_y).sum()) / int(
        np.ceil(len(active_months) / settings["batch_months"]))
    best, best_epoch, state, stale, trace = np.inf, None, None, 0, []
    started = time.monotonic()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        permutation = rng.permutation(active_months)
        squared, count = 0., 0
        for start in range(0, len(permutation), settings["batch_months"]):
            months = permutation[start:start + settings["batch_months"]]
            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs["windows"][months][:, rows], inputs["environment"][rows],
                         inputs["season"][months], train_graph)
            truth = target[months]
            mask = torch.isfinite(truth)
            error = (pred[mask] - truth[mask]).square()
            # Constant denominator: every observed training cell receives the
            # same weight across the month panels, including the short last batch.
            loss = error.sum() / expected_batch_cells
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite training loss in {arm}")
            loss.backward()
            nn_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(nn_norm):
                raise ValueError("nonfinite gradients")
            optimizer.step()
            squared += float(error.detach().sum())
            count += int(mask.sum())
        val_pred = native_prediction(predict(model, inputs, val_graph, val_months,
                                             settings["batch_months"]), norm)
        score = float(np.abs(val_pred[visible] - val_truth[visible]).mean())
        improved = score < best - 1e-6
        if improved:
            best, best_epoch, stale = score, epoch, 0
            state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        row = {"epoch": epoch, "train_log_mse": squared / count, "validation_mae": score,
               "best_epoch": best_epoch, "elapsed_seconds": time.monotonic() - started}
        trace.append(row)
        if epoch == 1 or epoch % 5 == 0:
            progress(arm, row)
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    summary = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_mae": best,
               "training_seconds": time.monotonic() - started,
               "parameters": sum(p.numel() for p in model.parameters()),
               "target_normalization": norm}
    return model, summary, trace


def verify_run(run, config):
    complete = json.loads((run / "complete.json").read_text())
    if complete["config_hash"] != digest(config):
        raise ValueError("completed configuration changed")
    for name, expected in complete["files"].items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"changed completed artifact: {run / name}")


def run_one(root, region, seed, protocol):
    run = root / "runs" / f"huc4_{region}_seed{seed}"
    config = {"experiment": ROOT.name, "target_huc4": region, "seed": seed,
        "settings": protocol["settings"], "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]), "dataset_path": str(DATASET),
        "dataset_hash": protocol["dataset_hash"], "mask_path": str(MASKS / f"huc4_{region}.npz"),
        "mask_hash": protocol["mask_hashes"][region]}
    if (run / "config.json").exists():
        saved = json.loads((run / "config.json").read_text())
        if any(saved[key] != value for key, value in config.items()):
            raise ValueError("existing run has different provenance")
        config = saved
    else:
        config["started_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run / "config.json", config)
    if (run / "complete.json").exists():
        verify_run(run, config)
        print(f"{run.name}: verified completed comparison", flush=True)
        return
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    _, t = data["y"].shape
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    role_rows = {role: np.unique(split[role] // t) for role in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(role_rows[a], role_rows[b]).size:
            raise ValueError("whole-station roles overlap")
    # Assemble covariates without ever passing the label grid into the model.
    inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                              role_rows["train"], lookback=protocol["settings"]["lookback"])
    train_graph = graph_view(data["edge_index"], inputs["order"], role_rows["train"])
    val_graph = graph_view(data["edge_index"], inputs["order"],
                           np.sort(np.r_[role_rows["train"], role_rows["val"]]))
    test_graph = graph_view(data["edge_index"], inputs["order"],
                            np.sort(np.r_[role_rows["train"], role_rows["test"]]))
    train_y = role_labels(data, split, "train", train_graph["rows"].numpy())
    val_y = role_labels(data, split, "val", val_graph["rows"].numpy())
    write_json(run / "normalization.json", inputs["normalization"])
    write_json(run / "node_roles.json", {role: rows.tolist() for role, rows in role_rows.items()})
    write_json(run / "graph_sizes.json", {role: {"nodes": len(graph["rows"]),
        "edges": graph["edge_count"]} for role, graph in
        (("train", train_graph), ("val", val_graph), ("test", test_graph))})

    def progress(arm, row):
        record = {"run": run.name, "arm": arm, **row}
        write_json(root / "progress.json", record)
        print(json.dumps(record), flush=True)

    products, summaries = [], {}
    for arm in ARMS:
        checkpoint = run / f"{arm}.pt"
        info_path = run / f"{arm}.json"
        marker = run / f"{arm}_complete.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            if record["config_hash"] != digest(config):
                raise ValueError("changed saved arm identity")
            for name, expected in record["files"].items():
                if sha256_file(run / name) != expected:
                    raise ValueError("changed saved arm artifact")
            payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
            model = RiverArchitectureModel(arm, **payload["architecture"])
            model.load_state_dict(payload["state"])
            summary = json.loads(info_path.read_text())
        else:
            model, summary, trace = train_one(arm, seed, inputs, train_graph, val_graph,
                train_y, val_y, protocol["settings"], progress)
            architecture = {key: protocol["settings"][key] for key in
                            ("hidden", "temporal_hidden", "heads", "layers", "dropout")}
            torch.save({"state": model.state_dict(), "architecture": architecture}, checkpoint)
            pd.DataFrame(trace).to_csv(run / f"{arm}_trace.csv", index=False)
            write_json(info_path, summary)
            write_json(marker, {"config_hash": digest(config), "files": {name: sha256_file(run / name)
                for name in (f"{arm}.pt", f"{arm}.json", f"{arm}_trace.csv")}})
        started = time.monotonic()
        grid = native_prediction(predict(model, inputs, test_graph, np.arange(t),
                                        protocol["settings"]["batch_months"]),
                                 summary["target_normalization"])
        summary = {**summary, "inference_seconds": time.monotonic() - started,
                   "inference_cells": int(grid.size)}
        summaries[arm] = summary
        products.append((arm, grid))
        del model
        gc.collect()
    # All four fitted states and validation selections are immutable before
    # accessing test DOC for metrics. Test y is never an input or selection key.
    cells = split["test"]
    receiver_index = np.searchsorted(test_graph["rows"].numpy(), cells // t)
    frames = []
    names, dates = np.asarray(data["site_no"], str), np.asarray(data["months"], str)
    truth = np.asarray(data["y"]).ravel()[cells]
    tail_threshold = float(np.quantile(train_y[torch.isfinite(train_y)].numpy(), .9))
    for arm, grid in products:
        frames.append(pd.DataFrame({"cell": cells, "station": names[cells // t], "month": dates[cells % t],
            "model_name": arm, "y_true": truth, "y_pred": grid[cells % t, receiver_index],
            "visibility_role": "test", "target_huc4": region, "seed": seed, "k": 0,
            "stream_order": np.asarray(inputs["order"])[cells // t],
            "high_doc": truth >= tail_threshold}))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(run / "predictions.parquet", index=False)
    write_json(run / "timing.json", summaries)
    files = ["config.json", "normalization.json", "node_roles.json", "graph_sizes.json", "timing.json"]
    files += [f"{arm}{suffix}" for arm in ARMS for suffix in
              (".pt", ".json", "_trace.csv", "_complete.json")]
    write_json(run / "predictions.meta.json", {"config": config, "config_hash": digest(config),
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"],
                                                   config["runtime_snapshot_hash"]),
        "prediction_sha256": sha256_file(run / "predictions.parquet"), "rows": len(frame),
        "model_files": {name: sha256_file(run / name) for name in files},
        "selection_role": "source_validation", "observed_query_coverage": 1.,
        "full_grid_saved": False, "scope": protocol["scope"]})
    files += ["predictions.parquet", "predictions.meta.json"]
    write_json(run / "complete.json", {"config_hash": digest(config),
        "files": {name: sha256_file(run / name) for name in files}})
    print(json.dumps({"run": run.name, "completed_arms": len(ARMS),
                       "query_cells": len(cells)}), flush=True)


def benchmark(settings):
    """Synthetic timing only: no DOC observations or model outcomes inspected."""
    n, b = 300, settings["batch_months"]
    graph = graph_view(np.array([np.arange(n - 1), np.arange(1, n)]),
                       np.repeat(np.arange(1, 11), n // 10), np.arange(n))
    windows, env, season = torch.randn(b, n, 12, 8), torch.randn(n, 15), torch.randn(b, 2)
    for arm in ARMS:
        model = RiverArchitectureModel(arm, hidden=settings["hidden"],
            temporal_hidden=settings["temporal_hidden"], heads=settings["heads"],
            layers=settings["layers"], dropout=settings["dropout"])
        started = time.monotonic()
        for _ in range(3):
            model.zero_grad()
            model(windows, env, season, graph).square().mean().backward()
        print(json.dumps({"arm": arm, "synthetic_step_seconds": (time.monotonic() - started) / 3,
                           "parameters": sum(p.numel() for p in model.parameters())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--region", choices=HUC4_BLOCKS)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-months", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    settings = {"hidden": 24, "temporal_hidden": 12, "heads": 3, "layers": 2, "dropout": .1,
                "lookback": 12, "learning_rate": .001, "weight_decay": .0001,
                "max_epochs": args.max_epochs, "patience": 10, "min_epochs": 15,
                "batch_months": args.batch_months, "torch_threads": args.threads}
    if args.benchmark_only:
        benchmark(settings)
        return
    protocol = freeze(args.root, settings)
    if args.freeze_only:
        print(json.dumps({"protocol_hash": digest(protocol), "fits": 60}), flush=True)
        return
    for region in ([args.region] if args.region else HUC4_BLOCKS):
        for seed in ([args.seed] if args.seed is not None else (42, 43, 44)):
            run_one(args.root, region, seed, protocol)
    if args.region is None and args.seed is None:
        write_json(args.root / "batch_complete.json", {"fits": 60, "protocol_hash": digest(protocol),
            "completed_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
