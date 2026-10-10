"""Sequential source-validation selection of environmental, time and fusion modules.

Run via run_ladder.py --experiment doc-fusion-component-comparison-v1.
Each region owns its artifacts. Shared fits are reused, never fitted twice.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_river_architecture_comparison_v1 import (
    DATASET,
    HUC4_BLOCKS,
    MASKS,
    NODES,
    digest,
    native_prediction,
    predict,
    role_labels,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.doc_fusion_comparison import (
    ENVIRONMENT_CHOICES,
    FUSION_CHOICES,
    TEMPORAL_CHOICES,
    DOCFusionModel,
)
from river_graph.models.river_architecture_comparison import (
    covariate_inputs,
    graph_view,
)

ROOT = Path("experiments/phase4_transfer/doc_fusion_component_comparison_v1")
SEEDS = (42, 43, 44)
CODE = ("scripts/run_ladder.py", "scripts/run_doc_fusion_component_comparison_v1.py",
        "scripts/run_doc_river_architecture_comparison_v1.py",
        "src/river_graph/models/doc_fusion_comparison.py",
        "src/river_graph/models/river_architecture_comparison.py",
        "src/river_graph/experiments/provenance.py",
        "src/river_graph/experiments/unmonitored_doc.py",
        "tests/test_doc_fusion_comparison.py", "tests/test_doc_fusion_protocol_v1.py")
STAGES = {
    "environment": {"choices": ENVIRONMENT_CHOICES, "eligible": ("mlp", "linear", "residual_mlp")},
    "time": {"choices": TEMPORAL_CHOICES, "eligible": ("gru", "lag_mlp", "lstm", "transformer")},
    "fusion": {"choices": FUSION_CHOICES, "eligible": FUSION_CHOICES},
}


def select_choice(scores, eligible):
    """Seed-average validation only. Declared order breaks exact ties."""
    means = {choice: float(np.mean(scores[choice])) for choice in eligible}
    if any(len(scores[choice]) != len(SEEDS) for choice in eligible):
        raise ValueError("all three seeds required before component selection")
    if not all(np.isfinite(value) for value in means.values()):
        raise ValueError("nonfinite validation score")
    return min(eligible, key=lambda choice: means[choice])


def specification(environment="mlp", temporal="gru", fusion="concat"):
    return {"environment": environment, "temporal": temporal, "fusion": fusion}


def spec_name(spec):
    return f"env_{spec['environment']}__time_{spec['temporal']}__fusion_{spec['fusion']}"


def architecture(settings):
    return {key: settings[key] for key in
            ("hidden", "temporal_hidden", "heads", "layers", "dropout", "lookback")}


def freeze(root, settings):
    snapshot = {name: sha256_file(name) for name in CODE}
    protocol = {"experiment": ROOT.name, "settings": settings, "regions": list(HUC4_BLOCKS),
        "seeds": list(SEEDS), "stages": STAGES, "unique_fits": 165,
        "dataset": str(DATASET), "dataset_hash": sha256_file(DATASET),
        "nodes_hash": sha256_file(NODES),
        "mask_hashes": {region: sha256_file(MASKS / f"huc4_{region}.npz") for region in HUC4_BLOCKS},
        "runtime_snapshot": snapshot,
        "primary_endpoint": "native DOC MAE, equal observed query-cell weight, seed errors averaged",
        "secondary_endpoints": ["basin-macro MAE", "station-macro MAE", "RMSE", "R2",
                                "source-training Q90 DOC tail MAE", "parameter count"],
        "selection": "per outer region, mean whole-validation-HUC4 MAE across seeds 42/43/44; "
                     "environment then time then fusion; no test values or scores in selection",
        "diagnostic_exclusions": "constant environment and current-only time are information ablations; "
                                 "excluded from selecting a full environmental/history model",
        "ties": "strict minimum, declared eligible order breaks exact ties",
        "test_access": "all three stage selections and fit hashes frozen before test labels scored",
        "training_graph": "source-station-induced graph; no held-out covariates in gradients",
        "inference_graph": "source+validation or source+target station graph, covariates only",
        "inputs": "source-normalized 15 static/ecological features; 12 months of temperature/log discharge, "
                  "masks, observation ages and season; directed river-network relation biases",
        "excluded_inputs": ["DOC history", "test DOC", "pH", "conductance", "station embeddings"],
        "loss": "source-standardized log1p DOC MSE, equal observed-cell weighting per epoch",
        "shared_trunk": "two graph Transformer blocks, hidden 24, three spatial heads",
        "capacity": "actual parameter counts reported; common trunk initialization identical; "
                    "residual fusion and environmental conditioning both add 600 parameters",
        "scope": "internal development comparison on previously inspected five HUC4 tasks; "
                 "not a global optimum, independent external validation, all-reach field or future forecast",
        "old_protocols_modified": False}
    # JSON conversion makes tuple/list identity independent of Python containers.
    protocol = json.loads(json.dumps(protocol))
    destination = root / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen component experiment changed; create a new version")
        if json.loads((root / "manifest.json").read_text())["protocol_hash"] != digest(protocol):
            raise ValueError("changed manifest")
        for name, expected in snapshot.items():
            if sha256_file(root / "code_snapshot" / name) != expected:
                raise ValueError("changed frozen source copy")
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


def train_one(spec, seed, inputs, graphs, labels, settings, progress):
    torch.manual_seed(seed)
    model = DOCFusionModel(**spec, **architecture(settings))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    train_y, val_y = labels["train"], labels["val"]
    source = torch.log1p(train_y[torch.isfinite(train_y)])
    norm = {"mean": float(source.mean()), "std": max(float(source.std()), .1)}
    target = (torch.log1p(train_y) - norm["mean"]) / norm["std"]
    active = torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    val_months = torch.where(torch.isfinite(val_y).any(1))[0].numpy()
    truth = val_y[val_months].numpy()
    visible = np.isfinite(truth)
    denominator = int(torch.isfinite(train_y).sum()) / int(np.ceil(len(active) / settings["batch_months"]))
    rng, rows = np.random.default_rng(seed), graphs["train"]["rows"]
    best, best_epoch, best_state, stale, trace = np.inf, None, None, 0, []
    started = time.monotonic()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = rng.permutation(active)
        squared, count = 0., 0
        for start in range(0, len(order), settings["batch_months"]):
            months = order[start:start + settings["batch_months"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs["windows"][months][:, rows], inputs["environment"][rows],
                               inputs["season"][months], graphs["train"])
            finite = torch.isfinite(target[months])
            error = (prediction[finite] - target[months][finite]).square()
            loss = error.sum() / denominator
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise ValueError("nonfinite gradients")
            optimizer.step()
            squared += float(error.detach().sum())
            count += int(finite.sum())
        predicted = native_prediction(predict(model, inputs, graphs["val"], val_months,
                                             settings["batch_months"]), norm)
        score = float(np.abs(predicted[visible] - truth[visible]).mean())
        if not np.isfinite(score):
            raise ValueError("nonfinite validation score")
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        row = {"epoch": epoch, "train_log_mse": squared / count, "validation_mae": score,
               "best_epoch": best_epoch, "elapsed_seconds": time.monotonic() - started}
        trace.append(row)
        if epoch == 1 or epoch % 5 == 0:
            progress(spec, seed, row)
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    summary = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_mae": best,
        "training_seconds": time.monotonic() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_normalization": norm}
    return model, summary, trace


def verify_package(directory, marker, config=None):
    record = json.loads((directory / marker).read_text())
    if config is not None and record["config_hash"] != digest(config):
        raise ValueError("changed fit identity")
    for name, expected in record["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"changed artifact {directory / name}")
    return record


def load_model(directory):
    payload = torch.load(directory / "checkpoint.pt", weights_only=False, map_location="cpu")
    model = DOCFusionModel(**payload["specification"], **payload["architecture"])
    model.load_state_dict(payload["state"])
    return model


def fit_one(directory, spec, seed, context, region_config, progress):
    config = {**region_config, "seed": seed, "specification": spec}
    if (directory / "config.json").exists():
        saved = json.loads((directory / "config.json").read_text())
        if any(saved[key] != value for key, value in config.items()):
            raise ValueError("existing fit provenance differs")
        config = saved
    else:
        config["started_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "config.json", config)
    if (directory / "fit_complete.json").exists():
        verify_package(directory, "fit_complete.json", config)
        return json.loads((directory / "info.json").read_text())
    model, summary, trace = train_one(spec, seed, context["inputs"], context["graphs"],
                                     context["labels"], config["settings"], progress)
    torch.save({"state": model.state_dict(), "specification": spec,
                "architecture": architecture(config["settings"])}, directory / "checkpoint.pt")
    write_json(directory / "info.json", summary)
    pd.DataFrame(trace).to_csv(directory / "trace.csv", index=False)
    cells = context["split"]["val"]
    graph = context["graphs"]["val"]
    grid = native_prediction(predict(model, context["inputs"], graph, np.arange(context["months"]),
                                    config["settings"]["batch_months"]), summary["target_normalization"])
    rows = np.searchsorted(graph["rows"].numpy(), cells // context["months"])
    frame = pd.DataFrame({"cell": cells, "y_true": context["data"]["y"].numpy().ravel()[cells],
        "y_pred": grid[cells % context["months"], rows], "visibility_role": "val"})
    frame.to_parquet(directory / "validation_predictions.parquet", index=False)
    write_json(directory / "validation_predictions.meta.json", {
        "config_hash": digest(config), "dataset_sha256": config["dataset_hash"],
        "mask_sha256": config["mask_hash"], "checkpoint_sha256": sha256_file(directory / "checkpoint.pt"),
        "prediction_sha256": sha256_file(directory / "validation_predictions.parquet"),
        "selection_role": "source_validation", "rows": len(frame)})
    files = ("config.json", "checkpoint.pt", "info.json", "trace.csv",
             "validation_predictions.parquet", "validation_predictions.meta.json")
    write_json(directory / "fit_complete.json", {"config_hash": digest(config),
        "files": {name: sha256_file(directory / name) for name in files}})
    del model
    gc.collect()
    return summary


def test_one(directory, context, region, seed, selections_hash):
    config = json.loads((directory / "config.json").read_text())
    verify_package(directory, "fit_complete.json", config)
    if (directory / "test_complete.json").exists():
        record = verify_package(directory, "test_complete.json", config)
        if record["selections_hash"] != selections_hash:
            raise ValueError("changed selections after testing")
        return pd.read_parquet(directory / "test_predictions.parquet")
    model = load_model(directory)
    info = json.loads((directory / "info.json").read_text())
    started = time.monotonic()
    graph, t = context["graphs"]["test"], context["months"]
    grid = native_prediction(predict(model, context["inputs"], graph, np.arange(t),
                                    config["settings"]["batch_months"]), info["target_normalization"])
    seconds = time.monotonic() - started
    cells = context["split"]["test"]
    rows = np.searchsorted(graph["rows"].numpy(), cells // t)
    data = context["data"]
    truth = np.asarray(data["y"]).ravel()[cells]
    training = context["labels"]["train"]
    threshold = float(np.quantile(training[torch.isfinite(training)].numpy(), .9))
    frame = pd.DataFrame({"cell": cells, "station": np.asarray(data["site_no"], str)[cells // t],
        "month": np.asarray(data["months"], str)[cells % t], "y_true": truth,
        "y_pred": grid[cells % t, rows], "visibility_role": "test", "target_huc4": region,
        "seed": seed, "k": 0, "high_doc": truth >= threshold,
        "stream_order": np.asarray(context["inputs"]["order"])[cells // t],
        **config["specification"]})
    frame.to_parquet(directory / "test_predictions.parquet", index=False)
    write_json(directory / "test_predictions.meta.json", {
        "config": config, "config_hash": digest(config), "dataset_sha256": config["dataset_hash"],
        "mask_sha256": config["mask_hash"], "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"],
                                                   config["runtime_snapshot_hash"]),
        "checkpoint_sha256": sha256_file(directory / "checkpoint.pt"),
        "prediction_sha256": sha256_file(directory / "test_predictions.parquet"),
        "selections_sha256": selections_hash, "rows": len(frame), "observed_query_coverage": 1.,
        "full_grid_saved": False, "inference_seconds": seconds, "inference_cells": int(grid.size)})
    write_json(directory / "test_complete.json", {"config_hash": digest(config),
        "selections_hash": selections_hash, "files": {name: sha256_file(directory / name)
            for name in ("test_predictions.parquet", "test_predictions.meta.json")}})
    del model
    gc.collect()
    return frame


def run_region(root, region, protocol):
    run = root / "regions" / f"huc4_{region}"
    config = {"experiment": ROOT.name, "target_huc4": region, "settings": protocol["settings"],
        "protocol_hash": digest(protocol), "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "dataset_hash": protocol["dataset_hash"], "mask_hash": protocol["mask_hashes"][region]}
    if (run / "complete.json").exists():
        verify_package(run, "complete.json", config)
        for directory in sorted((run / "fits").glob("*/seed*")):
            verify_package(directory, "fit_complete.json")
            verify_package(directory, "test_complete.json")
        print(f"huc4_{region}: verified complete", flush=True)
        return
    write_json(run / "config.json", config)
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    t = data["y"].shape[1]
    with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    rows = {role: np.unique(split[role] // t) for role in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(rows[a], rows[b]).size:
            raise ValueError("whole-station roles overlap")
    inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                              rows["train"], lookback=protocol["settings"]["lookback"])
    graphs = {role: graph_view(data["edge_index"], inputs["order"],
        rows["train"] if role == "train" else np.sort(np.r_[rows["train"], rows[role]]))
        for role in ("train", "val", "test")}
    labels = {role: role_labels(data, split, role, graphs[role]["rows"].numpy())
              for role in ("train", "val")}
    context = {"data": data, "inputs": inputs, "graphs": graphs, "labels": labels,
               "split": split, "months": t}
    write_json(run / "normalization.json", inputs["normalization"])
    write_json(run / "node_roles.json", {role: value.tolist() for role, value in rows.items()})
    write_json(run / "graph_sizes.json", {role: {"nodes": len(value["rows"]), "edges": value["edge_count"]}
                                          for role, value in graphs.items()})

    def progress(spec, seed, row):
        record = {"region": region, "specification": spec, "seed": seed, **row}
        write_json(run / "progress.json", record)
        print(json.dumps(record), flush=True)

    registry, selections, aliases = {}, {}, {}
    env, temporal = "mlp", "gru"
    for stage, definition in STAGES.items():
        scores = {}
        for choice in definition["choices"]:
            spec = specification(choice if stage == "environment" else env,
                                 choice if stage == "time" else temporal,
                                 choice if stage == "fusion" else "concat")
            key = spec_name(spec)
            scores[choice] = []
            aliases[f"{stage}__{choice}"] = key
            for seed in SEEDS:
                directory = run / "fits" / key / f"seed{seed}"
                info = fit_one(directory, spec, seed, context, config, progress)
                registry[(key, seed)] = directory
                scores[choice].append(info["validation_mae"])
        selected = select_choice(scores, definition["eligible"])
        selections[stage] = {"selected": selected, "validation_scores": scores,
                             "eligible": list(definition["eligible"])}
        if stage == "environment":
            env = selected
        elif stage == "time":
            temporal = selected
        aliases[f"selected__{stage}"] = aliases[f"{stage}__{selected}"]
        print(json.dumps({"region": region, "stage_complete": stage, "selected": selected,
                           "validation_means": {key: float(np.mean(value)) for key, value in scores.items()}}),
              flush=True)
    aliases["selected__reference"] = spec_name(specification())
    if len(registry) != 33:
        raise ValueError("sequential reuse must produce exactly 33 unique fits per region")
    selection_record = {"target_huc4": region, "stages": selections, "aliases": aliases,
        "protocol_hash": digest(protocol), "fit_hashes": {f"{key}/seed{seed}": sha256_file(
            directory / "fit_complete.json") for (key, seed), directory in registry.items()},
        "test_scores_used": False}
    selection_path = run / "selections.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection_record:
        raise ValueError("frozen stage selection changed")
    write_json(selection_path, selection_record)
    selections_hash = sha256_file(selection_path)
    products = {(key, seed): test_one(directory, context, region, seed, selections_hash)
                for (key, seed), directory in registry.items()}
    frames = []
    for alias, key in aliases.items():
        for seed in SEEDS:
            frame = products[(key, seed)].copy()
            frame["model_name"] = alias
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(run / "predictions.parquet", index=False)
    write_json(run / "predictions.meta.json", {"config": config, "config_hash": digest(config),
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "selections_sha256": selections_hash, "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "prediction_sha256": sha256_file(run / "predictions.parquet"), "rows": len(frame),
        "aliases": aliases, "unique_fits": 33, "coverage": 1.})
    files = ("config.json", "normalization.json", "node_roles.json", "graph_sizes.json", "selections.json",
             "predictions.parquet", "predictions.meta.json")
    write_json(run / "complete.json", {"config_hash": digest(config),
        "files": {name: sha256_file(run / name) for name in files}})
    print(json.dumps({"region": region, "complete": True, "unique_fits": 33,
                       "query_cells": len(split["test"])}), flush=True)


def benchmark(settings):
    n, batch = 240, settings["batch_months"]
    graph = graph_view(np.array([np.arange(n - 1), np.arange(1, n)]),
                       np.repeat(np.arange(1, 9), n // 8), np.arange(n))
    windows = torch.randn(batch, n, settings["lookback"], 8)
    env, season = torch.randn(n, 15), torch.randn(batch, 2)
    specs = [specification(temporal=choice) for choice in TEMPORAL_CHOICES]
    for spec in specs:
        model = DOCFusionModel(**spec, **architecture(settings))
        started = time.monotonic()
        for _ in range(3):
            model.zero_grad()
            model(windows, env, season, graph).square().mean().backward()
        print(json.dumps({**spec, "step_seconds": (time.monotonic() - started) / 3,
                          "parameters": sum(value.numel() for value in model.parameters())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--region", choices=HUC4_BLOCKS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-months", type=int, default=32)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    settings = {"hidden": 24, "temporal_hidden": 12, "heads": 3, "layers": 2, "dropout": .1,
        "lookback": 12, "learning_rate": .001, "weight_decay": .0001,
        "max_epochs": args.max_epochs, "min_epochs": 15, "patience": 10,
        "batch_months": args.batch_months, "torch_threads": args.threads}
    if args.benchmark_only:
        benchmark(settings)
        return
    protocol = freeze(args.root, settings)
    if args.freeze_only:
        print(json.dumps({"protocol_hash": digest(protocol), "unique_fits": 165}), flush=True)
        return
    for region in ([args.region] if args.region else HUC4_BLOCKS):
        run_region(args.root, region, protocol)
    if args.region is None:
        write_json(args.root / "batch_complete.json", {"unique_fits": 165, "protocol_hash": digest(protocol),
                   "completed_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
