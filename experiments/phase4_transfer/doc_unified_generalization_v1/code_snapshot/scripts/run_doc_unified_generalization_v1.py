"""Same-model transfer optimization under held-out source river units."""
from __future__ import annotations

import argparse
import json
import os
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
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_fusion_benchmark import metrics
from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import covariate_inputs
from river_graph.models.transfer_robust_doc import (
    GATED,
    HYDRO_DROPOUT,
    RECIPES,
    TransferRobustDOC,
    basin_weights,
    coordinate_policy,
    sensor_dropout,
)
from river_graph.models.unified_river_doc import controlled_graph, portable_inputs

ROOT = Path("experiments/phase4_transfer/doc_unified_generalization_v1")
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DOMAINS = ("val05_test10", "val10_test07", "val07_test05")
SEEDS = (42, 43, 44)
PREVIOUS = Path("experiments/phase4_transfer/doc_unified_transfer_v2")
CODE = ("scripts/run_ladder.py", "scripts/run_doc_unified_generalization_v1.py", "scripts/start_doc_unified_generalization_v1.py",
        "scripts/prepare_doc_unified_generalization_v1.py", "scripts/verify_doc_unified_generalization_v1.py",
        "src/river_graph/models/transfer_robust_doc.py", "src/river_graph/models/unified_river_doc.py",
        "src/river_graph/models/river_relation_comparison.py", "src/river_graph/models/river_architecture_comparison.py",
        "scripts/run_doc_river_architecture_comparison_v1.py", "scripts/run_doc_fusion_benchmark_development_v1.py",
        "src/river_graph/experiments/doc_fusion_benchmark.py", "src/river_graph/experiments/provenance.py",
        "tests/test_doc_unified_generalization_v1.py", "docs/doc_unified_generalization_plan_20261012.md")


def tasks():
    return [{"id": f"{domain}_{recipe}_s{seed}", "domain": domain, "recipe": recipe, "seed": seed}
            for domain in DOMAINS for seed in SEEDS for recipe in RECIPES]


def freeze(root, repository, workers):
    confirmation = json.loads((root/"confirmation/complete.json").read_text())
    for name, expected in confirmation["files"].items():
        if sha256_file(root/"confirmation"/name) != expected:
            raise ValueError("prepared independent river data changed")
    previous = json.loads((PREVIOUS/"protocol.json").read_text())
    settings = {**previous["settings"], "torch_threads": 2}
    protocol = {"experiment": ROOT.name, "tasks": tasks(), "recipes": list(RECIPES), "domains": list(DOMAINS),
        "seeds": list(SEEDS), "settings": settings, "workers": workers, "source_dataset": str(DATA),
        "dataset_sha256": sha256_file(DATA), "source_domains_sha256": sha256_file(root/"source_domains.csv"),
        "mask_sha256": {d: sha256_file(root/"masks"/f"{d}.npz") for d in DOMAINS},
        "runtime_snapshot": {name: sha256_file(name) for name in CODE},
        "development": {"inputs": str(PREVIOUS/"external/inputs.npz"), "input_sha256": previous["external_input_sha256"],
            "labels": previous["external_labels"], "labels_sha256": previous["external_labels_sha256"], "history": "already inspected Delaware case; diagnostic/development only"},
        "confirmation": {"inputs": str(root/"confirmation/inputs.npz"), "input_sha256": confirmation["files"]["inputs.npz"],
            "labels": confirmation["labels"], "labels_sha256": confirmation["labels_sha256"],
            "eligibility_sha256": confirmation["files"]["eligibility.npz"], "history": "availability-selected Hackensack-Passaic; no prior model-error inspection"},
        "diagnosis_sha256": sha256_file(root/"diagnosis/complete.json"), "preparation_sha256": sha256_file(root/"preparation.json"),
        "prior_paper_endpoints_modified": False, "joint_primary": ["mae", "log1p_mae"],
        "model_lineage": "same environmental MLP15->24->24, GRU8->12/lookback12, two24-wide upstream/downstream/global three-head blocks; no DOC input, anchors, forest or station/river identity embeddings",
        "parameters": {r: 12571 if r in GATED else 12565 for r in RECIPES},
        "reference": "full source training graph, cell-weighted loss; validation and all tests already use receiving-unit-only graphs",
        "episode_cells": "same loss/optimizer budget; source HUC2 graphs evaluated separately in each month panel; one summed gradient update per panel, equal cell weights",
        "episode_balanced": "isolated source HUC2 graphs; loss weight N/(G*N_g) from source observed training cells, equal total basin weight per epoch",
        "balanced_no_coords": "balanced episodes; normalize using source statistics and then zero normalized latitude/longitude in training and inference",
        "balanced_hydro_dropout": "balanced episodes; independently drop temperature/discharge sensor histories with fixed p=.1/.5 per station/month panel during training only; preserve season, recompute ages; no receiving statistics fitted",
        "balanced_gated": "balanced episodes; six learned cross-message sigmoid gates initialized logit-2, scaled by .2+.8*attention-weighted causal hydro-mask availability; self message retained, .01 gate penalty",
        "combined": "balanced episodes + neutral coordinates + source hydro dropout + six reliability gates",
        "split_reason": "new version for cross-river generalization: original random station splits do not simulate unseen river units; rotate whole HUC2 validation/test units05,10,07 and train on the other four units; old metrics remain unchanged and are not direct source-completion comparisons",
        "huc_parsing": "huc_cd only; handle mixed HUC8/HUC12 values and numeric leading-zero loss; no station-prefix inference",
        "training_and_selection": "same source target normalization, source-only input statistics, AdamW, shuffled month panels, max60/min15/patience10; best validation native MAE epoch; all63 selections frozen before external prediction",
        "architecture_selection": "source-validation only: paired native/log ratios to fresh reference averaged equally over9 panels; candidates eligible when both ratios<1 and joint mean<=.99; lowest eligible joint score, otherwise reference; no external calibration or test-selected recipe/seed",
        "external_gate": "seal all63 source fits and recipe choice, then all126 full label-free receiving fields, then open independent confirmation DOC for scoring",
        "external_inference": "each receiving river runs alone, frozen source normalization; no receiving DOC, support, calibration, node context from source, or receiving-fitted statistics",
        "aggregation": "arithmetic mean of9 fit-level metrics, no prediction ensemble; independent-confirmation queries identical across recipes; source rows and labels exactly matched within each panel",
        "scope": "three source regional river-unit partitions plus one diagnosed and one new external HUC8 network; not global generalization or arbitrary continuous-space/time validation"}
    if (root/"protocol.json").exists():
        if json.loads((root/"protocol.json").read_text()) != protocol:
            raise ValueError("frozen optimization changed; create a new version")
    else:
        write_json(root/"protocol.json", protocol)
        for name in CODE:
            path = root/"code_snapshot"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(CODE[1]), "dataset_sha256": protocol["dataset_sha256"],
            "mask_sha256": protocol["mask_sha256"], "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
            "prior_paper_endpoints_modified": False})
    return protocol


def configuration(protocol, task):
    return {**task, "protocol_hash": digest(protocol), "dataset_sha256": protocol["dataset_sha256"],
        "mask_sha256": protocol["mask_sha256"][task["domain"]], "runtime_snapshot_hash": digest(protocol["runtime_snapshot"])}


def context(root, task, data):
    with np.load(root/"masks"/f"{task['domain']}.npz") as saved:
        split = {r: saved[r].copy() for r in ("train", "val", "test")}
    t = data["y"].shape[1]
    rows = {r: np.unique(c//t) for r, c in split.items()}
    domains = pd.read_csv(root/"source_domains.csv", dtype={"site_no": str, "huc2": str}).set_index("site_no").loc[data["site_no"]].huc2.to_numpy()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(rows[a], rows[b]).size or np.intersect1d(domains[rows[a]], domains[rows[b]]).size:
            raise ValueError("whole river-unit roles overlap")
    inputs = covariate_inputs({k: v for k, v in data.items() if k not in ("y", "y_mask")}, rows["train"])
    inputs = coordinate_policy(inputs, task["recipe"])
    graphs = {r: controlled_graph(data["edge_index"], inputs["order"], rows[r], "direction_heads") for r in rows}
    group_ids = np.unique(domains[rows["train"]]) if task["recipe"] != "reference" else ["all_source"]
    source_graphs = [graphs["train"] if domain == "all_source" else controlled_graph(data["edge_index"], inputs["order"],
        rows["train"][domains[rows["train"]] == domain], "direction_heads") for domain in group_ids]
    return {"split": split, "rows": rows, "inputs": inputs, "graphs": graphs, "source_graphs": source_graphs,
        "group_ids": list(group_ids), "months": t}


def new_model(task, settings):
    return TransferRobustDOC(task["recipe"], **{k: settings[k] for k in ("hidden", "temporal_hidden", "heads", "layers", "dropout")})


def train_model(task, ctx, data, settings, progress):
    torch.manual_seed(task["seed"])
    model = new_model(task, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    train_rows = ctx["rows"]["train"]
    train_y = role_labels(data, ctx["split"], "train", train_rows)
    val_y = role_labels(data, ctx["split"], "val", ctx["rows"]["val"])
    logs = torch.log1p(train_y[torch.isfinite(train_y)])
    norm = {"mean": float(logs.mean()), "std": max(float(logs.std()), .1)}
    target = (torch.log1p(train_y)-norm["mean"])/norm["std"]
    positions = [np.searchsorted(train_rows, graph["rows"].numpy()) for graph in ctx["source_graphs"]]
    counts = [int(torch.isfinite(train_y[:, pos]).sum()) for pos in positions]
    weights = basin_weights(counts, task["recipe"] not in ("reference", "episode_cells"))
    active = torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    val_months = torch.where(torch.isfinite(val_y).any(1))[0].numpy()
    visible = np.isfinite(val_y[val_months].numpy())
    denominator = int(torch.isfinite(train_y).sum())/int(np.ceil(len(active)/settings["batch_months"]))
    rng = np.random.default_rng(task["seed"])
    augmentation = torch.Generator().manual_seed(task["seed"]+10007)
    best, state, stale, trace = np.inf, None, 0, []
    best_epoch = None
    started = time.monotonic()
    for epoch in range(1, settings["max_epochs"]+1):
        model.train()
        permutation = rng.permutation(active)
        weighted_squared, seen = 0., 0
        for start in range(0, len(permutation), settings["batch_months"]):
            selected = permutation[start:start+settings["batch_months"]]
            optimizer.zero_grad(set_to_none=True)
            objective = next(model.parameters()).new_zeros(())
            for graph, pos, weight in zip(ctx["source_graphs"], positions, weights, strict=True):
                truth = target[selected][:, pos]
                mask = torch.isfinite(truth)
                if not mask.any():
                    continue
                windows = ctx["inputs"]["windows"][selected][:, graph["rows"]]
                if task["recipe"] in HYDRO_DROPOUT:
                    windows = sensor_dropout(windows, augmentation)
                estimate = model(windows, ctx["inputs"]["environment"][graph["rows"]], ctx["inputs"]["season"][selected], graph)
                squared = (estimate[mask]-truth[mask]).square().sum()*float(weight)
                objective = objective+squared/denominator
                weighted_squared += float(squared.detach())
                seen += int(mask.sum())
            objective = objective+model.gate_penalty()
            if not torch.isfinite(objective):
                raise ValueError("nonfinite transfer objective")
            objective.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise ValueError("nonfinite transfer gradient")
            optimizer.step()
        val_pred = native_prediction(predict(model, ctx["inputs"], ctx["graphs"]["val"], val_months, settings["batch_months"]), norm)
        value = float(np.abs(val_pred[visible]-val_y[val_months].numpy()[visible]).mean())
        if value < best-1e-6:
            best, best_epoch, stale = value, epoch, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        row = {"epoch": epoch, "weighted_train_log_mse": weighted_squared/seen, "validation_mae": value,
            "best_epoch": best_epoch, "elapsed_seconds": time.monotonic()-started}
        trace.append(row)
        if epoch == 1 or epoch % 5 == 0:
            progress(row)
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    info = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_mae": best,
        "training_seconds": time.monotonic()-started, "parameters": sum(p.numel() for p in model.parameters()),
        "target_normalization": norm, "source_group_ids": ctx["group_ids"], "source_group_counts": counts,
        "source_group_loss_weights": weights.tolist()}
    if task["recipe"] in GATED:
        info["learned_sigmoid_gates"] = model.cross_gate_logits.sigmoid().detach().tolist()
    return model, info, trace


def load_model(directory, task, protocol):
    model = new_model(task, protocol["settings"])
    model.load_state_dict(torch.load(directory/"model.pt", weights_only=True, map_location="cpu"))
    return model, json.loads((directory/"info.json").read_text())


def cell_prediction(model, ctx, role, norm, batch):
    cells, t = ctx["split"][role], ctx["months"]
    times = np.unique(cells % t)
    graph = ctx["graphs"][role]
    grid = native_prediction(predict(model, ctx["inputs"], graph, times, batch), norm)
    return grid[np.searchsorted(times, cells % t), np.searchsorted(graph["rows"].numpy(), cells//t)]


def frame(data, cells, values, task, population):
    t = len(data["months"])
    return pd.DataFrame({"cell": cells, "station": cells//t, "month": cells % t, "y_true": np.asarray(data["y"]).ravel()[cells],
        "y_pred": values, "high_doc": False, "task": task["domain"], "seed": task["seed"], "recipe": task["recipe"], "population": population})


def fit(root, task, protocol, data):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"selection_complete.json").exists():
        verify(directory, "selection_complete.json", config)
        return
    directory.mkdir(parents=True, exist_ok=True)
    ctx = context(root, task, data)
    def progress(item):
        write_json(root/f"progress_{task['id']}.json", {"stage": "training", "task": task["id"], **item})
    model, info, trace = train_model(task, ctx, data, protocol["settings"], progress)
    if info["parameters"] != protocol["parameters"][task["recipe"]]:
        raise ValueError("unexpected model parameter count")
    torch.save(model.state_dict(), directory/"model.pt")
    write_json(directory/"info.json", info)
    write_json(directory/"trace.json", trace)
    write_json(directory/"normalization.json", ctx["inputs"]["normalization"])
    write_json(directory/"config.json", config)
    part = frame(data, ctx["split"]["val"], cell_prediction(model, ctx, "val", info["target_normalization"],
        protocol["settings"]["batch_months"]), task, "source_validation_unit")
    save_predictions(directory, "validation_predictions", part, config)
    write_json(directory/"selection.json", {"validation_metrics": metrics(part), "source_test_used": False,
        "development_doc_used_for_fit_or_selection": False, "confirmation_doc_used": False})
    bind(directory, "selection_complete.json", config, ["model.pt", "info.json", "trace.json", "normalization.json", "config.json",
        "validation_predictions.parquet", "validation_predictions.meta.json", "selection.json"])
    write_json(root/f"progress_{task['id']}.json", {"stage": "selection_complete", "task": task["id"]})
    print(json.dumps({"stage": "selection_complete", "task": task["id"]}), flush=True)


def choose(root, protocol):
    rows = [{**t, **json.loads((root/"fits"/t["id"]/"selection.json").read_text())["validation_metrics"]} for t in tasks()]
    table = pd.DataFrame(rows)
    base = table[table.recipe == "reference"][["domain", "seed", "mae", "log1p_mae"]]
    pairs = table.merge(base, on=["domain", "seed"], suffixes=("", "_ref"), validate="many_to_one")
    pairs["native_ratio"], pairs["log_ratio"] = pairs.mae/pairs.mae_ref, pairs.log1p_mae/pairs.log1p_mae_ref
    ratios = pairs.groupby("recipe")[["native_ratio", "log_ratio"]].mean()
    ratios["joint"] = ratios.mean(axis=1)
    eligible = [r for r in RECIPES if r != "reference" and ratios.loc[r, "native_ratio"] < 1 and
        ratios.loc[r, "log_ratio"] < 1 and ratios.loc[r, "joint"] <= .99]
    selected = min(eligible, key=lambda r: ratios.loc[r, "joint"]) if eligible else "reference"
    result = {"selected_recipe": selected, "source_validation_only": True, "validation_ratios": ratios.to_dict(orient="index"),
        "primary_checkpoint_id": f"val05_test10_{selected}_s42", "rule": protocol["architecture_selection"],
        "source_test_or_external_errors_used": False}
    write_json(root/"selected_model.json", result)
    write_json(root/"all_selections.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
        "selected_model_sha256": sha256_file(root/"selected_model.json"), "confirmation_doc_used": False,
        "fit_receipts": {t["id"]: sha256_file(root/"fits"/t["id"]/"selection_complete.json") for t in tasks()}})


def archive(path):
    with np.load(path) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    data["feature_channels"] = ["temperature", "discharge"]
    return data


def generate_field(root, task, protocol, population):
    directory = root/"fits"/task["id"]
    config = {**configuration(protocol, task), "population": population, "input_sha256": protocol[population]["input_sha256"]}
    marker = f"{population}_field_complete.json"
    if (directory/marker).exists():
        verify(directory, marker, config)
        return
    raw = archive(protocol[population]["inputs"])
    model, info = load_model(directory, task, protocol)
    norm = json.loads((directory/"normalization.json").read_text())
    inputs = coordinate_policy(portable_inputs(raw, norm), task["recipe"])
    graph = controlled_graph(raw["edge_index"], inputs["order"], np.arange(len(raw["site_no"])), "direction_heads")
    values = native_prediction(predict(model, inputs, graph, np.arange(len(raw["months"])),
        protocol["settings"]["batch_months"]), info["target_normalization"]).T
    if not np.isfinite(values).all():
        raise ValueError("nonfinite receiving field")
    name = f"{population}_field"
    np.savez_compressed(directory/f"{name}.npz", y_pred=values, site_no=raw["site_no"], months=raw["months"])
    write_json(directory/f"{name}.meta.json", {"config": config, "config_hash": digest(config),
        "model_sha256": sha256_file(directory/"model.pt"), "info_sha256": sha256_file(directory/"info.json"),
        "normalization_sha256": sha256_file(directory/"normalization.json"), "prediction_sha256": sha256_file(directory/f"{name}.npz"),
        "shape": list(values.shape), "units": "mg/L", "receiving_doc_used": False, "receiving_statistics_fitted": False})
    bind(directory, marker, config, [f"{name}.npz", f"{name}.meta.json"])


def score(root, task, protocol, data, receiving):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"test_complete.json").exists():
        verify(directory, "test_complete.json", config)
        return
    ctx = context(root, task, data)
    model, info = load_model(directory, task, protocol)
    part = frame(data, ctx["split"]["test"], cell_prediction(model, ctx, "test", info["target_normalization"],
        protocol["settings"]["batch_months"]), task, "source_heldout_unit")
    save_predictions(directory, "source_predictions", part, config)
    for population, labels in receiving.items():
        cells = np.flatnonzero(np.asarray(labels["y_mask"]).ravel())
        with np.load(directory/f"{population}_field.npz") as field:
            if list(field["site_no"]) != list(labels["site_no"]) or list(field["months"]) != list(labels["months"]):
                raise ValueError("receiving field and labels misaligned")
            values = field["y_pred"].ravel()[cells]
        part = frame(labels, cells, values, task, population)
        external_config = {**config, "dataset_sha256": protocol[population]["labels_sha256"],
            "mask_sha256": digest(cells.tolist()), "population": population,
            "input_sha256": protocol[population]["input_sha256"]}
        save_predictions(directory, f"{population}_predictions", part, external_config)
    bind(directory, "test_complete.json", config, [f"{p}_predictions.{ext}" for p in ("source", "development", "confirmation")
        for ext in ("parquet", "meta.json")])


def analyze(root, protocol):
    rows = []
    for task in tasks():
        for population in ("source", "development", "confirmation"):
            part = pd.read_parquet(root/"fits"/task["id"]/f"{population}_predictions.parquet")
            rows.append({**task, "population": population, **metrics(part)})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["population", "recipe"], as_index=False).mean(numeric_only=True)
    for index, row in summary.iterrows():
        base = summary[(summary.population == row.population) & (summary.recipe == "reference")].iloc[0]
        for metric in ("mae", "log1p_mae"):
            summary.loc[index, metric+"_improvement_percent"] = 100*(base[metric]-row[metric])/base[metric]
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    detail.to_csv(analysis/"by_fit.csv", index=False)
    summary.to_csv(analysis/"summary.csv", index=False)
    bind(analysis, "complete.json", {"protocol_hash": digest(protocol)}, ["by_fit.csv", "summary.csv"])


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
    data = torch.load(DATA, weights_only=False, map_location="cpu")
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    if args.task:
        with threadpool_limits(limits=2):
            fit(args.root, next(t for t in tasks() if t["id"] == args.task), protocol, data)
        return
    pending, running = list(tasks()), []
    (args.root/"worker_logs").mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running) < args.workers:
            task = pending.pop(0)
            log = (args.root/"worker_logs"/f"{task['id']}.log").open("a")
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
                    raise RuntimeError(f"source fitting failed at {task['id']}")
        running = remaining
    choose(args.root, protocol)
    with threadpool_limits(limits=2):
        for task in tasks():
            for population in ("development", "confirmation"):
                generate_field(args.root, task, protocol, population)
        write_json(args.root/"all_fields.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
            "selection_gate_sha256": sha256_file(args.root/"all_selections.json"), "confirmation_doc_used": False,
            "fields": {f"{t['id']}:{p}": sha256_file(args.root/"fits"/t["id"]/f"{p}_field_complete.json")
                for t in tasks() for p in ("development", "confirmation")}})
        receiving = {}
        for population in ("development", "confirmation"):
            if sha256_file(protocol[population]["labels"]) != protocol[population]["labels_sha256"]:
                raise ValueError("receiving label archive changed")
            labels = torch.load(protocol[population]["labels"], weights_only=False, map_location="cpu")
            if population == "confirmation":
                with np.load(args.root/"confirmation/eligibility.npz") as eligibility:
                    selected = eligibility["original_rows"]
                labels = {**labels, "y": np.asarray(labels["y"])[selected], "y_mask": np.asarray(labels["y_mask"])[selected],
                    "site_no": np.asarray(labels["site_no"])[selected].tolist()}
            receiving[population] = labels
        for task in tasks():
            score(args.root, task, protocol, data, receiving)
    analyze(args.root, protocol)
    write_json(args.root/"batch_complete.json", {"protocol_hash": digest(protocol), "tasks": 63,
        "completed_at": utc_now(), "selection_gate_sha256": sha256_file(args.root/"all_selections.json"),
        "field_gate_sha256": sha256_file(args.root/"all_fields.json")})
    print(json.dumps({"stage": "complete", "tasks": 63}), flush=True)


if __name__ == "__main__":
    main()
