"""Prospective spatial contribution and external K0 replication, run_ladder entry."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_doc_river_relation_v2 as trainer
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
from river_graph.models.unified_river_doc import (
    ARMS,
    UnifiedRiverDOC,
    controlled_graph,
    portable_inputs,
)

ROOT = Path("experiments/phase4_transfer/doc_unified_transfer_v1")
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
MASKS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks")
EXTERNAL_LABELS = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/external_inputs/02040104/dataset_scoring_only.pt")
SPLITS, SEEDS = (142, 143, 144), (42, 43, 44)
CODE = ("scripts/run_ladder.py", "scripts/run_doc_unified_transfer_v1.py", "scripts/start_doc_unified_transfer_v1.py",
        "scripts/prepare_doc_unified_transfer_v1.py", "scripts/verify_doc_unified_transfer_v1.py",
        "tests/test_doc_unified_transfer_v1.py", "src/river_graph/models/unified_river_doc.py",
        "src/river_graph/models/river_relation_comparison.py", "src/river_graph/models/river_architecture_comparison.py",
        "scripts/run_doc_river_relation_v2.py", "scripts/run_doc_river_architecture_comparison_v1.py",
        "scripts/run_doc_fusion_benchmark_development_v1.py", "scripts/run_doc_information_ablation_v1.py",
        "src/river_graph/experiments/doc_information_ablation.py", "src/river_graph/experiments/doc_fusion_benchmark.py",
        "src/river_graph/experiments/provenance.py", "src/river_graph/topology/build_edges.py",
        "docs/doc_unified_transfer_plan_20261012.md")


def tasks():
    return [{"id": f"split{split}_{arm}_s{seed}", "split": split, "seed": seed, "arm": arm}
            for split in SPLITS for seed in SEEDS for arm in ARMS]


def freeze(root, repository, workers):
    external = root/"external"
    metadata = json.loads((external/"complete.json").read_text())
    for name, expected in metadata["files"].items():
        if sha256_file(external/name) != expected:
            raise ValueError("receiving input preparation changed")
    for split in SPLITS:
        source = MASKS/f"split{split}.npz"
        destination = root/"masks"/source.name
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("user-selected split changed")
        destination.write_bytes(source.read_bytes())
    settings = json.loads(Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1/protocol.json").read_text())["settings"]
    settings["torch_threads"] = 2
    protocol = {"experiment": ROOT.name, "tasks": tasks(), "splits": list(SPLITS), "seeds": list(SEEDS),
        "arms": list(ARMS), "settings": settings, "workers": workers, "neural_parameters": 12565,
        "source_dataset": str(DATA), "dataset_sha256": sha256_file(DATA),
        "mask_sha256": {str(s): sha256_file(root/"masks"/f"split{s}.npz") for s in SPLITS},
        "external_labels": str(EXTERNAL_LABELS), "external_labels_sha256": sha256_file(EXTERNAL_LABELS),
        "external_input_sha256": sha256_file(external/"inputs.npz"), "external_metadata_sha256": sha256_file(external/"complete.json"),
        "runtime_snapshot": {name: sha256_file(name) for name in CODE},
        "joint_primary": ["mae", "log1p_mae"], "prior_paper_endpoints_modified": False,
        "model": "one jointly trained network: environmental MLP15->24->24, hydrological GRU8->12 over causal12 months, two spatial Transformer blocks24/3 heads, readout",
        "matching": "identical 12565 parameter tensors and seed initialization; identical inputs, labels, loss, optimizer, month order, early-stopping rule; only attention masks/relation categories change",
        "self_only": "all three heads see self only; preserve projection/feed-forward/residual self path and all static river/environment inputs",
        "permuted_river": "100 label-free node permutations, select minimum real-connected overlap; exact graph category/degree distribution preserved but degrees reassigned to identities; residual true-connected overlap reported, NOT guaranteed non-connected control",
        "source_training_graph": "source stations only; known omitted-node paths may supply topology but no omitted covariates or DOC",
        "source_inference_graph": "source+validation or source+test permitted exogenous covariates; no DOC inputs",
        "external_inference_graph": "receiving 130 stations only, no source node context; source-fitted normalization frozen, receiving DOC never an input; K0 no adaptation",
        "source_selection": "best validation native MAE epoch, inherited60 max/10 patience/15 minimum; all36 selections sealed before tests",
        "final_architecture_selection": "validation only: average paired ratios to self_only across9 panels; direction_heads and generic_dense eligible only if BOTH endpoint ratios<1 and joint mean<=0.99; select lowest eligible joint score, otherwise self_only; permuted_river diagnostic only",
        "primary_checkpoint": "split142 seed42 of selected architecture, fixed before any result; other8 fits assess partition/optimization variability; no prediction ensemble",
        "external_gate": "all36 source validation selections and architecture choice sealed, then all36 label-free external fields sealed, then external DOC archive opened for scoring",
        "aggregation": "equal-weight arithmetic mean of9 fit-level metrics, no prediction ensemble; all observed external cells scored identically (including original support cells, K0)",
        "external_history": metadata["history"],
        "scope": "3 previously inspected random whole-station source partitions plus ONE external HUC8 receiving river network; replication, not global river generalization; monthly station predictions, not all-reach or continuous-time validation",
        "versions": {"numpy": np.__version__, "torch": torch.__version__}}
    if (root/"protocol.json").exists():
        if json.loads((root/"protocol.json").read_text()) != protocol:
            raise ValueError("frozen protocol changed; create a new version")
    else:
        write_json(root/"protocol.json", protocol)
        for name in CODE:
            target = root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(CODE[1]), "dataset_sha256": protocol["dataset_sha256"],
            "mask_sha256": protocol["mask_sha256"], "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
            "prior_paper_endpoints_modified": False})
    return protocol


def configuration(protocol, task):
    return {"experiment": ROOT.name, **task, "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]), "dataset_sha256": protocol["dataset_sha256"],
        "mask_sha256": protocol["mask_sha256"][str(task["split"])], "external_input_sha256": protocol["external_input_sha256"],
        "external_labels_sha256": protocol["external_labels_sha256"]}


def context(root, task, data):
    with np.load(root/"masks"/f"split{task['split']}.npz") as saved:
        split = {k: saved[k].copy() for k in ("train", "val", "test")}
    t = data["y"].shape[1]
    rows = {k: np.unique(v//t) for k, v in split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(rows[a], rows[b]).size or np.intersect1d(split[a], split[b]).size:
            raise ValueError("whole-station roles overlap")
    label_free = {k: v for k, v in data.items() if k not in ("y", "y_mask")}
    inputs = covariate_inputs(label_free, rows["train"])
    portable = portable_inputs(label_free, inputs["normalization"])
    for key in ("windows", "environment", "season", "order"):
        torch.testing.assert_close(inputs[key], portable[key], rtol=0, atol=0)
    graphs = {role: controlled_graph(data["edge_index"], inputs["order"],
        rows["train"] if role == "train" else np.union1d(rows["train"], rows[role]), task["arm"])
        for role in ("train", "val", "test")}
    return {"split": split, "rows": rows, "inputs": inputs, "graphs": graphs, "months": t}


def load_model(directory, task, protocol):
    settings = protocol["settings"]
    model = UnifiedRiverDOC(task["arm"], **{k: settings[k] for k in ("hidden", "temporal_hidden", "heads", "layers", "dropout")})
    model.load_state_dict(torch.load(directory/"model.pt", weights_only=True, map_location="cpu"))
    return model, json.loads((directory/"info.json").read_text())


def cell_prediction(model, ctx, role, norm, batch):
    cells, t = ctx["split"][role], ctx["months"]
    times = np.unique(cells % t)
    graph = ctx["graphs"][role]
    grid = native_prediction(predict(model, ctx["inputs"], graph, times, batch), norm)
    return grid[np.searchsorted(times, cells % t), np.searchsorted(graph["rows"].numpy(), cells//t)]


def frame(data, cells, values, task, domain):
    t = len(data["months"])
    return pd.DataFrame({"cell": cells, "station": cells//t, "month": cells % t,
        "y_true": np.asarray(data["y"]).ravel()[cells], "y_pred": values,
        "split": task["split"], "seed": task["seed"], "model_name": task["arm"], "domain": domain})


def fit(root, task, protocol, data):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"selection_complete.json").exists():
        verify(directory, "selection_complete.json", config)
        return
    directory.mkdir(parents=True, exist_ok=True)
    ctx = context(root, task, data)
    train_y = role_labels(data, ctx["split"], "train", ctx["graphs"]["train"]["rows"].numpy())
    val_y = role_labels(data, ctx["split"], "val", ctx["graphs"]["val"]["rows"].numpy())
    # Inject this model factory into the unchanged audited training loop.
    # Workers are separate Python processes; the historical code stays intact.
    trainer.RiverRelationModel = UnifiedRiverDOC
    def progress(_arm, item):
        write_json(root/f"progress_{task['id']}.json", {"stage": "training", "task": task["id"], **item})
    model, info, trace = trainer.train_one(task["arm"], task["seed"], ctx["inputs"], ctx["graphs"]["train"],
        ctx["graphs"]["val"], train_y, val_y, protocol["settings"], progress)
    if info["parameters"] != protocol["neural_parameters"]:
        raise ValueError("capacity differs")
    torch.save(model.state_dict(), directory/"model.pt")
    write_json(directory/"info.json", info)
    write_json(directory/"trace.json", trace)
    write_json(directory/"config.json", config)
    write_json(directory/"normalization.json", ctx["inputs"]["normalization"])
    part = frame(data, ctx["split"]["val"], cell_prediction(model, ctx, "val", info["target_normalization"],
        protocol["settings"]["batch_months"]), task, "source_validation")
    save_predictions(directory, "validation_predictions", part, config)
    write_json(directory/"selection.json", {"test_used": False, "external_doc_used": False,
        "validation_metrics": metrics(part), "graph_pairs": {role: {key: value for key, value in graph.items()
            if key.startswith("permutation_")} for role, graph in ctx["graphs"].items()}})
    bind(directory, "selection_complete.json", config, ["model.pt", "config.json", "info.json", "trace.json",
        "normalization.json", "validation_predictions.parquet", "validation_predictions.meta.json", "selection.json"])
    write_json(root/f"progress_{task['id']}.json", {"stage": "selection_complete", "task": task["id"]})
    print(json.dumps({"stage": "selection_complete", "task": task["id"]}), flush=True)


def choose(root, protocol):
    rows = [{**task, **json.loads((root/"fits"/task["id"]/"selection.json").read_text())["validation_metrics"]} for task in tasks()]
    table = pd.DataFrame(rows)
    baseline = table[table.arm == "self_only"][["split", "seed", "mae", "log1p_mae"]]
    pairs = table.merge(baseline, on=["split", "seed"], suffixes=("", "_self"), validate="many_to_one")
    pairs["native_ratio"] = pairs.mae / pairs.mae_self
    pairs["log_ratio"] = pairs.log1p_mae / pairs.log1p_mae_self
    ratios = pairs.groupby("arm")[["native_ratio", "log_ratio"]].mean()
    ratios["joint"] = ratios.mean(axis=1)
    eligible = [arm for arm in ("direction_heads", "generic_dense") if ratios.loc[arm, "native_ratio"] < 1
                and ratios.loc[arm, "log_ratio"] < 1 and ratios.loc[arm, "joint"] <= .99]
    selected = min(eligible, key=lambda arm: ratios.loc[arm, "joint"]) if eligible else "self_only"
    result = {"selected_arm": selected, "source_validation_only": True, "test_used": False, "external_doc_used": False,
        "validation_ratios": ratios.to_dict(orient="index"), "rule": protocol["final_architecture_selection"],
        "primary_checkpoint_id": f"split142_{selected}_s42"}
    write_json(root/"selected_model.json", result)
    write_json(root/"all_selections.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
        "selected_model_sha256": sha256_file(root/"selected_model.json"), "external_doc_used": False,
        "fit_receipts": {t["id"]: sha256_file(root/"fits"/t["id"]/"selection_complete.json") for t in tasks()}})
    return result


def external_archive(root):
    with np.load(root/"external/inputs.npz") as archive:
        dataset = {k: archive[k].copy() for k in archive.files}
    dataset["feature_channels"] = ["temperature", "discharge"]
    return dataset


def generate_field(root, task, protocol, external):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"field_complete.json").exists():
        verify(directory, "field_complete.json", config)
        return
    model, info = load_model(directory, task, protocol)
    inputs = portable_inputs(external, json.loads((directory/"normalization.json").read_text()))
    graph = controlled_graph(external["edge_index"], inputs["order"], np.arange(len(external["site_no"])), task["arm"])
    grid = native_prediction(predict(model, inputs, graph, np.arange(len(external["months"])),
        protocol["settings"]["batch_months"]), info["target_normalization"]).T
    if not np.isfinite(grid).all():
        raise ValueError("nonfinite external field")
    np.savez_compressed(directory/"external_field.npz", y_pred=grid, site_no=external["site_no"], months=external["months"])
    write_json(directory/"external_field.meta.json", {"config": config, "config_hash": digest(config),
        "prediction_sha256": sha256_file(directory/"external_field.npz"), "receiving_doc_accessed": False,
        "units": "mg/L", "shape": list(grid.shape), "full_station_month_grid_saved": True,
        "model_sha256": sha256_file(directory/"model.pt"), "source_statistics_sha256": sha256_file(directory/"normalization.json"),
        "graph_pairs": {k: v for k, v in graph.items() if k.startswith("permutation_")}})
    bind(directory, "field_complete.json", config, ["external_field.npz", "external_field.meta.json"])


def score(root, task, protocol, data, external_labels):
    directory = root/"fits"/task["id"]
    config = configuration(protocol, task)
    if (directory/"test_complete.json").exists():
        verify(directory, "test_complete.json", config)
        return
    ctx = context(root, task, data)
    model, info = load_model(directory, task, protocol)
    part = frame(data, ctx["split"]["test"], cell_prediction(model, ctx, "test", info["target_normalization"],
        protocol["settings"]["batch_months"]), task, "source_hidden_station")
    save_predictions(directory, "source_predictions", part, config)
    cells = np.flatnonzero(np.asarray(external_labels["y_mask"]).ravel())
    with np.load(directory/"external_field.npz") as archive:
        if list(archive["site_no"]) != list(external_labels["site_no"]) or list(archive["months"]) != list(external_labels["months"]):
            raise ValueError("external label/field order mismatch")
        values = archive["y_pred"].ravel()[cells]
    part = frame(external_labels, cells, values, task, "external_river_K0")
    external_config = {**config, "dataset_sha256": protocol["external_labels_sha256"], "mask_sha256": digest(cells.tolist())}
    save_predictions(directory, "external_predictions", part, external_config)
    bind(directory, "test_complete.json", config, ["source_predictions.parquet", "source_predictions.meta.json",
        "external_predictions.parquet", "external_predictions.meta.json"])


def analyze(root, protocol):
    rows = []
    for task in tasks():
        for name in ("source", "external"):
            part = pd.read_parquet(root/"fits"/task["id"]/f"{name}_predictions.parquet")
            station_mae = part.assign(error=np.abs(part.y_pred-part.y_true)).groupby("station").error.mean().mean()
            rows.append({**task, "domain": part.domain.iloc[0], **metrics(part), "station_macro_mae": station_mae,
                "query_count": len(part), "coverage": float(np.isfinite(part.y_pred).mean())})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["domain", "arm"], as_index=False).mean(numeric_only=True)
    for index, row in summary.iterrows():
        ref = summary[(summary.domain == row.domain) & (summary.arm == "self_only")].iloc[0]
        for metric in ("mae", "log1p_mae"):
            summary.loc[index, metric+"_improvement_vs_self_percent"] = 100*(ref[metric]-row[metric])/ref[metric]
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
    threads = protocol["settings"]["torch_threads"]
    torch.set_num_threads(threads)
    if args.task:
        task = next(t for t in tasks() if t["id"] == args.task)
        with threadpool_limits(limits=threads):
            fit(args.root, task, protocol, data)
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
                    raise RuntimeError(f"worker failed: {task['id']}; inspect log")
        running = remaining
    choose(args.root, protocol)
    external = external_archive(args.root)
    with threadpool_limits(limits=threads):
        for task in tasks():
            generate_field(args.root, task, protocol, external)
        write_json(args.root/"all_fields.json", {"protocol_hash": digest(protocol), "frozen_at": utc_now(),
            "selection_gate_sha256": sha256_file(args.root/"all_selections.json"), "external_doc_used": False,
            "fields": {t["id"]: sha256_file(args.root/"fits"/t["id"]/"field_complete.json") for t in tasks()}})
        if sha256_file(EXTERNAL_LABELS) != protocol["external_labels_sha256"]:
            raise ValueError("external labels changed")
        external_labels = torch.load(EXTERNAL_LABELS, weights_only=False, map_location="cpu")
        for task in tasks():
            score(args.root, task, protocol, data, external_labels)
    analyze(args.root, protocol)
    write_json(args.root/"batch_complete.json", {"protocol_hash": digest(protocol), "tasks": len(tasks()),
        "completed_at": utc_now(), "source_selection_gate_sha256": sha256_file(args.root/"all_selections.json"),
        "external_fields_gate_sha256": sha256_file(args.root/"all_fields.json")})
    print(json.dumps({"stage": "complete", "tasks": len(tasks())}), flush=True)


if __name__ == "__main__":
    main()
