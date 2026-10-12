"""Versioned common-query DOC comparison; invoked by isolated run_ladder."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import (
    DATASET,
    MASKS,
    NODES,
    bind,
    context,
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

from river_graph.experiments.doc_domain_work import features
from river_graph.experiments.doc_fusion_benchmark import (
    REGIONS,
    SEEDS,
    concentration_prediction,
    metrics,
    paired_basin_interval,
    select_recipe,
)
from river_graph.experiments.doc_network_work import (
    PHYST_RECIPES,
    SSN_RECIPES,
    PhySTArchitecture,
    SpatialStreamNetwork,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import graph_view

ROOT = Path("experiments/phase4_transfer/doc_network_work_v2")
REFERENCE = Path("experiments/phase4_transfer/doc_domain_work_v1")
PREPARATION = Path("cache/doc_network_work_v1/preparation.json")
SOURCE = Path("cache/doc_network_work_v1/sources/PhyST-HC/scripts/Functions_DOC.py")
RECIPES = {"ssn_covariance_adapt": SSN_RECIPES, "physt_architecture_adapt": PHYST_RECIPES}
REUSED = ("doc_ecological_rf", "doc_hydro_gam", "doc_xgboost", "rf_history_control", "fixed_fusion")
MODELS = (*REUSED, *RECIPES)
CODE = ("scripts/run_ladder.py", "scripts/run_doc_network_work_v2.py",
        "scripts/start_doc_network_work_v2.py", "scripts/prepare_doc_network_work_v1.py",
        "scripts/verify_doc_network_work_v2.py", "src/river_graph/experiments/doc_network_work.py",
        "tests/test_doc_network_work_v1.py", "tests/test_doc_network_work_v2.py", "scripts/run_doc_fusion_benchmark_development_v1.py",
        "scripts/run_doc_river_architecture_comparison_v1.py",
        "src/river_graph/models/river_architecture_comparison.py",
        "src/river_graph/topology/river_structure.py", "src/river_graph/experiments/doc_domain_work.py",
        "src/river_graph/experiments/doc_fusion_benchmark.py", "src/river_graph/experiments/provenance.py")


def freeze(root, repository, threads, workers):
    previous = json.loads((REFERENCE/"protocol.json").read_text())
    preparation = json.loads(PREPARATION.read_text())
    for name, expected in preparation["files"].items():
        if sha256_file(name) != expected:
            raise ValueError(f"prepared dependency changed: {name}")
    dependencies = {}
    for region in REGIONS:
        directory = REFERENCE/"regions"/f"huc4_{region}"
        verify(directory, "test_complete.json")
        dependencies[str(directory)] = {name: sha256_file(directory/name) for name in
            ("predictions.parquet", "predictions.meta.json", "test_complete.json", "selection_complete.json")}
    settings = json.loads(Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1/protocol.json").read_text())["settings"]
    protocol = {"experiment": ROOT.name, "scope": "development; five previously inspected Mississippi HUC4 tasks",
        "regions": list(REGIONS), "seeds": list(SEEDS), "models": list(MODELS), "new_recipes": RECIPES,
        "dataset_sha256": sha256_file(DATASET), "nodes_sha256": sha256_file(NODES),
        "mask_sha256": {r: sha256_file(MASKS/f"huc4_{r}.npz") for r in REGIONS},
        "runtime_snapshot": {name: sha256_file(name) for name in CODE}, "preparation": preparation,
        "preparation_sha256": sha256_file(PREPARATION), "reference_dependencies": dependencies,
        "reference_protocol_sha256": sha256_file(REFERENCE/"protocol.json"),
        "reference_gate_sha256": sha256_file(REFERENCE/"all_selections.json"),
        "threads": threads, "workers": workers, "neural_settings": settings, "selection": previous["selection"],
        "execution_revision": "v2: source-only timing selected one CPU thread and five independent region workers; model recipes and evaluation unchanged; incomplete serial v1 preserved and never target-scored",
        "preliminary_protocol_sha256": sha256_file(Path("experiments/phase4_transfer/doc_network_work_v1/protocol.json")),
        "runtime_benchmark": json.loads(Path("cache/doc_network_work_v1/source_only_runtime_benchmark.json").read_text()),
        "joint_primary": previous["joint_primary"], "postprocessing": previous["postprocessing"],
        "permitted_information": previous["permitted_information"],
        "new_inputs": {"ssn_covariance_adapt": "E15 + current T8; NHD mainstem distance/connectivity/area weighting and ECEF chord distance",
            "physt_architecture_adapt": "E15 repeated + T8 for 12 months, one zero pad; same source-only normalization and station graph as fusion"},
        "ssn_model": "linear environmental/current-driver mean plus static station network GP; independent observation noise; exact grouped REML; two source-likelihood starts, maxiter=100",
        "physt_adaptation": "official two architecture classes unchanged; replace SWAT-C/predicted-Q process states with permitted covariates; monthly concentration response; common log-MSE optimizer and source MAE early stopping; compact width search",
        "native_reproductions": False, "unique_new_fits": 45,
        "target_access": "freeze all five source-validation selections before any new target scoring",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "original_dispatcher_sha256": sha256_file(repository/"scripts/run_ladder.py"),
        "prior_paper_endpoints_modified": False}
    for key in ("dataset_sha256", "nodes_sha256", "mask_sha256"):
        if protocol[key] != previous[key]:
            raise ValueError("reference cohort or masks changed")
    destination = root/"protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen comparison changed; use a new version")
        for name, expected in protocol["runtime_snapshot"].items():
            if sha256_file(root/"code_snapshot"/name) != expected:
                raise ValueError("execution snapshot changed")
    else:
        write_json(destination, protocol)
        for name in protocol["runtime_snapshot"]:
            target = root/"code_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        write_json(root/"manifest.json", {"frozen_at": utc_now(), "protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(CODE[1]), "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
            "dataset_sha256": protocol["dataset_sha256"], "nodes_sha256": protocol["nodes_sha256"],
            "mask_sha256": protocol["mask_sha256"], "prepared_artifact_hashes": preparation["files"]})
    return protocol


def config_for(protocol, region):
    return {"experiment": ROOT.name, "target_huc4": region, "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]), "dataset_sha256": protocol["dataset_sha256"],
        "mask_sha256": protocol["mask_sha256"][region]}


def train_physt(recipe, seed, ctx, data, settings, progress):
    torch.manual_seed(seed)
    model = PhySTArchitecture(SOURCE, recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    inputs, split, rows = ctx["inputs"], ctx["split"], ctx["rows"]
    graphs = {"train": graph_view(data["edge_index"], inputs["order"], rows["train"]),
        "val": graph_view(data["edge_index"], inputs["order"], np.union1d(rows["train"], rows["val"]))}
    labels = {role: role_labels(data, split, role, graph["rows"].numpy()) for role, graph in graphs.items()}
    train_y, val_y = labels["train"], labels["val"]
    source = torch.log1p(train_y[torch.isfinite(train_y)])
    norm = {"mean": float(source.mean()), "std": max(float(source.std()), .1)}
    target = (torch.log1p(train_y)-norm["mean"])/norm["std"]
    active = torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    val_months = torch.where(torch.isfinite(val_y).any(1))[0].numpy()
    truth = val_y[val_months].numpy()
    visible = np.isfinite(truth)
    denominator = int(torch.isfinite(train_y).sum())/int(np.ceil(len(active)/settings["batch_months"]))
    rng, graph_rows = np.random.default_rng(seed), graphs["train"]["rows"]
    best, best_epoch, best_state, stale, trace = np.inf, None, None, 0, []
    started = time.monotonic()
    for epoch in range(1, settings["max_epochs"]+1):
        model.train()
        order = rng.permutation(active)
        squared, count = 0., 0
        for start in range(0, len(order), settings["batch_months"]):
            months = order[start:start+settings["batch_months"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs["windows"][months][:, graph_rows], inputs["environment"][graph_rows],
                               inputs["season"][months], graphs["train"])
            finite = torch.isfinite(target[months])
            error = (prediction[finite]-target[months][finite]).square()
            loss = error.sum()/denominator
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            squared += float(error.detach().sum())
            count += int(finite.sum())
        predicted = native_prediction(predict(model, inputs, graphs["val"], val_months, settings["batch_months"]), norm)
        score = float(np.abs(predicted[visible]-truth[visible]).mean())
        if not np.isfinite(score):
            raise ValueError("nonfinite source validation score")
        if score < best-1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        trace.append({"epoch": epoch, "validation_mae": score, "train_log_mse": squared/count,
                      "elapsed_seconds": time.monotonic()-started})
        if epoch == 1 or epoch % 5 == 0:
            progress({"epoch": epoch, "validation_mae": score, "best_epoch": best_epoch})
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {"training_seconds": time.monotonic()-started, "best_epoch": best_epoch,
        "epochs_run": epoch, "target_normalization": norm, "parameters": sum(p.numel() for p in model.parameters())}, trace


def neural_cells(model, ctx, data, cells, role, norm, settings):
    inputs, months = ctx["inputs"], ctx["months"]
    graph = graph_view(data["edge_index"], inputs["order"], np.union1d(ctx["rows"]["train"], ctx["rows"][role]))
    times = np.unique(cells % months)
    raw = native_prediction(predict(model, inputs, graph, times, settings["batch_months"]), norm)
    locations = {int(node): index for index, node in enumerate(graph["rows"])}
    return raw[np.searchsorted(times, cells % months), [locations[int(node)] for node in cells//months]]


def fit_region(root, region, protocol, data, geometry):
    run, config = root/"regions"/f"huc4_{region}", config_for(protocol, region)
    if (run/"selection_complete.json").exists():
        verify(run, "selection_complete.json", config)
        selected = json.loads((run/"selection.json").read_text())
        for path, expected in selected["fit_receipts"].items():
            if sha256_file(run/path/"fit_complete.json") != expected:
                raise ValueError("source fit receipt changed")
            verify(run/path, "fit_complete.json")
        return
    ctx = context(data, region)
    split, inputs, months = ctx["split"], ctx["inputs"], ctx["months"]
    train_y, val_y = np.asarray(data["y"]).ravel()[split["train"]], np.asarray(data["y"]).ravel()[split["val"]]
    compact_train = features(inputs, split["train"], months, "doc_ecological_rf")
    compact_val = features(inputs, split["val"], months, "doc_ecological_rf")
    selections, receipts = {}, {}
    write_json(run/"normalization.json", inputs["normalization"])
    for family, recipes in RECIPES.items():
        scores = {name: [] for name in recipes}
        for name, recipe in recipes.items():
            for seed in ((0,) if family.startswith("ssn") else SEEDS):
                directory = run/"fits"/family/name/f"seed{seed}"
                fit_config = {**config, "family": family, "recipe_id": name, "recipe": recipe, "seed": seed}
                if (directory/"fit_complete.json").exists():
                    verify(directory, "fit_complete.json", fit_config)
                    info = json.loads((directory/"info.json").read_text())
                else:
                    write_json(directory/"config.json", fit_config)
                    status = {"stage": "source_validation_fitting", "region": region,
                              "family": family, "recipe": name, "seed": seed}
                    status_items = tuple(status.items())
                    def progress(extra, bound_status=status_items):
                        record = {**dict(bound_status), **extra}
                        write_json(root/f"progress_huc4_{region}.json", record)
                        print(json.dumps(record), flush=True)
                    progress({"started": True})
                    started = time.monotonic()
                    if family.startswith("ssn"):
                        y = np.log1p(train_y) if recipe["target"] == "log1p" else train_y
                        model = SpatialStreamNetwork(geometry, recipe["kernels"]).fit(compact_train, y, split["train"]//months)
                        predicted, clips = concentration_prediction(model.predict(compact_val, split["val"]//months), recipe["target"])
                        info, trace = {"training_seconds": time.monotonic()-started, "optimizer": model.info, "clips": clips}, []
                        joblib.dump(model, directory/"model.joblib", compress=3)
                        artifact = "model.joblib"
                    else:
                        model, info, trace = train_physt(recipe, seed, ctx, data, protocol["neural_settings"], progress)
                        predicted = neural_cells(model, ctx, data, split["val"], "val", info["target_normalization"], protocol["neural_settings"])
                        torch.save(model.state_dict(), directory/"model.pt")
                        artifact = "model.pt"
                    frame = pd.DataFrame({"cell": split["val"], "y_true": val_y, "y_pred": predicted,
                        "station": np.asarray(data["site_no"], str)[split["val"]//months],
                        "high_doc": val_y >= np.quantile(train_y, .9), "visibility_role": "val"})
                    info.update(metrics(frame))
                    write_json(directory/"info.json", info)
                    write_json(directory/"trace.json", trace)
                    save_predictions(directory, "validation_predictions", frame, fit_config, selection_role="source_validation")
                    bind(directory, "fit_complete.json", fit_config, ("config.json", artifact, "info.json", "trace.json",
                        "validation_predictions.parquet", "validation_predictions.meta.json"))
                    progress({"completed": True, "seconds": info["training_seconds"]})
                    del model
                scores[name].append({key: info[key] for key in ("mae", "log1p_mae")})
                receipts[str(directory.relative_to(run))] = sha256_file(directory/"fit_complete.json")
        selections[family] = {"recipe_id": select_recipe(scores), "validation_scores": scores, "test_used": False}
    write_json(run/"selection.json", {"selections": selections, "fit_receipts": receipts, "test_used": False, "frozen_at": utc_now()})
    bind(run, "selection_complete.json", config, ("selection.json", "normalization.json"))


def evaluate_region(root, region, protocol, data, gate_hash):
    run = root/"regions"/f"huc4_{region}"
    config = {**config_for(protocol, region), "all_selections_sha256": gate_hash}
    if (run/"test_complete.json").exists():
        verify(run, "test_complete.json", config)
        return pd.read_parquet(run/"predictions.parquet")
    ctx, frames, costs = context(data, region), [], []
    cells, months = ctx["split"]["test"], ctx["months"]
    choices = json.loads((run/"selection.json").read_text())["selections"]
    reference = REFERENCE/"regions"/f"huc4_{region}"
    for name, expected in protocol["reference_dependencies"][str(reference)].items():
        if sha256_file(reference/name) != expected:
            raise ValueError("reference artifact changed")
    old = pd.read_parquet(reference/"predictions.parquet")
    train_y = np.asarray(data["y"]).ravel()[ctx["split"]["train"]]
    query = {"cell": cells, "y_true": np.asarray(data["y"]).ravel()[cells], "target_huc4": region,
        "station": np.asarray(data["site_no"], str)[cells//months], "month": np.asarray(data["months"], str)[cells % months],
        "visibility_role": "test", "high_doc": np.asarray(data["y"]).ravel()[cells] >= np.quantile(train_y, .9), "k": 0}
    compact = features(ctx["inputs"], cells, months, "doc_ecological_rf")
    ssn_cache = {}
    for seed in SEEDS:
        for family in REUSED:
            part = old[(old.model_name == family) & (old.seed == seed)]
            np.testing.assert_array_equal(part.cell, cells)
            np.testing.assert_array_equal(part.y_true, query["y_true"])
            frames.append(pd.DataFrame({**query, "seed": seed, "model_name": family, "y_pred": part.y_pred.to_numpy()}))
        for family in RECIPES:
            name = choices[family]["recipe_id"]
            fit_seed = 0 if family.startswith("ssn") else seed
            directory = run/"fits"/family/name/f"seed{fit_seed}"
            fit_config = json.loads((directory/"config.json").read_text())
            verify(directory, "fit_complete.json", fit_config)
            info = json.loads((directory/"info.json").read_text())
            started = time.monotonic()
            if family.startswith("ssn"):
                if family not in ssn_cache:
                    model = joblib.load(directory/"model.joblib")
                    ssn_cache[family], _ = concentration_prediction(model.predict(compact, cells//months), fit_config["recipe"]["target"])
                    del model
                predicted = ssn_cache[family]
            else:
                model = PhySTArchitecture(SOURCE, fit_config["recipe"])
                model.load_state_dict(torch.load(directory/"model.pt", weights_only=True, map_location="cpu"))
                predicted = neural_cells(model, ctx, data, cells, "test", info["target_normalization"], protocol["neural_settings"])
                del model
            frames.append(pd.DataFrame({**query, "seed": seed, "model_name": family, "y_pred": predicted}))
            costs.append({"model_name": family, "seed": seed, "fit_seed": fit_seed, "seconds": time.monotonic()-started})
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["model_name", "seed", "cell"]).any() or not np.isfinite(frame[["y_true", "y_pred"]]).all().all():
        raise ValueError("duplicate or nonfinite query results")
    write_json(run/"costs.json", costs)
    save_predictions(run, "predictions", frame, config, all_selections_sha256=gate_hash)
    bind(run, "test_complete.json", config, ("predictions.parquet", "predictions.meta.json", "costs.json"))
    print(json.dumps({"stage": "target_scored", "region": region, "query_cells": len(cells)}), flush=True)
    return frame


def analyze(root, frame, protocol):
    out = root/"analysis"
    summaries, folds = [], []
    for (name, seed, region), part in frame.groupby(["model_name", "seed", "target_huc4"]):
        folds.append({"model_name": name, "seed": seed, "target_huc4": region, **metrics(part)})
    by_region = pd.DataFrame(folds)
    for name in MODELS:
        group = frame[frame.model_name == name]
        by_seed = pd.DataFrame([metrics(part) for _, part in group.groupby("seed")])
        summaries.append({"model_name": name, **{key: float(by_seed[key].mean()) if by_seed[key].notna().any() else None for key in by_seed},
            "basin_macro_mae": float(by_region[by_region.model_name == name].mae.mean()),
            "unique_query_cells": int(group.cell.nunique()), "coverage": 1.,
            "training_seed_mae_std": float(by_seed.mae.std(ddof=0))})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/"summary.csv", index=False)
    by_region.to_csv(out/"by_region_seed.csv", index=False)
    write_json(out/"paired_comparisons.json", [paired_basin_interval(frame, name, endpoint) for name in MODELS
        if name != "fixed_fusion" for endpoint in ("mae", "log1p_mae")])
    bind(out, "complete.json", {"protocol_hash": digest(protocol)}, ("summary.csv", "by_region_seed.csv", "paired_comparisons.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--region", choices=list(REGIONS))
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.threads < 1 or not 1 <= args.workers <= len(REGIONS):
        raise ValueError("positive threads and one to five workers required")
    protocol = freeze(root, args.repository.resolve(), args.threads, args.workers)
    if args.freeze_only:
        print(json.dumps({"stage": "frozen", "protocol_hash": digest(protocol), "new_fits": 45}))
        return
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        with np.load("cache/doc_network_work_v1/geometry.npz", allow_pickle=False) as saved:
            geometry = {name: saved[name].copy() for name in saved.files}
        data = torch.load(DATASET, weights_only=False, map_location="cpu")
        if args.region:
            fit_region(root, args.region, protocol, data, geometry)
            return
        pending, running = list(REGIONS), {}
        logs = root/"worker_logs"
        logs.mkdir(exist_ok=True)
        while pending or running:
            while pending and len(running) < args.workers:
                region = pending.pop(0)
                command = ["uv", "run", "--no-sync", "python", "scripts/run_ladder.py",
                    "--repository", str(args.repository.resolve()), "--root", str(root),
                    "--threads", str(args.threads), "--workers", str(args.workers), "--region", region]
                with (logs/f"huc4_{region}.log").open("a") as log:
                    running[region] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                print(json.dumps({"stage": "region_worker_started", "region": region, "pid": running[region].pid}), flush=True)
            for region, worker in list(running.items()):
                result = worker.poll()
                if result is not None:
                    if result != 0:
                        for other in running.values():
                            other.terminate()
                        raise RuntimeError(f"region worker failed: {region}, exit {result}; inspect its log")
                    del running[region]
                    print(json.dumps({"stage": "region_selection_complete", "region": region}), flush=True)
            if running:
                time.sleep(1.)
        gc.collect()
        gate = {"protocol_hash": digest(protocol), "regional_selections": {r: sha256_file(root/"regions"/f"huc4_{r}"/"selection_complete.json") for r in REGIONS}, "test_used": False}
        if (root/"all_selections.json").exists() and json.loads((root/"all_selections.json").read_text()) != gate:
            raise ValueError("selection gate changed")
        write_json(root/"all_selections.json", gate)
        frames = [evaluate_region(root, r, protocol, data, sha256_file(root/"all_selections.json")) for r in REGIONS]
        frame = pd.concat(frames, ignore_index=True)
        expected = np.concatenate([context(data, r)["split"]["test"] for r in REGIONS])
        for name in MODELS:
            for seed in SEEDS:
                np.testing.assert_array_equal(frame[(frame.model_name == name) & (frame.seed == seed)].cell, expected)
        analyze(root, frame, protocol)
    write_json(root/"batch_complete.json", {"completed_at": utc_now(), "protocol_hash": digest(protocol),
        "analysis_receipt_sha256": sha256_file(root/"analysis/complete.json"), "new_fits": 45,
        "unique_query_cells": int(frame.cell.nunique()), "models": len(MODELS), "native_reproductions": False})
    write_json(root/"progress.json", {"stage": "complete"})
    print(json.dumps({"stage": "complete", "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
