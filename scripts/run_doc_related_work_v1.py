"""Train pinned official SAITS/GRIN adapters on common DOC geographical tasks.

Execute through the versioned workspace's scripts/run_ladder.py. All source
validation recipe choices are frozen before any target labels are scored.
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
from run_doc_fusion_benchmark_development_v1 import (
    DATASET,
    MASKS,
    NODES,
    PARENT,
    bind,
    context,
    save_predictions,
    verify,
)
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    role_labels,
    verify_run,
    write_json,
)

from river_graph.experiments.doc_fusion_benchmark import (
    REGIONS,
    SEEDS,
    metrics,
    paired_basin_interval,
    select_recipe,
    structure_metrics,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_related_work import (
    OFFICIAL,
    RECIPES,
    THIRD_PARTY,
    build_related_model,
    official_file_hashes,
)
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    graph_view,
)

ROOT = Path("experiments/phase4_transfer/doc_related_work_v1")
MODELS = ("fixed_fusion", "saits_doc_covariates", "grin_doc_covariates")
CODE = ("scripts/run_ladder.py", "scripts/run_doc_related_work_v1.py",
        "scripts/start_doc_related_work_v1.py", "scripts/benchmark_doc_related_work_v1.py",
        "scripts/run_doc_fusion_benchmark_development_v1.py",
        "scripts/run_doc_river_architecture_comparison_v1.py",
        "src/river_graph/models/doc_related_work.py",
        "src/river_graph/models/river_architecture_comparison.py",
        "src/river_graph/experiments/doc_fusion_benchmark.py",
        "src/river_graph/experiments/unmonitored_doc.py",
        "src/river_graph/experiments/provenance.py", "tests/test_doc_related_work_v1.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def graph_for(data, inputs, rows):
    graph = graph_view(data["edge_index"], inputs["order"], rows)
    graph["adjacency"] = (graph["relation"] == 1).float()
    return graph


def freeze(root, repository, settings):
    upstream = official_file_hashes()
    for method, record in OFFICIAL.items():
        directory = THIRD_PARTY / record["directory"]
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != record["commit"]:
            raise ValueError(f"{method}: official checkout revision changed")
        subprocess.run(["git", "-C", str(directory), "diff", "--exit-code", "HEAD", "--", *record["files"]], check=True)
    snapshot = {name: sha256_file(name) for name in CODE}
    parents = {}
    for region in REGIONS:
        for seed in SEEDS:
            run = PARENT / "runs" / f"huc4_{region}_seed{seed}"
            verify_run(run, json.loads((run / "config.json").read_text()))
            parents[str(run)] = {name: sha256_file(run / name) for name in (
                "complete.json", "config.json", "normalization.json", "graph_transformer.pt",
                "graph_transformer.json", "predictions.parquet", "predictions.meta.json")}
    protocol = {"experiment": ROOT.name, "scope": "development comparison of published backbones adapted to covariate-only DOC; previously inspected internal HUC4 tasks",
        "regions": list(REGIONS), "seeds": list(SEEDS), "models": list(MODELS),
        "dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET), "nodes_sha256": sha256_file(NODES),
        "mask_sha256": {r: sha256_file(MASKS / f"huc4_{r}.npz") for r in REGIONS},
        "settings": settings, "recipes": RECIPES, "new_fits": 60, "reused_reference_fits": 15,
        "runtime_snapshot": snapshot, "official_sources": upstream, "reference_dependencies": parents,
        "original_dispatcher_sha256": sha256_file(repository / "scripts/run_ladder.py"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "inputs": "same 15 environment features and 12-month, eight-channel driver/mask/age/season windows; no DOC context at source or target",
        "supervision": "only observed source DOC at final window month; target DOC never enters an input, normalization, gradient or selection",
        "scaling": "identical source-only covariate scaling; source log1p DOC mean/std; common native clamp [0, expm1(15)]",
        "training_graph": "source-induced only; source+validation or source+target covariates permitted at inference",
        "saits_adapter": "official two DMSA blocks and combination preserved; 23 covariate features plus always-hidden DOC channel; original ORT + masked DOC MIT losses; local station windows treated as samples",
        "grin_adapter": "official BiGRIL with native exogenous u_size=23, n_nodes=None, embedding_size=0; zero h0; variable-node graph; original main MAE + four auxiliary-head MAE terms; DOC x/mask always zero",
        "training_recipe_changes": "common AdamW, 60-epoch maximum and native-validation-MAE checkpoint selection; target adaptation uses final-month labels; not the original papers' datasets or native masking training recipes",
        "selection": "per outer fold/method, minimum validation native MAE averaged across all three seeds; exact ties log1p MAE then lexical recipe ID",
        "test_gate": "all regions' source-side selections hashed before scoring any new target prediction",
        "joint_primary": ["native_mae", "log1p_mae"],
        "secondary": ["RMSE", "predictive_R2", "bias", "station_macro_MAE", "basin_macro_MAE", "source_Q90_tail_MAE", "observed_temporal_spatial_structure"],
        "aggregation": "metrics per training seed first, then mean; pooled cells and basin macro reported separately; no implicit prediction ensemble",
        "uncertainty": "5000 paired basin-bootstrap draws; five clusters; exploratory unadjusted intervals; no simultaneous superiority claim",
        "capacity": "two frozen widths per published family; report parameters/epochs/cost; reference retained from earlier development; no equal-compute optimality claim",
        "recent_methods": {"CoFILL": {"repository": "https://github.com/joyHJL/CoFILL",
            "checked_commit": "d461621b213df7d682034a1da99721f2ba65b1ab", "status": "not_in_this_run",
            "reason": "official implementation binds feature/node embeddings, adaptive graph vectors and dataset-specific graph loaders; matched unseen-node covariate interface needs further adaptation or a separate within-network DOC-assisted track"}},
        "claims": "ranking among these declared adapters only; not native-paper reproduction, fresh independent river confirmation or global SOTA",
        "existing_paper_endpoints_modified": False}
    parent = json.loads((PARENT / "protocol.json").read_text())
    if parent["dataset_hash"] != protocol["dataset_sha256"] or parent["mask_hashes"] != protocol["mask_sha256"]:
        raise ValueError("reference dataset/masks differ")
    if snapshot["src/river_graph/models/river_architecture_comparison.py"] != parent["runtime_snapshot"]["src/river_graph/models/river_architecture_comparison.py"]:
        raise ValueError("reference implementation changed")
    destination = root / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen related-work experiment changed; create a new version")
        if json.loads((root / "manifest.json").read_text())["protocol_hash"] != digest(protocol):
            raise ValueError("changed related-work manifest")
        for name, expected in snapshot.items():
            if sha256_file(root / "code_snapshot" / name) != expected:
                raise ValueError("changed frozen adapter source")
    else:
        write_json(destination, protocol)
        for name in snapshot:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        write_json(root / "manifest.json", {"frozen_at": utc_now(), "protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": snapshot[CODE[1]], "runtime_snapshot_hash": digest(snapshot),
            "official_sources_hash": digest(upstream), "dataset_sha256": protocol["dataset_sha256"],
            "nodes_sha256": protocol["nodes_sha256"], "mask_sha256": protocol["mask_sha256"],
            "repository": str(repository), "existing_paper_endpoints_modified": False})
    return protocol


def run_config(protocol, region):
    return {"experiment": ROOT.name, "target_huc4": region, "protocol_hash": digest(protocol),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "official_sources_hash": digest(protocol["official_sources"]),
        "dataset_sha256": protocol["dataset_sha256"], "mask_sha256": protocol["mask_sha256"][region]}


def related_predict(model, method, ctx, role, settings):
    inputs, cells, months = ctx["inputs"], ctx["split"][role], ctx["months"]
    model.eval()
    outputs = []
    with torch.no_grad():
        if method == "saits":
            for start in range(0, len(cells), settings["batch_cells"]):
                selected = cells[start:start + settings["batch_cells"]]
                nodes, times = selected // months, selected % months
                z = model(inputs["windows"][times, nodes].unsqueeze(0), inputs["environment"][nodes], None, None)
                outputs.append(z[0].numpy())
            return np.concatenate(outputs)
        graph = ctx["graphs"][role]
        selected_months = np.unique(cells % months)
        for start in range(0, len(selected_months), settings["batch_months"]):
            times = selected_months[start:start + settings["batch_months"]]
            outputs.append(model(inputs["windows"][times][:, graph["rows"]],
                                 inputs["environment"][graph["rows"]], None, graph).numpy())
        grid = np.concatenate(outputs)
        return grid[np.searchsorted(selected_months, cells % months),
                    np.searchsorted(graph["rows"].numpy(), cells // months)]


def training_data(data, region):
    ctx = context(data, region)
    rows, inputs = ctx["rows"], ctx["inputs"]
    ctx["graphs"] = {role: graph_for(data, inputs, rows["train"] if role == "train"
        else np.sort(np.r_[rows["train"], rows[role]])) for role in ("train", "val", "test")}
    # Only source labels are put in the optimization target, not validation/test DOC.
    ctx["train_y"] = role_labels(data, ctx["split"], "train", rows["train"])
    return ctx


def train_one(method, recipe, seed, ctx, data, settings, progress):
    torch.manual_seed(seed)
    model = build_related_model(method, recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    train_y = ctx["train_y"]
    log_y = torch.log1p(train_y[torch.isfinite(train_y)])
    normalization = {"mean": float(log_y.mean()), "std": max(float(log_y.std()), .1)}
    target = (torch.log1p(train_y) - normalization["mean"]) / normalization["std"]
    source_cells = ctx["split"]["train"]
    source_map = {int(node): i for i, node in enumerate(ctx["rows"]["train"])}
    source_nodes = np.array([source_map[int(cell // ctx["months"])] for cell in source_cells])
    active_months = torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    val_y = np.asarray(data["y"]).ravel()[ctx["split"]["val"]]
    best, best_epoch, best_state, stale, trace = np.inf, None, None, 0, []
    inputs, graph, months = ctx["inputs"], ctx["graphs"]["train"], ctx["months"]
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = rng.permutation(len(source_cells)) if method == "saits" else rng.permutation(active_months)
        batch = settings["batch_cells"] if method == "saits" else settings["batch_months"]
        expected_cells = len(source_cells) / int(np.ceil(len(order) / batch))
        weighted_loss, cells_seen = 0., 0
        for start in range(0, len(order), batch):
            selected = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            if method == "saits":
                cells = source_cells[selected]
                nodes, times = cells // months, cells % months
                label = target[times, source_nodes[selected]].unsqueeze(0)
                visible = torch.ones_like(label, dtype=torch.bool)
                loss = model.objective(inputs["windows"][times, nodes].unsqueeze(0),
                                       inputs["environment"][nodes], label, visible, None)
            else:
                label = target[selected]
                visible = torch.isfinite(label)
                loss = model.objective(inputs["windows"][selected][:, graph["rows"]],
                    inputs["environment"][graph["rows"]], label, visible, graph)
            count = int(visible.sum())
            scaled = loss * count / expected_cells
            if not torch.isfinite(scaled):
                raise ValueError("nonfinite related-work loss")
            scaled.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite related-work gradients")
            optimizer.step()
            weighted_loss += float(loss.detach()) * count
            cells_seen += count
        validation = native_prediction(related_predict(model, method, ctx, "val", settings), normalization)
        score = float(np.abs(validation.astype(float) - val_y.astype(float)).mean())
        if not np.isfinite(score):
            raise ValueError("nonfinite source validation score")
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        row = {"epoch": epoch, "train_objective": weighted_loss / cells_seen,
               "validation_mae": score, "best_epoch": best_epoch, "elapsed_seconds": time.monotonic() - started}
        trace.append(row)
        if epoch == 1 or epoch % 5 == 0:
            progress(row)
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    info = {"validation_mae": best, "best_epoch": best_epoch, "epochs_run": epoch,
        "target_normalization": normalization, "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": time.monotonic() - started}
    return model, info, trace


def fit_one(directory, config, ctx, data, progress):
    expected = dict(config)
    if (directory / "config.json").exists():
        saved = json.loads((directory / "config.json").read_text())
        if any(saved[key] != value for key, value in expected.items()):
            raise ValueError("existing related-work fit config changed")
        config = saved
    else:
        config = {**config, "started_at": utc_now()}
        write_json(directory / "config.json", config)
    if (directory / "fit_complete.json").exists():
        verify(directory, "fit_complete.json", config)
        return json.loads((directory / "info.json").read_text())
    method, recipe = config["method"], config["recipe"]
    model, info, trace = train_one(method, recipe, config["seed"], ctx, data, config["settings"], progress)
    torch.save({"method": method, "recipe": recipe, "state": model.state_dict()}, directory / "checkpoint.pt")
    pd.DataFrame(trace).to_csv(directory / "trace.csv", index=False)
    cells = ctx["split"]["val"]
    predicted = native_prediction(related_predict(model, method, ctx, "val", config["settings"]), info["target_normalization"])
    truth = np.asarray(data["y"]).ravel()[cells]
    frame = pd.DataFrame({"cell": cells, "y_true": truth, "y_pred": predicted,
        "station": np.asarray(data["site_no"], str)[cells // ctx["months"]], "visibility_role": "val",
        "high_doc": truth >= np.quantile(np.asarray(data["y"]).ravel()[ctx["split"]["train"]], .9)})
    scores = metrics(frame)
    info["validation_log1p_mae"] = scores["log1p_mae"]
    if abs(scores["mae"] - info["validation_mae"]) > 1e-5:
        raise ValueError("saved validation predictions disagree with selection score")
    write_json(directory / "info.json", info)
    save_predictions(directory, "validation_predictions", frame, config, selection_role="source_validation",
                     checkpoint_sha256=sha256_file(directory / "checkpoint.pt"))
    bind(directory, "fit_complete.json", config, ("config.json", "checkpoint.pt", "trace.csv",
         "info.json", "validation_predictions.parquet", "validation_predictions.meta.json"))
    del model
    gc.collect()
    return info


def fit_region(root, region, protocol, data):
    run, config = root / "regions" / f"huc4_{region}", run_config(protocol, region)
    if (run / "selection_complete.json").exists():
        verify(run, "selection_complete.json", config)
        record = json.loads((run / "selection.json").read_text())
        for name, expected in record["fit_receipts"].items():
            directory = run / name
            if sha256_file(directory / "fit_complete.json") != expected:
                raise ValueError("related-work fit changed after selection")
            verify(directory, "fit_complete.json")
        return
    ctx = training_data(data, region)
    write_json(run / "normalization.json", ctx["inputs"]["normalization"])
    selections, receipts = {}, {}
    for method, recipes in RECIPES.items():
        scores = {name: [] for name in recipes}
        for name, recipe in recipes.items():
            for seed in SEEDS:
                directory = run / "fits" / method / name / f"seed{seed}"
                fit_config = {**config, "method": method, "recipe_id": name, "recipe": recipe,
                              "seed": seed, "settings": protocol["settings"]}

                def progress(row, method=method, name=name, seed=seed):
                    status = {"stage": "source_fitting", "region": region, "method": method,
                              "recipe": name, "seed": seed, **row}
                    write_json(root / "progress.json", status)
                    print(json.dumps(status), flush=True)

                info = fit_one(directory, fit_config, ctx, data, progress)
                scores[name].append({"mae": info["validation_mae"], "log1p_mae": info["validation_log1p_mae"]})
                receipts[str(directory.relative_to(run))] = sha256_file(directory / "fit_complete.json")
        chosen = select_recipe(scores)
        selections[method] = {"recipe_id": chosen, "validation_scores": scores, "test_used": False}
    write_json(run / "selection.json", {"selections": selections, "fit_receipts": receipts,
                                       "test_used": False, "frozen_at": utc_now()})
    bind(run, "selection_complete.json", config, ("selection.json", "normalization.json"))


def load_model(directory):
    payload = torch.load(directory / "checkpoint.pt", weights_only=False, map_location="cpu")
    model = build_related_model(payload["method"], payload["recipe"])
    model.load_state_dict(payload["state"])
    return model


def replay_validation(model, method, directory, ctx, settings, normalization):
    saved = pd.read_parquet(directory / "validation_predictions.parquet")
    np.testing.assert_array_equal(saved.cell.to_numpy(), ctx["split"]["val"])
    prediction = native_prediction(related_predict(model, method, ctx, "val", settings), normalization)
    np.testing.assert_allclose(prediction, saved.y_pred.to_numpy(), rtol=1e-6, atol=1e-6)
    return float(np.abs(prediction - saved.y_pred.to_numpy()).max())


def reference_prediction(region, seed, ctx, data, protocol):
    parent = PARENT / "runs" / f"huc4_{region}_seed{seed}"
    for name, expected in protocol["reference_dependencies"][str(parent)].items():
        if sha256_file(parent / name) != expected:
            raise ValueError("reference dependency changed")
    verify_run(parent, json.loads((parent / "config.json").read_text()))
    if json.loads((parent / "normalization.json").read_text()) != ctx["inputs"]["normalization"]:
        raise ValueError("reference and related-work input scaling differs")
    payload = torch.load(parent / "graph_transformer.pt", weights_only=False, map_location="cpu")
    model = RiverArchitectureModel("graph_transformer", **payload["architecture"])
    model.load_state_dict(payload["state"])
    info = json.loads((parent / "graph_transformer.json").read_text())
    cells, months, graph = ctx["split"]["test"], ctx["months"], ctx["graphs"]["test"]
    times = np.unique(cells % months)
    started = time.monotonic()
    z = predict(model, ctx["inputs"], graph, times, 32)
    grid = native_prediction(z, info["target_normalization"])
    predicted = grid[np.searchsorted(times, cells % months), np.searchsorted(graph["rows"].numpy(), cells // months)]
    saved = pd.read_parquet(parent / "predictions.parquet")
    saved = saved[saved.model_name == "graph_transformer"]
    np.testing.assert_array_equal(cells, saved.cell.to_numpy())
    np.testing.assert_allclose(predicted, saved.y_pred.to_numpy(), rtol=1e-6, atol=1e-6)
    return saved.y_pred.to_numpy(), {"reused": True, "parent": str(parent), "parameters": info["parameters"],
        "training_seconds": info["training_seconds"], "query_inference_seconds": time.monotonic() - started,
        "maximum_reference_replay_difference": float(np.abs(predicted - saved.y_pred.to_numpy()).max())}


def evaluate_region(root, region, protocol, data, gate_hash):
    run = root / "regions" / f"huc4_{region}"
    config = {**run_config(protocol, region), "all_selections_sha256": gate_hash}
    if (run / "test_complete.json").exists():
        verify(run, "test_complete.json", config)
        return pd.read_parquet(run / "predictions.parquet")
    verify(run, "selection_complete.json", run_config(protocol, region))
    choices = json.loads((run / "selection.json").read_text())["selections"]
    ctx = training_data(data, region)
    cells, months = ctx["split"]["test"], ctx["months"]
    truth = np.asarray(data["y"]).ravel()[cells]
    threshold = np.quantile(np.asarray(data["y"]).ravel()[ctx["split"]["train"]], .9)
    query = {"cell": cells, "station": np.asarray(data["site_no"], str)[cells // months],
        "month": np.asarray(data["months"], str)[cells % months], "y_true": truth,
        "high_doc": truth >= threshold, "target_huc4": region, "visibility_role": "test", "k": 0}
    frames, costs = [], []
    for seed in SEEDS:
        reference, cost = reference_prediction(region, seed, ctx, data, protocol)
        frames.append(pd.DataFrame({**query, "model_name": "fixed_fusion", "seed": seed, "y_pred": reference}))
        costs.append({"model_name": "fixed_fusion", "seed": seed, **cost})
        for method in RECIPES:
            directory = run / "fits" / method / choices[method]["recipe_id"] / f"seed{seed}"
            fit_config = json.loads((directory / "config.json").read_text())
            verify(directory, "fit_complete.json", fit_config)
            test_config = {**fit_config, "all_selections_sha256": gate_hash}
            if (directory / "test_complete.json").exists():
                verify(directory, "test_complete.json", test_config)
                part = pd.read_parquet(directory / "test_predictions.parquet")
                np.testing.assert_array_equal(part.cell.to_numpy(), cells)
                np.testing.assert_array_equal(part.y_true.to_numpy(), truth)
                frames.append(part)
                costs.append(json.loads((directory / "test_cost.json").read_text()))
                continue
            model = load_model(directory)
            info = json.loads((directory / "info.json").read_text())
            replay_difference = replay_validation(model, method, directory, ctx, protocol["settings"], info["target_normalization"])
            started = time.monotonic()
            predicted = native_prediction(related_predict(model, method, ctx, "test", protocol["settings"]), info["target_normalization"])
            name = f"{method}_doc_covariates"
            part = pd.DataFrame({**query, "model_name": name, "seed": seed, "y_pred": predicted})
            if len(predicted) != len(cells) or not np.isfinite(predicted).all():
                raise ValueError("incomplete/nonfinite related-work query coverage")
            frames.append(part)
            cost = {"model_name": name, "seed": seed, "fit": str(directory),
                    "query_inference_seconds": time.monotonic() - started,
                    "maximum_validation_replay_difference": replay_difference, **info}
            costs.append(cost)
            write_json(directory / "test_cost.json", cost)
            save_predictions(directory, "test_predictions", part, test_config,
                checkpoint_sha256=sha256_file(directory / "checkpoint.pt"), all_selections_sha256=gate_hash)
            bind(directory, "test_complete.json", test_config,
                 ("test_predictions.parquet", "test_predictions.meta.json", "test_cost.json"))
            del model
    output = pd.concat(frames, ignore_index=True)
    write_json(run / "costs.json", costs)
    save_predictions(run, "predictions", output, config, all_selections_sha256=gate_hash)
    bind(run, "test_complete.json", config, ("predictions.parquet", "predictions.meta.json", "costs.json"))
    return output


def analysis(root, frame, protocol, data):
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows, summaries, structures = [], [], []
    for (name, seed, region), part in frame.groupby(["model_name", "seed", "target_huc4"]):
        rows.append({"model_name": name, "seed": seed, "target_huc4": region, **metrics(part)})
        structures.append({"model_name": name, "seed": seed, "target_huc4": region,
            **structure_metrics(part, data["edge_index"], data["y"].shape[1])})
    region_table = pd.DataFrame(rows)
    maximum = 0.
    for name in MODELS:
        parts = [part for _, part in frame[frame.model_name == name].groupby("seed")]
        per_seed = pd.DataFrame([metrics(part) for part in parts])
        summary = {"model_name": name, **{key: float(per_seed[key].mean()) if per_seed[key].notna().any() else None for key in per_seed.columns}}
        summary["basin_macro_mae"] = float(region_table[region_table.model_name == name].mae.mean())
        summary["basin_macro_log1p_mae"] = float(region_table[region_table.model_name == name].log1p_mae.mean())
        summary["training_seed_mae_std"] = float(per_seed.mae.std(ddof=0))
        summaries.append(summary)
        errors = []
        for part in parts:
            truth = np.asarray(data["y"]).ravel()[part.cell.to_numpy()].astype(float)
            np.testing.assert_array_equal(truth, part.y_true.to_numpy().astype(float))
            errors.append(float(np.abs(part.y_pred.to_numpy().astype(float) - truth).sum() / len(truth)))
        maximum = max(maximum, abs(float(np.mean(errors)) - summary["mae"]))
    table = pd.DataFrame(summaries)
    region_table.to_csv(out / "by_region_seed.csv", index=False)
    table.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(structures).to_csv(out / "structure_by_region_seed.csv", index=False)
    write_json(out / "paired_comparisons.json", [paired_basin_interval(frame, name, endpoint)
        for name in MODELS if name != "fixed_fusion" for endpoint in ("mae", "log1p_mae")])
    if maximum > 1e-10:
        raise ValueError("independent DOC metric audit differs")
    write_json(out / "audit.json", {"maximum_independent_mae_difference": maximum,
        "same_queries": True, "query_cells": int(frame.cell.nunique()),
        "protocol_sha256": sha256_file(root / "protocol.json"), "all_selections_sha256": sha256_file(root / "all_selections.json"),
        "regional_receipts": {r: sha256_file(root / "regions" / f"huc4_{r}" / "test_complete.json") for r in REGIONS},
        "scope": protocol["scope"], "sota_established": False})
    (out / "research_decision.md").write_text("# Published-backbone DOC comparison\n\n"
        "Pinned official SAITS and GRIN code with documented covariate-only/inductive adapters. "
        "This is an internal development comparison, not the papers' original datasets or an independent SOTA confirmation. "
        "All source-side selections preceded new target scoring. Report both primary endpoints.\n\n"
        + table.to_markdown(index=False) + "\n\nFive basin clusters give exploratory, unadjusted intervals. "
        "Additional recent competitors and reserved independent river systems are still required before stronger claims.\n")
    bind(out, "complete.json", {"protocol_hash": digest(protocol)}, ("summary.csv", "by_region_seed.csv",
        "structure_by_region_seed.csv", "paired_comparisons.json", "audit.json", "research_decision.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("positive thread count required")
    torch.set_num_threads(args.threads)
    settings = {"learning_rate": .001, "weight_decay": .0001, "max_epochs": 60,
        "min_epochs": 15, "patience": 10, "batch_months": 8, "batch_cells": 256, "torch_threads": args.threads}
    root = args.root.resolve()
    protocol = freeze(root, args.repository.resolve(), settings)
    if args.freeze_only:
        print(json.dumps({"stage": "frozen", "protocol_hash": digest(protocol), "new_fits": 60}), flush=True)
        return
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    for region in REGIONS:
        fit_region(root, region, protocol, data)
        gc.collect()
    if official_file_hashes() != protocol["official_sources"]:
        raise ValueError("official sources changed during training")
    gate = {"protocol_hash": digest(protocol), "selections": {
        r: sha256_file(root / "regions" / f"huc4_{r}" / "selection_complete.json") for r in REGIONS}, "test_used": False}
    if (root / "all_selections.json").exists():
        if json.loads((root / "all_selections.json").read_text()) != gate:
            raise ValueError("related-work selection gate changed")
    else:
        write_json(root / "all_selections.json", gate)
    gate_hash = sha256_file(root / "all_selections.json")
    frame = pd.concat([evaluate_region(root, r, protocol, data, gate_hash) for r in REGIONS], ignore_index=True)
    if frame.duplicated(["model_name", "seed", "cell"]).any():
        raise ValueError("duplicate related-work query rows")
    expected = []
    for region in REGIONS:
        with np.load(MASKS / f"huc4_{region}.npz") as split:
            expected.append(split["test"].copy())
    expected = np.concatenate(expected)
    for name in MODELS:
        for seed in SEEDS:
            np.testing.assert_array_equal(frame[(frame.model_name == name) & (frame.seed == seed)].cell.to_numpy(), expected)
    analysis(root, frame, protocol, data)
    write_json(root / "batch_complete.json", {"completed_at": utc_now(), "protocol_hash": digest(protocol),
        "analysis_receipt_sha256": sha256_file(root / "analysis/complete.json"), "sota_established": False})
    write_json(root / "progress.json", {"stage": "complete", "completed_at": utc_now()})


if __name__ == "__main__":
    main()
