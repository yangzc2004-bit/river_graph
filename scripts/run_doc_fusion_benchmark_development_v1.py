"""Frozen matched-input DOC baselines; enter through the isolated run_ladder.

Fit all development-side candidates and freeze all selections before scoring
any new target prediction. Previously audited neural fits are dependencies,
not new independent confirmation. Existing paper artifacts are not modified.
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

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    verify_run,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_fusion_benchmark import (
    CANDIDATES,
    MODELS,
    REGIONS,
    SEEDS,
    concentration_prediction,
    make_estimator,
    metrics,
    paired_basin_interval,
    seasonal_medians,
    select_recipe,
    structure_metrics,
    tabular_features,
)
from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)

ROOT = Path("experiments/phase4_transfer/doc_fusion_benchmark_development_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
MASKS = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks")
PARENT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")
CODE = ("scripts/run_ladder.py", "scripts/run_doc_fusion_benchmark_development_v1.py",
        "scripts/start_doc_fusion_benchmark_development_v1.py",
        "scripts/run_doc_river_architecture_comparison_v1.py",
        "src/river_graph/experiments/doc_fusion_benchmark.py",
        "src/river_graph/experiments/provenance.py", "src/river_graph/experiments/unmonitored_doc.py",
        "src/river_graph/models/river_architecture_comparison.py",
        "tests/test_doc_fusion_benchmark_development_v1.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def bind(directory, marker, config, names):
    write_json(directory / marker, {"config_hash": digest(config),
        "files": {name: sha256_file(directory / name) for name in names}})


def verify(directory, marker, config=None):
    record = json.loads((directory / marker).read_text())
    if config is not None and record["config_hash"] != digest(config):
        raise ValueError("existing package config changed; use a new version")
    for name, expected in record["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"changed package artifact: {directory / name}")


def freeze(root, repository, threads):
    parent_protocol = json.loads((PARENT / "protocol.json").read_text())
    if parent_protocol["regions"] != list(REGIONS) or parent_protocol["seeds"] != list(SEEDS):
        raise ValueError("neural parent has different tasks/seeds")
    snapshots = {name: sha256_file(name) for name in CODE}
    dependencies = {}
    for region in REGIONS:
        for seed in SEEDS:
            run = PARENT / "runs" / f"huc4_{region}_seed{seed}"
            config = json.loads((run / "config.json").read_text())
            verify_run(run, config)
            dependencies[str(run)] = {name: sha256_file(run / name) for name in
                ("complete.json", "config.json", "normalization.json", "predictions.parquet",
                 "predictions.meta.json", "local.pt", "graph_transformer.pt",
                 "local.json", "graph_transformer.json")}
    protocol = {
        "experiment": ROOT.name, "stage": "development_matched_input_baselines",
        "scope": "previously inspected five internal HUC4 tasks; not fresh confirmation or SOTA",
        "regions": list(REGIONS), "seeds": list(SEEDS), "models": list(MODELS),
        "dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET),
        "nodes": str(NODES), "nodes_sha256": sha256_file(NODES),
        "mask_sha256": {r: sha256_file(MASKS / f"huc4_{r}.npz") for r in REGIONS},
        "runtime_snapshot": snapshots, "parent_dependencies": dependencies,
        "parent_protocol_sha256": sha256_file(PARENT / "protocol.json"),
        "original_repository_dispatcher_sha256": sha256_file(repository / "scripts/run_ladder.py"),
        "isolation": "versioned run workspace; original dispatcher and frozen products unchanged",
        "threads": threads, "candidates": CANDIDATES,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "sklearn": sklearn.__version__},
        "budgets": {"recipes_per_tabular_family": 3, "forest_trees": 300,
            "boosting_iterations": 200, "new_unique_candidate_fits": 105,
            "seasonal_source_summaries": 5, "reused_neural_checkpoints": 30,
            "ridge_deterministic": True,
            "neural": "existing fixed recipes, 60 epochs maximum with source validation early stopping; no new tuning",
            "matching": "family-specific budgets disclosed; not an equal-compute optimality search"},
        "permitted_information": {
            "doc": "source training supervision and whole-source-validation basin labels for selection only",
            "local_features": "15 environment + 12*8 driver window + 2 current season = 113 columns",
            "excluded": ["DOC input/context", "pH", "conductance", "station embeddings", "query labels"],
            "graph": "source-induced during neural training; source+heldout covariates at inference",
            "comparison": "tabular methods receive local covariates; graph Transformer can aggregate permitted other-node covariates",
            "window": "12 months ending at query, current-month drivers allowed; retrospective reconstruction"},
        "postprocessing": "common clamp to [0, expm1(15)] mg/L; matches legacy neural cap; count raw clips",
        "selection": "per region/family: minimum seed-mean native validation MAE, then log1p MAE, then lexical recipe ID",
        "target_access": "all five regions' selections bound before scoring new target predictions",
        "joint_primary": ["native_mae", "log1p_mae"],
        "secondary": ["native_rmse", "native_r2", "bias", "station_macro_mae", "basin_macro_mae", "source_q90_mae"],
        "high_doc": "source-training DOC Q90 per fold; threshold never fit on target",
        "structure_diagnostics": {
            "seasonal": "centered mean calendar-month profile, >=24 paired observations and >=6 occupied calendar months per station",
            "temporal": "within-station Pearson and std ratio at >=12 observations; correlation undefined if either series is constant",
            "increments": "difference error for consecutive observed calendar months only",
            "spatial": "concentration contrast error on direct river edges with both query endpoints observed in same month",
            "truth": "same real hidden observed dates only; not complete-field ground truth"},
        "aggregation": "metrics per seed first, then arithmetic seed mean; report pooled cells and basin macro separately",
        "uncertainty": "5000 paired basin bootstrap draws after seed-error averaging; 5 clusters; descriptive unadjusted intervals, no simultaneous superiority claim",
        "deterministic_evaluation": "ridge and seasonal predictions repeated under 3 evaluation seeds; only fitted once per recipe/fold",
        "prior_paper_endpoints_modified": False,
    }
    if (parent_protocol["dataset_hash"] != protocol["dataset_sha256"]
            or parent_protocol["nodes_hash"] != protocol["nodes_sha256"]
            or parent_protocol["mask_hashes"] != protocol["mask_sha256"]):
        raise ValueError("legacy data/masks do not match benchmark")
    for name in ("src/river_graph/models/river_architecture_comparison.py",
                 "scripts/run_doc_river_architecture_comparison_v1.py"):
        if parent_protocol["runtime_snapshot"][name] != snapshots[name]:
            raise ValueError("reused neural recipe differs from its execution-time code")
    destination = root / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen benchmark changed; create a new version")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["protocol_hash"] != digest(protocol):
            raise ValueError("changed protocol manifest")
        for name, expected in snapshots.items():
            if sha256_file(root / "code_snapshot" / name) != expected:
                raise ValueError("changed benchmark source snapshot")
    else:
        write_json(destination, protocol)
        for name in snapshots:
            target = root / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        write_json(root / "manifest.json", {
            "frozen_at": utc_now(), "protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": snapshots[CODE[1]], "runtime_snapshot_hash": digest(snapshots),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
            "dataset_sha256": protocol["dataset_sha256"], "nodes_sha256": protocol["nodes_sha256"],
            "mask_sha256": protocol["mask_sha256"], "repository": str(repository),
            "prior_paper_endpoints_modified": False})
    return protocol


def context(data, region):
    months = data["y"].shape[1]
    with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test")}
    rows = {role: np.unique(cells // months) for role, cells in split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(rows[a], rows[b]).size:
            raise ValueError("geographical station roles overlap")
    # DOC is not passed to feature construction, including the target grid.
    inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")}, rows["train"])
    return {"months": months, "split": split, "rows": rows, "inputs": inputs}


def base_config(protocol, region):
    return {"experiment": ROOT.name, "target_huc4": region,
        "protocol_hash": digest(protocol), "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "dataset_sha256": protocol["dataset_sha256"], "mask_sha256": protocol["mask_sha256"][region]}


def save_predictions(directory, name, frame, config, **extra):
    frame.to_parquet(directory / f"{name}.parquet", index=False)
    write_json(directory / f"{name}.meta.json", {"config": config, "config_hash": digest(config),
        "dataset_sha256": config["dataset_sha256"], "mask_sha256": config["mask_sha256"],
        "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "run_identity_sha256": run_identity_sha256(digest(config), config.get("started_at", "dependency_replay"), config["runtime_snapshot_hash"]),
        "prediction_sha256": sha256_file(directory / f"{name}.parquet"), "rows": len(frame),
        "observed_query_coverage": 1., "full_grid_saved": False, **extra})


def fit_region(root, region, protocol, data):
    run = root / "regions" / f"huc4_{region}"
    config = base_config(protocol, region)
    if (run / "selection_complete.json").exists():
        verify(run, "selection_complete.json", config)
        selected = json.loads((run / "selection.json").read_text())
        for relative, expected in selected["fit_receipts"].items():
            directory = run / relative
            if sha256_file(directory / "fit_complete.json") != expected:
                raise ValueError("fit receipt changed after selection")
            verify(directory, "fit_complete.json")
        return
    ctx = context(data, region)
    split, inputs, months = ctx["split"], ctx["inputs"], ctx["months"]
    train_x = tabular_features(inputs, split["train"], months)
    val_x = tabular_features(inputs, split["val"], months)
    if train_x.shape[1] != 113:
        raise ValueError("declared feature dimension changed")
    flat = np.asarray(data["y"]).ravel()
    train_y, val_y = flat[split["train"]], flat[split["val"]]
    if not np.isfinite(train_y).all() or not np.isfinite(val_y).all() or np.any(train_y < 0):
        raise ValueError("invalid source DOC labels")
    dates = np.asarray(data["months"], str)
    month_numbers = np.asarray([int(date[5:7]) for date in dates])
    seasonal = seasonal_medians(train_y, month_numbers[split["train"] % months])
    write_json(run / "seasonal_medians.json", seasonal)
    write_json(run / "normalization.json", inputs["normalization"])
    selections, receipts = {}, {}
    for family, recipes in CANDIDATES.items():
        scores = {name: [] for name in recipes}
        for name, recipe in recipes.items():
            for seed in ((0,) if family == "ridge" else SEEDS):
                directory = run / "fits" / family / name / f"seed{seed}"
                fit_config = {**config, "family": family, "recipe_id": name, "recipe": recipe, "seed": seed}
                if (directory / "config.json").exists():
                    saved = json.loads((directory / "config.json").read_text())
                    if any(saved[key] != value for key, value in fit_config.items()):
                        raise ValueError("existing fit configuration changed")
                    fit_config = saved
                if (directory / "fit_complete.json").exists():
                    verify(directory, "fit_complete.json", fit_config)
                    info = json.loads((directory / "info.json").read_text())
                else:
                    fit_config.setdefault("started_at", utc_now())
                    write_json(directory / "config.json", fit_config)
                    status = {"stage": "source_validation_fitting", "region": region, "family": family,
                              "recipe": name, "seed": seed, "started_at": utc_now()}
                    write_json(root / "progress.json", status)
                    print(json.dumps(status), flush=True)
                    started = time.monotonic()
                    model = make_estimator(family, recipe, seed, protocol["threads"])
                    target = np.log1p(train_y) if recipe["target"] == "log1p" else train_y
                    model.fit(train_x, target)
                    seconds = time.monotonic() - started
                    predicted, clips = concentration_prediction(model.predict(val_x), recipe["target"])
                    frame = pd.DataFrame({"cell": split["val"], "y_true": val_y, "y_pred": predicted,
                        "station": np.asarray(data["site_no"], str)[split["val"] // months],
                        "high_doc": val_y >= np.quantile(train_y, .9), "visibility_role": "val"})
                    info = {**metrics(frame), "training_seconds": seconds, "clips": clips,
                        "training_rows": len(train_y), "features": train_x.shape[1],
                        "deterministic": family == "ridge"}
                    joblib.dump(model, directory / "model.joblib", compress=3)
                    write_json(directory / "info.json", info)
                    save_predictions(directory, "validation_predictions", frame, fit_config, selection_role="source_validation")
                    bind(directory, "fit_complete.json", fit_config,
                         ("config.json", "model.joblib", "info.json", "validation_predictions.parquet", "validation_predictions.meta.json"))
                    print(json.dumps({**status, "completed": True, "validation_mae": info["mae"],
                                      "training_seconds": seconds}), flush=True)
                    del model
                scores[name].append({key: info[key] for key in ("mae", "log1p_mae")})
                receipts[str(directory.relative_to(run))] = sha256_file(directory / "fit_complete.json")
        chosen = select_recipe(scores)
        selections[family] = {"recipe_id": chosen, "validation_scores": scores, "test_used": False}
    write_json(run / "selection.json", {"config": config, "selections": selections,
        "fit_receipts": receipts, "frozen_at": utc_now(), "test_used": False})
    bind(run, "selection_complete.json", config, ("selection.json", "normalization.json", "seasonal_medians.json"))
    print(json.dumps({"region": region, "stage": "selection_frozen",
                      "recipes": {key: value["recipe_id"] for key, value in selections.items()}}), flush=True)


def neural_predictions(region, seed, ctx, protocol, data):
    parent = PARENT / "runs" / f"huc4_{region}_seed{seed}"
    for name, expected in protocol["parent_dependencies"][str(parent)].items():
        if sha256_file(parent / name) != expected:
            raise ValueError("frozen neural dependency changed")
    verify_run(parent, json.loads((parent / "config.json").read_text()))
    if json.loads((parent / "normalization.json").read_text()) != ctx["inputs"]["normalization"]:
        raise ValueError("neural/tabular normalization differs")
    inputs, months, cells = ctx["inputs"], ctx["months"], ctx["split"]["test"]
    graph = graph_view(data["edge_index"], inputs["order"], np.sort(np.r_[ctx["rows"]["train"], ctx["rows"]["test"]]))
    saved = pd.read_parquet(parent / "predictions.parquet")
    result, costs = {}, {}
    for arm, name in (("local", "local_no_messages"), ("graph_transformer", "fixed_fusion")):
        payload = torch.load(parent / f"{arm}.pt", weights_only=False, map_location="cpu")
        model = RiverArchitectureModel(arm, **payload["architecture"])
        model.load_state_dict(payload["state"])
        info = json.loads((parent / f"{arm}.json").read_text())
        torch.set_num_threads(4)  # Preserve the neural parent's execution recipe.
        started = time.monotonic()
        grid = native_prediction(predict(model, inputs, graph, np.arange(months), 32), info["target_normalization"])
        seconds = time.monotonic() - started
        predicted = grid[cells % months, np.searchsorted(graph["rows"].numpy(), cells // months)]
        part = saved[saved.model_name == arm]
        np.testing.assert_array_equal(part.cell.to_numpy(), cells)
        difference = float(np.abs(predicted - part.y_pred.to_numpy()).max())
        np.testing.assert_allclose(predicted, part.y_pred.to_numpy(), rtol=1e-6, atol=1e-6)
        result[name] = predicted
        costs[name] = {"reused": True, "parent": str(parent), "arm": arm,
            "checkpoint_sha256": sha256_file(parent / f"{arm}.pt"), "inference_seconds": seconds,
            "parameters": info["parameters"], "training_seconds": info["training_seconds"],
            "maximum_replay_difference": difference}
        del model
    return result, costs


def evaluate_region(root, region, protocol, data, gate_hash):
    run = root / "regions" / f"huc4_{region}"
    config = {**base_config(protocol, region), "all_selections_sha256": gate_hash}
    if (run / "test_complete.json").exists():
        verify(run, "test_complete.json", config)
        return pd.read_parquet(run / "predictions.parquet")
    verify(run, "selection_complete.json", base_config(protocol, region))
    ctx = context(data, region)
    inputs, cells, months = ctx["inputs"], ctx["split"]["test"], ctx["months"]
    test_x = tabular_features(inputs, cells, months)
    choices = json.loads((run / "selection.json").read_text())["selections"]
    seasonal = json.loads((run / "seasonal_medians.json").read_text())
    seasonal_prediction = np.asarray([seasonal[date[5:7].lstrip("0")] for date in np.asarray(data["months"], str)[cells % months]])
    train_y = np.asarray(data["y"]).ravel()[ctx["split"]["train"]]
    query = {"cell": cells, "station": np.asarray(data["site_no"], str)[cells // months],
        "month": np.asarray(data["months"], str)[cells % months],
        "y_true": np.asarray(data["y"]).ravel()[cells], "target_huc4": region,
        "visibility_role": "test", "high_doc": np.asarray(data["y"]).ravel()[cells] >= np.quantile(train_y, .9),
        "stream_order": inputs["order"].numpy()[cells // months], "k": 0}
    frames, costs = [], []
    deterministic = {"source_seasonal_median": seasonal_prediction}
    for seed in SEEDS:
        predictions = dict(deterministic)
        for family in CANDIDATES:
            name = choices[family]["recipe_id"]
            directory = run / "fits" / family / name / f"seed{0 if family == 'ridge' else seed}"
            fit_config = json.loads((directory / "config.json").read_text())
            verify(directory, "fit_complete.json", fit_config)
            if family in deterministic:
                predicted = deterministic[family]
            else:
                model = joblib.load(directory / "model.joblib")
                started = time.monotonic()
                predicted, clips = concentration_prediction(model.predict(test_x), fit_config["recipe"]["target"])
                costs.append({"model_name": family, "seed": seed, "region": region,
                    "inference_seconds": time.monotonic() - started, "clips": clips,
                    "fit": str(directory), "training_seconds": json.loads((directory / "info.json").read_text())["training_seconds"]})
                del model
                if family == "ridge":
                    deterministic[family] = predicted
            predictions[family] = predicted
        neural, records = neural_predictions(region, seed, ctx, protocol, data)
        predictions.update(neural)
        costs.extend({"model_name": name, "seed": seed, "region": region, **record} for name, record in records.items())
        for name in MODELS:
            frames.append(pd.DataFrame({**query, "model_name": name, "seed": seed, "y_pred": predictions[name]}))
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["model_name", "seed", "cell"]).any() or not np.isfinite(output[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("incomplete or duplicate test predictions")
    write_json(run / "costs.json", costs)
    save_predictions(run, "predictions", output, config, all_selections_sha256=gate_hash, query_cells=len(cells))
    bind(run, "test_complete.json", config, ("predictions.parquet", "predictions.meta.json", "costs.json"))
    print(json.dumps({"stage": "target_scored", "region": region, "query_cells": len(cells), "models": len(MODELS)}), flush=True)
    return output


def analyze(root, frame, protocol, data):
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fold_rows, structure_rows, summaries, station_rows = [], [], [], []
    for (name, seed, region), part in frame.groupby(["model_name", "seed", "target_huc4"]):
        fold_rows.append({"model_name": name, "seed": seed, "target_huc4": region, **metrics(part)})
        structure_rows.append({"model_name": name, "seed": seed, "target_huc4": region,
            **structure_metrics(part, data["edge_index"], data["y"].shape[1])})
        for station, group in part.groupby("station"):
            station_rows.append({"model_name": name, "seed": seed, "target_huc4": region,
                                 "station": station, **metrics(group)})
    fold_table = pd.DataFrame(fold_rows)
    for name in MODELS:
        group = frame[frame.model_name == name]
        per_seed = pd.DataFrame([metrics(part) for _, part in group.groupby("seed")])
        summary = {"model_name": name, **{key: float(per_seed[key].mean())
            if per_seed[key].notna().any() else None for key in per_seed.columns}}
        summary["basin_macro_mae"] = float(fold_table[fold_table.model_name == name].mae.mean())
        summary["basin_macro_log1p_mae"] = float(fold_table[fold_table.model_name == name].log1p_mae.mean())
        summary["training_seed_mae_std"] = float(per_seed.mae.std(ddof=0))
        summary["unique_query_cells"] = int(group.cell.nunique())
        summaries.append(summary)
    summary_table = pd.DataFrame(summaries)
    fold_table.to_csv(out / "by_region_seed.csv", index=False)
    summary_table.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(structure_rows).to_csv(out / "structure_by_region_seed.csv", index=False)
    pd.DataFrame(station_rows).to_csv(out / "station_metrics.csv", index=False)
    comparisons = [paired_basin_interval(frame, name, endpoint) for name in MODELS if name != "fixed_fusion"
                   for endpoint in ("mae", "log1p_mae")]
    write_json(out / "paired_comparisons.json", comparisons)
    # Independent arithmetic from original truth and saved prediction rows.
    maximum = 0.
    for name in MODELS:
        values = []
        for _, part in frame[frame.model_name == name].groupby("seed"):
            truth = np.asarray(data["y"]).ravel()[part.cell.to_numpy()].astype(float)
            np.testing.assert_array_equal(truth, part.y_true.to_numpy().astype(float))
            values.append(float(np.sum(np.abs(part.y_pred.to_numpy().astype(float) - truth)) / len(truth)))
        recorded = float(summary_table.loc[summary_table.model_name == name, "mae"].iloc[0])
        maximum = max(maximum, abs(float(np.mean(values)) - recorded))
    if maximum > 1e-10:
        raise ValueError("independent MAE audit differs")
    sources = {str(root / "protocol.json"): sha256_file(root / "protocol.json"),
        str(root / "all_selections.json"): sha256_file(root / "all_selections.json"),
        **{str(root / "regions" / f"huc4_{r}" / "test_complete.json"):
           sha256_file(root / "regions" / f"huc4_{r}" / "test_complete.json") for r in REGIONS}}
    write_json(out / "audit.json", {"maximum_independent_mae_difference": maximum,
        "identical_query_coverage": True, "models": list(MODELS), "unique_query_cells": int(frame.cell.nunique()),
        "sources": sources, "scope": protocol["scope"]})
    text = ("# Matched-input DOC fusion development benchmark\n\n"
        "All tabular selections used source validation only. The five geographical tasks were previously "
        "inspected during development; this is not independent confirmation or a SOTA claim. "
        "Neural checkpoints are reused with audited inference replay. Native MAE and log1p MAE "
        "are joint primary; native MAE selected the tabular recipes.\n\n" + summary_table.to_markdown(index=False)
        + "\n\nReport all regions and both primary endpoints. The basin bootstrap has only five clusters; "
        "intervals are exploratory and unadjusted for multiple comparisons. Structure diagnostics use "
        "paired real observations, not complete time-series or continuous reach truth. "
        "No output verifies naturally missing DOC values.\n")
    (out / "research_decision.md").write_text(text)
    bind(out, "complete.json", {"protocol_hash": digest(protocol)},
        ("summary.csv", "by_region_seed.csv", "structure_by_region_seed.csv", "station_metrics.csv",
         "paired_comparisons.json", "audit.json", "research_decision.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("positive thread count required")
    root = args.root.resolve()
    protocol = freeze(root, args.repository.resolve(), args.threads)
    if args.freeze_only:
        print(json.dumps({"stage": "frozen", "protocol_hash": digest(protocol), "unique_candidate_fits": 105}), flush=True)
        return
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        data = torch.load(DATASET, weights_only=False, map_location="cpu")
        for region in REGIONS:
            fit_region(root, region, protocol, data)
            gc.collect()
        selection_records = {r: sha256_file(root / "regions" / f"huc4_{r}" / "selection_complete.json") for r in REGIONS}
        gate = {"protocol_hash": digest(protocol), "regional_selections": selection_records, "test_used": False}
        if (root / "all_selections.json").exists():
            if json.loads((root / "all_selections.json").read_text()) != gate:
                raise ValueError("selection gate changed")
        else:
            write_json(root / "all_selections.json", gate)
        gate_hash = sha256_file(root / "all_selections.json")
        frames = [evaluate_region(root, r, protocol, data, gate_hash) for r in REGIONS]
        frame = pd.concat(frames, ignore_index=True)
        expected_cells = []
        for region in REGIONS:
            with np.load(MASKS / f"huc4_{region}.npz") as split:
                expected_cells.append(split["test"].copy())
        expected = np.concatenate(expected_cells)
        for name in MODELS:
            for seed in SEEDS:
                actual = frame[(frame.model_name == name) & (frame.seed == seed)].cell.to_numpy()
                np.testing.assert_array_equal(actual, expected)
        analyze(root, frame, protocol, data)
    write_json(root / "batch_complete.json", {"protocol_hash": digest(protocol), "completed_at": utc_now(),
        "analysis_receipt_sha256": sha256_file(root / "analysis/complete.json"),
        "unique_query_cells": int(frame.cell.nunique()), "models": len(MODELS), "sota_established": False})
    write_json(root / "progress.json", {"stage": "complete", "completed_at": utc_now()})
    print(json.dumps({"stage": "complete", "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
