"""Fit frozen DOC literature baselines through an isolated run_ladder entry."""
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
import sklearn
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
from run_doc_river_architecture_comparison_v1 import digest, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_domain_work import (
    CANDIDATES,
    DETERMINISTIC,
    FAMILIES,
    MODELS,
    dependency_versions,
    features,
    make_estimator,
    optional_dependencies,
)
from river_graph.experiments.doc_fusion_benchmark import (
    REGIONS,
    SEEDS,
    concentration_prediction,
    metrics,
    paired_basin_interval,
    select_recipe,
    structure_metrics,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_domain_work_v1")
REFERENCE = Path("experiments/phase4_transfer/doc_fusion_benchmark_development_v1")
LITERATURE = Path("configs/doc_domain_work_v1.json")
CODE = ("scripts/run_ladder.py", "scripts/run_doc_domain_work_v1.py",
    "scripts/start_doc_domain_work_v1.py", "scripts/verify_doc_domain_work_v1.py",
    "scripts/prepare_doc_domain_dependencies_v1.py",
    "scripts/run_doc_fusion_benchmark_development_v1.py",
    "scripts/run_doc_river_architecture_comparison_v1.py",
    "src/river_graph/experiments/doc_domain_work.py",
    "src/river_graph/experiments/doc_fusion_benchmark.py",
    "src/river_graph/experiments/provenance.py",
    "src/river_graph/models/river_architecture_comparison.py",
    "tests/test_doc_domain_work_v1.py", str(LITERATURE), "cache/doc_domain_work_v1/dependency_setup.json")


def dependency_hashes():
    directory = optional_dependencies()
    return {path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
        and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def freeze(root, repository, threads):
    previous = json.loads((REFERENCE / "protocol.json").read_text())
    if previous["regions"] != list(REGIONS) or previous["seeds"] != list(SEEDS):
        raise ValueError("reference task definition differs")
    snapshots = {name: sha256_file(name) for name in CODE}
    dependencies = {}
    for region in REGIONS:
        directory = REFERENCE / "regions" / f"huc4_{region}"
        verify(directory, "test_complete.json")
        dependencies[str(directory)] = {name: sha256_file(directory / name) for name in
            ("test_complete.json", "predictions.parquet", "predictions.meta.json", "costs.json")}
    sources = Path("cache/doc_domain_work_v1/sources")
    protocol = {
        "experiment": ROOT.name, "stage": "development_DOC_literature_family_comparison",
        "scope": "previously inspected five internal Mississippi HUC4 tasks; not new independent rivers or SOTA",
        "adaptation_status": json.loads(LITERATURE.read_text())["adaptation_status"],
        "literature": json.loads(LITERATURE.read_text()),
        "regions": list(REGIONS), "seeds": list(SEEDS), "models": list(MODELS),
        "dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET),
        "nodes": str(NODES), "nodes_sha256": sha256_file(NODES),
        "mask_sha256": {r: sha256_file(MASKS / f"huc4_{r}.npz") for r in REGIONS},
        "runtime_snapshot": snapshots, "optional_package_files": dependency_hashes(),
        "optional_dependency_setup": json.loads(Path("cache/doc_domain_work_v1/dependency_setup.json").read_text()),
        "primary_source_files": {name: sha256_file(sources / name) for name in
            ("aqua_oc_2025.pdf", "aqua_oc_supplement.docx")},
        "reference_dependencies": dependencies,
        "reference_protocol_sha256": sha256_file(REFERENCE / "protocol.json"),
        "reference_selection_sha256": sha256_file(REFERENCE / "all_selections.json"),
        "original_repository_dispatcher_sha256": sha256_file(repository / "scripts/run_ladder.py"),
        "isolation": "versioned runtime; original dispatcher and existing protocols unchanged",
        "threads": threads, "candidates": CANDIDATES,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "sklearn": sklearn.__version__, **dependency_versions()},
        "budgets": {"recipes_per_family": 3, "rf_trees": 300, "xgb_trees": 200,
            "gam_max_iterations": 100, "unique_candidate_fits": 120,
            "deterministic_families": list(DETERMINISTIC),
            "svr": "RBF kernel; exact libsvm; no iteration cap; training-cell-only StandardScaler",
            "reference": "audited frozen predictions reused, no additional tuning",
            "matching": "family-specific limited budgets, not an equal-compute optimality claim"},
        "permitted_information": previous["permitted_information"],
        "new_feature_sets": {
            "doc_ecological_rf": "23 = environment15 + current driver/mask/age/season8; no history",
            "doc_hydro_gam": "same23; smooth environment/T/Q + linear headwater/masks/ages/season + area-by-Q tensor",
            "doc_xgboost": "same full113 local covariates as reference: environment + 12-month windows + season",
            "doc_svr": "same full113; extra StandardScaler fitted on source training cells only",
            "rf_history_control": "original 113-column RF; reused frozen predictions"},
        "selection": previous["selection"], "target_access": previous["target_access"],
        "joint_primary": previous["joint_primary"], "secondary": previous["secondary"],
        "postprocessing": previous["postprocessing"], "high_doc": previous["high_doc"],
        "aggregation": previous["aggregation"], "uncertainty": previous["uncertainty"],
        "structure_diagnostics": previous["structure_diagnostics"],
        "prior_paper_endpoints_modified": False,
    }
    for key in ("dataset_sha256", "nodes_sha256", "mask_sha256"):
        if previous[key] != protocol[key]:
            raise ValueError("reference data or masks differ")
    destination = root / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen protocol changed; use a new version")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["protocol_hash"] != digest(protocol):
            raise ValueError("protocol manifest changed")
        for name, expected in snapshots.items():
            if sha256_file(root / "code_snapshot" / name) != expected:
                raise ValueError("source snapshot changed")
    else:
        write_json(destination, protocol)
        for name in snapshots:
            destination = root / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
        write_json(root / "manifest.json", {
            "frozen_at": utc_now(), "protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": snapshots[CODE[1]],
            "runtime_snapshot_hash": digest(snapshots),
            "dataset_sha256": protocol["dataset_sha256"], "nodes_sha256": protocol["nodes_sha256"],
            "mask_sha256": protocol["mask_sha256"], "optional_package_hash": digest(protocol["optional_package_files"]),
            "prior_paper_endpoints_modified": False})
    return protocol


def base_config(protocol, region):
    return {"experiment": ROOT.name, "target_huc4": region,
        "protocol_hash": digest(protocol), "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "dataset_sha256": protocol["dataset_sha256"], "mask_sha256": protocol["mask_sha256"][region]}


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
    inputs, split, months = ctx["inputs"], ctx["split"], ctx["months"]
    flat = np.asarray(data["y"]).ravel()
    train_y, val_y = flat[split["train"]], flat[split["val"]]
    if not np.isfinite(train_y).all() or not np.isfinite(val_y).all() or np.any(train_y < 0):
        raise ValueError("invalid source DOC labels")
    write_json(run / "normalization.json", inputs["normalization"])
    selections, receipts = {}, {}
    for family in FAMILIES:
        train_x = features(inputs, split["train"], months, family)
        val_x = features(inputs, split["val"], months, family)
        scores = {}
        for name, recipe in CANDIDATES[family].items():
            scores[name] = []
            for seed in ((0,) if family in DETERMINISTIC else SEEDS):
                directory = run / "fits" / family / name / f"seed{seed}"
                fit_config = {**config, "family": family, "recipe_id": name, "recipe": recipe, "seed": seed}
                if (directory / "fit_complete.json").exists():
                    saved_config = json.loads((directory / "config.json").read_text())
                    if {k: v for k, v in saved_config.items() if k != "started_at"} != fit_config:
                        raise ValueError("fitted recipe configuration changed")
                    verify(directory, "fit_complete.json", saved_config)
                    info = json.loads((directory / "info.json").read_text())
                else:
                    fit_config["started_at"] = utc_now()
                    write_json(directory / "config.json", fit_config)
                    status = {"stage": "source_validation_fitting", "region": region,
                        "family": family, "recipe": name, "seed": seed}
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
                        "deterministic": family in DETERMINISTIC}
                    joblib.dump(model, directory / "model.joblib", compress=3)
                    write_json(directory / "info.json", info)
                    save_predictions(directory, "validation_predictions", frame, fit_config, selection_role="source_validation")
                    bind(directory, "fit_complete.json", fit_config,
                        ("config.json", "model.joblib", "info.json", "validation_predictions.parquet", "validation_predictions.meta.json"))
                    print(json.dumps({**status, "completed": True, "seconds": seconds}), flush=True)
                    del model
                scores[name].append({key: info[key] for key in ("mae", "log1p_mae")})
                receipts[str(directory.relative_to(run))] = sha256_file(directory / "fit_complete.json")
        selections[family] = {"recipe_id": select_recipe(scores), "validation_scores": scores, "test_used": False}
    write_json(run / "selection.json", {"config": config, "selections": selections,
        "fit_receipts": receipts, "frozen_at": utc_now(), "test_used": False})
    bind(run, "selection_complete.json", config, ("selection.json", "normalization.json"))
    print(json.dumps({"stage": "selection_frozen", "region": region}), flush=True)


def evaluate_region(root, region, protocol, data, gate_hash):
    run = root / "regions" / f"huc4_{region}"
    config = {**base_config(protocol, region), "all_selections_sha256": gate_hash}
    if (run / "test_complete.json").exists():
        verify(run, "test_complete.json", config)
        return pd.read_parquet(run / "predictions.parquet")
    verify(run, "selection_complete.json", base_config(protocol, region))
    ctx = context(data, region)
    inputs, cells, months = ctx["inputs"], ctx["split"]["test"], ctx["months"]
    choices = json.loads((run / "selection.json").read_text())["selections"]
    ref = REFERENCE / "regions" / f"huc4_{region}"
    for name, expected in protocol["reference_dependencies"][str(ref)].items():
        if sha256_file(ref / name) != expected:
            raise ValueError("reference predictions changed")
    verify(ref, "test_complete.json")
    saved = pd.read_parquet(ref / "predictions.parquet")
    train_y = np.asarray(data["y"]).ravel()[ctx["split"]["train"]]
    query = {"cell": cells, "station": np.asarray(data["site_no"], str)[cells // months],
        "month": np.asarray(data["months"], str)[cells % months],
        "y_true": np.asarray(data["y"]).ravel()[cells], "target_huc4": region,
        "visibility_role": "test", "high_doc": np.asarray(data["y"]).ravel()[cells] >= np.quantile(train_y, .9),
        "stream_order": inputs["order"].numpy()[cells // months], "k": 0}
    frames, costs, deterministic = [], [], {}
    for seed in SEEDS:
        for family in FAMILIES:
            name = choices[family]["recipe_id"]
            fit_seed = 0 if family in DETERMINISTIC else seed
            directory = run / "fits" / family / name / f"seed{fit_seed}"
            fit_config = json.loads((directory / "config.json").read_text())
            verify(directory, "fit_complete.json", fit_config)
            if family in deterministic:
                predicted = deterministic[family]
            else:
                model = joblib.load(directory / "model.joblib")
                started = time.monotonic()
                predicted, clips = concentration_prediction(model.predict(features(inputs, cells, months, family)), fit_config["recipe"]["target"])
                costs.append({"model_name": family, "seed": seed, "fit_seed": fit_seed,
                    "inference_seconds": time.monotonic() - started, "clips": clips, "fit": str(directory)})
                del model
                if family in DETERMINISTIC:
                    deterministic[family] = predicted
            frames.append(pd.DataFrame({**query, "model_name": family, "seed": seed, "y_pred": predicted}))
        for old, new in (("random_forest", "rf_history_control"), ("fixed_fusion", "fixed_fusion")):
            part = saved[(saved.model_name == old) & (saved.seed == seed)]
            np.testing.assert_array_equal(part.cell.to_numpy(), cells)
            np.testing.assert_array_equal(part.y_true.to_numpy(), query["y_true"])
            frames.append(pd.DataFrame({**query, "model_name": new, "seed": seed, "y_pred": part.y_pred.to_numpy()}))
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["model_name", "seed", "cell"]).any() or not np.isfinite(output[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("duplicate or nonfinite query predictions")
    write_json(run / "costs.json", costs)
    save_predictions(run, "predictions", output, config, all_selections_sha256=gate_hash, query_cells=len(cells))
    bind(run, "test_complete.json", config, ("predictions.parquet", "predictions.meta.json", "costs.json"))
    print(json.dumps({"stage": "target_scored", "region": region, "query_cells": len(cells)}), flush=True)
    return output


def analyze(root, frame, protocol, data):
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    folds, structures, summaries = [], [], []
    for (name, seed, region), part in frame.groupby(["model_name", "seed", "target_huc4"]):
        folds.append({"model_name": name, "seed": seed, "target_huc4": region, **metrics(part)})
        structures.append({"model_name": name, "seed": seed, "target_huc4": region,
            **structure_metrics(part, data["edge_index"], data["y"].shape[1])})
    fold_table = pd.DataFrame(folds)
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
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    fold_table.to_csv(out / "by_region_seed.csv", index=False)
    pd.DataFrame(structures).to_csv(out / "structure_by_region_seed.csv", index=False)
    write_json(out / "paired_comparisons.json", [paired_basin_interval(frame, name, endpoint)
        for name in MODELS if name != "fixed_fusion" for endpoint in ("mae", "log1p_mae")])
    bind(out, "complete.json", {"protocol_hash": digest(protocol)},
        ("summary.csv", "by_region_seed.csv", "structure_by_region_seed.csv", "paired_comparisons.json"))


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
        print(json.dumps({"stage": "frozen", "protocol_hash": digest(protocol), "unique_candidate_fits": 120}), flush=True)
        return
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        data = torch.load(DATASET, weights_only=False, map_location="cpu")
        for region in REGIONS:
            fit_region(root, region, protocol, data)
            gc.collect()
        gate = {"protocol_hash": digest(protocol), "regional_selections": {
            r: sha256_file(root / "regions" / f"huc4_{r}" / "selection_complete.json") for r in REGIONS}, "test_used": False}
        if (root / "all_selections.json").exists():
            if json.loads((root / "all_selections.json").read_text()) != gate:
                raise ValueError("selection gate changed")
        else:
            write_json(root / "all_selections.json", gate)
        frames = [evaluate_region(root, r, protocol, data, sha256_file(root / "all_selections.json")) for r in REGIONS]
        frame = pd.concat(frames, ignore_index=True)
        expected = np.concatenate([context(data, r)["split"]["test"] for r in REGIONS])
        for name in MODELS:
            for seed in SEEDS:
                np.testing.assert_array_equal(frame[(frame.model_name == name) & (frame.seed == seed)].cell.to_numpy(), expected)
        analyze(root, frame, protocol, data)
    write_json(root / "batch_complete.json", {"protocol_hash": digest(protocol), "completed_at": utc_now(),
        "analysis_receipt_sha256": sha256_file(root / "analysis/complete.json"),
        "unique_query_cells": int(frame.cell.nunique()), "models": len(MODELS), "sota_established": False})
    write_json(root / "progress.json", {"stage": "complete", "completed_at": utc_now()})
    print(json.dumps({"stage": "complete", "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
