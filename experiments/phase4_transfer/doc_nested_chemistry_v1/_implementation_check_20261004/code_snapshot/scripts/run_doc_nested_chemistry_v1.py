"""Develop a separate linear chemical increment on frozen DOC validation tasks."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_chemical_kernel_v1 import linear_preclip
from run_doc_chemistry_support_v1 import add_selected, make_panels
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import (
    bind_files,
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.nested_chemical_adapter import (
    NestedChemicalAdapter,
    NestedChemicalEpisode,
)
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_nested_chemistry_v1")
PRIOR = Path("experiments/phase4_transfer/doc_chemistry_confirmation_v1")
PARTITIONS, SEEDS, KS = (242, 243, 244), (42, 43, 44), (0, 1, 3, 5)
LEGACY = "neural_chemistry_integrated_legacy"
REFERENCES = ("point_integrated_legacy", "neural_chemistry_integrated_selected", LEGACY,
              "tree_chemistry_selected")
NESTED = {"chemistry": "neural_chemistry_integrated_nested",
          "masks": "neural_chemistry_integrated_nested_masks"}
MODELS = (*REFERENCES, *NESTED.values())


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    for name in ("scripts/run_ladder.py", "scripts/run_doc_nested_chemistry_v1.py",
                 "scripts/run_doc_chemical_kernel_v1.py", "scripts/run_doc_chemistry_support_v1.py",
                 "scripts/run_doc_auxiliary_chemistry_v1.py", "scripts/run_unified_doc_spatial.py",
                 "scripts/run_doc_tail_residual_v1.py", "src/river_graph/models/nested_chemical_adapter.py",
                 str(ROOT / "study_plan.md")):
        snapshot[name] = sha256_file(name)
    path = root / "runtime_snapshot.json"
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("Execution changed; use a new development directory")
        return verify_runtime_snapshot(root)
    write_json(path, snapshot)
    for name in snapshot:
        destination = root / "code_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def validation_label_view(dataset, split):
    labels = np.full(np.asarray(dataset["y"]).size, np.nan)
    labels[split["val"]] = np.asarray(dataset["y"]).ravel()[split["val"]]
    return labels


def load_parent(prior):
    config = json.loads((prior / "config.json").read_text())
    verify_files(prior, "complete.json", config)
    for name in ("dataset", "mask"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError(f"Changed parent {name}")
    dataset = torch.load(config["dataset_path"], weights_only=False)
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {name: saved[name].copy() for name in saved.files}
    full = pd.read_parquet(prior / "full_grid.parquet")
    labels = validation_label_view(dataset, split)
    months = np.asarray(dataset["y"]).shape[1]
    del dataset
    with np.load(prior / "chemical/representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in ("legacy", "masks_aug", "chemistry_aug")}
        chemical = {name: saved[f"{name}_chemical"].copy() for name in NESTED}
        active, source_ids = saved["active"].copy(), saved["source_station_ids"].copy()
    if np.intersect1d(source_ids, np.unique(np.r_[split["val"], split["test"]] // months)).size:
        raise ValueError("Coordinate fitting source stations overlap held stations")
    if not np.isnan(labels[np.r_[split["train"], split["test"], split["context"]]]).all():
        raise ValueError("Only source-validation DOC may enter development")
    states = {name: json.loads((prior / "chemical" / f"{name}.json").read_text())
              for name in ("adapters", "mixers", "legacy_adapters", "legacy_mixers", "basis_selection")}
    adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in states["adapters"].items()}
    mixers = {name: SupportAwareResidualTransfer.from_dict(state) for name, state in states["mixers"].items()}
    bases = {name: full[f"{name}_pred"].to_numpy() for name in ("neural_chemistry", "tree_chemistry")}
    panels = add_selected(make_panels(full, bases, shapes, adapters, mixers,
        states["legacy_adapters"], states["legacy_mixers"], labels, split, months, active, role="val"),
        states["basis_selection"])
    panels = panels[panels.model_name.isin(REFERENCES)].copy()
    panels["y_true"] = labels[panels.cell.to_numpy()]
    return config, split, full, labels, months, shapes["legacy"], chemical, active, source_ids, states, panels


def completed_support_predictions(full, legacy, states, labels, support, query, *, months, k):
    """Reproduce the legacy fitted values on support without relaxing overlap rules."""
    model = SupportAwareResidualTransfer.from_dict(
        states["legacy_mixers"]["neural_chemistry_integrated_gru_tuned_anchor"])
    base = model.selected_base(full.context_pred.to_numpy(), full.neural_chemistry_pred.to_numpy(),
                               full.ecological_memory.to_numpy(), k=k)
    choice = model.to_dict()["selection_by_k"][str(k)]
    qz, sz, _ = linear_preclip(base, legacy, labels, support, query, months=months, k=k,
        alpha=choice["alpha"], ridge_strength=choice["ridge_strength"])
    return np.maximum(0., np.expm1(qz)), np.maximum(0., np.expm1(sz))


def make_episodes(full, panels, labels, split, months, legacy, states, coordinates, active):
    result = {}
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        parent = panels[panels.model_name.eq(LEGACY) & panels.k.eq(k)].sort_values("cell")
        np.testing.assert_array_equal(parent.cell.to_numpy(), query)
        candidate, spred = completed_support_predictions(full, legacy, states, labels, support, query,
                                                         months=months, k=k)
        np.testing.assert_allclose(candidate[active[query]], parent.y_pred.to_numpy()[active[query]],
                                   rtol=0, atol=1e-12)
        result[k] = NestedChemicalEpisode(k, query, labels[query], parent.y_pred.to_numpy(),
            support, labels[support], spred, coordinates[query], coordinates[support], active[query], active[support])
    return result


def apply_episode(adapter, episode, *, choice=None):
    kwargs = {} if choice is None else dict(choice)
    return adapter.adapt_components(episode.query_prediction, episode.query_cells,
        episode.support_prediction, episode.support_cells, episode.support_values,
        query_chemical=episode.query_chemical, support_chemical=episode.support_chemical,
        query_active=episode.query_active, support_active=episode.support_active, k=episode.k, **kwargs)


def build_products(adapter, episodes, references, model, *, cross_validation):
    rows = []
    for k, episode in episodes.items():
        row = references[references.model_name.eq(LEGACY) & references.k.eq(k)].sort_values("cell").copy()
        applied = apply_episode(adapter, episode)
        fold_ids = np.full(len(row), -1, dtype=int)
        if cross_validation and k >= 3:
            for fold in adapter.fold_diagnostics_:
                q = np.isin(episode.query_cells // adapter.n_months, fold["held_stations"])
                held = apply_episode(adapter, episode, choice=fold["choice"])
                for name in applied:
                    applied[name][q] = held[name][q]
                fold_ids[q] = fold["fold"]
            if (fold_ids < 0).any():
                raise ValueError("Conditional CV did not cover every validation station")
        row["legacy_y_pred"] = row.y_pred
        row["legacy_adaptation_delta"] = row.adaptation_delta
        row["legacy_candidate_y_pred"] = row.candidate_y_pred
        row["model_name"], row["y_pred"] = model, applied["y_pred"]
        row["candidate_y_pred"] = row.y_pred
        row["adaptation_delta"] = np.log1p(row.y_pred) - np.log1p(row.base_pred)
        row["chemical_delta"] = applied["chemical_delta"]
        row["chemical_support_count"] = applied["chemical_support_count"]
        row["conditional_fold"] = fold_ids
        for i in range(2):
            row[f"chemical_coefficient_{i}"] = applied["chemical_coefficients"][:, i]
        if k < 3:
            np.testing.assert_array_equal(row.y_pred.to_numpy(), episode.query_prediction)
        np.testing.assert_array_equal(row.y_pred.to_numpy()[~episode.query_active],
                                      episode.query_prediction[~episode.query_active])
        if not np.isfinite(row[["y_pred", "chemical_delta"]].to_numpy()).all():
            raise FloatingPointError("Nonfinite nested prediction")
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def run_one(root, prior_root, partition, seed, runtime):
    prior = prior_root / "runs" / f"split{partition}_seed{seed}"
    run = root / "runs" / prior.name
    if (run / "complete.json").exists():
        config = json.loads((run / "config.json").read_text())
        if config["runtime_snapshot_hash"] != runtime or config["parent_completion_hash"] != sha256_file(prior / "complete.json"):
            raise ValueError("Cached nested increment has changed sources")
        verify_files(run, "complete.json", config)
        print(f"{run.name}: verified complete", flush=True)
        return
    pc, split, full, labels, months, legacy, coordinates, active, source_ids, states, references = load_parent(prior)
    run.mkdir(parents=True, exist_ok=True)
    config = {"experiment": "doc_nested_chemistry_v1", "split_seed": partition, "seed": seed,
        "runtime_snapshot_hash": runtime, "parent_run": str(prior),
        "parent_completion_hash": sha256_file(prior / "complete.json"),
        "parent_bindings": {name: sha256_file(prior / name) for name in (
            "full_grid.parquet", "chemical/representations.npz", "chemical/legacy_mixers.json",
            "chemical/legacy_adapters.json", "chemical/adapters.json", "chemical/mixers.json", "chemical/basis_selection.json")},
        **{name: pc[name] for name in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train")},
        "models": list(MODELS), "k_values": list(KS), "target_analyte": "doc", "target_transform": "log1p",
        "ridge_values": [.1, 1., 10., 100.], "strength_values": [0., .25, .5, 1.],
        "selection_role": "source_validation", "selection_weights": "K equal then station equal",
        "selection_folds": 5, "fold_seed": 4100 + partition, "parent_frozen": True,
        "new_neural_fits": 0, "new_forest_fits": 0, "target_evaluation": False,
        "study_role": "conditional source-validation adapter development; parent previously used validation",
        "plan_hash": sha256_file(ROOT / "study_plan.md")}
    write_json(run / "config.json", config)
    outputs = {False: [references.copy()], True: [references.copy()]}
    for mode, name in NESTED.items():
        episodes = make_episodes(full, references, labels, split, months, legacy, states, coordinates[mode], active)
        grid = coordinates[mode].reshape(-1, months, 2)[source_ids]
        source_active = active.reshape(-1, months)[source_ids]
        adapter = NestedChemicalAdapter(months, fold_seed=4100 + partition).fit_coordinates(
            grid, source_active, source_role="source_training").fit(
                [episodes[3], episodes[5]], selection_role="source_validation")
        write_json(run / f"nested_{mode}.json", adapter.to_dict())
        for cv, groups in outputs.items():
            groups.append(build_products(adapter, episodes, references, name, cross_validation=cv))
        print(f"{run.name}/{mode}: {adapter.selection_}", flush=True)
    for cv, groups in outputs.items():
        frame = pd.concat(groups, ignore_index=True)
        frame["split_seed"], frame["seed"] = partition, seed
        frame["evaluation_kind"] = "conditional_adapter_cv" if cv else "validation_tuning"
        for field in ("chemical_delta", "chemical_support_count", "conditional_fold",
                      "chemical_coefficient_0", "chemical_coefficient_1"):
            frame[field] = frame[field].fillna(-1 if field == "conditional_fold" else 0)
        frame["legacy_y_pred"] = frame["legacy_y_pred"].fillna(frame.y_pred)
        frame["legacy_adaptation_delta"] = frame["legacy_adaptation_delta"].fillna(frame.adaptation_delta)
        frame["legacy_candidate_y_pred"] = frame["legacy_candidate_y_pred"].fillna(frame.candidate_y_pred)
        filename = "cv_predictions.parquet" if cv else "validation.parquet"
        frame.to_parquet(run / filename, index=False)
        bind_product(run, filename, config, runtime, ["nested_chemistry.json", "nested_masks.json"])
    files = sorted(path for path in run.iterdir() if path.is_file() and path.name != "complete.json")
    bind_files(run, "complete.json", files, config)
    print(f"{run.name}: conditional station validation complete; target DOC unopened", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(PARTITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.root.mkdir(parents=True, exist_ok=True)
    verify_runtime_snapshot(args.prior_root)
    runtime = freeze_runtime(args.root)
    for partition in args.split_seeds:
        for seed in args.seeds:
            run_one(args.root, args.prior_root, partition, seed, runtime)
            gc.collect()


if __name__ == "__main__":
    main()
