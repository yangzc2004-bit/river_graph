"""Independently replay the source-validation chemical residual interpolation pilot."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import read_source
from verify_doc_chemistry_support_v1 import (
    PIPELINES,
    _add_selected,
    _panels,
    _summary,
)
from verify_doc_daily_hydro_support_basis_v1 import _manual_base, _sidecar

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_chemical_kernel_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
MODES, ETAS, BANDWIDTHS = ("masks", "chemistry"), (0., .25, .5, 1.), (.5, 1., 2.)
MODELS = tuple(f"{pipe}_{mode}" for pipe in PIPELINES.values() for mode in ("parent", *MODES))


def _distance_definition(chemical, active, source_ids):
    """Rebuild the source-only distance inventory, without production helpers."""
    positive, inventory = [], []
    for station in np.sort(source_ids):
        months = np.flatnonzero(active[station])
        n_active = len(months)
        if len(months) > 24:
            positions = np.linspace(0, len(months)-1, 24, dtype=np.int64)
            months = months[positions]
        values = chemical[station, months]
        left, right = np.triu_indices(len(months), 1)
        distance = np.linalg.norm(values[left]-values[right], axis=1)
        if not np.isfinite(distance).all():
            raise ValueError("Nonfinite source-only chemical distances")
        nonzero = distance[distance > 0]
        positive.extend(nonzero.tolist())
        inventory.append({"station_index": int(station), "n_active_months": n_active,
            "sampled_month_indices": months.tolist(), "n_sampled_months": len(months),
            "n_pairs": len(distance), "n_positive_pairs": len(nonzero)})
    fallback = not len(positive)
    return {"version": 1, "source_role": "source_training", "distance_scale": 1. if fallback else float(np.median(positive)),
        "fallback": fallback, "fallback_reason": "no_positive_within_station_distance" if fallback else None,
        "n_source_stations": len(source_ids), "n_active_source_stations": sum(row["n_active_months"] > 0 for row in inventory),
        "n_sampled_months": sum(row["n_sampled_months"] for row in inventory),
        "n_pairs": sum(row["n_pairs"] for row in inventory), "n_positive_pairs": len(positive),
        "max_months_per_station": 24,
        "sampling": "evenly spaced indices in sorted active months, including endpoints; floor integer positions",
        "distance": "Euclidean in frozen 2D chemical coordinates; median of positive within-station unordered pairs",
        "source_station_indices": np.sort(source_ids).tolist(), "station_inventory": inventory}


def _linear_remainder(support_values, support_base, support_basis, *, alpha, ridge):
    """Original all-K linear correction evaluated before inverse/clipping."""
    residual = np.log1p(support_values)-np.log1p(support_base)
    mean = residual.mean()
    adjustment = np.full(len(residual), alpha*mean)
    if len(residual) > 1 and np.isfinite(ridge):
        centered = support_basis-support_basis.mean(axis=0)
        coefficient = np.linalg.solve(centered.T @ centered/len(residual)+ridge*np.eye(centered.shape[1]),
            centered.T @ (residual-mean)/len(residual))
        adjustment += centered @ coefficient
    return residual-adjustment, adjustment


def _linear_preclip(query_base, support_values, support_base, query_basis, support_basis, *, alpha, ridge):
    result = np.log1p(query_base)
    if not len(support_values):
        return result
    residual = np.log1p(support_values)-np.log1p(support_base)
    mean = residual.mean()
    result += alpha*mean
    if len(residual) > 1 and np.isfinite(ridge):
        mean_basis = support_basis.mean(axis=0)
        centered = support_basis-mean_basis
        coefficient = np.linalg.solve(centered.T @ centered/len(residual)+ridge*np.eye(centered.shape[1]),
            centered.T @ (residual-mean)/len(residual))
        result += (query_basis-mean_basis) @ coefficient
    return result


def _kernel(query, support, query_active, support_active, remainder, scale, bandwidth, eta):
    """Independent normalized RBF convex interpolation; no kernel-ridge solve."""
    result = np.zeros(len(query))
    qa, sa = np.asarray(query_active, bool), np.asarray(support_active, bool)
    if eta == 0 or sa.sum() < 2 or not qa.any():
        return result
    q, s, r = query[qa], support[sa], remainder[sa]
    if np.all(s == s[0]):
        return result
    centered = r-r.mean()
    distances = np.sum(((q[:, None, :]-s[None, :, :])/(scale*bandwidth))**2, axis=-1)
    log_weights = -.5*distances
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-14)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid RBF interpolation weights")
    correction = eta*(weights @ centered)
    tolerance = 1e-14*(1+np.abs(centered).max())
    if ((correction < eta*centered.min()-tolerance).any()
            or (correction > eta*centered.max()+tolerance).any()):
        raise ValueError("Kernel correction exceeds centered donor convex bounds")
    result[qa] = np.clip(correction, centered.min(), centered.max())
    return result


def _csv_check(path, expected, *, station=False):
    saved = pd.read_csv(path, dtype={"station": str} if station else None)
    expected = expected[saved.columns]
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)


def _selected_linear(pipe, variant, k, full, adapters, mixers):
    if pipe == "neural_chemistry_integrated":
        model = mixers[variant]
        choice = model.to_dict()["selection_by_k"][str(k)]
        base = _manual_base(full.context_pred.to_numpy(), full.neural_chemistry_pred.to_numpy(),
            full.ecological_memory.to_numpy(), choice["gamma"])
    else:
        alias = "neural" if pipe == "neural_chemistry" else "tree"
        choice = {**adapters[(alias, variant)].to_dict()["selection_by_k"][str(k)], "gamma": 0.}
        base = full[f"{pipe}_pred"].to_numpy()
    return base, choice


def _linear_all_stations(base, basis, labels, support, query, months, alpha, ridge):
    qz, sz, remainder = np.log1p(base[query]), np.log1p(base[support]), np.empty(len(support))
    if not len(support):
        return qz, sz, remainder
    for station in np.unique(query//months):
        q, s = query//months == station, support//months == station
        qz[q] = _linear_preclip(base[query[q]], labels[support[s]], base[support[s]],
            basis[query[q]], basis[support[s]], alpha=alpha, ridge=ridge)
        sz[s] = _linear_preclip(base[support[s]], labels[support[s]], base[support[s]],
            basis[support[s]], basis[support[s]], alpha=alpha, ridge=ridge)
        remainder[s] = np.log1p(labels[support[s]])-sz[s]
        algebra, _ = _linear_remainder(labels[support[s]], base[support[s]], basis[support[s]], alpha=alpha, ridge=ridge)
        np.testing.assert_allclose(remainder[s], algebra, rtol=0, atol=2e-14)
    return qz, sz, remainder


def _correction(coordinates, active, remainder, support, query, months, scale, bandwidth, eta):
    correction, count = np.zeros(len(query)), np.zeros(len(query), dtype=np.int64)
    for station in np.unique(query//months):
        q, s = query//months == station, support//months == station
        count[q] = int(active[support[s]].sum())
        correction[q] = _kernel(coordinates[query[q]], coordinates[support[s]], active[query[q]],
            active[support[s]], remainder[s], scale, bandwidth, eta)
    return correction, count


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    expected = {"experiment": "doc_chemical_kernel_v1", "split_seed": split_seed, "seed": seed,
        "runtime_snapshot_hash": runtime, "pipelines": list(PIPELINES.values()), "models": list(MODELS),
        "kernel_modes": list(MODES), "k_values": list(KS), "eta_values": list(ETAS),
        "bandwidth_values": list(BANDWIDTHS), "max_source_months_per_station": 24,
        "selection_role": "source_validation", "target_evaluation": False,
        "new_neural_fits": 0, "new_forest_fits": 0, "linear_calibrators_refitted": False}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Pilot scope, source roles or candidate grid differs")
    completion = verify_files(run, "complete.json", config)
    model_files = {"distances.json", "kernel_selection.json", "linear_remainder_checks.json",
        "all_candidate_scores.csv", "candidate_station_scores.csv"}
    if set(completion["files"]) != {"config.json", *model_files, "source_validation.csv", "station_responses.csv",
            "timing.json", "validation_predictions.parquet", "validation_predictions.meta.json"}:
        raise ValueError("Incomplete source-validation pilot package")
    prior = Path(config["prior_run"])
    if sha256_file(prior / "complete.json") != config["prior_completion_hash"]:
        raise ValueError("Frozen linear support parent changed")
    pc, _, dataset, split, full = read_source(prior)
    for name, expected_hash in config["parent_bindings"].items():
        if sha256_file(prior / name) != expected_hash:
            raise ValueError("A bound parent calibration or representation changed")
    for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[key] != pc[key]:
            raise ValueError("Source dataset/protocol differs from the parent")
    parent_report = prior.parent.parent / "verification/replay_checks.json"
    records = [row for row in json.loads(parent_report.read_text())["results"] if row["run"] == prior.name]
    if (len(records) != 1 or records[0]["status"] != "verified"
            or records[0]["completion_sha256"] != config["prior_completion_hash"]
            or records[0]["runtime_snapshot_hash"] != verify_runtime_snapshot(prior.parent.parent)):
        raise ValueError("Parent numerical verification does not bind this package")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    labels = np.full(np.prod(shape), np.nan)
    labels[split["val"]] = np.asarray(dataset["y"]).ravel()[split["val"]]
    del dataset
    if not np.isnan(labels[np.r_[split["train"], split["test"]]]).all():
        raise ValueError("Only source validation labels may be extracted")
    with np.load(prior / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in ("legacy", "masks_aug", "chemistry_aug")}
        coordinates = {mode: saved[f"{mode}_chemical"].copy() for mode in MODES}
        active, source_ids = saved["active"].copy(), saved["source_station_ids"].copy()
    np.testing.assert_array_equal(source_ids, np.unique(split["train"]//months))
    np.testing.assert_array_equal(source_ids, config["source_station_ids"])
    distances = {mode: _distance_definition(coordinates[mode].reshape(*shape, 2), active.reshape(shape), source_ids) for mode in MODES}
    if distances != json.loads((run / "distances.json").read_text()):
        raise ValueError("Independent source-only median distance inventory differs")
    adapter_states = json.loads((prior / "adapters.json").read_text())
    mixer_states = json.loads((prior / "mixers.json").read_text())
    adapters = {(alias, variant): SupportShapeAdapter.from_dict(adapter_states[f"{PIPELINES[alias]}_{variant}"])
        for alias in ("neural", "tree") for variant in shapes}
    mixers = {variant: SupportAwareResidualTransfer.from_dict(mixer_states[f"neural_chemistry_integrated_{variant}"])
        for variant in shapes}
    selection = json.loads((prior / "basis_selection.json").read_text())
    decoder = Path(config["decoder_run"])
    dc = json.loads((decoder / "config.json").read_text())
    verify_files(decoder, "complete.json", dc)
    if (str(decoder) != pc["prior_run"] or sha256_file(decoder / "complete.json") != config["decoder_completion_hash"]
            or config["decoder_completion_hash"] != pc["prior_completion_hash"]):
        raise ValueError("The frozen nonlinear decoder parent changed")
    old_adapters, old_mixers = (json.loads((decoder / name).read_text()) for name in ("adapters.json", "mixers.json"))
    bases = {"neural": full.neural_chemistry_pred.to_numpy(), "tree": full.tree_chemistry_pred.to_numpy()}
    parents = _add_selected(_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
        labels, split, months, active, "val"), selection)
    _csv_check(prior / "source_validation.csv", _summary(parents, labels, active))
    product = pd.read_parquet(run / "validation_predictions.parquet")
    _sidecar(run, "validation_predictions.parquet", config, runtime, product, completion, model_files)
    _, fixed_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    if (len(fixed_query) != config["validation_query_cells"]
            or not np.isin(product.cell, split["val"]).all() or np.intersect1d(product.cell, split["test"]).size
            or not product.evaluation_role.eq("source_validation").all()
            or not product.split_seed.eq(split_seed).all() or not product.seed.eq(seed).all()
            or product.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, product[["model_name", "k"]].drop_duplicates().to_numpy())) != {(name, k) for name in MODELS for k in KS}
            or not np.isfinite(product.select_dtypes(include="number")).all().all()):
        raise ValueError("Validation-only product identities or coverage differ")
    candidates, station_candidates, remainder_checks, output_panels, query_checks = [], [], [], {}, []
    choices = {pipe: {mode: {} for mode in MODES} for pipe in PIPELINES.values()}
    for pipe in PIPELINES.values():
        for k in KS:
            support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
            np.testing.assert_array_equal(query, fixed_query)
            parent = parents[(f"{pipe}_selected", k)]
            np.testing.assert_array_equal(parent["cell"], query)
            representation = selection["choices"][pipe][str(k)]["basis"]
            base, choice = _selected_linear(pipe, representation, k, full, adapters, mixers)
            ridge = np.inf if choice["ridge_strength"] == "infinity" else choice["ridge_strength"]
            qz, sz, remainder = _linear_all_stations(base, shapes[representation], labels, support, query,
                months, choice["alpha"], ridge)
            reconstructed = np.maximum(0, np.expm1(qz))
            np.testing.assert_allclose(reconstructed, parent["candidate_y_pred"], rtol=0, atol=1e-12)
            final = reconstructed.copy()
            final[~active[query]] = parent["y_pred"][~active[query]]
            np.testing.assert_allclose(final, parent["y_pred"], rtol=0, atol=1e-12)
            remainder_checks.append({"pipeline": pipe, "k": k, "parent_basis": representation,
                "alpha": choice["alpha"], "ridge_strength": choice["ridge_strength"], "gamma": choice["gamma"],
                "query_candidate_max_abs_difference": float(np.abs(reconstructed-parent["candidate_y_pred"]).max()),
                "query_final_max_abs_difference": float(np.abs(final-parent["y_pred"]).max()),
                "support_rows": len(support), "query_rows": len(query),
                "support_preclip_log_min": float(sz.min()) if len(sz) else None,
                "support_remainder_abs_mean": float(np.abs(remainder).mean()) if len(remainder) else 0.,
                "support_preclip_used": True, "all_k_support_moments": True})
            donor_count = np.zeros(len(query), dtype=np.int64)
            for station in np.unique(query//months):
                donor_count[query//months == station] = active[support[support//months == station]].sum()
            parent_columns = {**parent, "parent_candidate_y_pred": parent["candidate_y_pred"],
                "parent_y_pred": parent["y_pred"], "parent_preclip_z": qz,
                "kernel_log_correction": np.zeros(len(query)), "active_support_count": donor_count}
            parent_columns.pop("candidate_y_pred")
            output_panels[(f"{pipe}_parent", k)] = (parent_columns, "parent", 0., 1., representation)
            for mode in MODES:
                scale = distances[mode]["distance_scale"]
                trials, outputs = [], {}
                for eta in ((0.,) if k < 2 else ETAS):
                    for bandwidth in ((1.,) if k < 2 else BANDWIDTHS):
                        delta, donors = _correction(coordinates[mode], active, remainder, support, query,
                            months, scale, bandwidth, eta)
                        prediction = parent["y_pred"].copy()
                        changed = delta != 0
                        prediction[changed] = np.maximum(0, np.expm1(qz[changed]+delta[changed]))
                        if not np.isfinite(prediction).all():
                            raise ValueError("A candidate native kernel prediction is nonfinite")
                        np.testing.assert_array_equal(prediction[~active[query]], parent["y_pred"][~active[query]])
                        error = np.abs(prediction-labels[query])
                        item = {"pipeline": pipe, "kernel_mode": mode, "k": k, "parent_basis": representation,
                            "eta": eta, "bandwidth_scale": bandwidth, "source_distance_scale": scale,
                            "valid": True, "locked": k < 2, "n": len(query), "n_active": int(active[query].sum()),
                            "mae": float(error.mean()), "active_mae": float(error[active[query]].mean())}
                        candidates.append(item)
                        trials.append(item)
                        outputs[eta, bandwidth] = prediction, delta, donors
                        if k >= 3:
                            for station in np.unique(query//months):
                                member = query//months == station
                                valid = member & active[query]
                                station_candidates.append({**{key: item[key] for key in ("pipeline", "kernel_mode", "k", "parent_basis", "eta", "bandwidth_scale", "source_distance_scale", "valid")},
                                    "station_index": int(station), "station": str(full.iloc[query[member]]["station"].iloc[0]),
                                    "n": int(member.sum()), "n_active": int(valid.sum()),
                                    "error_sum": float(error[member].sum()), "active_error_sum": float(error[valid].sum()),
                                    "split_seed": split_seed, "seed": seed})
                chosen = min(trials, key=lambda row: (row["active_mae"], row["eta"], row["bandwidth_scale"] != 1., row["bandwidth_scale"]))
                choices[pipe][mode][str(k)] = chosen
                prediction, delta, donors = outputs[chosen["eta"], chosen["bandwidth_scale"]]
                expected_panel = {**parent_columns, "y_pred": prediction, "kernel_log_correction": delta,
                    "active_support_count": donors, "adaptation_delta": parent["adaptation_delta"].copy()}
                changed = delta != 0
                expected_panel["adaptation_delta"][changed] = np.log1p(prediction[changed])-np.log1p(parent["base_pred"][changed])
                output_panels[(f"{pipe}_{mode}", k)] = (expected_panel, mode, chosen["eta"], chosen["bandwidth_scale"], representation)
    _csv_check(run / "all_candidate_scores.csv", pd.DataFrame(candidates))
    _csv_check(run / "candidate_station_scores.csv", pd.DataFrame(station_candidates), station=True)
    selected = {"version": 1, "selection_role": "source_validation", "score": "active_validation_native_mae",
        "tie_rule": "smaller eta; bandwidth1; smaller bandwidth", "eta_values": list(ETAS),
        "bandwidth_values": list(BANDWIDTHS), "pipelines": choices}
    if selected != json.loads((run / "kernel_selection.json").read_text()):
        raise ValueError("Independent active-validation kernel selection differs")
    expected_remainders = {"parent_validation_mae_tolerance": 1e-12,
        "parent_validation_groups_reproduced": 52, "label_scope": "split.val only", "checks": remainder_checks}
    if expected_remainders != json.loads((run / "linear_remainder_checks.json").read_text()):
        raise ValueError("Independent preclip all-K remainder diagnostics differ")
    descriptors = ("station", "month", "analyte", "visibility_role", "ecological_novelty", "upstream_support",
                   "ph_available", "ec_available", "aux_available", "doc_observed")
    summary_panels = {}
    for (name, k), (panel, mode, eta, bandwidth, representation) in output_panels.items():
        stored = product[product.model_name.eq(name) & product.k.eq(k)].sort_values("cell")
        for column, values in panel.items():
            np.testing.assert_array_equal(stored[column], values)
        for column in descriptors:
            np.testing.assert_array_equal(stored[column], full.iloc[fixed_query][column])
        for column, value in (("kernel_mode", mode), ("pipeline", name.removesuffix(f"_{mode}")), ("eta", eta), ("bandwidth_scale", bandwidth),
                              ("parent_basis", representation)):
            if not stored[column].eq(value).all():
                raise ValueError("Selected kernel product metadata differs")
        np.testing.assert_array_equal(stored.y_true, labels[fixed_query])
        if k < 2 or eta == 0:
            np.testing.assert_array_equal(stored.y_pred, stored.parent_y_pred)
        summary_panels[(name, k)] = panel
        query_checks.append({"model_name": name, "k": k, "n_query": len(stored), "max_abs_difference": 0.,
            "bitwise_exact": True, "inactive_and_zero_kernel_parent_identity": True})
    _csv_check(run / "source_validation.csv", _summary(summary_panels, labels, active))
    station_rows = []
    for (model, k, station), group in product.groupby(["model_name", "k", "station"], sort=True):
        residual = group.y_true.to_numpy()-group.y_pred.to_numpy()
        station_rows.append({"model_name": model, "k": k, "station": station, "n": len(group),
            "n_active": int(group.aux_available.sum()), "mae": float(np.abs(residual).mean()),
            "parent_mae": float(np.abs(group.y_true-group.parent_y_pred).mean()), "mean_residual": float(residual.mean()),
            "mean_abs_kernel_correction": float(group.kernel_log_correction.abs().mean()), "split_seed": split_seed, "seed": seed})
    _csv_check(run / "station_responses.csv", pd.DataFrame(station_rows), station=True)
    return {"run": run.name, "status": "verified", "completion_sha256": sha256_file(run / "complete.json"),
        "runtime_snapshot_hash": runtime, "source_distance_inventories": 2, "all_candidate_scores": len(candidates),
        "station_candidate_scores": len(station_candidates), "selected_kernel_choices": 24,
        "preclip_remainder_checks": len(remainder_checks), "query_replays": query_checks,
        "source_validation_rows": len(summary_panels), "label_scope": "split.val only",
        "neural_forest_or_linear_calibrators_refitted": False, "normalized_rbf_formula_independent": True,
        "same_station_convex_bounds_verified": True, "parent_replay_path": str(parent_report),
        "parent_replay_sha256": sha256_file(parent_report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requested = [(s, r) for s in args.split_seeds for r in args.seeds]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Replay requires a nonempty unique run panel")
    runtime = verify_runtime_snapshot(args.root)
    results = []
    for split, seed in requested:
        print(f"Replaying source validation split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime))
        gc.collect()
    output = args.output or args.root / "verification/replay_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(s, r) for s in SPLITS for r in SEEDS},
        "verifier_sha256": sha256_file(__file__), "query_replay_rtol_atol": 0,
        "target_labels_extracted": False, "target_outcomes_evaluated": False,
        "new_model_or_linear_adapter_fits": 0, "shared_helpers_sha256": {name: sha256_file(name) for name in (
            "scripts/verify_doc_chemistry_support_v1.py", "scripts/verify_doc_daily_hydro_support_basis_v1.py")},
        "results": results}, indent=2)+"\n")
    print(f"Verified {len(results)} complete source-validation packages: {output}", flush=True)


if __name__ == "__main__":
    main()
