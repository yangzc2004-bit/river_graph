"""Phase-2B batch planner / driver (GNN via run2b_executor, tabular in-process).

Batch matrix (2B pilot: 6 arms x 3 seeds x 8 key masks = 144 configs):
    batch 1: 4 GNN arms x 3 seeds x e1_r20_seed{42,43,44}        (36 units)
    batch 2: 4 GNN arms x 3 seeds x {e2a_strict, e2b_partial}    (24 units)
    batch 3: 4 GNN arms x 3 seeds x e3_spatial_seed{42,43}       (24 units)
    batch 4: 4 GNN arms x 3 seeds x e3_spatial_seed44            (12 units)
             + eco_RF/eco_MLP x 3 seeds x 8 masks                (48 fits)
    batch 5: resume — run exactly the missing (arm, seed, mask) cells

Usage:
    python scripts/run2b_dispatch.py --count
    python scripts/run2b_dispatch.py --batch 1
    python scripts/run2b_dispatch.py --smoke-all
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from run2b_executor import (  # type: ignore  # same-dir script import
    ARMS,
    OUT_DIR,
    load_and_check_policies,
    run_unit,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = OUT_DIR / "batch_records"
SEEDS = [42, 43, 44]
MASKS = [
    "e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44",
    "e2a_strict", "e2b_partial",
    "e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44",
]
BATCHES = {
    1: ["e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44"],
    2: ["e2a_strict", "e2b_partial"],
    3: ["e3_spatial_seed42", "e3_spatial_seed43"],
    4: ["e3_spatial_seed44"],
}


def gnn_units(masks: list[str], seeds: list[int] | None = None):
    seeds = seeds or SEEDS
    for arm in ARMS:
        for seed in seeds:
            for mask in masks:
                yield arm, seed, mask


def expected_sidecar(arm: str, seed: int, mask: str, smoke: bool = False) -> Path:
    prefix = f"P2A_SMOKE_{arm}" if smoke else f"P2X_{arm}"
    sub = "smoke" if smoke else "."
    return (OUT_DIR / sub / "predictions" /
            f"{prefix}_river_s{seed}__{mask}.meta.json")


def sidecar_is_valid(meta_path: Path, pol: dict) -> bool:
    """Cache-hit validation: a skip requires a verifiable artifact.

    File existence alone is not cache validity. A skippable unit must have its
    parquet + sidecar pair, a config hash that recomputes under the sidecar's
    own schema, a run identity that re-derives from its stored fields,
    dataset/mask content hashes matching the frozen policy, and finite metrics.
    """
    from river_graph.experiments.provenance import (
        config_hash as _config_hash,
    )
    from river_graph.experiments.provenance import (
        run_identity_sha256 as _run_id,
    )

    parquet = Path(str(meta_path).replace(".meta.json", ".parquet"))
    if not meta_path.is_file() or not parquet.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    cfg = meta.get("config") or {}
    version = int(meta.get("config_hash_version") or 1)
    if _config_hash(cfg, version=version) != meta.get("config_hash"):
        return False
    if _run_id(meta.get("config_hash") or "",
               meta.get("run_started_at") or "",
               meta.get("runtime_code_snapshot_sha256") or "",
               ) != meta.get("run_identity_sha256"):
        return False
    if (meta.get("dataset") or {}).get("sha256") != pol["pol2c"]["dataset_sha256"]:
        return False
    want_mask = pol["pol2c"]["mask_sha256s"].get(str(meta.get("mask_name")))
    if not want_mask or (meta.get("mask") or {}).get("sha256") != want_mask:
        return False
    mae = (meta.get("metrics") or {}).get("mae")
    return mae is not None and np.isfinite(mae)


def cached_ok(meta_path: Path, pol: dict) -> bool:
    return meta_path.is_file() and sidecar_is_valid(meta_path, pol)


def tabular_units(seeds: list[int] | None = None):
    seeds = seeds or SEEDS
    for model in ("eco_RF", "eco_MLP"):
        for seed in seeds:
            for mask in MASKS:
                yield model, seed, mask


def run_tabular(pol: dict, model: str, seed: int, mask: str,
                *, max_epochs: int = 300, smoke: bool = False) -> dict:
    """Fit one eco RF/MLP configuration and store predictions + provenance."""
    import torch

    from river_graph.baselines.baselines import EcoMLP, EcoRandomForest
    from river_graph.experiments.evaluate import load_mask, metrics
    from river_graph.experiments.predictions import save_predictions
    from river_graph.experiments.provenance import (
        build_meta,
        run_identity_sha256,
        runtime_code_snapshot_sha256,
    )

    out_root = OUT_DIR / ("smoke" if smoke else ".")
    ds = torch.load(pol["pol2c"]["dataset_path"], weights_only=False)
    split = load_mask(mask, Path(pol["pol2c"]["masks_path"]))
    if model == "eco_RF":
        clf = EcoRandomForest(n_estimators=200, seed=seed)
        arch = "tabular_rf"
    else:
        clf = EcoMLP(hidden=(128, 64), max_epochs=max_epochs, patience=20, seed=seed)
        arch = "tabular_mlp"
    started = datetime.now(timezone.utc).isoformat()
    snap = runtime_code_snapshot_sha256()
    pred = clf.fit_predict(ds, split)
    y = ds["y"].numpy()
    m = metrics(y.ravel()[split["test"]], np.asarray(pred).ravel()[split["test"]])
    prefix = f"P2A_SMOKE_{model}" if smoke else f"P2X_{model}"
    mname = f"{prefix}_s{seed}"
    params = {
        "script": "scripts/run2b_dispatch.py", "model_name": mname, "tag": model,
        "architecture": arch, "variant": "none", "seed": seed,
        "lr": 0.0, "weight_decay": 0.0, "edge_dropout": 0.0,
        "share_weights": False, "env_groups": None, "env_encoder": False,
        "edge_set": "empty", "edge_direction": "both",
        "hidden": 128 if model == "eco_MLP" else 0, "layers": 2, "dropout": 0.0,
        "max_epochs": 0 if model == "eco_RF" else max_epochs,
        "patience": 0 if model == "eco_RF" else 20,
        "env_emb": 0,
    }
    meta = build_meta(
        model_name=mname, mask_name=mask,
        dataset_path=pol["pol2c"]["dataset_path"], split=split, params=params,
        results_path=out_root / "results" / f"{mname}.json",
        masks_dir=pol["pol2c"]["masks_path"], caller="scripts/run2b_dispatch.py",
    )
    meta["metrics"] = m
    meta["run_started_at"] = started
    meta["runtime_code_snapshot_sha256"] = snap
    meta["run_identity_sha256"] = run_identity_sha256(
        meta["config_hash"], started, snap
    )
    meta["train_budget"] = {"max_epochs": max_epochs, "patience": 20, "smoke": smoke}
    meta["early_stop"] = getattr(clf, "early_stop_", None)
    meta["feature_names"] = getattr(clf, "feature_names_", None)
    out, _mp = save_predictions(
        pred, ds, split, mname, mask, Path(pol["pol2c"]["dataset_path"]).stem,
        out_dir=out_root / "predictions", meta=meta,
    )
    jpath = out_root / "results" / f"{mname}.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
    merged[mask] = m
    jpath.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return {
        "arm": model, "seed": seed, "mask": mask, "model_name": mname,
        "smoke": smoke,
        "metrics": m, "config_hash": meta["config_hash"],
        "run_identity_sha256": meta["run_identity_sha256"],
        "runtime_code_snapshot_sha256": snap,
        "prediction_file": out.name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def write_record(batch: str, units: list[dict], planned: int,
                 started_at: str, extra: dict | None = None) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"batch_{batch}.json"
    payload = {
        "batch": batch, "planned_units": planned, "executed_units": len(units),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "units": units,
        "note": "Directional pilot only — never a paper claim (spec §7).",
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"batch record -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--smoke-all", action="store_true")
    ap.add_argument("--tabular-only", action="store_true",
                    help="2B-R3: run only the eco RF/MLP arms (staged "
                         "visibility), keeping cached-and-valid units")
    ap.add_argument("--ensemble-readiness", action="store_true",
                    help="Phase 3 uncertainty ensemble completion "
                         "(H2X, H2X_nomsg, eco_RF x seeds 45,46 x 8 masks); "
                         "not a post hoc Phase 2 claim test")
    args = ap.parse_args()
    pol = load_and_check_policies()

    if args.count:
        for n, masks in BATCHES.items():
            print(f"batch {n}: {4 * len(SEEDS) * len(masks)} GNN units")
        print(f"batch 4 tabular: {2 * len(SEEDS) * len(MASKS)} fits")
        print(f"batch 5 resume: fills whatever is missing of "
              f"{4 * len(SEEDS) * len(MASKS)} GNN + {2 * len(SEEDS) * len(MASKS)} tabular")
        return

    if args.smoke_all:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        units = []
        for arm in ARMS:
            for mask in ("e2b_partial", "e3_spatial_seed42"):
                units.append(run_unit(pol, arm, 42, mask, smoke=True))
        for model in ("eco_RF", "eco_MLP"):
            rec = run_tabular(pol, model, 42, "e2b_partial", max_epochs=15,
                              smoke=True)
            rec["smoke"] = True
            units.append(rec)
        # chain checks: identity fields present, hashes distinct, finite metrics
        hashes = [u["config_hash"] for u in units]
        assert len(set(hashes)) == len(hashes), "config hashes collide"
        for u in units:
            assert u["run_identity_sha256"] and u["runtime_code_snapshot_sha256"]
            assert np.isfinite(
                float((u.get("metrics") or {}).get("mae", float("nan")))
            )
        report = {
            "version": "phase2b_smoke_v1", "started_at": started,
            "duration_sec": round(time.time() - t0, 1),
            "units": units, "checks": {
                "config_hashes_distinct": True,
                "run_identity_fields_present": True,
                "metrics_finite": True,
                "claims_allowed": False,
            },
        }
        path = OUT_DIR / "smoke_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"smoke report -> {path}")
        return

    if args.ensemble_readiness:
        # Phase 3 uncertainty ensemble completion (frozen route decision):
        # "not a post hoc Phase 2 claim test."
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        units: list[dict] = []
        en_gnn = ["H2X", "H2X_nomsg"]
        en_tab = ["eco_RF"]
        en_seeds = [45, 46]
        planned = (len(en_gnn) + len(en_tab)) * len(en_seeds) * len(MASKS)
        for arm in en_gnn:
            for seed in en_seeds:
                for mask in MASKS:
                    if cached_ok(expected_sidecar(arm, seed, mask), pol):
                        print(f"[cached] {arm} s{seed} {mask}", flush=True)
                        continue
                    units.append(run_unit(pol, arm, seed, mask, smoke=False))
                    print(f"[ok] {arm} s{seed} {mask}", flush=True)
        for model in en_tab:
            for seed in en_seeds:
                for mask in MASKS:
                    meta_p = (OUT_DIR / "predictions" /
                              f"P2X_{model}_s{seed}__{mask}.meta.json")
                    if cached_ok(meta_p, pol):
                        print(f"[cached] {model} s{seed} {mask}", flush=True)
                        continue
                    units.append(run_tabular(pol, model, seed, mask))
                    print(f"[ok] {model} s{seed} {mask}", flush=True)
        write_record("P3_ensemble", units, planned, started, {
            "duration_sec": round(time.time() - t0, 1),
            "role": ("Phase 3 uncertainty ensemble completion; "
                     "not a post hoc Phase 2 claim test."),
            "route_decision": ("experiments/phase2_ablation_stcore_v1/"
                               "frozen/phase2_route_decision.md"),
        })
        return

    if args.tabular_only:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        units = []
        for model, seed, mask in tabular_units():
            meta_p = (OUT_DIR / "predictions" /
                      f"P2X_{model}_s{seed}__{mask}.meta.json")
            if cached_ok(meta_p, pol):
                print(f"[cached] {model} s{seed} {mask}", flush=True)
                continue
            units.append(run_tabular(pol, model, seed, mask))
            print(f"[ok] {model} s{seed} {mask}", flush=True)
        write_record("R3_tabular", units,
                     2 * len(SEEDS) * len(MASKS), started,
                     {"duration_sec": round(time.time() - t0, 1)})
        return

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    units: list[dict] = []
    if args.batch in (1, 2, 3, 4):
        masks = BATCHES[args.batch]
        planned = 4 * len(SEEDS) * len(masks)
        for arm, seed, mask in gnn_units(masks):
            if cached_ok(expected_sidecar(arm, seed, mask), pol):
                print(f"[cached] {arm} s{seed} {mask}", flush=True)
                continue
            units.append(run_unit(pol, arm, seed, mask, smoke=False))
            print(f"[ok] {arm} s{seed} {mask}", flush=True)
        if args.batch == 4:
            planned += 2 * len(SEEDS) * len(MASKS)
            for model, seed, mask in tabular_units():
                meta_p = (OUT_DIR / "predictions" /
                          f"P2X_{model}_s{seed}__{mask}.meta.json")
                if cached_ok(meta_p, pol):
                    print(f"[cached] {model} s{seed} {mask}", flush=True)
                    continue
                units.append(run_tabular(pol, model, seed, mask))
                print(f"[ok] {model} s{seed} {mask}", flush=True)
        write_record(str(args.batch), units, planned, started,
                     {"duration_sec": round(time.time() - t0, 1)})
        return

    # batch 5: resume exactly the missing or invalid cells of the 2B matrix
    planned = 4 * len(SEEDS) * len(MASKS) + 2 * len(SEEDS) * len(MASKS)
    for arm, seed, mask in gnn_units(MASKS):
        if cached_ok(expected_sidecar(arm, seed, mask), pol):
            continue
        units.append(run_unit(pol, arm, seed, mask, smoke=False))
        print(f"[ok] {arm} s{seed} {mask}", flush=True)
    for model, seed, mask in tabular_units():
        meta_p = (OUT_DIR / "predictions" /
                  f"P2X_{model}_s{seed}__{mask}.meta.json")
        if cached_ok(meta_p, pol):
            continue
        units.append(run_tabular(pol, model, seed, mask))
        print(f"[ok] {model} s{seed} {mask}", flush=True)
    write_record("5", units, planned, started,
                 {"duration_sec": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
