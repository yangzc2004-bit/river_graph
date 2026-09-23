"""Phase-2B/2C GNN executor (fail-closed).

Drives ``scripts/run_gnn.py``'s ``run()`` in-process for one arm over a
(seed, mask) grid. Before any training it verifies every frozen policy hash
(h2x policy, dataset, masks, run policy, runtime snapshot pin); after every
save it refuses results whose sidecar lacks the run-identity fields.

Usage:
    python scripts/run2b_executor.py --arm H2 --seeds 42 43 44 \
        --masks e2a_strict e2b_partial
    python scripts/run2b_executor.py --arm H2X_nomsg --seeds 42 \
        --masks e2b_partial --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot_sha256,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase2_ablation_stcore_v1.json"
POL2C_PATH = ROOT / "configs" / "phase2_2c_policy.json"
H2X_PATH = ROOT / "experiments" / "phase2_ablation_stcore_v1" / "h2x_policy.json"
RUNPOL_PATH = (
    ROOT / "experiments" / "phase2_ablation_stcore_v1" / "phase2_run_policy.json"
)
OUT_DIR = ROOT / "experiments" / "phase2_ablation_stcore_v1"
RECORDS = OUT_DIR / "batch_records"
# Each matrix generation pins its own runtime snapshot; the historical 2B pin
# stays at runtime_snapshot.pin. R3 (post-2B-R2 fixes) defaults to 2b-r3.
MATRIX_ID = os.environ.get("PHASE2_MATRIX_ID", "2b-r3")
SNAPSHOT_PIN = RECORDS / f"runtime_snapshot_{MATRIX_ID}.pin"

ARMS = {
    "H2": {"arch": "transport", "env_groups": ["hydro"], "edge_set": "river"},
    "H2E": {"arch": "transport", "env_groups": None, "edge_set": "river"},
    "H2X": {"arch": "transport_enc", "env_groups": None, "edge_set": "river"},
    "H2X_nomsg": {"arch": "transport_enc", "env_groups": None, "edge_set": "empty"},
}


class PolicyError(RuntimeError):
    """A frozen policy check failed; nothing may be trained."""


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise PolicyError(f"missing policy file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_check_policies() -> dict:
    """Fail-closed verification of every frozen hash and required key."""
    cfg, pol2c, h2x, runpol = map(
        _load_json, (CFG_PATH, POL2C_PATH, H2X_PATH, RUNPOL_PATH)
    )
    h2x_sha = sha256_file(H2X_PATH)
    if pol2c.get("h2x_policy_sha256") != h2x_sha:
        raise PolicyError("h2x_policy_sha256 mismatch in configs/phase2_2c_policy.json")
    if cfg.get("policy_binding", {}).get("h2x_policy_sha256") != h2x_sha:
        raise PolicyError("h2x_policy_sha256 mismatch in phase2_ablation_stcore_v1.json")
    for key in ("allow_nearest_distance_km_max", "hard_input_exclusions",
                "treatment_statement"):
        if key not in h2x:
            raise PolicyError(f"h2x_policy.json missing required key: {key}")
    if not h2x["hard_input_exclusions"]:
        raise PolicyError("h2x_policy.json has empty hard_input_exclusions")
    ds = ROOT / pol2c["dataset_path"]
    if not ds.is_file() or sha256_file(ds) != pol2c["dataset_sha256"]:
        raise PolicyError("dataset missing or dataset_sha256 mismatch")
    for name, want in pol2c["mask_sha256s"].items():
        mpath = ROOT / pol2c["masks_path"] / f"{name}.npz"
        if not mpath.is_file() or sha256_file(mpath) != want:
            raise PolicyError(f"mask missing or sha256 mismatch: {name}")
    for legacy in runpol["never_write"]:
        if str(OUT_DIR) in str((ROOT / legacy).resolve()):
            raise PolicyError(f"never_write inside output dir: {legacy}")
    return {"cfg": cfg, "pol2c": pol2c, "h2x": h2x, "runpol": runpol,
            "h2x_sha": h2x_sha}


def check_or_pin_runtime_snapshot() -> str:
    """Pin the runtime code snapshot for the whole matrix; refuse drift."""
    snap = runtime_code_snapshot_sha256()
    RECORDS.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT_PIN.exists():
        pinned = SNAPSHOT_PIN.read_text(encoding="utf-8").strip()
        if pinned != snap:
            raise PolicyError(
                "stale runtime snapshot: src/river_graph changed mid-matrix "
                f"(pinned {pinned[:12]}, current {snap[:12]}); finish or "
                "re-freeze the matrix deliberately instead of mixing code"
            )
    else:
        SNAPSHOT_PIN.write_text(snap + "\n", encoding="utf-8")
    return snap


def _run_gnn_module():
    spec = importlib.util.spec_from_file_location(
        "run_gnn_p2", ROOT / "scripts" / "run_gnn.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_gnn_p2"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_args(pol: dict, arm: str, seed: int, *, smoke: bool) -> argparse.Namespace:
    spec = ARMS[arm]
    prefix = f"P2A_SMOKE_{arm}" if smoke else f"P2X_{arm}"
    out = OUT_DIR / ("smoke" if smoke else ".")
    import argparse as _ap

    ns = _ap.Namespace(
        variants=["river"], only=None, variant=None,
        dataset=pol["pol2c"]["dataset_path"],
        masks_dir=pol["pol2c"]["masks_path"],
        model_name=prefix, tag=arm, arch=spec["arch"], seed=seed,
        lr=1e-3, wd=0.0, edge_dropout=0.0, share_weights=False,
        env_groups=spec["env_groups"], edge_set=spec["edge_set"],
        edge_direction="both",
        save_predictions=True, rebuild_predictions=False,
        force=False, repair_metrics=False,
        train_kw={"max_epochs": 5, "patience": 2} if smoke else {},
    )
    ns._results_dir = out / "results"
    ns._pred_dir = out / "predictions"
    return ns


def run_unit(pol: dict, arm: str, seed: int, mask: str, *, smoke: bool) -> dict:
    if arm not in ARMS:
        raise PolicyError(f"unknown GNN arm: {arm}")
    snap = check_or_pin_runtime_snapshot()
    ns = build_args(pol, arm, seed, smoke=smoke)
    ns.only = mask
    run_gnn = _run_gnn_module()
    mname = run_gnn.model_name_for(ns, "river")
    report = run_gnn.run(
        ns, results_dir=ns._results_dir,
        masks_dir=Path(ns.masks_dir), pred_dir=ns._pred_dir,
    )
    # fail closed on missing run identity
    sidecar = ns._pred_dir / f"{mname}__{mask}.meta.json"
    if not sidecar.is_file():
        raise PolicyError(f"no provenance sidecar written: {sidecar.name}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    for field in ("config_hash", "run_identity_sha256",
                  "runtime_code_snapshot_sha256"):
        if not meta.get(field):
            raise PolicyError(f"missing run identity field {field}: {sidecar.name}")
    if meta["runtime_code_snapshot_sha256"] != snap:
        raise PolicyError("sidecar runtime snapshot disagrees with the pinned one")
    recomputed = run_identity_sha256(
        meta["config_hash"], meta.get("run_started_at", ""),
        meta["runtime_code_snapshot_sha256"],
    )
    if recomputed != meta["run_identity_sha256"]:
        raise PolicyError(f"run identity does not recompute: {sidecar.name}")
    return {
        "arm": arm, "seed": seed, "mask": mask, "model_name": mname,
        "smoke": smoke, "trained": report.trained, "skipped": report.skipped,
        "metrics": meta.get("metrics"),
        "config_hash": meta["config_hash"],
        "run_identity_sha256": meta["run_identity_sha256"],
        "runtime_code_snapshot_sha256": meta["runtime_code_snapshot_sha256"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--masks", nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    pol = load_and_check_policies()
    units = []
    for seed in args.seeds:
        for mask in args.masks:
            rec = run_unit(pol, args.arm, seed, mask, smoke=args.smoke)
            units.append(rec)
            print(f"[ok] {rec['model_name']} @ {mask} "
                  f"identity={rec['run_identity_sha256'][:12]}", flush=True)
    print(json.dumps({"units": len(units)}, indent=2))


if __name__ == "__main__":
    main()
