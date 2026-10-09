"""T10: versioned run identity and artifact storage for the H3-A round.

Deliberately separate from the frozen gate-run identities: nothing here changes
a historical hash, and a new model family gets its own schema version.  A run
is a directory of hashed artifacts plus a manifest that is the commit marker.

Guarantees enforced here:

  * a different configuration can never overwrite an existing run record --
    the config hash is over every parameter that determines the numbers;
  * a record may only say "complete" when the checkpoint, the epoch log and the
    independent hidden-validation predictions all exist and hash correctly;
  * an existing record is reused only after it has actually been loaded and
    re-verified (audit_run recomputes the metrics from the stored files);
  * pilot and expand stages never compute or store an outer-test metric.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import KEY_SCENARIOS, load_mask
from river_graph.experiments.h3_training import (
    ARMS,
    H3Trainer,
    load_protocol,
    restrict_to_stations,
)
from river_graph.experiments.provenance import sha256_file

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[3]
DEV_MASKS = ("e1_r20_seed42", "e2b_partial", "e3_internal_seed42")
PILOT_SEEDS = (0, 1, 2)
EXPAND_SEEDS = (0, 1, 2, 3, 4)
SMOKE_STATIONS = 150
SMOKE_EPOCHS = 3
STAGES = ("smoke", "pilot", "expand", "frozen_eval")

SOURCE_FILES = (
    "configs/h3a_v1.json",
    "src/river_graph/models/h3.py",
    "src/river_graph/models/gcn.py",
    "src/river_graph/models/hydro.py",
    "src/river_graph/experiments/h3_masks.py",
    "src/river_graph/experiments/h3_training.py",
    "src/river_graph/experiments/h3_runs.py",
    "src/river_graph/experiments/evaluate.py",
    "scripts/build_h3_masks.py",
    "scripts/run_h3.py",
)

ARTIFACT_KEYS = ("validation", "checkpoint", "epochs")


def atomic_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path, value) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def tensor_digest(dataset: dict) -> str:
    """Content digest of the tensors a run actually reads."""
    digest = hashlib.sha256()
    for key in sorted(dataset):
        value = dataset[key]
        digest.update(key.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, list):
            digest.update("\u0000".join(str(v) for v in value).encode("utf-8"))
        else:
            digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def dataset_scope(dataset: dict, dataset_path: str, stage: str) -> dict:
    """Identity of the exact data a run reads (full file, or a smoke subset)."""
    if stage == "smoke":
        subset, _ = restrict_to_stations(
            dataset, {"train": np.array([0], dtype=np.int64)}, SMOKE_STATIONS
        )
        return {
            "dataset_path": normalized_path(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "subset_stations": min(SMOKE_STATIONS, int(dataset["y"].shape[0])),
            "subset_sha256": tensor_digest(subset),
            "subset_rule": "first N stations, cells outside the prefix dropped",
        }
    return {
        "dataset_path": normalized_path(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "subset_stations": None,
        "subset_sha256": None,
        "subset_rule": None,
    }


def current_scope(config: dict) -> dict:
    """Recompute the dataset identity of a stored record from disk, right now.

    Used by the verification and frozen-evaluation paths: they must check the
    identity the record SHOULD have given the current sources, protocol, data
    and masks, not simply trust the record's own hash.
    """
    stored = config["dataset"]
    path = stored["dataset_path"]
    dataset = torch.load(path, weights_only=False)
    stage = "smoke" if stored.get("subset_stations") else "pilot"
    return dataset_scope(dataset, path, stage)


def identity_mismatches(stored: dict, expected: dict) -> list[str]:
    """Top-level differences between two identity payloads."""
    problems = []
    for key in sorted(set(stored) | set(expected)):
        if stored.get(key) != expected.get(key):
            if key == "source_sha256":
                old = stored.get(key) or {}
                new = expected.get(key) or {}
                for name in sorted(set(old) | set(new)):
                    if old.get(name) != new.get(name):
                        problems.append(
                            "source_sha256[" + name + "]: stored="
                            + str(old.get(name))[:16] + " current="
                            + str(new.get(name))[:16]
                        )
            else:
                problems.append(
                    key + ": stored=" + repr(stored.get(key))
                    + " current=" + repr(expected.get(key))
                )
    return problems


def scenario_of(mask: str) -> str:
    if mask.startswith("e1_"):
        return "E1"
    if mask.startswith("e2a_"):
        return "E2a"
    if mask.startswith("e2b_"):
        return "E2b"
    if mask.startswith("e3_"):
        return "E3"
    raise ValueError("cannot classify mask: " + str(mask))


def name_for(arm: str, protocol_version: str) -> str:
    return "H3_" + arm.upper() + "_" + protocol_version


def run_stem(arm: str, seed: int, mask: str, protocol_version: str) -> str:
    return name_for(arm, protocol_version) + "_s" + str(seed) + "__" + mask


def stem_for(
    arm: str, seed: int, mask: str, protocol_version: str, subset_stations=None
) -> str:
    """Run stem; a station-subset stage gets its own namespace."""
    stem = run_stem(arm, seed, mask, protocol_version)
    if subset_stations:
        stem += "__sub" + str(int(subset_stations))
    return stem


def dependency_versions() -> dict:
    versions = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def identity(
    arm: str,
    seed: int,
    mask: str,
    protocol: dict,
    scope: dict,
    masks_dir: Path,
    cpu_threads: int = 1,
    edge_set: str = "river",
) -> tuple[dict, str]:
    """Full configuration identity of one (arm, seed, mask) run."""
    init = protocol["init"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": protocol["protocol_version"],
        "model": name_for(arm, protocol["protocol_version"]),
        "arm": arm,
        "mask": mask,
        "scenario": scenario_of(mask),
        "seed": int(seed),
        "model_params": {
            "arms": protocol["arms"],
            "edge_set": edge_set,
        },
        "resolved_init_seeds": {
            "base": int(seed) + int(init["base_seed_offset"]),
            "correction": int(seed) + int(init["correction_seed_offset"]),
            "h2x": int(seed) + int(init["h2x_seed_offset"]),
        },
        "training": protocol["training"],
        "prediction": protocol["prediction"],
        "metrics_definition": protocol["metrics"]["definition_source"],
        "dataset": scope,
        "mask_path": normalized_path(Path(masks_dir) / (mask + ".npz")),
        "mask_sha256": sha256_file(Path(masks_dir) / (mask + ".npz")),
        "source_sha256": {p: sha256_file(ROOT / p) for p in SOURCE_FILES},
        "dependencies": dependency_versions(),
        "python": platform.python_version(),
        "cpu_threads": int(cpu_threads),
        "outer_test_policy": "hidden in pilot/expand; only the frozen evaluation entry may read it",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload, digest


def normalized_path(path) -> str:
    """Repository-relative path when possible, so identities are portable.

    Storing an absolute path would make the same run hash differently on
    another checkout, which is exactly the kind of drift the identity is
    supposed to detect.
    """
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def payload_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def effective_protocol(stage: str, protocol: dict) -> dict:
    """The protocol actually used for a stage (smoke shortens the budget)."""
    if stage != "smoke":
        return protocol
    effective = json.loads(json.dumps(protocol))
    effective["training"]["max_epochs"] = min(
        effective["training"]["max_epochs"], SMOKE_EPOCHS
    )
    effective["training"]["patience"] = min(
        effective["training"]["patience"], SMOKE_EPOCHS
    )
    return effective


def expected_config_for(record: dict, protocol: dict, masks_dir: Path):
    """Recompute the identity a stored record SHOULD have, from the current
    sources, protocol, dataset content and mask content.

    Verification and export use this instead of the record's own hash, so a
    changed source file, a changed protocol, a changed dataset or a changed
    mask is a hard failure rather than a silent pass.
    """
    config = record["config"]
    scope = current_scope(config)
    stage = "smoke" if scope.get("subset_stations") else "pilot"
    effective = effective_protocol(stage, protocol)
    expected, digest = identity(
        config["arm"],
        config["seed"],
        config["mask"],
        effective,
        scope,
        Path(masks_dir),
        config.get("cpu_threads", 1),
        config.get("model_params", {}).get("edge_set", "river"),
    )
    return expected, digest


def build_split(arm: str, seed: int, stage: str, protocol: dict, mask: str,
                masks_dir: Path, dataset_path: str, cpu_threads: int = 1,
                edge_set: str = "river"):
    """Everything one run needs, including the exact split it will read."""
    dataset = torch.load(dataset_path, weights_only=False)
    split = load_mask(Path(masks_dir) / (mask + ".npz"))
    if stage == "smoke":
        dataset, split = restrict_to_stations(dataset, split, SMOKE_STATIONS)
    scope = dataset_scope(dataset, dataset_path, stage)
    # The identity must describe the protocol that will ACTUALLY be used, so a
    # shortened smoke budget appears in the stored configuration instead of the
    # full budget the run never had.
    config, config_hash = identity(
        arm, seed, mask, effective_protocol(stage, protocol), scope,
        Path(masks_dir), cpu_threads, edge_set,
    )
    return dataset, split, config, config_hash


def train_and_store(
    arm: str,
    seed: int,
    stage: str,
    protocol: dict,
    mask: str,
    masks_dir: Path,
    dataset_path: str,
    root: Path,
    cpu_threads: int = 1,
    edge_set: str = "river",
    dry_run: bool = False,
    audit_only: bool = False,
    report=None,
) -> dict:
    """Train one configuration (or verify the cached one) and return its record."""
    root = Path(root)
    dataset, split, config, config_hash = build_split(
        arm, seed, stage, protocol, mask, masks_dir, dataset_path, cpu_threads,
        edge_set,
    )
    if stage == "smoke":
        protocol = effective_protocol(stage, protocol)

    stem = stem_for(
        arm, seed, mask, protocol["protocol_version"],
        config["dataset"]["subset_stations"],
    )
    manifest = root / "runs" / (stem + ".json")
    intent = root / "runs" / (stem + ".intent.json")

    if manifest.exists():
        record = audit_run(root, manifest, dataset, split, config_hash,
                           expected_config=config)
        if report is not None:
            report.skipped += 1
        print(stem + ": verified cached", flush=True)
        return record
    if audit_only:
        raise ValueError(stem + ": incomplete or missing run")
    if intent.exists():
        previous = json.loads(intent.read_text(encoding="utf-8"))["config_hash"]
        if previous != config_hash:
            raise ValueError(stem + ": incomplete run has a different configuration")
    if dry_run:
        if report is not None:
            report.pending += 1
        print(
            stem + ": pending config=" + config_hash[:12] + " scenario="
            + config["scenario"],
            flush=True,
        )
        return {"status": "pending", "config": config, "config_hash": config_hash}

    atomic_json(intent, {"config": config, "config_hash": config_hash})
    print(stem + ": training", flush=True)
    torch.set_num_threads(max(1, int(cpu_threads)))
    trainer = H3Trainer(arm, seed, protocol)
    trainer.fit(dataset, split)
    if trainer.validation_ is None:
        raise RuntimeError(stem + ": training produced no validation output")

    paths = write_artifacts(root, stem, trainer, dataset, split, config, config_hash)
    val = trainer.validation_
    vmetrics = finite_metrics(val["y_true"], val["y_pred"])
    raw = trainer.raw_log_metrics()
    record = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "config_hash": config_hash,
        "config": config,
        "training": trainer.training_info_,
        "validation_metrics": vmetrics,
        "raw_log_metrics": raw,
        "test_metrics": None,
        "test_metrics_note": "pilot/expand never compute an outer-test metric",
        "validation_visibility": config["training"]["visible_roles"],
        "completed_at": time.time(),
        "artifacts": {
            key: {"path": path.relative_to(root).as_posix(),
                  "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }
    pending = root / "runs" / (stem + ".pending.json")
    atomic_json(pending, record)
    audit_run(root, pending, dataset, split, config_hash, expected_config=config)
    os.replace(pending, manifest)
    if report is not None:
        report.trained += 1
    print(
        stem + ": complete epochs=" + str(trainer.training_info_["epochs"])
        + " best=" + str(trainer.training_info_["best_epoch"])
        + " seconds=" + format(trainer.training_info_["elapsed_seconds"], ".1f")
        + " val_log_rmse=" + format(vmetrics["log_rmse"], ".5f"),
        flush=True,
    )
    return record


def write_artifacts(root, stem, trainer, dataset, split, config, config_hash) -> dict:
    """Store validation predictions, the best checkpoint and the epoch log."""
    root = Path(root)
    val = trainer.validation_
    ntime = dataset["y"].shape[1]
    cells = val["cells"]
    frame = pd.DataFrame(
        {
            "cell": cells,
            "station": [dataset["site_no"][int(c) // ntime] for c in cells],
            "month": [str(dataset["months"][int(c) % ntime]) for c in cells],
            "role": "val",
            "y_true": val["y_true"],
            "y_true_log": val["y_true_log"],
            "base_log": val["base_log"],
            "correction_log": val["correction_log"],
            "total_log": val["total_log"],
            "pred_log_clipped": val["pred_log_clipped"],
            "y_pred": val["y_pred"],
            "clipped": val["clipped"],
        }
    )
    frame.attrs["clip_log_bounds"] = val["clip_log_bounds"]

    validation_path = root / "validation" / (stem + ".parquet")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    temp = validation_path.with_name(validation_path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, validation_path)

    checkpoint_path = root / "checkpoints" / (stem + ".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        {
            "state_dict": trainer.best_state_,
            "config": config,
            "config_hash": config_hash,
            "training": trainer.training_info_,
            "clip_log_bounds": val["clip_log_bounds"],
        },
        temp,
    )
    os.replace(temp, checkpoint_path)

    epoch_path = root / "epochs" / (stem + ".jsonl")
    atomic_text(
        epoch_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in trainer.epoch_log_),
    )
    return {
        "validation": validation_path,
        "checkpoint": checkpoint_path,
        "epochs": epoch_path,
    }


def finite_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("empty or nonfinite validation coverage")
    result = metrics(y_true, y_pred)
    if result["n"] != len(y_true) or not all(
        np.isfinite(v) for v in result.values()
    ):
        raise ValueError("nonfinite metrics or incomplete coverage")
    return result


def check_metrics(saved: dict, actual: dict, tolerance: float = 1e-6) -> None:
    if saved.keys() != actual.keys():
        raise ValueError("stored metric keys disagree with the recomputation")
    for key, value in saved.items():
        if not np.isfinite(value) or abs(value - actual[key]) > tolerance:
            raise ValueError("stored metric " + key + " disagrees with the records")


def audit_run(
    root,
    manifest_path,
    dataset,
    split,
    expected_hash=None,
    expected_config=None,
    strict_identity: bool = False,
) -> dict:
    """Re-derive everything a record claims, from the files on disk."""
    root = Path(root)
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload_hash(record["config"]) != record["config_hash"]:
        raise ValueError("manifest configuration was altered")
    if expected_hash is not None and record["config_hash"] != expected_hash:
        # No compatibility escape hatch exists here on purpose: a different
        # configuration, a different seed or a changed source file means a
        # different run and it must not silently reuse these artifacts.
        raise ValueError(
            "configuration identity conflict (stored "
            + str(record["config_hash"])[:12] + ", current "
            + str(expected_hash)[:12] + ")"
        )
    if strict_identity:
        # The training path can pass the record's own hash back in, which only
        # proves internal consistency.  Verification and export must instead
        # pass a freshly recomputed identity and have EVERY field compared,
        # including the source files, the frozen protocol, the dataset content
        # hash and the mask content hash.
        if expected_config is None:
            raise ValueError("strict identity checking needs the expected config")
        problems = identity_mismatches(record["config"], expected_config)
        if problems:
            raise ValueError(
                "stored run identity disagrees with the current sources, "
                "protocol, data or masks:\n  " + "\n  ".join(problems)
            )
    if record.get("status") != "complete":
        raise ValueError("run has no completion marker")
    if record.get("test_metrics") is not None:
        raise ValueError("a non-frozen stage stored an outer-test metric")
    if set(record["artifacts"]) != set(ARTIFACT_KEYS):
        raise ValueError("missing required artifact entry")
    for item in record["artifacts"].values():
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError("missing or corrupt artifact: " + str(path))

    val = pd.read_parquet(root / record["artifacts"]["validation"]["path"])
    if not np.array_equal(val["cell"].to_numpy(), split["val"]):
        raise ValueError("validation cell identities mismatch")
    truth = dataset["y"].numpy().ravel()
    if not np.array_equal(val.y_true.to_numpy(), truth[split["val"]]):
        raise ValueError("validation labels mismatch")
    ntime = dataset["y"].shape[1]
    expected_sites = [dataset["site_no"][int(c) // ntime] for c in split["val"]]
    expected_months = [str(dataset["months"][int(c) % ntime]) for c in split["val"]]
    if (
        val.station.tolist() != expected_sites
        or val.month.astype(str).tolist() != expected_months
    ):
        raise ValueError("validation station/month mismatch")
    if set(val.role.unique()) != {"val"}:
        raise ValueError("validation file must not carry any other role")
    hidden = set(split.get("test", np.empty(0, dtype=np.int64)).tolist())
    if hidden & set(val["cell"].tolist()):
        raise ValueError("an outer-test cell reached the validation file")

    check_metrics(record["validation_metrics"], finite_metrics(val.y_true, val.y_pred))
    raw = float(np.mean((val.y_true_log.to_numpy() - val.total_log.to_numpy()) ** 2))
    if abs(raw - record["training"]["best_val_mse_raw"]) > 1e-6:
        raise ValueError("validation raw loss does not match the selected checkpoint")
    stored_raw = record["raw_log_metrics"]
    residual = val.y_true_log.to_numpy() - val.total_log.to_numpy()
    if abs(float(np.sqrt(np.mean(residual**2))) - stored_raw["raw_log_rmse"]) > 1e-6:
        raise ValueError("stored raw log RMSE disagrees with the records")
    np.testing.assert_allclose(
        val.y_pred.to_numpy(), np.expm1(val.pred_log_clipped.to_numpy()),
        rtol=1e-12, atol=0,
    )
    lo, hi = val.pred_log_clipped.min(), val.pred_log_clipped.max()
    bounds = record["training"]["clip_log_bounds"]
    if lo < bounds[0] - 1e-6 or hi > bounds[1] + 1e-6:
        raise ValueError("stored prediction escapes the frozen clip bounds")

    epochs = [
        json.loads(line)
        for line in (root / record["artifacts"]["epochs"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(epochs) != record["training"]["epochs"]:
        raise ValueError("epoch log length disagrees with the record")
    best = [row["epoch"] for row in epochs if row["is_best"]]
    if not best or best[-1] != record["training"]["best_epoch"]:
        raise ValueError("best epoch is not the last improving epoch in the log")
    return record


NO_MESSAGE_ARM = "h3a_no_message"


def scenarios_for_stage(stage: str, protocol: dict) -> tuple:
    if stage == "smoke":
        return (protocol["masks"]["dev_scenarios"][1],)
    if stage in ("expand", "no_message"):
        return tuple(KEY_SCENARIOS)
    return tuple(protocol["masks"]["dev_scenarios"])


def seeds_for_stage(stage: str, protocol: dict) -> tuple:
    if stage == "smoke":
        return (0,)
    if stage in ("expand", "no_message"):
        return tuple(protocol["seeds"]["expand"])
    return tuple(protocol["seeds"]["pilot"])


def default_arms_for_stage(stage: str) -> tuple:
    """The no-message control is its own arm, never part of the three-arm grid."""
    return (NO_MESSAGE_ARM,) if stage == "no_message" else tuple(ARMS[:3])


def task_grid(stage: str, protocol: dict, arms=None, seeds=None, masks=None) -> list:
    """Exact list of (arm, seed, mask) configurations a stage will run.

    The T18 no-message control is 8 key scenarios x 5 seeds = 40 runs and is
    defined here on its own, so a completeness check can compare it against
    its own size instead of against the 120-run three-arm grid.
    """
    arms = tuple(arms) if arms else default_arms_for_stage(stage)
    for arm in arms:
        if arm not in ARMS:
            raise ValueError("unknown arm: " + str(arm))
    seeds = tuple(seeds) if seeds else seeds_for_stage(stage, protocol)
    masks = tuple(masks) if masks else scenarios_for_stage(stage, protocol)
    for mask in masks:
        if mask not in KEY_SCENARIOS:
            raise ValueError("unknown mask: " + str(mask))
    return [(arm, int(seed), mask) for mask in masks for seed in seeds for arm in arms]


def artifact_entries(records: list) -> list[dict]:
    """Identity of an exact run set: config hash and checkpoint hash per run."""
    entries = []
    for record, manifest in sorted(
        records, key=lambda item: Path(item[1]).stem
    ):
        root = Path(manifest).parent.parent
        entries.append(
            {
                "stem": Path(manifest).stem,
                "arm": record["config"]["arm"],
                "seed": record["config"]["seed"],
                "mask": record["config"]["mask"],
                "config_hash": record["config_hash"],
                "checkpoint_sha256": sha256_file(
                    root / record["artifacts"]["checkpoint"]["path"]
                ),
            }
        )
    return entries


def artifact_manifest_hash(entries: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


DECISION_FILES = {
    "pilot": "pilot_decision.json",
    "expand": "expansion_decision.json",
    "no_message": "no_message_decision.json",
}


def require_decision(root: Path, kind: str, action: str) -> dict:
    """Refuse an action until the decision that licenses it says promote.

    T17 (expand) needs the T16 pilot decision; T18 and the frozen test export
    need the T17 expansion decision.  The gate is code, not documentation.
    """
    path = Path(root) / DECISION_FILES[kind]
    if not path.is_file():
        raise SystemExit(
            "refusing to " + action + ": " + str(path) + " does not exist"
        )
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("outcome") != "promote":
        raise SystemExit(
            "refusing to " + action + ": the " + kind + " decision outcome is "
            + repr(decision.get("outcome")) + ", not 'promote'"
        )
    return decision


def load_default_protocol() -> dict:
    return load_protocol(ROOT / "configs/h3a_v1.json")
