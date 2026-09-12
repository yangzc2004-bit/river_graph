"""Provenance records for stored predictions.

The frozen benchmark can only be trustworthy if each stored prediction can
name the exact inputs that produced it. This module builds that record: a
config hash over every parameter that determines the numbers, plus file
identities (sha256) for the dataset and the mask.

The hash deliberately covers the training seed, the dataset path *and* its
content hash, and the mask name *and* its content hash, so that
"same name, different inputs" is detectable instead of silently overwriting a
historical result.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fields that determine the numbers produced by a training run. Missing
# entries are recorded as null rather than omitted, so the hash is stable
# across runs that share a configuration.
#
# ``dataset_sha256`` and ``mask_sha256`` are CONTENT hashes of the exact inputs.
# Including them is what makes a modified dataset or a regenerated mask a
# different configuration: comparing only paths and names would let a run reuse
# results that were computed on other data.
CONFIG_FIELDS = (
    "script",
    "model_name",
    "tag",
    "architecture",
    "variant",
    "seed",
    "lr",
    "weight_decay",
    "edge_dropout",
    "share_weights",
    "env_groups",
    "env_encoder",
    "dataset_path",
    "dataset_sha256",
    "mask_path",
    "mask_sha256",
)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a file (streamed, so large datasets are fine)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    """Path + size + mtime + content hash for one file."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(p),
    }


def config_hash(params: dict[str, Any]) -> str:
    """Stable hash over the parameters that determine the produced numbers."""
    payload = {key: params.get(key) for key in CONFIG_FIELDS}
    if isinstance(payload.get("env_groups"), list):
        payload["env_groups"] = sorted(payload["env_groups"])
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_meta(
    *,
    model_name: str,
    mask_name: str,
    dataset_path: str | Path,
    split: dict[str, Any],
    params: dict[str, Any],
    results_path: str | Path | None = None,
    masks_dir: str | Path = "experiments/masks",
    caller: str | None = None,
) -> dict[str, Any]:
    """Assemble the sidecar record for one (model, mask) prediction.

    ``params`` carries the training configuration; the config hash is computed
    here so callers cannot forget it.
    """
    mask_file = Path(masks_dir) / f"{mask_name}.npz"
    dataset_identity = file_identity(dataset_path)
    mask_identity = file_identity(mask_file)
    # The persisted config must carry the SAME identity fields that the hash is
    # computed over, otherwise a later run cannot tell that its inputs changed.
    full_config = config_payload({**params, "model_name": model_name},
                                 dataset_identity, mask_identity)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "mask_name": mask_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "caller": caller or params.get("script"),
        "python": platform.python_version(),
        "config": {key: full_config.get(key) for key in CONFIG_FIELDS},
        "dataset": dataset_identity,
        "mask": mask_identity,
        "results_path": str(results_path) if results_path else None,
        "split_sizes": {k: len(v) for k, v in split.items()
                        if hasattr(v, "__len__")},
        "export_scope": (
            "observed cells only (a real DOC label is required); this is not "
            "a full-grid missing-value imputation product"
        ),
    }
    payload["config_hash"] = config_hash(full_config)
    return payload


def config_payload(params: dict[str, Any], dataset_identity: dict[str, Any],
                   mask_identity: dict[str, Any]) -> dict[str, Any]:
    """Add the input content hashes to a parameter dict before hashing."""
    return {
        **params,
        "dataset_path": str(dataset_identity.get("path")),
        "dataset_sha256": dataset_identity.get("sha256"),
        "mask_path": str(mask_identity.get("path")),
        "mask_sha256": mask_identity.get("sha256"),
    }


def identity_problems(meta: dict[str, Any] | None,
                      expected: dict[str, Any]) -> list[str]:
    """List the identity fields where a stored sidecar disagrees with now.

    An empty list means they match. Used to refuse reusing a stored prediction
    that was produced from different inputs or settings.
    """
    if not meta:
        return ["no provenance sidecar"]
    stored = meta.get("config", {}) or {}
    problems = []
    for key in CONFIG_FIELDS:
        want = expected.get(key)
        got = stored.get(key)
        if key in ("dataset_path", "mask_path"):
            continue  # covered by the content hashes below
        if want != got:
            if key.endswith("_sha256"):
                want = str(want)[:12] if want else want
                got = str(got)[:12] if got else got
            problems.append(f"{key}: stored={got!r} current={want!r}")
    return problems


def describe(meta: dict[str, Any] | None) -> str:
    """One-line summary for logs."""
    if not meta:
        return "no provenance sidecar"
    cfg = meta.get("config", {})
    ds = (meta.get("dataset") or {}).get("sha256") or "?"
    mk = (meta.get("mask") or {}).get("sha256") or "?"
    return (f"config_hash={str(meta.get('config_hash'))[:12]} "
            f"seed={cfg.get('seed')} arch={cfg.get('architecture')} "
            f"dataset_sha={ds[:12]} mask_sha={mk[:12]}")
