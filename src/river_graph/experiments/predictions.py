"""Frozen prediction storage (Phase 0 of the paper-production stage).

After the benchmark is frozen, every model's per-cell predictions are stored
as parquet under experiments/predictions/. All downstream analysis
(residual maps, extreme-cell breakdowns, uncertainty) reads these files —
no retraining, no benchmark drift.

Row schema: station, month, y_true (mg/L), y_pred (mg/L), split role,
plus model / mask / dataset_version metadata.

Two invariants this module enforces (see docs/run_gnn_prediction_storage.md):

* **Atomic writes.** Files are written to a temporary sibling, flushed to
  disk, and then moved into place with ``os.replace``. An interrupted run can
  therefore never be mistaken for a complete cache entry.
* **Identity checking.** ``save_predictions`` writes a ``.meta.json`` sidecar
  carrying a config hash of everything that determines the numbers. Writing a
  different configuration over an existing prediction raises
  :class:`PredictionConflictError` unless the caller passes ``force=True``.

What is exported here is the set of *observed* cells (a cell needs a real DOC
label to be exported). It is NOT a full 571x652 missing-value imputation
product; a complete grid export would be a separate pipeline.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

PRED_DIR = Path("experiments/predictions")
META_SUFFIX = ".meta.json"


class PredictionConflictError(RuntimeError):
    """Raised when a prediction would be overwritten by a different config."""


class CacheState(str, Enum):
    """What exists on disk for one (model, mask) pair."""

    ABSENT = "absent"                  # neither metrics nor predictions
    METRICS_ONLY = "metrics_only"      # metrics JSON entry, no parquet
    PREDICTIONS_ONLY = "predictions_only"  # parquet exists, no metrics entry
    COMPLETE = "complete"              # both present


def prediction_path(model: str, mask_name: str, out_dir: Path = PRED_DIR) -> Path:
    return Path(out_dir) / f"{model}__{mask_name}.parquet"


def meta_path(model: str, mask_name: str, out_dir: Path = PRED_DIR) -> Path:
    return Path(out_dir) / f"{model}__{mask_name}{META_SUFFIX}"


def cache_state(
    model: str,
    mask_name: str,
    results_dir: Path | str = Path("experiments/results"),
    pred_dir: Path = PRED_DIR,
    metrics: dict | None = None,
) -> CacheState:
    """Classify the cache for one (model, mask) pair.

    ``metrics`` may be supplied to avoid re-reading the metrics JSON; when it
    is ``None`` the model's JSON file is read from ``results_dir``.
    """
    if metrics is None:
        jpath = Path(results_dir) / f"{model}.json"
        if jpath.exists():
            import json

            metrics = json.loads(jpath.read_text(encoding="utf-8"))
        else:
            metrics = {}
    has_metrics = mask_name in metrics
    has_preds = prediction_path(model, mask_name, pred_dir).exists()
    if has_metrics and has_preds:
        return CacheState.COMPLETE
    if has_metrics:
        return CacheState.METRICS_ONLY
    if has_preds:
        return CacheState.PREDICTIONS_ONLY
    return CacheState.ABSENT


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a flushed temp file + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def save_predictions(
    pred: np.ndarray,
    dataset: dict,
    split: dict[str, np.ndarray],
    model: str,
    mask_name: str,
    dataset_version: str,
    out_dir: Path = PRED_DIR,
    meta: dict | None = None,
    force: bool = False,
) -> tuple[Path, Path | None]:
    """Store per-cell predictions for every observed cell, atomically.

    Parameters
    ----------
    meta:
        Optional provenance payload (see
        :func:`river_graph.experiments.provenance.build_meta`). When given, a
        ``.meta.json`` sidecar is written next to the parquet.
    force:
        Overwrite an existing prediction whose recorded ``config_hash``
        differs from ``meta["config_hash"]``. Without it such a write raises
        :class:`PredictionConflictError`, so a new configuration can never
        silently reuse an old result file name.

    Returns
    -------
    (parquet_path, meta_path)
    """
    y = dataset["y"].numpy()
    obs_mask = dataset["y_mask"].numpy()
    sites = dataset["site_no"]
    months = dataset["months"]
    n, t = y.shape

    role = np.full(n * t, "", dtype=object)
    for key in ("train", "val", "test", "context"):
        if key in split and len(split[key]):
            role[split[key]] = key
    role = role.reshape(n, t)

    rows = []
    for i in range(n):
        cells = np.flatnonzero(obs_mask[i])
        if len(cells) == 0:
            continue
        rows.append(pd.DataFrame({
            "station": sites[i],
            "month": [months[j] for j in cells],
            "y_true": y[i, cells],
            "y_pred": pred[i, cells],
            "split": role[i, cells],
        }))
    df = pd.concat(rows, ignore_index=True)
    df["model"] = model
    df["mask"] = mask_name
    df["dataset_version"] = dataset_version

    out_dir = Path(out_dir)
    out = prediction_path(model, mask_name, out_dir)
    mpath = meta_path(model, mask_name, out_dir)

    if meta is not None and mpath.exists():
        existing = read_meta(model, mask_name, out_dir)
        old_hash = (existing or {}).get("config_hash")
        new_hash = meta.get("config_hash")
        if old_hash and new_hash and old_hash != new_hash and not force:
            raise PredictionConflictError(
                f"{out.name} already exists with a different configuration "
                f"(existing config_hash={old_hash[:12]}, new={new_hash[:12]}). "
                "Refusing to overwrite: give this run a distinct --model-name, "
                "or pass force=True to replace it deliberately."
            )

    _atomic_write_parquet(out, df)
    if meta is not None:
        import json

        payload = dict(meta)
        payload.setdefault("model_name", model)
        payload.setdefault("mask_name", mask_name)
        payload.setdefault("rows", len(df))
        payload.setdefault("observed_cells", int(obs_mask.sum()))
        _atomic_write_text(mpath, json.dumps(payload, indent=2,
                                             ensure_ascii=False) + "\n")
    return out, (mpath if meta is not None else None)


def load_predictions(model: str, mask_name: str, out_dir: Path = PRED_DIR) -> pd.DataFrame:
    return pd.read_parquet(prediction_path(model, mask_name, out_dir))


def read_meta(model: str, mask_name: str, out_dir: Path = PRED_DIR) -> dict | None:
    """Return the stored provenance sidecar, or ``None`` when absent."""
    import json

    mpath = meta_path(model, mask_name, out_dir)
    if not mpath.exists():
        return None
    return json.loads(mpath.read_text(encoding="utf-8"))
