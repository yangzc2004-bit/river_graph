"""Deterministic support-shuffle contracts for the Stage-2 integrity audit.

The helpers in this module operate on cells and observation masks only.  They
never inspect target values while constructing a shuffle manifest.  A
site-shuffle is deliberately fail-closed: when the frozen task has fewer than
K alternative observed sites in the same target basin and month, the control
is marked ``not_identifiable_by_design`` rather than silently reusing the
true support locations.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

SHUFFLE_MODES = ("true", "value_shuffle", "site_shuffle")


def stable_seed(*parts: object) -> int:
    """Return a reproducible 32-bit seed from protocol identifiers."""

    payload = json.dumps([str(p) for p in parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**32)


def value_shuffle_permutation(support_cells: np.ndarray, *, seed: int) -> np.ndarray:
    """Return a deterministic permutation of support positions.

    The returned array contains indices into ``support_cells``.  K=0 and K=1
    are valid but intrinsically non-identifiable controls; callers should
    report their status rather than treating them as evidence.
    """

    cells = np.asarray(support_cells, dtype=np.int64).reshape(-1)
    if len(np.unique(cells)) != len(cells):
        raise ValueError("support cells must be unique")
    return np.random.default_rng(seed).permutation(len(cells)).astype(np.int64)


def site_shuffle_destination(
    support_cells: np.ndarray,
    query_cells: np.ndarray,
    target_rows: np.ndarray,
    month_index: int,
    n_months: int,
    observed_flat: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Choose alternative same-month target-basin sites without labels.

    ``observed_flat`` is a boolean target-analyte mask.  Destination cells are
    sampled from target rows, excluding both true support and query cells.
    Values are transferred by the caller in source-support order.  If the
    eligible pool is too small, no destination is returned and the reason is
    explicit.
    """

    support = np.asarray(support_cells, dtype=np.int64).reshape(-1)
    query = np.asarray(query_cells, dtype=np.int64).reshape(-1)
    rows = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    mask = np.asarray(observed_flat, dtype=bool).reshape(-1)
    if n_months <= 0 or not 0 <= int(month_index) < n_months:
        raise ValueError("month_index outside grid")
    if len(np.unique(support)) != len(support) or len(np.unique(query)) != len(query):
        raise ValueError("support/query cells must be unique")
    if np.intersect1d(support, query).size:
        raise ValueError("support/query overlap")
    if len(mask) == 0 or len(mask) % n_months:
        raise ValueError("observed_flat must represent a station-month grid")
    n_sites = len(mask) // n_months
    if len(rows) and (rows.min() < 0 or rows.max() >= n_sites):
        raise ValueError("target row outside grid")
    support_set = set(map(int, support.tolist()))
    query_set = set(map(int, query.tolist()))
    pool = [
        int(row * n_months + month_index)
        for row in rows.tolist()
        if bool(mask[int(row * n_months + month_index)])
        and int(row * n_months + month_index) not in support_set
        and int(row * n_months + month_index) not in query_set
    ]
    k = len(support)
    if k == 0:
        return {
            "status": "not_applicable_k0",
            "candidate_count": len(pool),
            "destination_cells": [],
        }
    if k == 1:
        return {
            "status": "not_identifiable_k1",
            "candidate_count": len(pool),
            "destination_cells": [],
        }
    if len(pool) < k:
        return {
            "status": "not_identifiable_by_design",
            "candidate_count": len(pool),
            "destination_cells": [],
        }
    rng = np.random.default_rng(seed)
    destinations = np.asarray(pool, dtype=np.int64)[rng.permutation(len(pool))[:k]]
    if np.intersect1d(destinations, support).size or np.intersect1d(destinations, query).size:
        raise AssertionError("site shuffle selected a forbidden cell")
    return {
        "status": "identifiable",
        "candidate_count": len(pool),
        "destination_cells": destinations.tolist(),
    }


def validate_shuffle_record(record: dict, *, n_months: int) -> None:
    """Fail-closed validation for a frozen task-level shuffle record."""

    mode = record.get("mode")
    if mode not in SHUFFLE_MODES:
        raise ValueError(f"unknown shuffle mode: {mode}")
    support = np.asarray(record.get("support_cells", []), dtype=np.int64)
    query = np.asarray(record.get("query_cells", []), dtype=np.int64)
    if len(np.unique(support)) != len(support) or len(np.unique(query)) != len(query):
        raise ValueError("duplicate support/query cells")
    if np.intersect1d(support, query).size:
        raise ValueError("support/query overlap")
    if mode == "value_shuffle":
        perm = np.asarray(record.get("value_permutation", []), dtype=np.int64)
        if len(perm) != len(support) or not np.array_equal(np.sort(perm), np.arange(len(support))):
            raise ValueError("value shuffle is not a permutation")
    if mode == "site_shuffle" and record.get("status") == "identifiable":
        dest = np.asarray(record.get("destination_cells", []), dtype=np.int64)
        if len(dest) != len(support) or len(np.unique(dest)) != len(dest):
            raise ValueError("site shuffle destination count/uniqueness mismatch")
        if np.intersect1d(dest, support).size or np.intersect1d(dest, query).size:
            raise ValueError("site shuffle destination overlaps forbidden cells")
        if not np.all(dest % n_months == int(record["month_index"])):
            raise ValueError("site shuffle changed month")
