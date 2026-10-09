"""Station-held-out DOC tasks for a unified temporal and local-adaptation model.

Only the observation mask determines the partitions and support schedule.  The
full graph and station cohort remain unchanged.  Support calibration is
retrospective: support months can occur after a query month, but those labels
are never opened in the base model's temporal or environmental inputs.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from river_graph.experiments.spatial_fewshot import K_VALUES, support_schedule

SPLIT_SEEDS = (142, 143, 144)
TRAINING_SEEDS = (42, 43, 44)
ROLES = ("train", "val", "test", "context")


def _observation_mask(y_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(y_mask)
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("y_mask must be a nonempty station by month matrix")
    if not np.isin(mask, (False, True)).all():
        raise ValueError("y_mask must contain only boolean observation indicators")
    return mask.astype(bool, copy=False)


def _cells(value: np.ndarray, *, size: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise ValueError("cell identities must be a one-dimensional integer array")
    cells = arr.astype(np.int64, copy=False)
    if np.any(cells < 0) or np.any(cells >= size):
        raise ValueError("cell identities fall outside the station-month grid")
    if np.unique(cells).size != cells.size:
        raise ValueError("duplicate cell identities within a role")
    return cells


def validate_unified_spatial_split(
    y_mask: np.ndarray, split: Mapping[str, np.ndarray]
) -> None:
    """Require complete observed-cell coverage and disjoint station roles."""
    observed = _observation_mask(y_mask)
    n_months = observed.shape[1]
    if any(role not in split for role in ROLES):
        raise ValueError("train, val, test and context roles must all be explicit")
    cells = {role: _cells(split[role], size=observed.size) for role in ROLES}
    if cells["context"].size:
        raise ValueError("base station-transfer protocol has no context labels")
    for role in ("train", "val", "test"):
        if not cells[role].size:
            raise ValueError(f"{role} must contain observed cells")
        if not observed.ravel()[cells[role]].all():
            raise ValueError(f"{role} includes an unobserved cell")
    combined = np.concatenate(list(cells.values()))
    if not np.array_equal(np.sort(combined), np.flatnonzero(observed)):
        raise ValueError("roles must partition every observed cell exactly once")
    stations = {role: set(cells[role] // n_months) for role in cells}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if stations[first] & stations[second]:
            raise ValueError("train, validation and test stations must be disjoint")
    for role in ("val", "test"):
        _, counts = np.unique(cells[role] // n_months, return_counts=True)
        if (counts < 6).any():
            raise ValueError("held-out stations need five supports and at least one query")


def build_unified_spatial_split(
    y_mask: np.ndarray,
    *,
    seed: int,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, np.ndarray], dict]:
    """Create a value-blind random-station partition and inspectable metadata.

    Fractions refer to all support-eligible stations, not observed cells.  A
    station needs six observations to enter validation or test; less-observed
    stations remain in the source cohort.  For ST357 the defaults yield 232
    source, 54 validation and 71 test stations.  Validation labels are used for
    early stopping, fusion and calibration selection, never as model inputs.
    """
    observed = _observation_mask(y_mask)
    if not (0 < test_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("test and validation fractions must be between zero and one")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must leave source stations")
    n_stations, n_months = observed.shape
    observation_count = observed.sum(axis=1)
    eligible = np.flatnonzero(observation_count >= 6)
    n_test = max(1, round(len(eligible) * test_fraction))
    n_val = max(1, round(len(eligible) * validation_fraction))
    if n_test + n_val >= len(eligible):
        raise ValueError("too few support-eligible stations for station-held-out tasks")
    permutation = np.random.default_rng(seed).permutation(eligible)
    test_stations = np.sort(permutation[:n_test])
    val_stations = np.sort(permutation[n_test:n_test + n_val])
    source_stations = np.setdiff1d(np.arange(n_stations), permutation[:n_test + n_val])
    if np.count_nonzero(observation_count[source_stations]) < 5:
        raise ValueError("source stations must support station-blocked five-fold OOF")
    observed_cells = np.flatnonzero(observed)
    station_rows = observed_cells // n_months
    split = {
        "train": observed_cells[np.isin(station_rows, source_stations)],
        "val": observed_cells[np.isin(station_rows, val_stations)],
        "test": observed_cells[np.isin(station_rows, test_stations)],
        "context": np.empty(0, dtype=np.int64),
    }
    validate_unified_spatial_split(observed, split)
    role_stations = {"train": source_stations, "val": val_stations, "test": test_stations}
    metadata = {
        "protocol_version": "unified_spatial_v1",
        "split_seed": int(seed),
        "split_unit": "station",
        "generalization_task": "random station holdout within the same Mississippi cohort",
        "geographical_scope": "neither external-basin validation nor contiguous geographic holdout",
        "split_selection": "random permutation of eligible stations; no target values",
        "test_fraction_of_eligible_stations": float(test_fraction),
        "validation_fraction_of_eligible_stations": float(validation_fraction),
        "shape": [int(n_stations), int(n_months)],
        "observed_cells": int(observed.sum()),
        "support_eligible_stations": eligible.tolist(),
        "source_only_ineligible_stations": np.flatnonzero(observation_count < 6).tolist(),
        "station_observation_counts": observation_count.astype(int).tolist(),
        "station_roles": {role: stations.tolist() for role, stations in role_stations.items()},
        "station_counts": {role: len(stations) for role, stations in role_stations.items()},
        "cell_counts": {role: len(cells) for role, cells in split.items()},
        "K_values": list(K_VALUES),
        "support_order": ["first", "middle", "last", "first_quarter", "third_quarter"],
        "support_schedule": "five equally spaced observed-month ranks; nested prefix by K",
        "query_rule": "exclude all five reserved support cells at every K, including K=0",
        "adaptation_mode": "retrospective station calibration; support may postdate query",
        "base_input_roles": ["train"],
        "support_input_policy": "post-hoc station calibration only; never base-model inputs",
        "validation_use": "early stopping, model/fusion/calibration selection",
        "test_use": "final evaluation after source-only model and calibration selection",
        "graph_policy": "retain original complete cohort and graph; hold out labels by station",
        "target_tasks": {},
    }
    for role in ("val", "test"):
        schedule, query = support_schedule(split[role], n_months)
        metadata["target_tasks"][role] = {
            "support_candidates": {str(station): cells.tolist() for station, cells in schedule.items()},
            "reserved_support_cells": int(sum(len(cells) for cells in schedule.values())),
            "fixed_query_cells": len(query),
        }
    return split, metadata


def support_query_cells(
    split: Mapping[str, np.ndarray], *, target_role: str, k: int, n_months: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return nested support and fixed query identities without reading labels."""
    if target_role not in ("val", "test"):
        raise ValueError("target_role must be val or test")
    if k not in K_VALUES:
        raise ValueError("K must be 0, 1, 3 or 5")
    if n_months < 1:
        raise ValueError("n_months must be positive")
    target = np.asarray(split[target_role])
    if target.ndim != 1 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("target cell identities must be a one-dimensional integer array")
    if (target < 0).any() or np.unique(target).size != target.size:
        raise ValueError("target cell identities must be nonnegative and unique")
    schedule, query = support_schedule(target, n_months)
    opened = [cells[:k] for cells in schedule.values()]
    support = np.sort(np.concatenate(opened)) if opened else np.empty(0, dtype=np.int64)
    return support, query


def base_inference_split(split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep every held-out label hidden even for predictors opening val at test.

    Fit uses the original split so validation targets can evaluate early
    stopping.  This view is only for prediction/feature construction: both
    station-held-out roles are mapped to test and val/context are empty.
    """
    if np.asarray(split.get("context", [])).size:
        raise ValueError("base inference cannot include support/context labels")
    return {
        "train": np.asarray(split["train"], dtype=np.int64).copy(),
        "val": np.empty(0, dtype=np.int64),
        "test": np.sort(np.concatenate([split["val"], split["test"]])).astype(np.int64),
        "context": np.empty(0, dtype=np.int64),
    }
