"""Nested station support and fixed queries for retrospective DOC transfer."""
from __future__ import annotations

import numpy as np

K_VALUES = (0, 1, 3, 5)
ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def support_schedule(target_cells: np.ndarray, n_months: int) -> tuple[dict, np.ndarray]:
    """Five value-blind support candidates per station, with nested prefixes.

    The ordered candidates are first, middle, last, first quarter and third
    quarter among the station's observed months. They are all reserved before
    evaluating K=0, so every K uses exactly the same query population.
    """
    cells = np.unique(np.asarray(target_cells, dtype=np.int64))
    support: dict[int, np.ndarray] = {}
    reserve: list[np.ndarray] = []
    for station in np.unique(cells // n_months):
        local = cells[cells // n_months == station]
        if len(local) <= 5:
            raise ValueError("each target station needs five support months and at least one query")
        grid = np.linspace(0, len(local) - 1, 5, dtype=int)
        ordered = local[grid[[0, 2, 4, 1, 3]]]
        support[int(station)] = ordered
        reserve.append(ordered)
    reserved = np.sort(np.concatenate(reserve)) if reserve else np.empty(0, dtype=np.int64)
    return support, np.setdiff1d(cells, reserved, assume_unique=True)


def support_view(split: dict, *, target_role: str, k: int,
                 n_months: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Source labels plus K opened target supports; all other targets hidden.

    The training cells are unchanged. Validation target stations are mapped to
    the hidden test role so TEST_ROLES cannot accidentally open their labels.
    Any outer test cells in an internal split are omitted and remain invisible.
    """
    if k not in K_VALUES:
        raise ValueError("K must be 0, 1, 3 or 5")
    schedule, query = support_schedule(split[target_role], n_months)
    opened = [cells[:k] for cells in schedule.values()]
    support = np.sort(np.concatenate(opened)) if opened else np.empty(0, dtype=np.int64)
    source_val = np.asarray(split.get("val", []), dtype=np.int64)
    if target_role == "val":
        source_val = np.empty(0, dtype=np.int64)
    original_context = np.asarray(split.get("context", []), dtype=np.int64)
    view = {
        "train": np.asarray(split["train"], dtype=np.int64).copy(),
        "val": source_val.copy(),
        "context": np.unique(np.concatenate([original_context, support])),
        "test": query.copy(),
    }
    return view, support, query


def station_level_correction(values: np.ndarray, support: np.ndarray,
                             query: np.ndarray, base_prediction: np.ndarray,
                             *, n_months: int, alpha: float) -> np.ndarray:
    """Support mean minus label-free query prediction mean, in log1p units.

    This is retrospective reconstruction: support months span the record, and
    the level correction is applied across that record. It is not a forecast
    before those support observations have been acquired.
    """
    prediction = np.asarray(base_prediction, dtype=np.float64)
    query = np.asarray(query, dtype=np.int64)
    support = np.asarray(support, dtype=np.int64)
    if prediction.shape != query.shape or not np.isfinite(prediction).all():
        raise ValueError("predictions must be finite and aligned with query cells")
    if np.intersect1d(support, query).size:
        raise ValueError("support and query must be disjoint")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha == 0 or not len(support):
        return prediction.copy()
    # Access only the support labels, not an all-label reduction.
    support_z = np.log1p(np.asarray(values).reshape(-1)[support])
    if not np.isfinite(support_z).all():
        raise ValueError("support values must be finite and nonnegative")
    z = np.log1p(np.maximum(prediction, 0.0))
    out = z.copy()
    for station in np.unique(query // n_months):
        q = query // n_months == station
        s = support // n_months == station
        if s.any():
            out[q] += alpha * (support_z[s].mean() - z[q].mean())
    return np.maximum(np.expm1(out), 0.0)
