"""K-shot pseudo-new-watershed protocol (exploratory plan Task 1).

Hide one sub-basin's DOC labels during training, then at inference reveal K
same-month local observations (support) and score the remaining observed
cells of that sub-basin (query). Query sites are fixed within a task month
so the K curve is nested and comparable.

Cell indices are flat C-order indices into the (N, T) label matrix, matching
experiments/masks.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

K_LIST = (0, 1, 3, 5, 10)
MIN_QUERY = 2
VAL_FRAC = 0.1

# Dense same-month sampling pockets (see session density audit on ST357).
DEFAULT_REGIONS = ("10300101", "7050002")


@dataclass(frozen=True)
class KShotTask:
    """One evaluation month inside a pseudo-new watershed."""

    region: str
    month_index: int
    month: str
    station_idx: tuple[int, ...]
    query: tuple[int, ...]  # flat cell indices
    support_by_k: dict[int, tuple[int, ...]]  # nested supports, flat cells


def region_station_index(
    sites: list[str],
    nodes: pd.DataFrame,
    huc8: str,
    level: int = 8,
) -> np.ndarray:
    """Row indices in the dataset whose station sits in the given HUC prefix."""
    meta = pd.DataFrame({"station": [str(s) for s in sites]})
    meta = meta.merge(
        nodes[["site_no", "huc_cd"]].rename(columns={"site_no": "station"}),
        on="station",
        how="left",
        validate="one_to_one",
    )
    prefix = meta["huc_cd"].astype(str).str[:level]
    return np.flatnonzero(prefix.eq(str(huc8)).to_numpy())


def make_region_split(
    y_mask: np.ndarray,
    region_rows: Iterable[int],
    seed: int = 42,
    val_frac: float = VAL_FRAC,
) -> dict[str, np.ndarray]:
    """Train/val outside the region; every observed region cell is held out.

    Region cells never enter train/val/context, so training cannot see the
    pseudo-new watershed's DOC labels. Support/query assignment is done later
    at inference time from the held-out region cells.
    """
    n, t = y_mask.shape
    region = {int(i) for i in region_rows}
    if not region:
        raise ValueError("region_rows is empty")
    if max(region) >= n:
        raise ValueError(f"region row {max(region)} out of range for N={n}")

    train_val, test = [], []
    for flat in np.flatnonzero(y_mask.ravel()):
        row = flat // t
        (test if row in region else train_val).append(int(flat))
    if not train_val:
        raise ValueError("no train/val cells outside the region")
    if not test:
        raise ValueError("region has no observed cells")

    rng = np.random.default_rng(seed)
    tv = rng.permutation(np.asarray(train_val, dtype=np.int64))
    n_val = max(1, round(len(tv) * val_frac))
    return {
        "train": np.sort(tv[n_val:]),
        "val": np.sort(tv[:n_val]),
        "test": np.sort(np.asarray(test, dtype=np.int64)),
        "region_rows": np.asarray(sorted(region), dtype=np.int64),
    }


def make_support_query_tasks(
    y_mask: np.ndarray,
    region_rows: Iterable[int],
    months: list[str],
    k_list: tuple[int, ...] = K_LIST,
    min_query: int = MIN_QUERY,
    seed: int = 42,
) -> list[KShotTask]:
    """Nested same-month support / fixed query tasks inside the region.

    For a region-month with n observed stations:
    - feasible K are those with K + min_query <= n
    - query = the last (n - max_K) stations of a fixed permutation
    - support(K) = the first K stations of that same permutation

    Query cells are identical across K within a month. Stations used as
    support never appear as query for that task.
    """
    rows = [int(i) for i in region_rows]
    _n, t = y_mask.shape
    if not rows:
        return []
    if len(months) != t:
        raise ValueError("months length must match y_mask columns")

    tasks: list[KShotTask] = []
    for j in range(t):
        obs = [i for i in rows if y_mask[i, j]]
        if len(obs) < min(k_list) + min_query:
            continue
        feasible = [k for k in k_list if k + min_query <= len(obs)]
        if not feasible:
            continue
        max_k = max(feasible)
        rng = np.random.default_rng(seed + 1000 * (j + 1) + hash(rows[0]) % 1000)
        order = [obs[i] for i in rng.permutation(len(obs))]
        support_pool = order[:max_k]
        query_rows = order[max_k:]
        if not query_rows:
            continue
        query = tuple(i * t + j for i in query_rows)
        support_by_k = {
            k: tuple(i * t + j for i in support_pool[:k]) for k in feasible
        }
        tasks.append(
            KShotTask(
                region=",".join(str(i) for i in rows[:1]) + f"…n{len(rows)}",
                month_index=j,
                month=str(months[j]),
                station_idx=tuple(order),
                query=query,
                support_by_k=support_by_k,
            )
        )
    return tasks


def label_tasks(tasks: list[KShotTask], region_name: str) -> list[KShotTask]:
    """Attach a stable region label (used in output tables)."""
    out = []
    for task in tasks:
        out.append(
            KShotTask(
                region=region_name,
                month_index=task.month_index,
                month=task.month,
                station_idx=task.station_idx,
                query=task.query,
                support_by_k=task.support_by_k,
            )
        )
    return out


def tasks_to_frame(tasks: list[KShotTask]) -> pd.DataFrame:
    """One row per (task, K) with support/query cell lists as strings."""
    rows = []
    for task in tasks:
        for k, support in sorted(task.support_by_k.items()):
            rows.append(
                {
                    "region": task.region,
                    "month": task.month,
                    "month_index": task.month_index,
                    "k": k,
                    "n_query": len(task.query),
                    "n_support": len(support),
                    "query_cells": " ".join(map(str, task.query)),
                    "support_cells": " ".join(map(str, support)),
                    "n_stations_available": len(task.station_idx),
                }
            )
    return pd.DataFrame(rows)


def validate_split(split: dict[str, np.ndarray], y_mask: np.ndarray) -> list[str]:
    """Return a list of protocol violations (empty means OK)."""
    problems: list[str] = []
    n, t = y_mask.shape
    train = {int(x) for x in split.get("train", [])}
    val = {int(x) for x in split.get("val", [])}
    test = {int(x) for x in split.get("test", [])}
    region = {int(x) for x in split.get("region_rows", [])}

    if train & val:
        problems.append(f"train/val overlap: {len(train & val)}")
    if train & test:
        problems.append(f"train/test overlap: {len(train & test)}")
    if val & test:
        problems.append(f"val/test overlap: {len(val & test)}")

    for name, cells in ("train", train), ("val", val), ("test", test):
        for flat in cells:
            if flat < 0 or flat >= n * t:
                problems.append(f"{name} cell {flat} out of range")
                break
            if not y_mask.ravel()[flat]:
                problems.append(f"{name} cell {flat} is not an observed DOC cell")
                break

    for flat in test:
        if flat // t not in region:
            problems.append(f"test cell {flat} is outside region_rows")
            break
    for flat in train | val:
        if flat // t in region:
            problems.append(f"{'train' if flat in train else 'val'} cell {flat} is inside region")
            break
    if not test:
        problems.append("test is empty")
    if not train:
        problems.append("train is empty")
    return problems


def validate_tasks(
    tasks: list[KShotTask],
    y_mask: np.ndarray,
    k_list: tuple[int, ...] = K_LIST,
    min_query: int = MIN_QUERY,
) -> list[str]:
    """Return protocol violations for support/query construction (empty = OK)."""
    problems: list[str] = []
    _n, t = y_mask.shape
    flat_obs = {int(i) for i in np.flatnonzero(y_mask.ravel())}
    for task in tasks:
        query = set(task.query)
        if len(query) < min_query and task.month_index >= 0:
            # allowed only if this month contributed any K; still require min_query
            problems.append(f"{task.month}: query size {len(query)} < {min_query}")
        query_rows = {flat // t for flat in query}
        support_rows = set()
        for support in task.support_by_k.values():
            support_rows |= {flat // t for flat in support}
        if query_rows & support_rows:
            problems.append(f"{task.month}: station used as both support and query")
        for flat in query:
            if flat not in flat_obs:
                problems.append(f"{task.month}: query cell {flat} not observed")
        ks = sorted(task.support_by_k)
        if ks != sorted(k for k in k_list if k in task.support_by_k):
            problems.append(f"{task.month}: unexpected K keys {ks}")
        prev: tuple[int, ...] = ()
        for k in ks:
            support = set(task.support_by_k[k])
            if len(support) != k:
                problems.append(f"{task.month}: K={k} has {len(support)} support cells")
            if support & query:
                problems.append(f"{task.month}: K={k} support/query overlap")
            if not support >= set(prev):
                problems.append(f"{task.month}: K={k} support is not nested")
            if not support <= flat_obs:
                problems.append(f"{task.month}: K={k} support has unobserved cells")
            prev = task.support_by_k[k]
        # query fixed across K by construction (single query tuple)
        if task.support_by_k.get(0):
            problems.append(f"{task.month}: K=0 must have empty support")
    return problems
