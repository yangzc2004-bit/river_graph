"""Mask generation for the DOC reconstruction benchmark (design.md).

A "mask" assigns every observed (station, month) cell to exactly one of
train / val / test. Cell indices are flat C-order indices into the (N, T)
label matrix. Masks are generated once and stored under experiments/masks/;
all reported metrics must name their mask file and seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

E1_RATES = (0.2, 0.4, 0.6)
E1_SEEDS = (42, 43, 44)
E2_CUTOFF = "2020-12"  # test = months after this
E3_HOLDOUT_FRAC = 0.2
E3_SEED = 42
VAL_FRAC = 0.1


def observed_cells(y_mask: np.ndarray) -> np.ndarray:
    """Flat indices of observed cells of an (N, T) boolean matrix."""
    return np.flatnonzero(y_mask.ravel())


def _split_cells(cells: np.ndarray, test_frac: float, seed: int) -> dict[str, np.ndarray]:
    """Random train/val/test split of observed cells (val = VAL_FRAC of train)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cells)
    n_test = round(len(cells) * test_frac)
    test = perm[:n_test]
    rest = perm[n_test:]
    n_val = round(len(rest) * VAL_FRAC)
    return {"train": rest[n_val:], "val": rest[:n_val], "test": test}


def make_e1(y_mask: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """E1 random masks: {name: {train, val, test}} for rates x seeds."""
    cells = observed_cells(y_mask)
    return {
        f"e1_r{int(r * 100)}_seed{s}": _split_cells(cells, r, s)
        for r in E1_RATES
        for s in E1_SEEDS
    }


E2B_CONTEXT_FRAC = 0.2  # share of post-cutoff observed cells kept as context


def _pre_cutoff_train_val(
    y_mask: np.ndarray, months: pd.DatetimeIndex
) -> tuple[list[int], list[int], np.ndarray]:
    """Shared E2 helper: (train, val) cells up to E2_CUTOFF + test-month flags."""
    _n, t = y_mask.shape
    assert len(months) == t
    cutoff = pd.Timestamp(E2_CUTOFF)
    is_test_month = months > cutoff
    train_months = np.flatnonzero(~is_test_month)
    n_val_months = max(1, round(len(train_months) * VAL_FRAC))
    val_months = set(train_months[-n_val_months:].tolist())
    train_month_set = set(train_months[:-n_val_months].tolist())

    train, val = [], []
    for flat in observed_cells(y_mask):
        j = flat % t
        if j in val_months:
            val.append(flat)
        elif j in train_month_set:
            train.append(flat)
    return train, val, np.asarray(is_test_month)


def make_e2_strict(y_mask: np.ndarray, months: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """E2-a strict future forecasting: ALL observed cells after the cutoff are
    test; no DOC context exists in test months (feature-only prediction)."""
    train, val, is_test_month = _pre_cutoff_train_val(y_mask, months)
    t = y_mask.shape[1]
    test = [f for f in observed_cells(y_mask) if is_test_month[f % t]]
    return {"train": np.array(train), "val": np.array(val), "test": np.array(test)}


def make_e2_partial(
    y_mask: np.ndarray, months: pd.DatetimeIndex, seed: int = 42
) -> dict[str, np.ndarray]:
    """E2-b future reconstruction under a running network: in post-cutoff
    months, E2B_CONTEXT_FRAC of observed cells stay visible as context
    (never fitted, never scored); the rest are test."""
    train, val, is_test_month = _pre_cutoff_train_val(y_mask, months)
    t = y_mask.shape[1]
    post = np.array([f for f in observed_cells(y_mask) if is_test_month[f % t]])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(post)
    n_context = round(len(post) * E2B_CONTEXT_FRAC)
    return {
        "train": np.array(train),
        "val": np.array(val),
        "test": perm[n_context:],
        "context": perm[:n_context],
    }


def make_e3(
    y_mask: np.ndarray,
    edges: pd.DataFrame,
    sites: list[str],
    seed: int = E3_SEED,
) -> dict[str, np.ndarray]:
    """E3 spatial split: hold out all observed cells of 20% of stations in
    the largest weakly connected component."""
    import networkx as nx

    g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    largest = max(nx.weakly_connected_components(g), key=len)
    cand = sorted(set(sites) & largest)
    rng = np.random.default_rng(seed)
    held_out = set(rng.permutation(cand)[: round(len(cand) * E3_HOLDOUT_FRAC)])

    _n, t = y_mask.shape
    held_rows = {i for i, s in enumerate(sites) if s in held_out}
    train_val, test = [], []
    for flat in observed_cells(y_mask):
        (test if flat // t in held_rows else train_val).append(flat)
    rng2 = np.random.default_rng(seed + 1)
    tv = rng2.permutation(np.array(train_val))
    n_val = round(len(tv) * VAL_FRAC)
    return {
        "train": tv[n_val:],
        "val": tv[:n_val],
        "test": np.array(test),
        "held_out_sites": np.array(sorted(held_out)),
    }
