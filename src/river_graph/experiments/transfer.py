"""Data-only contracts for the transfer route. No model fitting lives here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

ANALYTES = ("doc", "ph", "spec_conductance")
UNITS = {"doc": "mg/L", "ph": "standard units", "spec_conductance": "uS/cm"}
DATASETS = {
    "doc": "data/processed/mississippi_graph_graphfix_st357.pt",
    "ph": "data/processed/mississippi_graph_ph_st357.pt",
    "spec_conductance": "data/processed/mississippi_graph_spec_conductance_st357.pt",
}
SHARED_FIELDS = ("edge_index", "edge_attr", "x", "x_mask", "static", "regime")
K_VALUES = (0, 1, 3, 5)


def canonical_huc8(value: str) -> str:
    """Canonicalize legacy HUC6 aliases and HUC8/HUC12 metadata."""
    code = str(value)
    if not code.isdigit() or len(code) not in (6, 7, 8, 11, 12):
        raise ValueError(f"invalid HUC8/HUC12 metadata: {value}")
    if len(code) == 6:
        # Historical protocol shorthand dropped the leading zero only for
        # the 051002 task; ordinary six-digit HUC6 labels stay unchanged.
        return "051002" if code == "510020" else code
    if len(code) % 2:
        code = "0" + code
    return code[:8]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def array_hash(value) -> str:
    a = np.ascontiguousarray(array(value))
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def validate_dataset(ds: dict) -> dict:
    """Fail closed on malformed grids, edges and undeclared input channels."""
    y, mask = array(ds["y"]), array(ds["y_mask"])
    if y.ndim != 2 or mask.shape != y.shape or mask.dtype != np.dtype(bool):
        raise ValueError("y/y_mask must be aligned 2D arrays with a boolean mask")
    n, t = y.shape
    sites = [str(s) for s in ds["site_no"]]
    months = [str(m)[:7] for m in ds["months"]]
    if len(sites) != n or len(set(sites)) != n or any(not s for s in sites):
        raise ValueError("missing/duplicate/misaligned station identifiers")
    parsed = pd.PeriodIndex(months, freq="M")
    if len(months) != t or parsed.has_duplicates or not parsed.is_monotonic_increasing:
        raise ValueError("missing/duplicate/unsorted/misaligned months")
    if len(parsed) and len(pd.period_range(parsed[0], parsed[-1], freq="M")) != t:
        raise ValueError("non-contiguous month grid")
    if not np.isfinite(y[mask]).all():
        raise ValueError("nonfinite observed labels")
    ei = array(ds["edge_index"])
    if ei.ndim != 2 or ei.shape[0] != 2 or ei.dtype.kind not in "iu":
        raise ValueError("edge_index must have integer shape (2,E)")
    if ei.size and (ei.min() < 0 or ei.max() >= n):
        raise ValueError("edge endpoint outside dataset")
    if array(ds["edge_attr"]).shape != (ei.shape[1], 6):
        raise ValueError("edge_attr must have six aligned channels")
    if list(ds.get("feature_channels", [])) != ["temperature", "discharge"]:
        raise ValueError("transfer inputs must explicitly declare temperature/discharge only")
    if array(ds["x"]).shape != (n, t, 2) or array(ds["x_mask"]).shape != (n, t, 2):
        raise ValueError("hydro feature/mask shape mismatch")
    if not np.isin(array(ds["x_mask"]), [0, 1]).all():
        raise ValueError("invalid hydro feature mask")
    if array(ds["static"]).shape != (n, 2) or array(ds["regime"]).shape != (n, 13):
        raise ValueError("static/regime feature shape mismatch")
    if any(not np.isfinite(array(ds[k])).all() for k in SHARED_FIELDS):
        raise ValueError("nonfinite shared features")
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(ei.T.tolist())
    largest = max((len(c) for c in nx.connected_components(g)), default=0)
    active = mask.any(1)
    active_graph = g.subgraph(np.flatnonzero(active))
    active_largest = max((len(c) for c in nx.connected_components(active_graph)), default=0)
    return {
        "stations": n, "months": t, "active_stations": int(active.sum()),
        "active_months": int(mask.any(0).sum()), "observed_station_months": int(mask.sum()),
        "month_start": months[0], "month_end": months[-1],
        "directed_edges": ei.shape[1], "largest_component_nodes": largest,
        "largest_component_fraction": largest / n,
        "active_largest_component_fraction": active_largest / max(int(active.sum()), 1),
        "identity": {"sites": object_hash(sites), "months": object_hash(months),
                     **{k: array_hash(ds[k]) for k in SHARED_FIELDS}},
    }


def load_bundle(paths: dict, nodes_path: str | Path, edges_path: str | Path) -> tuple:
    """Bind all three targets to identical graph/features and authoritative metadata."""
    datasets, summaries = {}, {}
    for analyte in ANALYTES:
        path = Path(paths[analyte])
        ds = torch.load(path, map_location="cpu", weights_only=False)
        summary = validate_dataset(ds)
        y = array(ds["y"])[array(ds["y_mask"])]
        if analyte == "ph" and ((y < 0) | (y > 14)).any():
            raise ValueError("pH outside frozen physical range")
        if analyte == "spec_conductance" and ((y < 0) | (y > 100000)).any():
            raise ValueError("conductance outside frozen physical range")
        if analyte == "doc" and (y < 0).any():
            raise ValueError("negative DOC needs explicit QC decision; not clipped")
        summaries[analyte] = {**summary, "path": str(path), "sha256": file_hash(path),
                              "unit": UNITS[analyte]}
        datasets[analyte] = ds
    ref = datasets["doc"]
    for a in ANALYTES:
        if summaries[a]["identity"] != summaries["doc"]["identity"]:
            raise ValueError(f"{a}: grid, graph or covariate identity differs from DOC")
    nodes = pd.read_csv(nodes_path, dtype=str).set_index("site_no", verify_integrity=True)
    sites = list(ref["site_no"])
    if set(nodes.index) != set(sites):
        raise ValueError("node CSV membership differs from dataset")
    nodes = nodes.loc[sites].copy()
    if not nodes["site_tp_cd"].eq("ST").all():
        raise ValueError("non-ST node in primary bundle")
    codes = nodes["huc_cd"]
    if codes.isna().any():
        raise ValueError("HUC metadata missing/invalid")
    nodes["huc8"] = codes.map(canonical_huc8)
    nodes["huc6"] = nodes["huc8"].str[:6]
    edges = pd.read_csv(edges_path, dtype=str)
    actual = {(sites[i], sites[j]) for i, j in array(ref["edge_index"]).T}
    listed = list(zip(edges.source, edges.target))
    if len(set(listed)) != len(listed) or set(listed) != actual:
        raise ValueError("edge CSV differs from dataset directed edges")
    return datasets, summaries, nodes


def availability_tasks(mask: np.ndarray, rows, months, seed: int) -> list[dict]:
    """Data-only task inventory: fixed query, five nested support cells, >=2 queries."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows, dtype=int)
    t = mask.shape[1]
    tasks = []
    for j, month in enumerate(months):
        observed = rows[mask[rows, j]]
        if len(observed) < 7:
            continue
        perm = rng.permutation(observed)
        tasks.append({
            "month": str(month)[:7], "month_index": j,
            "query_cells": (perm[5:] * t + j).tolist(),
            "support_cells_by_k": {str(k): (perm[:k] * t + j).tolist() for k in K_VALUES},
        })
    return tasks


def isolated_view(labels, observed, source_train, source_val, *, target: int,
                  target_rows, support=(), query=()) -> dict:
    """Materialize the only label views a future trainer/adapter may receive.

    Masks have shape (analyte, station, month); support/query are flat cells
    within the target analyte. Query values are never returned. Validation
    labels are returned only as a separate source-selection target.
    """
    y = array(labels)
    masks = [array(m) for m in (observed, source_train, source_val)]
    if y.ndim != 3 or any(m.shape != y.shape or m.dtype != bool for m in masks):
        raise ValueError("aligned boolean role masks required")
    obs, train, val = masks
    if not 0 <= target < y.shape[0]:
        raise ValueError("target index out of bounds")
    rows = np.asarray(target_rows, dtype=int)
    if not len(rows) or rows.min() < 0 or rows.max() >= y.shape[1]:
        raise ValueError("invalid target basin rows")
    if (train & val).any() or ((train | val) & ~obs).any():
        raise ValueError("overlapping or unobserved source roles")
    if train[target].any() or val[target].any() or train[:, rows].any() or val[:, rows].any():
        raise ValueError("target analyte/basin cannot enter source fit or selection")
    support, query = np.asarray(support, dtype=int), np.asarray(query, dtype=int)
    for cells in (support, query):
        if len(cells) != len(set(cells.tolist())):
            raise ValueError("duplicate role cells")
        if len(cells) and (cells.min() < 0 or cells.max() >= y.shape[1] * y.shape[2]):
            raise ValueError("role cell out of bounds")
        if not np.isin(cells // y.shape[2], rows).all():
            raise ValueError("support/query outside target basin")
        if not obs[target].ravel()[cells].all():
            raise ValueError("unobserved support/query")
    if np.intersect1d(support, query).size:
        raise ValueError("support/query overlap")
    # Per-source transform statistics; no statistic exists for the held-out target.
    stats = {}
    for a in range(y.shape[0]):
        if a == target:
            continue
        values = y[a][train[a]]
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("source training labels absent/nonfinite")
        stats[str(a)] = {"mean": float(values.mean()), "std": max(float(values.std()), 1e-8)}
    visible_support = np.zeros_like(y[target])
    visible_support.ravel()[support] = y[target].ravel()[support]
    if not np.isfinite(visible_support).all() or not np.isfinite(y[val]).all():
        raise ValueError("visible support or source validation nonfinite")
    return {
        "fit_labels": np.where(train, y, 0), "fit_mask": train.copy(),
        "selection_labels": np.where(val, y, 0), "selection_mask": val.copy(),
        "source_statistics": stats, "support_labels": visible_support,
        "support_cells": support.copy(), "query_cells": query.copy(),
    }


def external_checks(summaries: dict, *, limits: dict) -> dict:
    """Numeric availability gates; final approval ALSO requires metadata and provenance."""
    return {
        a: {
            "stations": summaries[a]["active_stations"] >= limits["minimum_stations"],
            "months": summaries[a]["active_months"] >= limits["minimum_months"],
            "station_months": summaries[a]["observed_station_months"]
            >= limits["minimum_station_months_per_analyte"],
            "largest_component": summaries[a]["active_largest_component_fraction"]
            >= limits["minimum_largest_component_fraction"],
        }
        for a in ANALYTES
    }
