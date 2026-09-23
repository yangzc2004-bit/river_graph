"""Phase-3 uncertainty product core (docs/paper/phase3_uncertainty_spec.md).

Implements the frozen spec: scenario visibility sets, the three ecological
covariates (visible-set only), the scaled-quantile central 90% interval in
log1p space ("empirical validation calibration" — never a conformal
guarantee), full-grid product writing with bound provenance hashes, and
hidden-test coverage evaluation with station-clustered bootstrap.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SPEC_PATH = Path("docs/paper/phase3_uncertainty_spec.md")
OUT_ROOT = Path("experiments/phase3_uncertainty_stcore_v1")
SEEDS = (42, 43, 44, 45, 46)
TOOLS = ("H2X", "H2X_nomsg", "eco_RF")
# Spec §1: which split cells' labels are visible at full-grid inference.
SCENARIO_VISIBILITY = {
    "E1": ("train",),
    "E2a": ("train",),
    "E2b": ("train", "context"),
    "E3": ("train",),
}


def family_of(mask: str) -> str:
    if mask.startswith("e1"):
        return "E1"
    if mask.startswith("e2a"):
        return "E2a"
    if mask.startswith("e2b"):
        return "E2b"
    return "E3"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat_cells(split: dict, keys) -> np.ndarray:
    parts = [np.asarray(split[k], dtype=np.int64) for k in keys
             if k in split and len(split[k])]
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def scenario_visibility(split: dict, family: str) -> np.ndarray:
    """Spec §1 inference-time visible label cells (flat indices)."""
    return flat_cells(split, SCENARIO_VISIBILITY[family])


def calibration_cells(split: dict, family: str) -> np.ndarray:
    """Spec §3.4 calibration cells: visible set + val (+ context for E2b)."""
    keys = set(SCENARIO_VISIBILITY[family]) | {"val"}
    return flat_cells(split, keys)


def visibility_role(split: dict, n: int, t: int) -> np.ndarray:
    """(N*T,) role label per cell: train/val/test/context/unobserved."""
    role = np.full(n * t, "unobserved", dtype=object)
    for key in ("train", "val", "test", "context"):
        if key in split and len(split[key]):
            role[np.asarray(split[key], dtype=np.int64)] = key
    return role


def _undirected_adj(n: int, edge_index: np.ndarray) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for a, b in np.asarray(edge_index).T.tolist():
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _hop_ball(adj: list[set[int]], src: int, radius: int) -> set[int]:
    seen = {src}
    frontier = [src]
    for _ in range(radius):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return seen


def covariates(dataset: dict, visible_flat: np.ndarray) -> dict[str, np.ndarray]:
    """Spec §4 covariates, scenario-conditional on the visible label set.

    support_count: visible DOC label cells at stations within <=2 undirected
    hops (self included), counted over all months.
    network_distance: undirected hops to the nearest station with >=1 visible
    label (0 = self has one; NaN only if none anywhere).
    ecological_novelty: mean Euclidean distance to the 5 nearest visible
    stations in 13-dim regime space standardized ON VISIBLE STATIONS ONLY;
    a station is excluded from its own neighbours.
    """
    y_mask = np.asarray(dataset["y_mask"])
    n, t = y_mask.shape
    ei = np.asarray(dataset["edge_index"])
    adj = _undirected_adj(n, ei)
    vis_cell = np.zeros(n * t, dtype=bool)
    if len(visible_flat):
        vis_cell[visible_flat] = True
    vis_cell = vis_cell.reshape(n, t)
    vis_per_station = vis_cell.sum(axis=1)  # visible label cells per station
    vis_stations = np.flatnonzero(vis_per_station > 0)

    support = np.zeros(n, dtype=float)
    net_dist = np.full(n, np.nan, dtype=float)
    for i in range(n):
        ball = _hop_ball(adj, i, 2)
        support[i] = float(sum(vis_per_station[j] for j in ball))
    if len(vis_stations):
        # BFS from all visible stations simultaneously (multi-source)
        dist = np.full(n, np.inf)
        frontier = [int(s) for s in vis_stations]
        for s in frontier:
            dist[s] = 0.0
        d = 0.0
        while frontier:
            d += 1.0
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[v] > d:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
        net_dist = np.where(np.isfinite(dist), dist, np.nan)

    regime = np.asarray(dataset["regime"], dtype=float)
    novelty = np.full(n, np.nan, dtype=float)
    if len(vis_stations):
        ref = regime[vis_stations]
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0) + 1e-8
        z_all = (regime - mu) / sd
        z_ref = (ref - mu) / sd
        for i in range(n):
            d = np.linalg.norm(z_ref - z_all[i], axis=1)
            if i in set(vis_stations.tolist()):
                d[vis_stations.tolist().index(i)] = np.inf
            order = np.sort(d[np.isfinite(d)])
            if len(order):
                novelty[i] = float(order[: min(5, len(order))].mean())
    return {
        "support_count": support,
        "network_distance": net_dist,
        "ecological_novelty": novelty,
    }


def calibrate_interval(
    seed_log1p: np.ndarray,
    y_log1p: np.ndarray,
    cal_flat: np.ndarray,
) -> dict:
    """Spec §3: scaled-quantile central 90% interval, log1p calibration space.

    Returns prediction_median/pi_* in mg/L and `uncertainty` as the calibrated
    log1p half-width. Empirical validation calibration — NOT conformal.
    """
    median = np.median(seed_log1p, axis=0)
    spread = seed_log1p.std(axis=0)
    # floor = 10th percentile of spreads (spec §3.2), with an absolute
    # epsilon guard against degenerate all-ties spreads (not a tunable)
    s_floor = float(max(np.quantile(spread, 0.10), 1e-3))
    s_eff = np.maximum(spread, s_floor)
    cal = np.asarray(cal_flat, dtype=np.int64)
    scores = (np.abs(y_log1p.reshape(-1)[cal] - median.reshape(-1)[cal])
              / s_eff.reshape(-1)[cal])
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration cells")
    level = min(np.ceil((n + 1) * 0.9) / n, 1.0)
    qhat = float(np.quantile(scores, level, method="higher"))
    half = qhat * s_eff
    return {
        "prediction_median": np.expm1(median),
        "pi_lower": np.expm1(np.maximum(median - half, 0.0)),
        "pi_upper": np.expm1(median + half),
        "uncertainty": half,
        "qhat": qhat,
        "s_floor": s_floor,
        "n_cal": n,
    }


def coverage_with_bootstrap(
    y_mg_l: np.ndarray,
    pi_lower: np.ndarray,
    pi_upper: np.ndarray,
    test_flat: np.ndarray,
    stations: list[str],
    t_len: int,
    draws: int = 2000,
    seed: int = 0,
) -> dict:
    """Hidden-test coverage + station-clustered bootstrap 95% CI."""
    te = np.asarray(test_flat, dtype=np.int64)
    y = y_mg_l.reshape(-1)[te]
    lo = pi_lower.reshape(-1)[te]
    hi = pi_upper.reshape(-1)[te]
    hit = ((y >= lo) & (y <= hi)).astype(float)
    st = np.asarray([stations[i // t_len] for i in te])
    uniq = np.unique(st)
    idx = {s: np.flatnonzero(st == s) for s in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(draws)
    for i in range(draws):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in picked])
        stats[i] = float(hit[sel].mean())
    return {
        "coverage": float(hit.mean()),
        "ci95_lo": float(np.percentile(stats, 2.5)),
        "ci95_hi": float(np.percentile(stats, 97.5)),
        "n_test": len(te),
    }


def monotonicity_check(
    uncertainty: np.ndarray,
    covar: np.ndarray,
    direction: str,
    test_flat: np.ndarray,
    vis_stations: np.ndarray,
    n: int,
    t_len: int,
) -> dict:
    """Spec §5 monotonicity: train-derived tertiles, top-vs-bottom direction.

    ``direction`` = "up" (uncertainty should grow with the covariate) or
    "down". Bin edges come from the covariate at VISIBLE stations only.
    """
    ref = covar[vis_stations]
    ref = ref[np.isfinite(ref)]
    if len(ref) < 9:
        return {"ok": None, "reason": "too few visible stations for tertiles"}
    edges = np.quantile(ref, [1 / 3, 2 / 3])
    te = np.asarray(test_flat, dtype=np.int64)
    c = covar[te // t_len]
    u = uncertainty.reshape(-1)[te]
    lo = u[c <= edges[0]]
    hi = u[c > edges[1]]
    if not len(lo) or not len(hi):
        return {"ok": None, "reason": "empty tertile on test cells"}
    bottom, top = float(np.mean(lo)), float(np.mean(hi))
    overall = float(np.mean(u))
    gap = abs(top - bottom)
    if direction == "up":
        ok = top >= bottom
    else:
        ok = top <= bottom
    strongly_reversed = (not ok) and gap >= 0.10 * max(overall, 1e-12)
    return {
        "ok": bool(ok and not strongly_reversed),
        "direction_correct": bool(ok),
        "strongly_reversed": bool(strongly_reversed),
        "bottom_mean": bottom, "top_mean": top, "edges": edges.tolist(),
    }


def product_hashes(dataset_path: str, mask_path: str) -> dict:
    return {
        "dataset_sha256": sha256_file(dataset_path),
        "mask_sha256": sha256_file(mask_path),
        "phase3_spec_sha256": sha256_file(SPEC_PATH),
        "runtime_code_snapshot_sha256": __import__(
            "river_graph.experiments.provenance",
            fromlist=["runtime_code_snapshot_sha256"],
        ).runtime_code_snapshot_sha256(),
    }


def write_product(path: Path, frame_rows: dict, provenance: dict) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(frame_rows)
    df["provenance_hashes"] = json.dumps(provenance, sort_keys=True)
    df.to_parquet(path, index=False)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "provenance_hashes": provenance,
        "calibration": "empirical validation calibration (not conformal)",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")
