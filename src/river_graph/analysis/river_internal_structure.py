"""Geometry-only internal river profiles and fixed-mean routing scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.analysis.river_routing_mechanisms import contract_paths, route_pulse

PROFILES = {
    0: "junction_free",
    1: "less_balanced_concentrated",
    2: "less_balanced_dispersed",
    3: "more_balanced_concentrated",
    4: "more_balanced_dispersed",
}


def path_distribution(area, distance):
    """Unique incremental areas; relative paths preserve weighted mean one."""
    a, d = np.asarray(area, float), np.asarray(distance, float)
    if a.ndim != 1 or a.shape != d.shape or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("aligned unique areas and paths required")
    keep = (a > 0) & np.isfinite(d) & (d >= 0)
    if not keep.any() or a.sum() <= 0:
        raise ValueError("positive reachable area required")
    w = a[keep]/a[keep].sum()
    mean = np.dot(w, d[keep])
    if mean <= 0:
        raise ValueError("positive mean path required")
    relative = d[keep]/mean
    cv = np.sqrt(np.sum(w*(relative-1)**2))
    return w, relative, {"path_mean_km": mean, "path_cv": cv,
        "represented_area_fraction": a[keep].sum()/a.sum(), "n_reachable_sources": int(keep.sum())}


def assign_profiles(frame, balance_cut, path_cut):
    f = frame.copy()
    if not np.isfinite(f[["tributary_balance", "path_cv"]].to_numpy()).all():
        raise ValueError("finite geometric descriptors required")
    if not np.isfinite([balance_cut, path_cut]).all() or min(balance_cut, path_cut) < 0:
        raise ValueError("finite nonnegative geometry cut points required")
    balanced = f.tributary_balance.ge(balance_cut)
    dispersed = f.path_cv.ge(path_cut)
    f["profile"] = 1+2*balanced.astype(int)+dispersed.astype(int)
    f.loc[f.n_junctions.eq(0), "profile"] = 0
    f["profile_name"] = f.profile.map(PROFILES)
    return f


def classify_geometry(frame):
    if frame.comid.duplicated().any():
        raise ValueError("one geometric vote per receiving COMID required")
    thresholds = {"balance_cut": float(frame.tributary_balance.median()),
                  "path_cut": float(frame.path_cv.median())}
    return assign_profiles(frame, **thresholds), thresholds


def representatives(frame):
    """Nearest to each group's geometric median; no DOC is accepted here."""
    columns = ("tributary_balance", "path_cv", "log_area")
    f = frame.copy()
    f["log_area"] = np.log1p(f.basin_area_km2)
    scale = f[list(columns)].std(ddof=0).to_numpy()
    scale = np.where(scale > 0, scale, 1.)
    rows = []
    for profile, group in f[f.profile.ne(0)].groupby("profile"):
        center = group[list(columns)].median().to_numpy()
        g = group.copy()
        g["representative_distance"] = np.linalg.norm((g[list(columns)].to_numpy()-center)/scale, axis=1)
        row = g.sort_values(["representative_distance", "station"]).iloc[0]
        rows.append({"profile": int(profile), "station": row.station, "comid": int(row.comid),
            "cluster": int(row.cluster), "representative_distance": row.representative_distance})
    return pd.DataFrame(rows)


def routing_scenarios(weights, paths):
    """Same input and mean delay; only the path spread contracts."""
    if not np.isclose(np.dot(weights, paths), 1., atol=1e-10):
        raise ValueError("relative path mean must be one")
    rows, traces = [], []
    for scenario, fraction in (("actual_spread", 1.), ("half_spread", .5), ("zero_spread", 0.)):
        delay = contract_paths(paths, weights, fraction)
        result, time, pulse = route_pulse(delay, weights)
        rows.append({"scenario": scenario, **result})
        traces.append(pd.DataFrame({"scenario": scenario, "relative_time": time, "outlet_anomaly": pulse}))
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True)
