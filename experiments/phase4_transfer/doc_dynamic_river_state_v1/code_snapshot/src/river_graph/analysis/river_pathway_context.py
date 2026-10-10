"""Whole form, cropped monitored corridors and receiving signal budgets."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from river_graph.analysis.river_monitored_arrivals import _project


def corridor_geometry(paths, gauges, receiver_comid, receiver_measure, waterbody_types,
                      waterbody_ids, layout):
    """Crop gauge endpoints; count the shared suffix once in corridor tables."""
    if (len(gauges) < 2 or gauges.comid.duplicated().any()
            or gauges.station.duplicated().any()):
        raise ValueError("at least two distinct disjoint upstream gauges required")
    types, wb = np.asarray(waterbody_types), np.asarray(waterbody_ids)
    if types.shape != paths.comids.shape or wb.shape != paths.comids.shape:
        raise ValueError("waterbody fields must align with routed reaches")
    storage = np.isin(types, ["LakePond", "Reservoir"]) & (wb > 0)
    lookup = {int(c): i for i, c in enumerate(paths.comids)}
    receiver = lookup[int(receiver_comid)]
    if paths.successor[receiver] != -1:
        raise ValueError("receiver must be the routed root")
    if not np.isfinite(receiver_measure) or not 0 <= receiver_measure <= 100:
        raise ValueError("receiver measure must lie in [0,100]")
    weights = gauges.area_weight.to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("positive finite source weights required")
    weights = weights/weights.sum()
    routes, rows = [], []
    for g in gauges.itertuples():
        measure = float(g.measure)
        if not np.isfinite(measure) or not 0 <= measure <= 100:
            raise ValueError("gauge measure must lie in [0,100]")
        i = lookup[int(g.comid)]
        route, seen = [], set()
        while i not in seen:
            seen.add(i)
            route.append(i)
            if i == receiver:
                break
            i = int(paths.successor[i])
            if i < 0:
                raise ValueError("source route does not reach receiving station")
        if route[-1] != receiver or route[0] == receiver:
            raise ValueError("distinct upstream source and acyclic connected route required")
        routes.append(route)
        for sequence, j in enumerate(route):
            start = measure if sequence == 0 else 100.
            end = receiver_measure if j == receiver else 0.
            length = float(paths.length[j]*(start-end)/100)
            if length < -1e-10:
                raise ValueError("cropped route length cannot be negative")
            rows.append({"source_station": g.station, "source_order": g.source_order,
                "comid": int(paths.comids[j]), "sequence": sequence,
                "start_measure": start, "end_measure": end, "length_km": max(length, 0.),
                "mapped_waterbody": bool(storage[j]), "waterbody_id": int(wb[j]) if storage[j] else 0,
                "area_weight": float(g.area_weight)})
    common_set = set.intersection(*(set(r) for r in routes))
    common = [i for i in routes[0] if i in common_set]
    for route in routes:
        if route[route.index(common[0]):] != common:
            raise ValueError("all source paths must share the same contiguous suffix")
    common_ids = {int(paths.comids[i]) for i in common}
    f = pd.DataFrame(rows)
    f["shared_by_all_sources"] = f.comid.isin(common_ids)
    lengths = f.groupby("source_station", sort=False).length_km.sum().reindex(gauges.station).to_numpy()
    storage_lengths = (f[f.mapped_waterbody].groupby("source_station").length_km.sum()
                       .reindex(gauges.station, fill_value=0).to_numpy())
    shared = f[f.source_station.eq(gauges.station.iloc[0]) & f.shared_by_all_sources]
    common_length = shared.length_km.sum()
    mean = float(weights@lengths)
    if mean <= 0 or common_length > lengths.min()+1e-8:
        raise ValueError("positive source paths must include the common suffix")
    main_lookup = {int(c): i for i, c in enumerate(layout.mainstem_comids)}
    entry, branch = [], []
    for route, length in zip(routes, lengths, strict=True):
        first = next(i for i in route if int(paths.comids[i]) in main_lookup)
        g = f[f.source_station.eq(gauges.station.iloc[len(entry)])]
        entry_length = float(g[g.sequence.ge(route.index(first))].length_km.sum())
        entry.append(entry_length)
        branch.append(length-entry_length)
    entry, branch = np.asarray(entry), np.asarray(branch)
    em, bm = float(weights@entry), float(weights@branch)
    evar, bvar = float(weights@((entry-em)**2)), float(weights@((branch-bm)**2))
    covariance = float(weights@((entry-em)*(branch-bm)))
    variance = float(weights@((lengths-mean)**2))
    np.testing.assert_allclose(variance, evar+bvar+2*covariance, atol=1e-7, rtol=1e-10)
    covered = f[f.length_km.gt(1e-10) & f.mapped_waterbody]
    return {"monitored_path_mean_km": mean,
        "monitored_path_cv": np.sqrt(variance)/mean,
        "monitored_common_km": float(common_length),
        "monitored_common_fraction": float(common_length/mean),
        "monitored_storage_length_fraction": float(weights@storage_lengths/mean),
        "monitored_storage_source_share": float(weights@(storage_lengths > 1e-10)),
        "n_monitored_corridor_waterbodies": int(covered.waterbody_id.nunique()),
        "n_whole_mapped_waterbodies": len(np.unique(wb[storage])),
        "monitored_entry_variance_km2": evar, "monitored_branch_variance_km2": bvar,
        "monitored_twice_entry_branch_covariance_km2": 2*covariance,
        "monitored_path_variance_km2": variance}, f


def variance_budget(sources, receiver, weights, dates):
    """Receiving = mixture + mismatch, all projected on one calendar design.

    Varying weights multiply native concentrations BEFORE projection. This
    signal identity is not DOC mass balance or a fitted physical source model.
    """
    a, y, w = np.asarray(sources, float), np.asarray(receiver, float), np.asarray(weights, float)
    if a.ndim != 2 or a.shape[0] != len(y) or a.shape[1] < 2 or len(y) < 12:
        raise ValueError("at least twelve common dates and two source columns required")
    if w.shape == (a.shape[1],):
        w = np.broadcast_to(w, a.shape)
    if (w.shape != a.shape or not np.isfinite(np.r_[a.ravel(), y, w.ravel()]).all()
            or min(a.min(), y.min()) < 0 or (w < 0).any() or (w.sum(axis=1) <= 0).any()):
        raise ValueError("finite nonnegative DOC and valid source shares required")
    if len(pd.DatetimeIndex(dates).unique()) != len(y):
        raise ValueError("unique common dates required")
    w = w/w.sum(axis=1, keepdims=True)
    mix = (a*w).sum(axis=1)
    residual = _project(np.column_stack([mix, y]), dates)
    m, r = residual[:, 0], residual[:, 1]
    mismatch = r-m
    vm, vy, vd = np.mean(m*m), np.mean(r*r), np.mean(mismatch*mismatch)
    cross = float(2*np.mean(m*mismatch))
    np.testing.assert_allclose(vy, vm+vd+cross, rtol=1e-10, atol=1e-10)
    correlation = float(np.mean(m*r)/np.sqrt(vm*vy)) if min(vm, vy) > 1e-12 else np.nan
    beta = float(np.mean(m*r)/vm) if vm > 1e-12 else np.nan
    out = {"n_months": len(y), "outlet_variance": vy, "mixture_variance": vm,
        "mismatch_variance": vd, "twice_mix_mismatch_covariance": cross,
        "mixture_variance_share": vm/vy if vy > 1e-12 else np.nan,
        "mismatch_variance_share": vd/vy if vy > 1e-12 else np.nan,
        "covariance_variance_share": cross/vy if vy > 1e-12 else np.nan,
        "outlet_mix_correlation": correlation,
        "outlet_mix_log_sd_ratio": .5*np.log(vy/vm) if min(vm, vy) > 1e-12 else np.nan,
        "mixture_amplitude_beta": beta,
        "unexplained_outlet_variance_fraction": 1-correlation**2 if np.isfinite(correlation) else np.nan}
    series = pd.DataFrame({"date": pd.DatetimeIndex(dates), "doc_mixture": mix,
        "doc_receiver": y, "mixture_anomaly": m, "receiver_anomaly": r,
        "mismatch_anomaly": mismatch})
    return out, series


def comparison_opportunities(inventory, *, area_ratio=2., coverage=.8):
    """Retain every same-region form pairing; no DOC response enters selection."""
    f = inventory[inventory.physical_receiver_representative].copy()
    if f.comid.duplicated().any() or f.station.duplicated().any():
        raise ValueError("one physical receiver representative required")
    rows = []
    for huc, group in f.groupby("huc4", sort=True):
        for a, b in product(group[group.cluster.eq(1)].itertuples(), group[group.cluster.eq(3)].itertuples()):
            ratio = max(a.basin_area_km2, b.basin_area_km2)/min(a.basin_area_km2, b.basin_area_km2)
            available = a.status == b.status == "included"
            covered = available and min(a.covered_area_fraction, b.covered_area_fraction) >= coverage
            rows.append({"huc4": huc, "elongated_station": a.station, "broad_station": b.station,
                "area_ratio": ratio, "area_comparable": ratio <= area_ratio,
                "both_observed": available, "both_high_coverage": covered,
                "elongated_coverage": a.covered_area_fraction, "broad_coverage": b.covered_area_fraction,
                "elongated_status": a.status, "broad_status": b.status,
                "usable_form_pair": ratio <= area_ratio and covered})
    return pd.DataFrame(rows, columns=["huc4", "elongated_station", "broad_station",
        "area_ratio", "area_comparable", "both_observed", "both_high_coverage",
        "elongated_coverage", "broad_coverage", "elongated_status", "broad_status", "usable_form_pair"])
