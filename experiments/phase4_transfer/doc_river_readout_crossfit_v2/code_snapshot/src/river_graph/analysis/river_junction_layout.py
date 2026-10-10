"""Non-overlapping lateral tributaries and outlet-arrival variance budgets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from river_graph.analysis.river_routing_mechanisms import route_pulse


@dataclass
class JunctionLayout:
    """Source arrays remain aligned to the saved routed reaches."""

    mainstem_comids: np.ndarray
    source_units: np.ndarray
    source_area: np.ndarray
    source_distance: np.ndarray
    unit_means: np.ndarray
    units: pd.DataFrame
    junctions: pd.DataFrame
    descriptors: dict


@dataclass
class ConditionalRoutes:
    comids: np.ndarray
    successor: np.ndarray
    length: np.ndarray
    area: np.ndarray
    distance: np.ndarray


def condition_geometric_mainstem(paths, hydroseq, mainstem, primary, secondary):
    """Keep the original mapped trunk on real links, with off-trunk routes fixed.

    This is a separate routing sensitivity. It never overwrites saved shortest
    distances. Every forced trunk link must be an original primary/minor edge.
    """
    hydro = np.asarray(hydroseq, np.int64)
    main = np.asarray(mainstem, np.int64)
    primary, secondary = np.asarray(primary), np.asarray(secondary)
    lookup = {int(c): i for i, c in enumerate(paths.comids)}
    if len(np.unique(main)) != len(main) or not len(main) or set(main)-set(lookup):
        raise ValueError("unique mapped mainstem within routed network required")
    successor = paths.successor.copy()
    if successor[lookup[int(main[0])]] != -1:
        raise ValueError("mapped mainstem must start at receiving root")
    for down, up in pairwise(main):
        i, j = lookup[int(up)], lookup[int(down)]
        if hydro[j] not in (primary[i], secondary[i]) or hydro[j] >= hydro[i]:
            raise ValueError("mapped trunk link must be a real downstream primary/minor edge")
        successor[i] = j
    distance = np.empty(len(hydro))
    for i in np.argsort(hydro):
        j = successor[i]
        distance[i] = paths.length[i]/2 if j < 0 else distance[j]+(paths.length[i]+paths.length[j])/2
    if np.any(distance < paths.distance-1e-7):
        raise ValueError("conditioning cannot shorten a saved shortest route")
    return ConditionalRoutes(paths.comids, successor, paths.length, paths.area, distance)


def _moments(values, weights):
    weight = np.asarray(weights, float)
    weight = weight/weight.sum()
    value = np.asarray(values, float)
    mean = weight@value
    return float(mean), float(weight@((value-mean)**2))


def partition_mainstem(paths, hydroseq, arbolatesu, *, geometric_mainstem=None, prescribed_mainstem=None):
    """Contiguous route mainstem, first-attachment subtrees and exact moments.

    Mainstem parents are selected by cumulative upstream channel length, never
    DOC. Every incremental source is assigned exactly once. The entry coordinate
    of a lateral unit is the upstream end of its mainstem attachment reach;
    mainstem-local sources enter at their own reach midpoint.
    """
    cid = np.asarray(paths.comids, np.int64)
    successor = np.asarray(paths.successor, int)
    length = np.asarray(paths.length, float)
    area = np.asarray(paths.area, float)
    distance = np.asarray(paths.distance, float)
    hydro = np.asarray(hydroseq, np.int64)
    arbolate = np.asarray(arbolatesu, float)
    arrays = (successor, length, area, distance, hydro, arbolate)
    if cid.ndim != 1 or not len(cid) or any(x.shape != cid.shape for x in arrays):
        raise ValueError("aligned nonempty routed reach arrays required")
    if len(np.unique(cid)) != len(cid) or len(np.unique(hydro)) != len(cid):
        raise ValueError("unique reach identities and hydroseq required")
    if not np.isfinite(np.concatenate((length, area, distance, arbolate))).all():
        raise ValueError("finite geometry, incremental areas and arbolatesu required")
    if min(length.min(), area.min(), distance.min(), arbolate.min()) < 0 or area.sum() <= 0:
        raise ValueError("nonnegative geometry with positive incremental area required")
    roots = np.flatnonzero(successor == -1)
    if len(roots) != 1 or (successor < -1).any() or (successor >= len(cid)).any():
        raise ValueError("one receiving root and valid downstream indices required")
    parents = [[] for _ in cid]
    for i, j in enumerate(successor):
        if j >= 0:
            if hydro[j] >= hydro[i]:
                raise ValueError("successors must be downstream in hydroseq order")
            parents[j].append(i)
            if not np.isclose(distance[i]-distance[j], (length[i]+length[j])/2, atol=1e-7, rtol=0):
                raise ValueError("distances must reproduce the contiguous saved route")
    if prescribed_mainstem is None:
        main = [int(roots[0])]
        while parents[main[-1]]:
            main.append(min(parents[main[-1]], key=lambda i: (-arbolate[i], int(cid[i]))))
        main = np.asarray(main, int)
    else:
        supplied = np.asarray(prescribed_mainstem, np.int64)
        lookup = {int(c): i for i, c in enumerate(cid)}
        if not len(supplied) or set(supplied)-set(lookup) or len(np.unique(supplied)) != len(supplied):
            raise ValueError("unique prescribed mainstem in this network required")
        main = np.array([lookup[int(c)] for c in supplied])
        if main[0] != roots[0] or any(successor[up] != down for down, up in pairwise(main)):
            raise ValueError("prescribed mainstem must follow the selected contiguous downstream route")
    main_length = float(length[main].sum())
    if main_length <= 0 or not np.isclose(distance[main[-1]]+length[main[-1]]/2, main_length, atol=1e-7):
        raise ValueError("positive contiguous mainstem ending at the receiving outlet required")
    on_main = np.zeros(len(cid), bool)
    on_main[main] = True
    unit_root = np.full(len(cid), -1, int)
    attachment = np.full(len(cid), -1, int)
    entry_distance = np.zeros(len(cid))
    for i in np.argsort(hydro):
        j = successor[i]
        if on_main[i]:
            unit_root[i] = attachment[i] = i
            entry_distance[i] = distance[i]
        elif on_main[j]:
            unit_root[i], attachment[i] = i, j
            entry_distance[i] = distance[j]+length[j]/2
        else:
            unit_root[i], attachment[i] = unit_root[j], attachment[j]
            entry_distance[i] = entry_distance[j]
    if (unit_root < 0).any() or (attachment < 0).any():
        raise ValueError("every reach must attach to the routed mainstem")
    local_distance = distance-entry_distance
    if local_distance.min() < -1e-7:
        raise ValueError("within-tributary distance cannot be negative")
    root_ids, unit = np.unique(unit_root, return_inverse=True)
    unit_area = np.bincount(unit, weights=area, minlength=len(root_ids))
    sums = np.bincount(unit, weights=area*distance, minlength=len(root_ids))
    means = np.divide(sums, unit_area, out=np.zeros(len(root_ids)), where=unit_area > 0)
    residual = distance-means[unit]
    variance_sums = np.bincount(unit, weights=area*residual**2, minlength=len(root_ids))
    unit_variance = np.divide(variance_sums, unit_area, out=np.zeros(len(root_ids)), where=unit_area > 0)
    records = pd.DataFrame({
        "unit_comid": cid[root_ids], "unit_kind": np.where(on_main[root_ids], "mainstem_local", "lateral"),
        "entry_comid": cid[attachment[root_ids]], "entry_distance_km": entry_distance[root_ids],
        "entry_position": entry_distance[root_ids]/main_length,
        "area_km2": unit_area, "area_fraction": unit_area/area.sum(),
        "mean_path_km": means, "within_unit_variance_km2": unit_variance,
        "mean_lateral_path_km": np.where(unit_area > 0, means-entry_distance[root_ids], np.nan),
        "n_reaches": np.bincount(unit, minlength=len(root_ids)),
        "n_positive_area_sources": np.bincount(unit, weights=(area > 0).astype(int), minlength=len(root_ids)).astype(int),
    })
    lateral = records[records.unit_kind.eq("lateral") & records.area_km2.gt(0)].copy()
    junctions = lateral.groupby("entry_comid", as_index=False).agg(
        entry_distance_km=("entry_distance_km", "first"), entry_position=("entry_position", "first"),
        lateral_area_km2=("area_km2", "sum"), n_lateral_units=("unit_comid", "size"))
    junctions["lateral_area_fraction"] = junctions.lateral_area_km2/area.sum()
    junctions["lateral_only_weight"] = (junctions.lateral_area_km2/lateral.area_km2.sum()
                                       if len(lateral) else np.nan)
    junctions = junctions.sort_values("entry_position").reset_index(drop=True)
    weight = area/area.sum()
    mean, total_variance = _moments(distance, weight)
    within = float(weight@(residual**2))
    unit_weight = unit_area/area.sum()
    between = float(unit_weight@((means-mean)**2))
    entry = entry_distance[root_ids]
    branch = means-entry
    positive = unit_area > 0
    entry_mean, entry_var = _moments(entry[positive], unit_weight[positive])
    branch_mean, branch_var = _moments(branch[positive], unit_weight[positive])
    covariance = float(unit_weight[positive]@((entry[positive]-entry_mean)*(branch[positive]-branch_mean)))
    np.testing.assert_allclose(total_variance, within+between, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(between, entry_var+branch_var+2*covariance, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(unit_area.sum(), area.sum(), rtol=1e-12)
    denominator = total_variance if total_variance > 1e-14 else np.nan
    if len(lateral):
        lm, lv = _moments(lateral.entry_position, lateral.area_km2)
        near = lateral.loc[lateral.entry_position.lt(1/3), "area_km2"].sum()/lateral.area_km2.sum()
        upper = lateral.loc[lateral.entry_position.ge(2/3), "area_km2"].sum()/lateral.area_km2.sum()
        hhi = float((junctions.lateral_only_weight**2).sum())
    else:
        lm = lv = near = upper = hhi = np.nan
    descriptors = {
        "n_routed_reaches": len(cid), "incremental_area_km2": area.sum(),
        "n_route_mainstem_reaches": len(main), "route_mainstem_length_km": main_length,
        "n_lateral_units": len(lateral), "n_lateral_entry_junctions": len(junctions),
        "n_units": len(records), "n_positive_area_units": int(positive.sum()),
        "lateral_area_fraction": lateral.area_km2.sum()/area.sum(),
        "lateral_entry_mean": lm, "lateral_entry_sd": np.sqrt(lv),
        "lower_third_lateral_share": near, "middle_third_lateral_share": 1-near-upper,
        "upper_third_lateral_share": upper, "entry_area_concentration": hhi,
        "effective_area_weighted_entries": 1/hhi if hhi > 0 else np.nan,
        "mean_path_km": mean, "path_variance_km2": total_variance,
        "path_cv": np.sqrt(total_variance)/mean if mean > 0 else np.nan,
        "within_unit_variance_km2": within, "between_unit_variance_km2": between,
        "entry_trunk_variance_km2": entry_var, "mean_branch_variance_km2": branch_var,
        "twice_entry_branch_covariance_km2": 2*covariance,
        "within_unit_variance_fraction": within/denominator,
        "between_unit_variance_fraction": between/denominator,
        "entry_trunk_variance_fraction": entry_var/denominator,
        "mean_branch_variance_fraction": branch_var/denominator,
        "entry_branch_covariance_fraction": 2*covariance/denominator,
        "entry_branch_mean_correlation": covariance/np.sqrt(entry_var*branch_var)
            if entry_var*branch_var > 1e-14 else np.nan,
    }
    if geometric_mainstem is not None:
        original = np.asarray(geometric_mainstem, np.int64)
        common = np.intersect1d(original, cid[main])
        lookup = {int(c): i for i, c in enumerate(cid)}
        if len(np.unique(original)) != len(original) or set(original)-set(cid):
            raise ValueError("unique original mainstem reaches within this network required")
        broken = sum(successor[lookup[int(up)]] != lookup[int(dn)]
                     for dn, up in pairwise(original))
        descriptors.update({"original_mainstem_link_mismatches": broken,
            "original_mainstem_reaches": len(original), "mainstem_reach_jaccard":
                len(common)/len(np.union1d(original, cid[main])),
            "mainstem_original_overlap_fraction": len(common)/len(original)})
    return JunctionLayout(cid[main], unit, area, distance, means, records, junctions, descriptors)


def arrival_scenarios(layout, *, dt=.005, sigma=.15, keep_curves=False):
    """Remove within- or between-unit dispersion; common translation control."""
    positive = layout.source_area > 0
    weights = layout.source_area[positive]/layout.source_area[positive].sum()
    mean = float(weights@layout.source_distance[positive])
    if mean <= 0:
        raise ValueError("positive area-weighted path mean required")
    actual = layout.source_distance[positive]/mean
    unit_mean = layout.unit_means[layout.source_units[positive]]/mean
    latest = float(unit_mean.max())
    waits = np.maximum(0., latest-unit_mean)
    rows, curves = [], []
    for name, delay in (("actual_paths", actual), ("within_unit_collapsed", unit_mean),
                        ("unit_means_aligned", actual+waits)):
        result, time, pulse = route_pulse(delay, weights, dt=dt, sigma=sigma)
        cdf = np.cumsum(pulse)/pulse.sum()
        t10, t90 = np.interp([.1, .9], cdf, time)
        result.update(scenario=name, duration_80=t90-t10, input_sd=sigma,
                      analytic_delay_variance=weights@((delay-weights@delay)**2))
        rows.append(result)
        if keep_curves:
            curves.append(pd.DataFrame({"scenario": name, "relative_time": time,
                "centered_time": time-result["pulse_centroid"], "outlet_anomaly": pulse}))
    original = rows[0]
    translated = {**original, "scenario": "common_translation",
                  "pulse_centroid": latest, "mean_travel_delay": latest,
                  "pulse_peak_time": original["pulse_peak_time"]+latest-1.}
    rows.append(translated)
    if keep_curves:
        f = curves[0].copy()
        f["scenario"] = "common_translation"
        f["relative_time"] += latest-1.
        curves.append(f)
    table = pd.DataFrame(rows)
    table["peak_change_vs_actual_pct"] = 100*(table.pulse_peak/original["pulse_peak"]-1)
    table["duration_change_vs_actual_pct"] = 100*(table.duration_80/original["duration_80"]-1)
    np.testing.assert_allclose(table.anomaly_mass_fraction, 1., atol=1e-10)
    np.testing.assert_allclose(table.retained_fraction, 1., atol=1e-12)
    return table, pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
