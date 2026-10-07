"""Observed tributary integration and longitudinal DOC pathway diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def permitted_doc(dataset, cells):
    truth = np.asarray(dataset["y"])
    cells = np.asarray(cells)
    if (cells.ndim != 1 or cells.dtype.kind not in "iu"
            or len(np.unique(cells)) != len(cells)
            or (cells < 0).any() or (cells >= truth.size).any()):
        raise ValueError("unique, in-range source cells required")
    result = np.full(truth.shape, np.nan, dtype=float)
    result.ravel()[cells] = truth.ravel()[cells]
    result[result < 0] = np.nan
    return result


def branch_status(comid_a, comid_b, members_a, members_b, area_lookup):
    a, b = np.unique(members_a), np.unique(members_b)
    shared = np.intersect1d(a, b, assume_unique=True)
    def area(ids):
        if isinstance(area_lookup, pd.Series):
            values = area_lookup.reindex(ids).to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("finite nonnegative unique catchment area required")
            return values.sum()
        return sum(area_lookup[int(c)] for c in ids)
    area_a, area_b = area(a), area(b)
    if min(area_a, area_b) <= 0:
        raise ValueError("positive source drainage areas required")
    overlap = area(shared)/min(area_a, area_b)
    nested = comid_a in b or comid_b in a
    status = "nested" if nested else "shared_catchment" if overlap > .01 else "independent"
    return {"branch_status": status, "source_a_area_km2": area_a,
            "source_b_area_km2": area_b, "area_overlap_fraction": overlap,
            "reach_overlap_fraction": len(shared)/min(len(a), len(b))}


def gauge_mapping_status(mapped_area, reported_area):
    if not np.isfinite(reported_area) or reported_area <= 0:
        return "official_area_missing"
    if not np.isfinite(mapped_area) or mapped_area <= 0:
        raise ValueError("mapped drainage must be positive")
    return "gross_area_mismatch" if not .5 <= mapped_area/reported_area <= 2. else "area_consistent"


def drainage_order_consistent(source_a_area, source_b_area, receiver_area):
    if not np.isfinite(source_a_area+source_b_area+receiver_area):
        return True  # Unknown is separately flagged, not verified.
    return source_a_area+source_b_area <= 1.1*receiver_area


def concentration_mix(a, b, weight_a, weight_b):
    a, b, qa, qb = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in (a, b, weight_a, weight_b)])
    good = np.isfinite(a+b+qa+qb) & (a >= 0) & (b >= 0) & (qa > 0) & (qb > 0)
    out = np.full(a.shape, np.nan)
    np.divide(a*qa+b*qb, qa+qb, out=out, where=good)
    return out


def detrend(values, dates, hydro=None):
    """Calendar harmonics + linear year; optional measured local hydro."""
    values = np.asarray(values, float)
    dates = pd.DatetimeIndex(dates)
    angle = 2*np.pi*dates.month.to_numpy()/12
    year = dates.year.to_numpy(float)
    columns = [np.ones(len(values)), np.sin(angle), np.cos(angle), year-year.mean()]
    if hydro is not None:
        columns.extend(np.asarray(hydro).T)
    design = np.column_stack(columns)
    if not np.isfinite(values).all() or not np.isfinite(design).all():
        raise ValueError("finite common population required for detrending")
    return values-design@np.linalg.lstsq(design, values, rcond=None)[0]


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(spearmanr(a, b).statistic) if np.std(a) > 1e-10 and np.std(b) > 1e-10 else np.nan


def mixing_summary(frame, *, weights, dates):
    """Paired summaries; the reference never chooses the better observed source."""
    a, b, target = (frame[c].to_numpy(float) for c in ("doc_a", "doc_b", "doc_target"))
    qa, qb = weights
    mix = concentration_mix(a, b, qa, qb)
    if not np.isfinite(mix).all():
        raise ValueError("positive weights and common finite DOC required")
    reference = (abs(a-target)+abs(b-target))/2
    cv = lambda x: np.std(x)/np.mean(x) if np.mean(x) > 0 else np.nan
    source_cv = (cv(a)+cv(b))/2
    seasonal = [detrend(np.log1p(v), dates) for v in (a, b, target, mix)]
    return {"n_records": len(frame), "n_months": len(pd.DatetimeIndex(dates).to_period("M").unique()),
            "reference_mae": reference.mean(), "mixture_mae": np.mean(abs(target-mix)),
            "unweighted_mix_mae": np.mean(abs(target-(a+b)/2)),
            "signed_departure_mg_L": np.mean(target-mix),
            "source_cv": source_cv, "downstream_cv": cv(target), "mixture_cv": cv(mix),
            "downstream_cv_change": cv(target)-source_cv,
            "branch_rho": correlation(seasonal[0], seasonal[1]),
            "mixture_downstream_rho": correlation(seasonal[3], seasonal[2]),
            "within_source_range_fraction": np.mean((target >= np.minimum(a, b)) & (target <= np.maximum(a, b)))}


def receiver_summary(cases):
    metrics = ["reference_mae", "mixture_mae", "unweighted_mix_mae", "signed_departure_mg_L", "source_cv",
               "downstream_cv", "mixture_cv", "downstream_cv_change", "branch_rho",
               "mixture_downstream_rho", "within_source_range_fraction"]
    keys = ["population", "weighting", "target", "huc4"]
    out = cases.groupby(keys, dropna=False, as_index=False)[metrics].mean()
    counts = cases.groupby(keys, dropna=False, as_index=False).agg(
        n_combinations=("pair_id", "size"), cluster=("cluster", "first"))
    if "component" in cases:
        component = cases.groupby(keys, dropna=False, as_index=False).component.first()
        counts = counts.merge(component, on=keys, validate="one_to_one")
    return out.merge(counts, on=keys, validate="one_to_one")


def shared_station_components(connections):
    """Connected observation sets, including shared source stations."""
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    for stations in connections:
        for station in stations[1:]:
            parents[find(station)] = find(stations[0])
    roots = sorted({find(x) for x in parents})
    labels = {root: i for i, root in enumerate(roots)}
    return {x: labels[find(x)] for x in parents}


def bootstrap_receivers(receivers, draws=5000):
    rows = []
    for (population, weighting), f in receivers.groupby(["population", "weighting"]):
        for group in ("all", "class_1", "class_2", "class_3"):
            s = f if group == "all" else f[f.cluster.eq(int(group[-1]))]
            if s.empty:
                continue
            for unit in ("target", "huc4", *(["component"] if "component" in s else [])):
                _, inverse = np.unique(s[unit].to_numpy(), return_inverse=True)
                n = inverse.max()+1
                rng = np.random.default_rng(42)
                weights = rng.multinomial(n, np.full(n, 1/n), size=draws)[:, inverse]
                for metric in ("mae_gain_pct", "weighting_gain_pct", "downstream_cv_change", "branch_rho",
                               "signed_departure_mg_L", "within_source_range_fraction"):
                    if metric in ("mae_gain_pct", "weighting_gain_pct"):
                        base = s.reference_mae.to_numpy() if metric == "mae_gain_pct" else s.unweighted_mix_mae.to_numpy()
                        candidate = s.mixture_mae.to_numpy()
                        point = 100*(1-candidate.mean()/base.mean()) if base.mean() > 0 else np.nan
                        denominator = weights@base
                        ratio = np.divide(weights@candidate, denominator, out=np.full(draws, np.nan), where=denominator > 0)
                        boot = 100*(1-ratio)
                    else:
                        values = s[metric].to_numpy()
                        valid = np.isfinite(values)
                        if not valid.any():
                            continue
                        point = np.mean(values[valid])
                        numerator = weights[:, valid]@values[valid]
                        denominator = weights[:, valid].sum(axis=1)
                        boot = np.divide(numerator, denominator, out=np.full(draws, np.nan), where=denominator > 0)
                    finite = boot[np.isfinite(boot)]
                    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) and n >= 2 else (np.nan, np.nan)
                    rows.append({"population": population, "weighting": weighting, "group": group,
                                 "metric": metric, "estimate": point, "ci_low": lo, "ci_high": hi,
                                 "resampling_unit": unit, "n_receivers": len(s), "n_huc4": s.huc4.nunique(),
                                 "n_components": s.component.nunique() if "component" in s else np.nan,
                                 "few_receivers": len(s) < 10, "few_regions": s.huc4.nunique() < 10,
                                 "interval_identifiable": n >= 2, "valid_bootstrap_draws": len(finite) if n >= 2 else 0})
    return pd.DataFrame(rows)


def covariance_identity(a, b, weight=.5):
    """Conservative fixed-weight identity, not an independent scientific test."""
    cov = np.cov(a, b, ddof=0)
    expected = weight**2*cov[0, 0]+(1-weight)**2*cov[1, 1]+2*weight*(1-weight)*cov[0, 1]
    return float(expected), float(np.var(weight*np.asarray(a)+(1-weight)*np.asarray(b)))


def simulate_networks(steps=480, branches=8):
    """Equal sources/flow/channel length; topology alters routing and path delay.

    Each reach is one synthetic time step. Distinct sources enter distinct nodes;
    mixing is flow weighted and removal acts on loads while water is conserved.
    """
    if branches != 8 or steps < 60:
        raise ValueError("fixed eight-source demonstration requires >=60 steps")
    # Each topology has exactly 14 reaches and eight equal-flow source inputs.
    networks = {
        "chain": (list(range(1, 14))+[-1], [*range(0, 14, 2), 13]),
        "balanced": ([8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, -1, 12], list(range(8))),
        "elongated": ([8, 8, 9, 9, 10, 10, 11, 11, 9, 10, 11, 12, 13, -1], list(range(8))),
    }
    # Balanced topology has two symmetric child trees ending at node 12.
    # Avoid pretending node count alone fixes hydraulic volume or travel time.
    t = np.arange(steps)
    rows, summaries = [], []
    for forcing in ("synchronous", "asynchronous"):
        phase = np.zeros(8) if forcing == "synchronous" else np.arange(8)*2*np.pi/8
        source = 5+2*np.sin(2*np.pi*t[:, None]/48+phase[None, :])
        source += np.exp(-((t[:, None]-240-phase[None, :]*4)/7)**2)*4
        for name, (downstream, source_nodes) in networks.items():
            source_nodes = np.asarray(source_nodes)
            if len(source_nodes) != 8 or len(downstream) != 14:
                raise AssertionError("fixed source/reach budget")
            # Chain: 8 sources at nodes 0,2,..12 and13 (the outlet).
            for removal in (0., .04):
                water, load = np.zeros((steps, 14)), np.zeros((steps, 14))
                paths = []
                for node in source_nodes:
                    visited, cur = set(), int(node)
                    while cur >= 0:
                        if cur in visited:
                            raise AssertionError("simulation cycle")
                        visited.add(cur)
                        cur = downstream[cur]
                    paths.append(len(visited))
                for ti in range(steps):
                    water[ti, source_nodes] += 1/8
                    load[ti, source_nodes] += source[ti]/8
                    for node, target in enumerate(downstream):
                        load[ti, node] *= np.exp(-removal)
                        if target >= 0 and ti+1 < steps:
                            water[ti+1, target] += water[ti, node]
                            load[ti+1, target] += load[ti, node]
                outlet = downstream.index(-1)
                valid = np.arange(steps) >= max(paths)
                concentration = load[valid, outlet]/water[valid, outlet]
                for ti, value in zip(t[valid], concentration, strict=True):
                    rows.append({"topology": name, "forcing": forcing, "removal_per_step": removal,
                                 "step": int(ti), "doc_normalized": value, "outlet_flow": water[ti, outlet]})
                summaries.append({"topology": name, "forcing": forcing, "removal_per_step": removal,
                                  "n_sources": 8, "n_reaches": 14, "input_flow": 1.,
                                  "mean_path_steps": np.mean(paths), "max_path_steps": max(paths),
                                  "mean_doc": concentration.mean(), "doc_cv": concentration.std()/concentration.mean(),
                                  "outlet_flow_min": water[valid, outlet].min(),
                                  "outlet_flow_max": water[valid, outlet].max()})
    return pd.DataFrame(rows), pd.DataFrame(summaries)
