"""Whole-network monitored frontiers and observed multi-source DOC coherence."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def availability_ok(mask, dates, minimum=24):
    selected = pd.DatetimeIndex(dates)[np.asarray(mask, bool)]
    return (len(selected) >= minimum and selected.year.nunique() >= 3
            and selected.month.nunique() >= 6)


def monitored_frontier(paths, hydroseq, gauges):
    """Nearest downstream eligible gauges give disjoint upstream catchments.

    Gauges must have unique COMIDs and exclude the receiving reach. The caller
    chooses aliases and availability using metadata only. Area is unique local
    catchment area, including the complete gauge reach.
    """
    hydro = np.asarray(hydroseq, np.int64)
    if hydro.shape != paths.comids.shape or len(np.unique(hydro)) != len(hydro):
        raise ValueError("aligned unique hydroseq required")
    if gauges.comid.duplicated().any() or set(gauges.comid)-set(paths.comids):
        raise ValueError("unique mapped candidate gauge COMIDs required")
    lookup = {int(c): i for i, c in enumerate(paths.comids)}
    area = paths.area.copy()
    for i in np.argsort(hydro)[::-1]:
        j = paths.successor[i]
        if j >= 0:
            if hydro[j] >= hydro[i]:
                raise ValueError("acyclic downstream routing required")
            area[j] += area[i]
    candidate = np.zeros(len(hydro), bool)
    for c in gauges.comid:
        i = lookup[int(c)]
        if paths.successor[i] < 0:
            raise ValueError("receiving reach cannot be an upstream gauge")
        candidate[i] = True
    below = np.full(len(hydro), -1, int)
    for i in np.argsort(hydro):
        j = paths.successor[i]
        if j >= 0:
            below[i] = j if candidate[j] else below[j]
    out = gauges.copy()
    positions = np.array([lookup[int(c)] for c in out.comid], int)
    out["routed_area_km2"] = area[positions]
    out["frontier"] = below[positions] < 0
    out["downstream_candidate_comid"] = [int(paths.comids[j]) if j >= 0 else -1 for j in below[positions]]
    return out


def select_source_set(frontier, observed, receiving_observed, dates, receiver_area=np.nan):
    """Largest-area eligible pair followed by coverage-ordered greedy additions."""
    f = frontier[frontier.frontier & frontier.routed_area_km2.gt(0)].copy()
    f = f.sort_values(["routed_area_km2", "station"], ascending=[False, True]).reset_index(drop=True)
    if len(f) < 2:
        return [], np.zeros(len(dates), bool), "fewer_than_two_frontier_gauges"

    def allowed(indices):
        mask = np.asarray(receiving_observed, bool).copy()
        for j in indices:
            mask &= np.asarray(observed[f.station.iloc[j]], bool)
        official = f.iloc[list(indices)].reported_area_km2.to_numpy(float)
        consistent = (not np.isfinite(receiver_area) or not np.isfinite(official).all()
                      or official.sum() <= 1.1*receiver_area)
        return mask, consistent and availability_ok(mask, dates)

    choices = []
    for a, b in combinations(range(len(f)), 2):
        mask, ok = allowed((a, b))
        if ok:
            choices.append((-(f.routed_area_km2.iloc[a]+f.routed_area_km2.iloc[b]),
                            -int(mask.sum()), f.station.iloc[a], f.station.iloc[b], a, b))
    if not choices:
        return [], np.zeros(len(dates), bool), "no_pair_with_common_sampling"
    selected = list(min(choices)[-2:])
    for j in range(len(f)):
        if j not in selected and allowed((*selected, j))[1]:
            selected.append(j)
    mask, _ = allowed(selected)
    return f.station.iloc[selected].tolist(), mask, "included"


def covered_geometry(paths, hydroseq, selected_comids, layout):
    """Exactly one owner per covered incremental catchment; no nested gauges."""
    ids = np.asarray(selected_comids, np.int64)
    if len(np.unique(ids)) != len(ids) or set(ids)-set(paths.comids):
        raise ValueError("unique selected mapped gauges required")
    lookup = {int(c): i for i, c in enumerate(paths.comids)}
    selected = {lookup[int(c)]: k for k, c in enumerate(ids)}
    owner = np.full(len(paths.comids), -1, int)
    for i in np.argsort(hydroseq):
        j = paths.successor[i]
        downstream = owner[j] if j >= 0 else -1
        if i in selected:
            if downstream >= 0:
                raise ValueError("selected upstream catchments must not be nested")
            owner[i] = selected[i]
        else:
            owner[i] = downstream
    covered = owner >= 0
    areas = np.bincount(owner[covered], weights=paths.area[covered], minlength=len(ids))
    total = paths.area.sum()
    w = paths.area/total
    full_mean = w@paths.distance
    full_var = w@((paths.distance-full_mean)**2)
    cw = paths.area[covered]/paths.area[covered].sum() if covered.any() else np.array([])
    mean = float(cw@paths.distance[covered]) if len(cw) else np.nan
    var = float(cw@((paths.distance[covered]-mean)**2)) if len(cw) else np.nan
    lateral_units = layout.units.unit_kind.eq("lateral").to_numpy()
    lateral_sources = lateral_units[layout.source_units]
    lateral_area = paths.area[lateral_sources].sum()
    covered_unit_area = np.bincount(layout.source_units[covered], weights=paths.area[covered],
                                    minlength=len(layout.units))
    positive = lateral_units & layout.units.area_km2.gt(0).to_numpy()
    share = np.divide(covered_unit_area, layout.units.area_km2.to_numpy(),
                      out=np.zeros(len(layout.units)), where=layout.units.area_km2.to_numpy() > 0)
    metrics = {"covered_area_fraction": float(areas.sum()/total), "n_selected_gauges": len(ids),
        "effective_gauges": float(areas.sum()**2/(areas@areas)) if areas.sum() > 0 else np.nan,
        "covered_lateral_area_fraction": float(paths.area[covered & lateral_sources].sum()/lateral_area) if lateral_area else np.nan,
        "n_lateral_units_covered": int(np.sum(positive & (covered_unit_area > 0))),
        "n_lateral_units_80pct_covered": int(np.sum(positive & (share >= .8))),
        "covered_path_cv": np.sqrt(var)/mean if mean > 0 else np.nan,
        "full_path_cv": np.sqrt(full_var)/full_mean if full_mean > 0 else np.nan,
        "covered_path_mean_km": mean,
        "covered_full_variance_fraction": float(w[covered]@((paths.distance[covered]-full_mean)**2)/full_var) if full_var > 0 else np.nan}
    return metrics, areas, owner


def receiver_components(members, station_groups):
    """Join overlapping complete catchments and reused physical gauge reaches."""
    parents = {s: s for s in members}

    def root(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    reach_owner = {}
    for receiver in sorted(members):
        for c in set(members[receiver]) | set(station_groups[receiver]):
            if c in reach_owner:
                parents[root(receiver)] = root(reach_owner[c])
            else:
                reach_owner[c] = receiver
    labels = {s: i for i, s in enumerate(sorted({root(s) for s in parents}))}
    return {s: labels[root(s)] for s in parents}


def _project(values, dates, adjustment="calendar_year"):
    date = pd.DatetimeIndex(dates)
    a = np.asarray(values, float)
    if adjustment == "within_month":
        _, codes = np.unique(date.to_period("M"), return_inverse=True)
        result = a.copy()
        for code in np.unique(codes):
            selected = codes == code
            result[selected] -= a[selected].mean(axis=0)
        return result
    if adjustment != "calendar_year":
        raise ValueError("calendar_year or within_month adjustment required")
    angle = 2*np.pi*date.month.to_numpy()/12
    year = date.year.to_numpy(float)
    x = np.column_stack([np.ones(len(date)), np.sin(angle), np.cos(angle), year-year.mean()])
    return a-x@np.linalg.lstsq(x, a, rcond=None)[0]


def _correlation(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/denominator, -1, 1)) if denominator > 1e-12 else np.nan


def signal_statistics(sources, receiver, weights, dates, *, shuffles=200, seed=42,
                      adjustment="calendar_year"):
    """Native-scale common-calendar identity; shuffle retains sample calendar."""
    a, y, w = np.asarray(sources, float), np.asarray(receiver, float), np.asarray(weights, float)
    if (a.ndim != 2 or a.shape != (len(y), len(w)) or len(w) < 2 or len(y) < 12
            or (w <= 0).any() or not np.isfinite(np.r_[a.ravel(), y, w]).all()
            or min(a.min(), y.min()) < 0 or len(pd.DatetimeIndex(dates).unique()) != len(y)):
        raise ValueError("finite common dates, nonnegative DOC and positive fixed shares required")
    w = w/w.sum()
    residual = _project(np.column_stack([a, y]), dates, adjustment)
    source, target = residual[:, :-1], residual[:, -1]
    mix = source@w
    np.testing.assert_allclose(_project(a@w, dates, adjustment), mix, atol=1e-10)
    variance = np.mean(source**2, axis=0)
    sd = np.sqrt(variance)
    reference, perfect = float(w@variance), float((w@sd)**2)
    independent = float((w*w)@variance)
    actual = float(np.mean(mix**2))
    denominator = perfect-independent
    coherence = (actual-independent)/denominator if denominator > 1e-12 else np.nan
    total_buffer = (reference-actual)/reference if reference > 1e-12 else np.nan
    asynchronous = (perfect-actual)/reference if reference > 1e-12 else np.nan
    amplitude = (reference-perfect)/reference if reference > 1e-12 else np.nan
    if np.isfinite(total_buffer) and not np.isclose(total_buffer, asynchronous+amplitude, atol=1e-10):
        raise ValueError("source-mixture variance identity failed")
    source_high = a >= np.quantile(a, .9, axis=0)
    receiver_high = y >= np.quantile(y, .9)
    multiple = source_high.sum(axis=1) >= 2
    high_count = int(multiple.sum())
    point = _correlation(mix, target)
    rng = np.random.default_rng(seed)
    calendar = (pd.DatetimeIndex(dates).to_period("M").astype(str).to_numpy()
                if adjustment == "within_month" else pd.DatetimeIndex(dates).month.to_numpy())
    control = []
    for _ in range(shuffles):
        permuted = a.copy()
        for month in np.unique(calendar):
            slots = np.flatnonzero(calendar == month)
            for j in range(len(w)):
                permuted[slots, j] = rng.permutation(a[slots, j])
        control.append(_correlation(_project(permuted, dates, adjustment)@w, target))
    available_control = np.asarray(control)[np.isfinite(control)]
    shuffled = float(available_control.mean()) if len(available_control) else np.nan
    result = {"n_months": len(y), "n_inputs": len(w), "source_coherence": coherence,
        "mixture_buffer_fraction": total_buffer, "asynchronous_buffer_fraction": asynchronous,
        "amplitude_balance_buffer_fraction": amplitude, "source_reference_variance": reference,
        "mixture_variance": actual, "source_perfect_variance": perfect,
        "source_independent_variance": independent, "outlet_mix_correlation": point,
        "shuffle_outlet_mix_correlation": shuffled, "real_minus_shuffle_correlation": point-shuffled,
        "outlet_mix_log_sd_ratio": .5*np.log(np.mean(target**2)/actual) if min(actual, np.mean(target**2)) > 1e-12 else np.nan,
        "n_receiver_high": int(receiver_high.sum()), "n_multiple_sources_high": high_count,
        "n_receiver_and_multiple_sources_high": int(np.sum(multiple & receiver_high)),
        "receiver_high_given_multiple_sources": float(receiver_high[multiple].mean()) if high_count else np.nan,
        "small_coincidence_denominator": high_count < 20, "n_valid_shuffle_draws": len(available_control)}
    series = pd.DataFrame({"date": pd.DatetimeIndex(dates), "doc_receiver": y, "doc_mixture": a@w,
        "receiver_anomaly": target, "mixture_anomaly": mix,
        "n_sources_high": source_high.sum(axis=1), "source_high_area_share": source_high@w,
        "receiver_high": receiver_high})
    return result, series


def same_day_activities(activities, stations, months):
    """One metadata-selected activity per station on each exactly common day.

    Known timestamps precede missing timestamps; DOC never selects an activity.
    A daily mean of all accepted activities provides a replicate sensitivity.
    """
    if len(stations) < 3 or len(set(stations)) != len(stations):
        raise ValueError("distinct sources and receiver required")
    f = activities[activities.site_no.isin(stations) & activities.month.isin(months)].copy()
    f["date"] = pd.to_datetime(f.date).dt.normalize()
    f["event_id"] = f.event_id.astype(str)
    means = f.groupby(["site_no", "date"], as_index=False).agg(
        doc_daily_mean=("doc", "mean"), n_daily_activities=("doc", "size"))
    selected = f.sort_values(["site_no", "date", "timestamp_utc", "event_id"], na_position="last")
    selected = selected.drop_duplicates(["site_no", "date"])
    dates = [set(selected.loc[selected.site_no.eq(s), "date"]) for s in stations]
    common = set.intersection(*dates)
    selected = selected[selected.date.isin(common)].merge(means, on=["site_no", "date"], validate="one_to_one")
    selected["source_order"] = selected.site_no.map({s: i if i < len(stations)-1 else -1 for i, s in enumerate(stations)})
    selected = selected.sort_values(["date", "source_order"], kind="stable").reset_index(drop=True)
    if selected.groupby("date").site_no.nunique().ne(len(stations)).any():
        raise ValueError("incomplete common sampling date")
    return selected


def select_activity_set(groups):
    """Minimum calendar-span set; DOC is copied only after metadata selection.

    Last group is the receiver. This algorithm avoids Cartesian products and
    uses a source-to-receiver date-offset tie break, not an optimized UTC lag.
    """
    if len(groups) < 3 or any(f.empty for f in groups):
        raise ValueError("at least two source groups and one receiver required")
    records = [f.to_dict("records") for f in groups]
    best = None
    for receiver in records[-1]:
        t = pd.Timestamp(receiver["date"])
        starts = sorted({pd.Timestamp(r["date"]) for g in records for r in g if pd.Timestamp(r["date"]) <= t})
        for start in starts:
            earliest = []
            for g in records[:-1]:
                eligible = [r for r in g if pd.Timestamp(r["date"]) >= start]
                if not eligible:
                    break
                earliest.append(min(pd.Timestamp(r["date"]) for r in eligible))
            if len(earliest) != len(records)-1:
                continue
            end = max(t, *earliest)
            chosen = [min((r for r in g if start <= pd.Timestamp(r["date"]) <= end),
                          key=lambda r: (abs((pd.Timestamp(r["date"])-t).days), pd.Timestamp(r["date"]), str(r["event_id"])))
                      for g in records[:-1]]+[receiver]
            dates = [pd.Timestamp(r["date"]) for r in chosen]
            key = ((max(dates)-min(dates)).days, sum(abs((d-t).days) for d in dates[:-1]),
                   t, *[str(r["event_id"]) for r in chosen])
            if best is None or key < best[0]:
                best = key, chosen
    if best is None:
        raise ValueError("no complete metadata set found")
    return best[1], best[0][0]


def association_table(frame, focals, outcomes, *, draws=5000):
    """Descriptive Pearson associations with whole-system resampling."""
    rows = []
    for focal in focals:
        for outcome in outcomes:
            f = frame.dropna(subset=[focal, outcome, "component"])
            x, y = f[focal].to_numpy(float), f[outcome].to_numpy(float)
            if len(f) < 3 or min(np.std(x), np.std(y)) < 1e-12:
                continue
            _, idx = np.unique(f.component, return_inverse=True)
            n = idx.max()+1
            weights = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, idx]
            total = weights.sum(axis=1)
            mx, my = weights@x/total, weights@y/total
            covariance = weights@(x*y)/total-mx*my
            denominator = np.sqrt(np.maximum(weights@(x*x)/total-mx*mx, 0)*np.maximum(weights@(y*y)/total-my*my, 0))
            boot = np.divide(covariance, denominator, out=np.full(draws, np.nan), where=denominator > 1e-12)
            finite = boot[np.isfinite(boot)]
            lo, hi = np.quantile(finite, [.025, .975]) if n >= 2 and len(finite) else (np.nan, np.nan)
            omitted = []
            for c in f.component.unique():
                a, b = x[f.component.ne(c)], y[f.component.ne(c)]
                if len(a) >= 3 and min(np.std(a), np.std(b)) > 1e-12:
                    omitted.append(np.corrcoef(a, b)[0, 1])
            rows.append({"focal": focal, "outcome": outcome, "correlation": np.corrcoef(x, y)[0, 1],
                "ci_low": lo, "ci_high": hi, "n_receivers": len(f), "n_components": int(n),
                "valid_draws": len(finite) if n >= 2 else 0,
                "omitted_min": min(omitted) if omitted else np.nan,
                "omitted_max": max(omitted) if omitted else np.nan})
    return pd.DataFrame(rows)


def form_contrasts(frame, metrics, *, draws=5000):
    """Exploratory broad-minus-elongated differences, same system draws."""
    rows = []
    for metric in metrics:
        f = frame[frame.cluster.isin((1, 3))].dropna(subset=[metric, "component"])
        x = f[metric].to_numpy(float)
        broad, elongated = f.cluster.eq(3).to_numpy(), f.cluster.eq(1).to_numpy()
        if not broad.any() or not elongated.any():
            continue
        _, codes = np.unique(f.component, return_inverse=True)
        n = codes.max()+1
        weight = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, codes]
        total_b, total_e = weight[:, broad].sum(axis=1), weight[:, elongated].sum(axis=1)
        mean_b = np.divide(weight[:, broad]@x[broad], total_b, out=np.full(draws, np.nan), where=total_b > 0)
        mean_e = np.divide(weight[:, elongated]@x[elongated], total_e, out=np.full(draws, np.nan), where=total_e > 0)
        differences = mean_b-mean_e
        finite = differences[np.isfinite(differences)]
        lo, hi = np.quantile(finite, [.025, .975]) if n >= 2 and len(finite) else (np.nan, np.nan)
        rows.append({"contrast": "broad_minus_elongated", "metric": metric,
            "estimate": x[broad].mean()-x[elongated].mean(), "ci_low": lo, "ci_high": hi,
            "n_broad": int(broad.sum()), "n_elongated": int(elongated.sum()), "n_components": n,
            "valid_draws": len(finite) if n >= 2 else 0,
            "scope": "unadjusted descriptive form contrast; shared catchment systems resampled together"})
    return pd.DataFrame(rows)
