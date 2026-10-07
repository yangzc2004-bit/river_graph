"""Matched-strength placement of conservative storage on real confluence paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from river_graph.analysis.river_storage_transport import gaussian_storage_response

PLACEMENTS = ("early_branch", "late_branch", "shared_trunk")
MATCHES = ("variance_matched", "mean_budget_matched")


def _continuous_peak(time, sampled, evaluate):
    """Refine local maxima; report both times if global peaks are nearly equal.

    Grid argmax alone jumps between symmetric separated peaks as dt changes.
    The peak height is continuous; the representative time is the earliest
    maximum within relative tolerance1e-8, with its ambiguity retained.
    """
    candidates = find_peaks(sampled)[0]
    candidates = candidates[sampled[candidates] >= sampled.max()*(1-1e-3)]
    if not len(candidates):
        raise ValueError("response grid must bracket a positive pulse peak")
    peaks = []
    for i in candidates:
        refined = minimize_scalar(lambda t: -float(evaluate(t)),
            bounds=(time[i-1], time[i+1]), method="bounded", options={"xatol": 1e-12})
        if not refined.success:
            raise ValueError("continuous peak refinement failed")
        peaks.append((float(refined.x), float(-refined.fun)))
    value = max(h for _, h in peaks)
    times = [t for t, h in peaks if h >= value*(1-1e-8)]
    return value, min(times), max(times), len(times)


def select_confluence(paths, hydroseq):
    """Largest balanced real upstream-area junction; source means choose midpoints."""
    hydroseq = np.asarray(hydroseq)
    if hydroseq.shape != paths.area.shape or len(np.unique(hydroseq)) != len(hydroseq):
        raise ValueError("unique hydroseq aligned with selected tree required")
    p = paths.successor
    order = np.argsort(hydroseq)
    area = paths.area.copy()
    moment = paths.area*paths.distance
    for i in order[::-1]:
        if p[i] >= 0:
            area[p[i]] += area[i]
            moment[p[i]] += moment[i]
    candidates = np.flatnonzero((p >= 0) & (area > 0))
    sorted_children = candidates[np.lexsort((paths.comids[candidates], -area[candidates], p[candidates]))]
    parents = p[sorted_children]
    starts = np.flatnonzero(np.r_[True, np.diff(parents) != 0]) if len(parents) else np.empty(0, int)
    starts = starts[(starts+1 < len(parents)) & (parents[np.minimum(starts+1, max(len(parents)-1, 0))] == parents[starts])]
    if not len(starts):
        return None, "no_two_positive_area_tributaries"
    a, b = sorted_children[starts], sorted_children[starts+1]
    junction = p[a]
    common = paths.distance[junction]+paths.length[junction]/2
    mean_a, mean_b = moment[a]/area[a], moment[b]/area[b]
    valid = (common > 0) & (mean_a > common) & (mean_b > common)
    if not valid.any():
        return None, "no_positive_three_segment_geometry"
    score = 2*area[a]*area[b]/(area[a]+area[b])
    eligible = np.flatnonzero(valid)
    pick = eligible[np.lexsort((paths.comids[junction[eligible]], -score[eligible]))[0]]
    aj, bj, joint = int(a[pick]), int(b[pick]), int(junction[pick])
    membership = np.full(len(p), -1, int)
    membership[aj], membership[bj] = 0, 1
    for i in order:
        if i not in (aj, bj) and p[i] >= 0:
            membership[i] = membership[p[i]]
    sources = []
    for arm, root in enumerate((aj, bj)):
        indices = np.flatnonzero((membership == arm) & (paths.area > 0))
        target_distance = moment[root]/area[root]
        chosen = indices[np.lexsort((paths.comids[indices], abs(paths.distance[indices]-target_distance)))[0]]
        sources.append(int(chosen))
    c = float(common[pick])
    la, lb = (float(paths.distance[s]-c) for s in sources)
    if min(la, lb) <= 0:
        return None, "representative_source_has_no_independent_branch"
    weights = area[[aj, bj]]/sum(area[[aj, bj]])
    mean = weights@np.array([la+c, lb+c])
    capacity = min(np.sqrt(weights[0])*la, np.sqrt(weights[1])*lb, c)/mean
    descriptor = {"junction_index": joint, "source_a_index": sources[0], "source_b_index": sources[1],
        "junction_comid": int(paths.comids[joint]), "source_a_comid": int(paths.comids[sources[0]]),
        "source_b_comid": int(paths.comids[sources[1]]), "branch_a_km": la, "branch_b_km": lb,
        "common_km": c, "weight_a": float(weights[0]), "mean_total_km": float(mean),
        "selected_pair_area_share": float(sum(area[[aj, bj]])/paths.area.sum()),
        "selection_score_area": float(score[pick]), "n_eligible_junctions": int(valid.sum()),
        "relative_arrival_cv": float(np.sqrt(weights[0]*weights[1])*abs(la-lb)/mean),
        "common_fraction": c/mean, "variance_capacity_sd": float(capacity),
        "representative_a_distance_error": float(abs(paths.distance[sources[0]]-mean_a[pick])),
        "representative_b_distance_error": float(abs(paths.distance[sources[1]]-mean_b[pick]))}
    return descriptor, "included"


def selected_corridor(paths, selection, scale=1.):
    joint = selection["junction_index"]
    rows = []
    for segment, start in (("branch_a", selection["source_a_index"]),
                            ("branch_b", selection["source_b_index"]), ("common", joint)):
        i, sequence, visited = start, 0, set()
        while i >= 0 and i not in visited:
            if segment != "common" and i == joint:
                break
            visited.add(i)
            measure = 50. if segment != "common" and sequence == 0 else 100.
            rows.append({"segment": segment, "sequence": sequence, "comid": int(paths.comids[i]),
                "start_measure": measure, "end_measure": 0.,
                "length_km": float(paths.length[i]*measure/100*scale)})
            i, sequence = int(paths.successor[i]), sequence+1
        if segment != "common" and i != joint:
            raise ValueError("representative branch does not reach selected junction")
    result = pd.DataFrame(rows)
    if result.comid.duplicated().any():
        raise ValueError("independent branches and common trunk must be unique")
    return result


def placement_parameters(branch_a, branch_b, common, weight, match, placement, fraction):
    if not np.isfinite([branch_a, branch_b, common, weight, fraction]).all():
        raise ValueError("finite geometry, shares and allocation required")
    if min(branch_a, branch_b, common) <= 0 or not 0 < weight < 1 or not 0 <= fraction <= 1:
        raise ValueError("positive three-segment geometry/shares and fraction in [0,1] required")
    if match not in MATCHES or placement not in PLACEMENTS:
        raise ValueError("recognized strength match and placement required")
    weights = np.array([weight, 1-weight])
    mean = weights@np.array([branch_a+common, branch_b+common])
    branches = np.array([branch_a, branch_b])/mean
    c = common/mean
    early = int(np.argmin(branches))
    late = 1-early
    if match == "variance_matched":
        capacity = min(*(np.sqrt(weights)*branches), c)
        budget = (fraction*capacity)**2
        tau = np.sqrt(budget) if placement == "shared_trunk" else np.sqrt(budget/weights[early if placement == "early_branch" else late])
    else:
        capacity = min(*(weights*branches), c)
        budget = fraction*capacity
        tau = budget if placement == "shared_trunk" else budget/weights[early if placement == "early_branch" else late]
    tau_vector = np.zeros(2)
    if placement == "shared_trunk":
        tau_vector[:] = tau
        remaining = c-tau
    else:
        arm = early if placement == "early_branch" else late
        tau_vector[arm] = tau
        remaining = branches[arm]-tau
    if remaining < -1e-12:
        raise ValueError("storage allocation exceeds causal segment budget")
    delays = branches+c
    return {"weights": weights, "delays": delays, "taus": tau_vector,
        "translation": np.maximum(delays-tau_vector, 0.), "early_index": early,
        "early_weight": weights[early], "arrival_gap": abs(branches[0]-branches[1]),
        "branch_variance": weights@((delays-1)**2), "storage_variance": weights@(tau_vector**2),
        "allocated_mean_budget": weights@tau_vector, "match_capacity": capacity,
        "minimum_remaining_segment": max(remaining, 0.)}


def placement_responses(branch_a, branch_b, common, weight, *, sigma=.15,
                        fractions=(.25, .5, 1.), dt=.00125, keep_curves=False):
    if not np.isfinite([sigma, dt]).all() or min(sigma, dt) <= 0:
        raise ValueError("positive finite pulse SD and numerical resolution required")
    fractions = np.asarray(fractions, float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or (fractions <= 0).any() or (fractions > 1).any() or len(np.unique(fractions)) != len(fractions):
        raise ValueError("distinct positive fractions up to one required")
    settings = [("baseline", "translation", 0.)]
    settings += [(match, placement, float(f)) for match in MATCHES for f in fractions for placement in PLACEMENTS]
    # A single absolute grid for all matched placements and their baseline.
    parameters = [placement_parameters(branch_a, branch_b, common, weight,
        "variance_matched" if match == "baseline" else match,
        "shared_trunk" if placement == "translation" else placement, f) for match, placement, f in settings]
    d = parameters[0]["delays"]
    maximum_tau = max(p["taus"].max() for p in parameters)
    low = min(p["translation"].min() for p in parameters)-9*sigma
    high = d.max()+9*sigma+32*maximum_tau
    time = np.arange(np.floor(low/dt), np.ceil(high/dt)+1)*dt
    rows, curves = [], []
    for (match, placement, fraction), p in zip(settings, parameters, strict=True):
        source = np.stack([gaussian_storage_response(time, delay, sigma, tau)
            for delay, tau in zip(p["translation"], p["taus"], strict=True)])
        weighted = p["weights"][:, None]*source
        response = weighted.sum(axis=0)
        mass = np.trapezoid(response, time)
        centroid = np.trapezoid(time*response, time)/mass
        sd = np.sqrt(np.trapezoid((time-centroid)**2*response, time)/mass)
        cdf = cumulative_trapezoid(response/mass, time, initial=0)
        q10, q50, q90 = np.interp([.1, .5, .9], cdf, time)
        refined = [_continuous_peak(time, weighted[j],
            lambda t, j=j, p=p: p["weights"][j]*gaussian_storage_response(t, p["translation"][j], sigma, p["taus"][j]))
            for j in range(2)]
        peaks = np.array([r[0] for r in refined])
        peak_times = np.array([r[1] for r in refined])
        envelope = peaks.sum()
        peak, first_peak, last_peak, n_peaks = _continuous_peak(time, response,
            lambda t, p=p: sum(p["weights"][j]*gaussian_storage_response(t, p["translation"][j], sigma, p["taus"][j])
                          for j in range(2)))
        row = {"strength_match": match, "placement": placement, "fraction": fraction, "input_sd": sigma,
            "pulse_peak": float(peak), "peak_time": first_peak, "last_near_equal_peak_time": last_peak,
            "near_equal_peak_count": n_peaks, "peak_time_is_ambiguous": n_peaks > 1,
            "pulse_centroid": float(centroid), "pulse_sd": float(sd), "analytic_centroid": 1.,
            "analytic_sd": float(np.sqrt(sigma*sigma+p["branch_variance"]+p["storage_variance"])),
            "branch_variance": p["branch_variance"], "storage_variance": p["storage_variance"],
            "allocated_mean_budget": p["allocated_mean_budget"], "match_capacity": p["match_capacity"],
            "minimum_remaining_segment": p["minimum_remaining_segment"],
            "early_index": p["early_index"], "early_weight": p["early_weight"], "arrival_gap": p["arrival_gap"],
            "tau_a": p["taus"][0], "tau_b": p["taus"][1], "t10": q10, "t50": q50, "t90": q90,
            "duration_80": q90-q10, "anomaly_area_fraction": mass/(np.sqrt(2*np.pi)*sigma), "steady_gain": 1.,
            "individual_peak_envelope": float(envelope), "alignment_ratio": float(peak/envelope),
            "source_a_peak_time": peak_times[0], "source_b_peak_time": peak_times[1],
            "source_peak_gap": abs(peak_times[0]-peak_times[1]), "dt": dt}
        rows.append(row)
        if keep_curves:
            curves.append(pd.DataFrame({"strength_match": match, "placement": placement, "fraction": fraction,
                "input_sd": sigma, "relative_time": time[::4], "outlet_anomaly": response[::4],
                "weighted_a": weighted[0, ::4], "weighted_b": weighted[1, ::4]}))
    frame = pd.DataFrame(rows)
    baseline = frame.iloc[0]
    frame["peak_reduction_pct"] = 100*(1-frame.pulse_peak/baseline.pulse_peak)
    frame["duration_change_pct"] = 100*(frame.duration_80/baseline.duration_80-1)
    frame["log_peak_change"] = np.log(frame.pulse_peak/baseline.pulse_peak)
    frame["log_envelope_change"] = np.log(frame.individual_peak_envelope/baseline.individual_peak_envelope)
    frame["log_alignment_change"] = np.log(frame.alignment_ratio/baseline.alignment_ratio)
    return frame, pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def placement_contrasts(frame):
    """Paired placement differences, never contrasts across different geometries."""
    keys = ["cohort", "case_id", "flow_rule", "input_sd", "strength_match", "fraction"]
    metrics = ["pulse_peak", "peak_reduction_pct", "duration_80", "duration_change_pct",
               "alignment_ratio", "individual_peak_envelope", "source_peak_gap"]
    nonbaseline = frame[frame.strength_match.ne("baseline")]
    wide = nonbaseline.set_index(keys+["placement"])
    if not wide.index.is_unique:
        raise ValueError("each geometry and placement must occur once")
    rows = []
    for later, earlier in (("late_branch", "early_branch"), ("shared_trunk", "early_branch"),
                            ("shared_trunk", "late_branch")):
        a = wide.xs(later, level="placement")
        b = wide.xs(earlier, level="placement").reindex(a.index)
        if b[metrics].isna().any().any():
            raise ValueError("all three matched placements required")
        out = (a[metrics]-b[metrics]).reset_index()
        out["contrast"] = f"{later}_minus_{earlier}"
        rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    meta = frame[["cohort", "case_id", "station", "huc4", "component", "cluster"]].drop_duplicates()
    return result.merge(meta, on=["cohort", "case_id"], validate="many_to_one")


def blocked_summary(frame, metrics, *, group, unit, draws=5000):
    """Unit-equal means; whole groups resampled, with variable group sizes retained."""
    if frame[unit].duplicated().any() or draws < 1:
        raise ValueError("one row per inferential unit and positive bootstrap count required")
    values = frame[list(metrics)].to_numpy(float)
    if not len(frame) or not np.isfinite(values).all() or frame[group].isna().any():
        raise ValueError("nonempty finite metrics and defined blocks required")
    labels, codes = np.unique(frame[group].astype(str), return_inverse=True)
    n = len(labels)
    count = np.bincount(codes)
    sums = np.zeros((n, len(metrics)))
    np.add.at(sums, codes, values)
    if n >= 2:
        frequencies = np.random.default_rng(42).multinomial(n, np.ones(n)/n, size=draws)
        estimates = frequencies@sums/(frequencies@count)[:, None]
        low, high = np.quantile(estimates, [.025, .975], axis=0)
    else:
        low, high = np.full(len(metrics), np.nan), np.full(len(metrics), np.nan)
    return pd.DataFrame({"metric": list(metrics), "mean": values.mean(axis=0),
        "median": np.median(values, axis=0), "q10": np.quantile(values, .1, axis=0),
        "q90": np.quantile(values, .9, axis=0), "ci_low": low, "ci_high": high,
        "n_units": len(frame), "n_blocks": n, "bootstrap_draws": draws,
        "interval_scope": "block-bootstrap geometry mean" if n >= 2 else "unavailable: one block"})
