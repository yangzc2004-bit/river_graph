"""Observed branch-wave overlap and local confluence DOC/geometry comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wave_overlap(a: np.ndarray, b: np.ndarray, baseline_quantile: float = 0.0) -> dict:
    """Compare incoming water signals on one observed clock; never search a lag."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Expected two matched water series")
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or min(a.min(), b.min()) < 0:
        raise ValueError("Flow must be observed, finite and nonnegative")
    if not 0 <= baseline_quantile < 1:
        raise ValueError("Invalid baseline quantile")
    a = np.maximum(a-np.quantile(a, baseline_quantile), 0)
    b = np.maximum(b-np.quantile(b, baseline_quantile), 0)
    if a.sum() == 0 or b.sum() == 0:
        return {"wave_overlap": np.nan, "incoming_peak_coincidence": np.nan,
                "amplitude_balanced_peak_coincidence": np.nan, "peak_excess_a_share": np.nan}
    return {"wave_overlap": float(np.minimum(a/a.sum(), b/b.sum()).sum()),
            "incoming_peak_coincidence": float((a+b).max()/(a.max()+b.max())),
            "amplitude_balanced_peak_coincidence": float((a/a.max()+b/b.max()).max()/2),
            "peak_excess_a_share": float(a.max()/(a.max()+b.max()))}


def maximum_clock(clock: pd.DatetimeIndex, values: np.ndarray) -> dict:
    """Retain tied seasonal maxima, including separated runs; do not pick one."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if len(clock) != len(values) or not clock.is_monotonic_increasing or not clock.is_unique:
        raise ValueError("Clock and values must align uniquely")
    if (values[valid] < 0).any():
        raise ValueError("Negative discharge")
    if not valid.any():
        return {"maximum": np.nan, "first": pd.NaT, "last": pd.NaT,
                "n_tied_samples": 0, "n_tied_runs": 0}
    maximum = float(values[valid].max())
    indices = np.flatnonzero(valid & (values == maximum))
    runs = 1 + int(np.sum(np.diff(indices) > 1))
    return {"maximum": maximum, "first": clock[indices[0]], "last": clock[indices[-1]],
            "n_tied_samples": len(indices), "n_tied_runs": runs}


def summarize_water_window(clock: pd.DatetimeIndex, a, b, receiver) -> dict:
    """Coverage, peak-clock bounds, water overlap and partial-input flow share."""
    arrays = [np.asarray(x, dtype=float) for x in (a, b, receiver)]
    if len(clock) < 2 or not np.all((clock[1:]-clock[:-1]) == pd.Timedelta(minutes=30)):
        raise ValueError("Expected an uninterrupted half-hour clock")
    if any(x.shape != (len(clock),) for x in arrays):
        raise ValueError("Input shapes differ from clock")
    if any(np.isinf(x).any() or (x[np.isfinite(x)] < 0).any() for x in arrays):
        raise ValueError("Invalid observed discharge")
    valid = np.column_stack([np.isfinite(x) for x in arrays])
    joint = valid.all(axis=1)
    peaks = [maximum_clock(clock, x) for x in arrays]
    result = {"n_expected_slots": len(clock), "n_joint_slots": int(joint.sum()),
              "joint_coverage": float(joint.mean()), "full_joint_clock": bool(joint.all())}
    for name, flags, peak in zip(("a", "b", "receiver"), valid.T, peaks):
        result[f"coverage_{name}"] = float(flags.mean())
        result.update({f"peak_{name}_{key}": value for key, value in peak.items()})
    for name, left, right in (("b_minus_a", peaks[0], peaks[1]),
                              ("receiver_minus_a", peaks[0], peaks[2]),
                              ("receiver_minus_b", peaks[1], peaks[2])):
        result[f"{name}_lower_hours"] = (right["first"]-left["last"]).total_seconds()/3600
        result[f"{name}_upper_hours"] = (right["last"]-left["first"]).total_seconds()/3600
    peaks_retained = all(np.any(joint & (x == peak["maximum"])) for x, peak in zip(arrays, peaks))
    result["all_site_maxima_present_on_joint_clock"] = peaks_retained
    result["primary_coverage_subset"] = bool(joint.mean() >= .95 and peaks_retained)
    if joint.sum() >= 2:
        aj, bj, rj = [x[joint] for x in arrays]
        for q, suffix in ((0., ""), (.1, "_p10")):
            result.update({key+suffix: value for key, value in wave_overlap(aj, bj, q).items()})
        positive = rj > 0
        result["n_positive_receiver_slots"] = int(positive.sum())
        result["branch_flow_share_median"] = float(np.median((aj[positive]+bj[positive])/rj[positive])) if positive.any() else np.nan
        result["branch_receiver_correlation"] = float(np.corrcoef(aj+bj, rj)[0, 1]) if np.std(aj+bj) > 0 and np.std(rj) > 0 else np.nan
    else:
        for key in ("wave_overlap", "incoming_peak_coincidence", "wave_overlap_p10",
                    "incoming_peak_coincidence_p10", "amplitude_balanced_peak_coincidence",
                    "amplitude_balanced_peak_coincidence_p10", "peak_excess_a_share", "peak_excess_a_share_p10",
                    "branch_flow_share_median", "branch_receiver_correlation"):
            result[key] = np.nan
        result["n_positive_receiver_slots"] = 0
    return result


def confluence_chemistry(database: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute mixtures without treating lateral positions as independent sites."""
    rows, points = [], []
    for (site, season), frame in database.groupby(["location", "season"], sort=True):
        if sorted(frame["transect.location"].tolist()) != ["L", "M", "R"]:
            raise ValueError("Expected the three lateral transect positions")
        for col in ("Q.Ls.main", "Q.Ls.trib", "Q.Ls"):
            if frame[col].nunique(dropna=False) != 1 or not np.isfinite(frame[col].iloc[0]) or frame[col].iloc[0] <= 0:
                raise ValueError(f"Invalid or inconsistent campaign discharge: {col}")
        qa, qb, qr = (float(frame[col].iloc[0]) for col in ("Q.Ls.main", "Q.Ls.trib", "Q.Ls"))
        wa, wb = qa/(qa+qb), qb/(qa+qb)
        row = {"confluence": site, "season": season, "n_lateral_positions": len(frame),
               "main_flow_ls": qa, "tributary_flow_ls": qb, "receiver_flow_ls": qr,
               "main_weight": wa, "tributary_weight": wb, "branch_flow_share": (qa+qb)/qr}
        for raw, name in (("DOC.mgL.mean", "doc"), ("SpC", "conductivity")):
            a, b, r = (frame[col].to_numpy(dtype=float) for col in (raw+".main", raw+".trib", raw))
            if not all(np.isfinite(v).all() and (v >= 0).all() for v in (a, b, r)):
                raise ValueError(f"Invalid campaign chemistry: {raw}")
            mixture = float(wa*a.mean()+wb*b.mean())
            if mixture <= 0:
                raise ValueError("Positive reference concentration is required")
            row.update({f"{name}_main_mean": float(a.mean()), f"{name}_trib_mean": float(b.mean()),
                        f"{name}_main_distinct_values": len(np.unique(a)), f"{name}_trib_distinct_values": len(np.unique(b)),
                        f"{name}_mix_reference": mixture, f"{name}_receiver_lateral_mean": float(r.mean()),
                        f"{name}_receiver_min": float(r.min()), f"{name}_receiver_max": float(r.max()),
                        f"{name}_receiver_minus_mix": float(r.mean()-mixture),
                        f"{name}_receiver_minus_mix_pct": float(100*(r.mean()-mixture)/mixture),
                        f"{name}_mix_within_lateral_range": bool(r.min() <= mixture <= r.max())})
        supplied_sums = frame["pQm.mtsum"]+frame["pQt.mtsum"]
        row["source_weight_sum_min"] = float(supplied_sums.min())
        row["source_weight_sum_max"] = float(supplied_sums.max())
        row["source_weights_differ_from_normalized"] = bool(
            not (np.allclose(frame["pQm.mtsum"], wa, atol=1e-8, rtol=0)
                 and np.allclose(frame["pQt.mtsum"], wb, atol=1e-8, rtol=0)))
        for point in frame.to_dict("records"):
            author_mix = point["pQm.mtsum"]*point["DOC.mgL.mean.main"]+point["pQt.mtsum"]*point["DOC.mgL.mean.trib"]
            points.append({"confluence": site, "season": season, "position": point["transect.location"],
                           "doc_measured": point["DOC.mgL.mean"], "doc_mix_reference": row["doc_mix_reference"],
                           "source_row_mix": author_mix, "source_weight_sum": point["pQm.mtsum"]+point["pQt.mtsum"]})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(points)


def channel_geometry(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One width per transect; depth points do not multiply the width sample size."""
    data = raw.copy()
    data["confluence"] = "Con-"+data.location.str.extract(r"^(\d+)\.", expand=False)
    keys = ["confluence", "reach", "transect"]
    if data.confluence.isna().any() or data[keys].isna().any().any():
        raise ValueError("Unrecognized survey site or transect")
    transects = []
    for key, frame in data.groupby(keys, sort=True):
        for col in ("wettedwidth.m",):
            if frame[col].nunique(dropna=False) != 1:
                raise ValueError(f"Conflicting transect {col}")
        width = float(frame["wettedwidth.m"].iloc[0])
        depth = frame["depth.m"].to_numpy(dtype=float)
        observed = np.isfinite(depth)
        if not np.isfinite(width) or width <= 0 or np.isinf(depth).any() or (depth[observed] < 0).any():
            raise ValueError("Invalid surveyed width/depth")
        transects.append({**dict(zip(keys, key)), "width_m": width,
                          "mean_depth_m": float(depth[observed].mean()) if observed.any() else np.nan,
                          "n_depth_points": len(depth), "n_observed_depth_points": int(observed.sum()),
                          "n_missing_depth_points": int((~observed).sum()),
                          "distance_m": frame["distance.to.confluence"].iloc[0] if frame["distance.to.confluence"].nunique(dropna=False) == 1 else np.nan,
                          "distance_min_m": frame["distance.to.confluence"].min(),
                          "distance_max_m": frame["distance.to.confluence"].max(),
                          "conflicting_distance": frame["distance.to.confluence"].nunique(dropna=False) > 1})
    transects = pd.DataFrame(transects)
    rows = []
    for site, frame in transects.groupby("confluence", sort=True):
        row = {"confluence": site}
        for source, name in (("1.upstream", "main"), ("2.downstream", "receiver")):
            sub = frame[frame.reach.eq(source)]
            if len(sub) < 2:
                raise ValueError(f"Missing surveyed reach: {site} {source}")
            row.update({f"{name}_n_transects": len(sub), f"{name}_mean_width_m": float(sub.width_m.mean()),
                        f"{name}_width_cv_pct": float(100*sub.width_m.std(ddof=1)/sub.width_m.mean()),
                        f"{name}_mean_depth_m": float(sub.mean_depth_m.mean()),
                        f"{name}_depth_cv_pct": float(100*sub.mean_depth_m.std(ddof=1)/sub.mean_depth_m.mean()),
                        f"{name}_n_missing_depth_points": int(sub.n_missing_depth_points.sum())})
        row["receiver_main_width_ratio"] = row["receiver_mean_width_m"]/row["main_mean_width_m"]
        row["receiver_main_depth_ratio"] = row["receiver_mean_depth_m"]/row["main_mean_depth_m"]
        rows.append(row)
    return pd.DataFrame(rows), transects


def descriptive_geometry_relations(fall: pd.DataFrame) -> pd.DataFrame:
    """Five junctions in one network: describe sensitivity, not causal estimates."""
    rows = []
    y = "doc_receiver_minus_mix_pct"
    for x in ("receiver_main_width_ratio", "receiver_main_depth_ratio", "receiver_width_cv_pct",
              "receiver_main_residence_ratio"):
        valid = fall[["confluence", x, y]].dropna()
        full = valid[[x, y]].corr(method="spearman").iloc[0, 1]
        dropped = []
        for site in valid.confluence:
            sub = valid[valid.confluence.ne(site)]
            dropped.append(sub[[x, y]].corr(method="spearman").iloc[0, 1])
        rows.append({"geometry_metric": x, "n_confluences": len(valid), "spearman": full,
                     "leave_one_min": min(dropped), "leave_one_max": max(dropped),
                     "interpretation": "descriptive_five_junctions_in_one_network"})
    return pd.DataFrame(rows)
