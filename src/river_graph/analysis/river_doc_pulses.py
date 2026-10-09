"""Flow-selected single-pulse timing measurements, retaining unresolved responses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from river_graph.analysis.river_kervidy_observations import occupied_record_coverage


def flow_pulses(flow, relative_prominence=.20, absolute_prominence=.020, sample_minutes=15):
    """Detect on uninterrupted recorded runs; no additional gap filling.

    Source preprocessing (if any) must be reported separately. Quarter-hour
    defaults preserve the preceding Kervidy analysis exactly.
    """
    if sample_minutes <= 0:
        raise ValueError("Sampling interval must be positive")
    valid = flow.loc[flow.flow_valid].sort_values("flow_timestamp_utc").reset_index(drop=True)
    runs = valid.flow_timestamp_utc.diff().dt.total_seconds().ne(sample_minutes*60).cumsum()
    rows = []
    for block, frame in valid.groupby(runs):
        frame = frame.reset_index(drop=True)
        q = frame.q_m3_s.to_numpy()
        distance = max(1, int(np.ceil(360/sample_minutes)))
        window = int(7*24*60/sample_minutes)//2*2+1
        peaks, properties = find_peaks(q, distance=distance, prominence=absolute_prominence, wlen=window,
                                       plateau_size=(None, None))
        accepted = properties["prominences"] >= relative_prominence*q[peaks]
        peaks = peaks[accepted]
        for name in properties:
            properties[name] = properties[name][accepted]
        for k, p in enumerate(peaks):
            prominence = properties["prominences"][k]
            threshold = q[p]-.9*prominence
            left = np.flatnonzero(q[int(properties["left_bases"][k]):p+1] <= threshold)
            right = np.flatnonzero(q[p:int(properties["right_bases"][k])+1] <= threshold)
            if not len(left) or not len(right):
                raise ValueError("Prominence bases must bracket the declared threshold")
            a = int(properties["left_bases"][k])+int(left[-1])
            b = p+int(right[0])
            start, end = frame.flow_timestamp_utc.iloc[[a, b]]
            n_other = int(((peaks > a) & (peaks < b) & (peaks != p)).sum())
            duration = (end-start).total_seconds()/3600
            rows.append({"event_id": f"{frame.flow_timestamp_utc.iloc[p]:%Y%m%dT%H%M}",
                "flow_block": int(block), "start_utc": start, "flow_return_utc": end,
                "flow_peak_utc": frame.flow_timestamp_utc.iloc[p], "q_peak_m3_s": q[p],
                "flow_prominence_m3_s": prominence, "flow_threshold_m3_s": threshold,
                "flow_span_hours": duration, "n_other_qualifying_flow_peaks": n_other,
                "isolated_flow_pulse": 3 <= duration <= 96 and n_other == 0,
                "flow_peak_censored": bool(frame.flow_at_reported_cap.iloc[p]),
                "flow_boundary_on_record_edge": a == 0 or b == len(frame)-1,
                "flow_plateau_start_utc": frame.flow_timestamp_utc.iloc[int(properties["left_edges"][k])],
                "flow_plateau_end_utc": frame.flow_timestamp_utc.iloc[int(properties["right_edges"][k])],
                "flow_run_first_utc": frame.flow_timestamp_utc.iloc[0],
                "flow_run_last_utc": frame.flow_timestamp_utc.iloc[-1]})
    if not rows:
        return pd.DataFrame(columns=["event_id", "flow_peak_utc", "start_utc", "flow_return_utc"])
    events = pd.DataFrame(rows).sort_values("flow_peak_utc").reset_index(drop=True)
    if events.event_id.duplicated().any():
        raise ValueError("Flow event identities must be unique")
    return events


def half_excess_width(clock, values, peak_index, baseline):
    """Quarter-hour-resolved observed brackets; no crossing across a long gap."""
    times = pd.DatetimeIndex(clock)
    y = np.asarray(values, float)
    threshold = baseline+.5*(y[peak_index]-baseline)
    left = np.flatnonzero(y[:peak_index+1] <= threshold)
    right = np.flatnonzero(y[peak_index:] <= threshold)
    result = {"half_threshold": threshold, "left_crossing_missing": not len(left),
              "right_crossing_missing": not len(right), "width_hours": np.nan,
              "crossing_gap_hours": np.nan, "left_crossing_utc": pd.NaT, "right_crossing_utc": pd.NaT}
    if len(left) and len(right):
        a, b = int(left[-1]), peak_index+int(right[0])
        gap = float(pd.Series(times[a:b+1]).diff().dt.total_seconds().max()/3600) if b > a else 0.
        result.update(crossing_gap_hours=gap, left_crossing_utc=times[a], right_crossing_utc=times[b])
        if gap <= 1:
            result["width_hours"] = (times[b]-times[a]).total_seconds()/3600
    return result


def measure_doc_responses(events, flow, doc, sample_minutes=15):
    rows = []
    doc = doc.sort_values("timestamp_utc")
    qsource = flow.loc[flow.flow_valid].sort_values("flow_timestamp_utc")
    for w in events.itertuples(index=False):
        r = w._asdict()
        antecedent_start = w.start_utc-pd.Timedelta(hours=6)
        antecedent_end = w.start_utc-pd.Timedelta(hours=1)
        later = events.loc[events.flow_peak_utc.gt(w.flow_peak_utc), "start_utc"]
        response_end = w.flow_return_utc+pd.Timedelta(hours=24)
        next_start = later.min() if len(later) else pd.NaT
        if len(later):
            # A later, larger candidate can begin before this smaller pulse.
            # Its overlapping envelope must not be silently ignored.
            response_end = max(w.start_utc, min(response_end, next_start))
        r.update(antecedent_start_utc=antecedent_start, antecedent_end_utc=antecedent_end,
                 response_end_utc=response_end,
                 next_flow_candidate_start_utc=next_start,
                 next_flow_envelope_overlaps_start=bool(pd.notna(next_start) and next_start <= w.start_utc),
                 followup_shortened_by_next_flow= response_end < w.flow_return_utc+pd.Timedelta(hours=24))
        dbase = doc.loc[doc.timestamp_utc.between(antecedent_start, antecedent_end)]
        qbase = qsource.loc[qsource.flow_timestamp_utc.between(antecedent_start, antecedent_end)]
        d = doc.loc[doc.timestamp_utc.between(w.start_utc, response_end)].reset_index(drop=True)
        qresponse = qsource.loc[qsource.flow_timestamp_utc.between(w.start_utc, response_end)].reset_index(drop=True)
        q = qsource.loc[qsource.flow_timestamp_utc.between(w.start_utc, w.flow_return_utc)].reset_index(drop=True)
        for name, clock, start, end in (("antecedent_doc", dbase.timestamp_utc, antecedent_start, antecedent_end),
                ("antecedent_flow", qbase.flow_timestamp_utc, antecedent_start, antecedent_end),
                ("response_doc", d.timestamp_utc, w.start_utc, response_end),
                ("response_flow", qresponse.flow_timestamp_utc, w.start_utc, response_end)):
            r.update({f"{name}_{key}": value for key, value in
                      occupied_record_coverage(clock, start, end, sample_minutes).items()})
        r["doc_baseline_mg_l"] = dbase.doc_mg_l.median()
        r["q_baseline_m3_s"] = qbase.q_m3_s.median()
        minimum_baseline = int(np.ceil(.75*(5*60/sample_minutes+1)))
        r["baseline_available"] = len(dbase) >= minimum_baseline and len(qbase) >= minimum_baseline
        r["flow_rise_above_antecedent"] = w.q_peak_m3_s > r["q_baseline_m3_s"]
        r["followup_interrupted_before_flow_return"] = response_end < w.flow_return_utc
        r["dense_response"] = r["baseline_available"] and all(
            r[f"{name}_coverage"] >= .9 and r[f"{name}_max_gap_hours"] <= 1
            for name in ("antecedent_doc", "antecedent_flow", "response_doc", "response_flow"))
        r["timing_eligible"] = (w.isolated_flow_pulse and not w.flow_peak_censored
            and not w.flow_boundary_on_record_edge and r["dense_response"]
            and r["flow_rise_above_antecedent"] and not r["followup_interrupted_before_flow_return"])
        r["positive_doc_response"] = False
        r["doc_peak_on_response_boundary"] = True
        r["doc_recovery_censored"] = True
        if len(d) and r["baseline_available"]:
            peak = int(d.doc_mg_l.to_numpy().argmax())
            excess = d.doc_mg_l.iloc[peak]-r["doc_baseline_mg_l"]
            ties = d.loc[d.doc_mg_l.eq(d.doc_mg_l.iloc[peak]), "timestamp_utc"]
            r.update(doc_peak_utc=d.timestamp_utc.iloc[peak], doc_peak_mg_l=d.doc_mg_l.iloc[peak],
                doc_peak_excess_mg_l=excess,
                doc_peak_on_response_boundary=bool(peak == 0 or peak == len(d)-1),
                positive_doc_response=bool(excess >= max(.5, .1*r["doc_baseline_mg_l"])),
                doc_peak_lag_hours=(d.timestamp_utc.iloc[peak]-w.flow_peak_utc).total_seconds()/3600,
                doc_lag_min_hours=(ties.min()-w.flow_plateau_end_utc).total_seconds()/3600,
                doc_lag_max_hours=(ties.max()-w.flow_plateau_start_utc).total_seconds()/3600)
            dwidth = half_excess_width(d.timestamp_utc, d.doc_mg_l, peak, r["doc_baseline_mg_l"])
            r.update({f"doc_{key}": value for key, value in dwidth.items()})
            qpeak = int(np.flatnonzero(q.flow_timestamp_utc.eq(w.flow_peak_utc))[0]) if len(q) else 0
            if len(q):
                qwidth = half_excess_width(q.flow_timestamp_utc, q.q_m3_s, qpeak, r["q_baseline_m3_s"])
                r.update({f"flow_{key}": value for key, value in qwidth.items()})
            recovery = d.loc[(d.index > peak) & d.doc_mg_l.le(r["doc_baseline_mg_l"]+.1*excess)]
            r["doc_recovery_censored"] = not len(recovery)
            r["doc_recovery_hours"] = (recovery.timestamp_utc.iloc[0]-d.timestamp_utc.iloc[peak]).total_seconds()/3600 if len(recovery) else np.nan
            r["doc_followup_after_peak_hours"] = (response_end-d.timestamp_utc.iloc[peak]).total_seconds()/3600
        r["lag_eligible"] = r["timing_eligible"] and r["positive_doc_response"] and not r["doc_peak_on_response_boundary"]
        r["width_pair_eligible"] = r["lag_eligible"] and np.isfinite(r.get("doc_width_hours", np.nan)) and r.get("flow_width_hours", 0) > 0
        r["doc_flow_width_ratio"] = r["doc_width_hours"]/r["flow_width_hours"] if r["width_pair_eligible"] else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def month_bootstrap(frame, field, draws=5000, seed=42):
    usable = frame.loc[np.isfinite(frame[field])].copy()
    usable["month_block"] = usable.flow_peak_utc.dt.strftime("%Y-%m")
    groups = [g[field].to_numpy() for _, g in usable.groupby("month_block")]
    if not groups:
        return {"median": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_events": 0, "n_months": 0}
    rng = np.random.default_rng(seed)
    medians = [np.median(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])) for _ in range(draws)]
    return {"median": float(usable[field].median()), "ci_low": float(np.quantile(medians, .025)),
            "ci_high": float(np.quantile(medians, .975)), "n_events": len(usable), "n_months": len(groups)}
