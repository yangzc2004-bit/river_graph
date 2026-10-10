"""Source timing and shared volume/flow responses on fixed river corridors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from river_graph.analysis.river_storage_placement import _continuous_peak
from river_graph.analysis.river_storage_transport import gaussian_storage_response

PHASES = (-1., 0., 1.)
VOLUMES = (1., 2.)
FLOWS = (1., 2.)
MIXING = (0., .5)


def junction_response(branch_a, branch_b, common, weight, *, phase=0., volume=1.,
                      flow=1., mixing=0., sigma=.15, dt=.0025, keep_curve=False):
    """Route identically shaped source pulses with explicit phase offsets.

    Mean path scaling is fixed before changing downstream V/Q. Source phases
    change input clocks, never the nonnegative causal routing delays.
    """
    values = np.array([branch_a, branch_b, common, weight, phase, volume, flow, mixing, sigma, dt])
    if not np.isfinite(values).all():
        raise ValueError("finite geometry and scenario parameters required")
    if min(branch_a, branch_b, volume, flow, sigma, dt) <= 0 or common < 0 or not 0 < weight < 1:
        raise ValueError("positive branches, shares, volume/flow and resolution required")
    if not -1 <= phase <= 1 or not 0 <= mixing <= 1:
        raise ValueError("phase in [-1,1] and mixing fraction in [0,1] required")
    weights = np.array([weight, 1-weight])
    mean = weights@np.array([branch_a, branch_b])+common
    branches = np.array([branch_a, branch_b])/mean
    c = common/mean
    bbar = weights@branches
    source_phases = phase*(branches-bbar)
    shared_mean = c*volume/flow
    tau = mixing*shared_mean
    delays = branches+(1-mixing)*shared_mean
    translated_centres = source_phases+delays
    low = translated_centres.min()-9*sigma
    high = translated_centres.max()+9*sigma+32*tau
    time = np.arange(np.floor(low/dt), np.ceil(high/dt)+1)*dt

    def evaluate(t):
        return sum(w*gaussian_storage_response(np.asarray(t)-p, d, sigma, tau)
                   for w, p, d in zip(weights, source_phases, delays, strict=True))

    response = evaluate(time)
    area = np.trapezoid(response, time)
    centroid = np.trapezoid(time*response, time)/area
    variance = np.trapezoid((time-centroid)**2*response, time)/area
    cdf = cumulative_trapezoid(response/area, time, initial=0)
    q10, q50, q90 = np.interp([.1, .5, .9], cdf, time)
    peak, first, last, count = _continuous_peak(time, response, evaluate)
    branch_var = weights@((branches-bbar)**2)
    mean_expected = bbar+shared_mean
    variance_expected = sigma*sigma+(1+phase)**2*branch_var+tau*tau
    row = {"source_phase_factor": phase, "volume_factor": volume,
           "flow_factor": flow, "mixing_fraction": mixing, "input_sd": sigma,
           "shared_mean_ratio": volume/flow, "shared_reference_mean": c,
           "shared_mean": shared_mean, "shared_tau": tau,
           "source_phase_a": source_phases[0], "source_phase_b": source_phases[1],
           "source_phase_weighted_mean": weights@source_phases,
           "junction_arrival_gap": abs((1+phase)*(branches[0]-branches[1])),
           "branch_variance": branch_var,
           "arrival_variance": (1+phase)**2*branch_var,
           "distributed_variance": tau*tau, "pulse_peak": peak,
           "peak_time": first, "last_near_equal_peak_time": last,
           "peak_time_is_ambiguous": count > 1, "near_equal_peak_count": count,
           "pulse_centroid": centroid, "analytic_centroid": mean_expected,
           "pulse_sd": np.sqrt(variance), "analytic_sd": np.sqrt(variance_expected),
           "t10": q10, "t50": q50, "t90": q90, "duration_80": q90-q10,
           "anomaly_area_fraction": area/(np.sqrt(2*np.pi)*sigma), "steady_gain": 1., "dt": dt}
    curve = pd.DataFrame()
    if keep_curve:
        branch_curves = [w*gaussian_storage_response(time-p, d, sigma, tau)
                         for w, p, d in zip(weights, source_phases, delays, strict=True)]
        curve = pd.DataFrame({"relative_time": time[::4], "outlet_anomaly": response[::4],
                              "weighted_a": branch_curves[0][::4], "weighted_b": branch_curves[1][::4]})
    return row, curve


def junction_contrasts(scenarios):
    """Pair within each geometry, share rule and duration; preserve every setting."""
    keys = ["cohort", "case_id", "flow_rule", "input_sd"]
    rows = []
    for _, frame in scenarios.groupby(keys, sort=True):
        reference = frame.set_index(["source_phase_factor", "volume_factor", "flow_factor", "mixing_fraction"])
        if not reference.index.is_unique:
            raise ValueError("one row per geometry and scenario required")
        base = reference.loc[(0., 1., 1., 0.)]
        for _, row in frame.iterrows():
            phase = reference.loc[(row.source_phase_factor, 1., 1., 0.)]
            translation = reference.loc[(row.source_phase_factor, row.volume_factor, row.flow_factor, 0.)]
            same_phase_base = reference.loc[(row.source_phase_factor, 1., 1., row.mixing_fraction)]
            aligned = reference.loc[(-1., row.volume_factor, row.flow_factor, row.mixing_fraction)]
            out = row.to_dict()
            out.update({
                "joint_peak_reduction_pct": 100*(1-row.pulse_peak/base.pulse_peak),
                "phase_peak_reduction_pct": 100*(1-phase.pulse_peak/base.pulse_peak),
                "mixing_peak_reduction_pct": 100*(1-row.pulse_peak/translation.pulse_peak),
                "mixing_duration_change_pct": 100*(row.duration_80/translation.duration_80-1),
                "volume_flow_peak_reduction_pct": 100*(1-row.pulse_peak/same_phase_base.pulse_peak),
                "volume_flow_duration_change_pct": 100*(row.duration_80/same_phase_base.duration_80-1),
                "arrival_peak_reduction_from_aligned_pct": 100*(1-row.pulse_peak/aligned.pulse_peak),
                "joint_log_peak_change": np.log(row.pulse_peak/base.pulse_peak),
                "phase_log_peak_change": np.log(phase.pulse_peak/base.pulse_peak),
                "mixing_log_peak_change": np.log(row.pulse_peak/translation.pulse_peak)})
            rows.append(out)
    return pd.DataFrame(rows)


def field_transit_proxies(transects, chemistry):
    """Survey rectangles and Q yield a diagnostic, not measured wetted area."""
    if transects.duplicated(["confluence", "reach", "transect"]).any():
        raise ValueError("one row per surveyed transect required")
    data = transects.copy()
    observed = data.mean_depth_m.notna()
    if (data.width_m <= 0).any() or (data.loc[observed, "mean_depth_m"] <= 0).any():
        raise ValueError("positive observed survey width/depth required")
    data["rectangular_area_proxy_m2"] = data.width_m*data.mean_depth_m
    grouped = data.groupby(["confluence", "reach"]).agg(
        area_proxy_m2=("rectangular_area_proxy_m2", "mean"),
        n_survey_units=("transect", "size"), n_usable_proxy_units=("rectangular_area_proxy_m2", "count"),
        n_missing_depth_points=("n_missing_depth_points", "sum"))
    rows = []
    fall = chemistry.loc[chemistry.season.eq("fall")].copy()
    if fall.confluence.duplicated().any():
        raise ValueError("one fall chemistry row per junction required")
    for _, row in fall.iterrows():
        main = grouped.loc[(row.confluence, "1.upstream")]
        receiver = grouped.loc[(row.confluence, "2.downstream")]
        if min(main.area_proxy_m2, receiver.area_proxy_m2, row.main_flow_ls, row.receiver_flow_ls) <= 0:
            raise ValueError("positive proxy and flow required")
        area_ratio = receiver.area_proxy_m2/main.area_proxy_m2
        flow_ratio = row.receiver_flow_ls/row.main_flow_ls
        rows.append({"confluence": row.confluence, "season": "fall",
            "main_area_proxy_m2": main.area_proxy_m2, "receiver_area_proxy_m2": receiver.area_proxy_m2,
            "area_proxy_ratio": area_ratio, "flow_ratio": flow_ratio,
            "same_length_transit_proxy_ratio": area_ratio/flow_ratio,
            "reported_tracer_transit_ratio": row.receiver_main_residence_ratio,
            "receiver_main_depth_ratio": row.receiver_main_depth_ratio,
            "receiver_main_width_ratio": row.receiver_main_width_ratio,
            "doc_receiver_minus_mix_pct": row.doc_receiver_minus_mix_pct,
            "main_survey_units": int(main.n_survey_units), "receiver_survey_units": int(receiver.n_survey_units),
            "main_usable_proxy_units": int(main.n_usable_proxy_units),
            "receiver_usable_proxy_units": int(receiver.n_usable_proxy_units),
            "n_missing_depth_points": int(main.n_missing_depth_points+receiver.n_missing_depth_points),
            "measurement_scope": "rectangular proxy; campaign flow; not synchronized event or fitted storage"})
    return pd.DataFrame(rows), data
