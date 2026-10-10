"""Observed tributary mixing identities and opportunities for unequal arrival."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.analysis.river_mechanisms import detrend
from river_graph.analysis.river_observed_transport import concentration_proxy


def variance_decomposition(a, b, weight):
    """Native-anomaly variance budget; no linearization of log concentration."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.ndim != 1 or a.shape != b.shape or len(a) < 2
            or not np.isfinite(np.column_stack([a, b])).all() or not 0 < weight < 1):
        raise ValueError("aligned finite signals and a positive branch share required")
    va, vb = np.var(a), np.var(b)
    sa, sb = np.sqrt(va), np.sqrt(vb)
    cov = np.mean((a-a.mean())*(b-b.mean()))
    rho = float(np.clip(cov/(sa*sb), -1., 1.)) if min(sa, sb) > 1e-10 else np.nan
    reference = weight*va+(1-weight)*vb
    mixture = np.var(weight*a+(1-weight)*b)
    amplitude = weight*(1-weight)*(sa-sb)**2
    asynchronous = 2*weight*(1-weight)*(sa*sb-cov)
    if not np.isclose(reference-mixture, amplitude+asynchronous, atol=1e-10, rtol=1e-10):
        raise AssertionError("variance identity failed")
    return {"source_a_variance": va, "source_b_variance": vb, "source_covariance": cov,
            "source_rho": rho, "source_reference_variance": reference,
            "mixture_variance": mixture, "amplitude_reduction_variance": amplitude,
            "asynchronous_reduction_variance": asynchronous,
            "mixture_buffer_fraction": (reference-mixture)/reference if reference > 1e-12 else np.nan,
            "amplitude_buffer_fraction": amplitude/reference if reference > 1e-12 else np.nan,
            "asynchronous_buffer_fraction": asynchronous/reference if reference > 1e-12 else np.nan,
            "equal_amplitude_buffer_fraction": 2*weight*(1-weight)*(1-rho),
            "balanced_equal_amplitude_buffer_fraction": .5*(1-rho),
            "synchronized_equal_amplitude_buffer_fraction": 0. if np.isfinite(rho) else np.nan,
            "branch_balance": 4*weight*(1-weight)}


def mixing_records(frame):
    """Fit a common calendar projection and retain connection-specific anomalies."""
    rows, products, fitted = [], [], []
    for pair, f in frame.groupby("pair_id", sort=True):
        f = f.sort_values("month_index").copy()
        if len(f) < 24 or f.month_index.duplicated().any() or f.weight_a.nunique() != 1:
            raise ValueError("fixed-share connection with at least 24 unique months required")
        r, w = f.iloc[0], float(f.weight_a.iloc[0])
        f["mixture_doc"] = w*f.doc_a_now+(1-w)*f.doc_b_now
        columns = ("doc_a_now", "doc_b_now", "mixture_doc", "y_true")
        values = f[list(columns)].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("common finite nonnegative DOC required")
        anomalies = [detrend(f[c], f.date) for c in columns]
        a, b, mixture, target = anomalies
        np.testing.assert_allclose(mixture, w*a+(1-w)*b, atol=1e-10, rtol=1e-10)
        metrics = variance_decomposition(a, b, w)
        vm, vr, vt = metrics["mixture_variance"], metrics["source_reference_variance"], np.var(target)
        metrics.update({"outlet_variance": vt,
                        "outlet_reference_variance_ratio": vt/vr if vr > 1e-12 else np.nan,
                        "outlet_mixture_log_sd_ratio": .5*np.log(vt/vm) if min(vt, vm) > 1e-12 else np.nan,
                        "outlet_reference_log_sd_ratio": .5*np.log(vt/vr) if min(vt, vr) > 1e-12 else np.nan})
        log_signals = [detrend(np.log1p(f[c]), f.date) for c in columns]
        log_reference = w*np.var(log_signals[0])+(1-w)*np.var(log_signals[1])
        log_mixture, log_outlet = np.var(log_signals[2]), np.var(log_signals[3])
        metrics.update({
            "logscale_outlet_reference_log_sd_ratio": .5*np.log(log_outlet/log_reference) if min(log_outlet, log_reference) > 1e-12 else np.nan,
            "logscale_outlet_mixture_log_sd_ratio": .5*np.log(log_outlet/log_mixture) if min(log_outlet, log_mixture) > 1e-12 else np.nan})
        rows.append({k: r[k] for k in ("pair_id", "target", "huc4", "component", "cluster",
                                      "weight_a", "source_drainage_coverage", "path_a_km", "path_b_km")}
                    | {"n_months": len(f), **metrics})
        for name, v in zip(("source_a_anomaly", "source_b_anomaly", "mixture_anomaly", "outlet_anomaly"), anomalies, strict=True):
            f[name] = v
        products.append(f)
        dates = pd.DatetimeIndex(f.date)
        year = dates.year.to_numpy(float)
        design = np.column_stack([np.ones(len(f)), np.sin(2*np.pi*dates.month/12),
                                  np.cos(2*np.pi*dates.month/12), year-year.mean()])
        fitted.append({"pair_id": pair, "year_center": year.mean(),
                       "design_rank": int(np.linalg.matrix_rank(design)),
                       "columns": ["intercept", "month_sin", "month_cos", "centered_year"],
                       "coefficients": {c: np.linalg.lstsq(design, f[c], rcond=None)[0].tolist() for c in columns}})
    return pd.DataFrame(rows), pd.concat(products, ignore_index=True), fitted


def arrival_opportunity(frame, max_path):
    """Exact difference between branch-specific and mean-path interpolation."""
    f = frame.copy()
    f["same_month_proxy"] = concentration_proxy(f, "same_month", 1., max_path)
    f["mean_delay_proxy"] = concentration_proxy(f, "mean_delay", 1., max_path)
    f["branch_arrival_proxy"] = concentration_proxy(f, "branch_arrival", 1., max_path)
    f["path_contrast"] = (f.path_a_km-f.path_b_km)/max_path
    f["source_change_contrast"] = (f.doc_a_previous-f.doc_a_now)-(f.doc_b_previous-f.doc_b_now)
    f["arrival_gap"] = f.weight_a*(1-f.weight_a)*f.path_contrast*f.source_change_contrast
    np.testing.assert_allclose(f.branch_arrival_proxy-f.mean_delay_proxy, f.arrival_gap, atol=1e-10, rtol=1e-10)
    f["arrival_gap_abs"] = abs(f.arrival_gap)
    f["arrival_gap_relative"] = f.arrival_gap_abs/(1+f.same_month_proxy)
    f["arrival_gap_over_1pct"] = (f.arrival_gap_relative >= .01).astype(float)
    f["arrival_gap_over_5pct"] = (f.arrival_gap_relative >= .05).astype(float)
    return f


def flow_budget_records(frame, qa, qb, qt):
    """Measured monthly flow supports a screening ledger, not a load budget."""
    f = frame.copy()
    for name, values in zip(("flow_a_cfs", "flow_b_cfs", "flow_receiver_cfs"), (qa, qb, qt), strict=True):
        values = np.asarray(values, float)
        if values.shape != (len(f),):
            raise ValueError("one measured or missing flow per row required")
        f[name] = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    f["three_flows_measured"] = f[["flow_a_cfs", "flow_b_cfs", "flow_receiver_cfs"]].notna().all(axis=1)
    f["flow_closure"] = (f.flow_a_cfs+f.flow_b_cfs)/f.flow_receiver_cfs
    f["area_screen"] = f.source_drainage_coverage >= .8
    f["monthly_flow_screen"] = f.three_flows_measured & f.flow_closure.between(.8, 1.2)
    f["budget_screen"] = f.area_screen & f.monthly_flow_screen
    f["flow_weight_a"] = f.flow_a_cfs/(f.flow_a_cfs+f.flow_b_cfs)
    f["flow_mixture_doc"] = f.flow_weight_a*f.doc_a_now+(1-f.flow_weight_a)*f.doc_b_now
    f["apparent_departure_mg_l"] = f.y_true-f.flow_mixture_doc
    f["area_flow_share_difference"] = f.flow_weight_a-f.weight_a
    return f
