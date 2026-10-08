"""Observed DOC combination on a common calendar at mapped confluences."""

from __future__ import annotations

import numpy as np
import pandas as pd

DOC_COLUMNS = ("doc_a", "doc_b", "doc_dynamic_mix", "doc_receiver")
METRICS = (
    "adjusted_outlet_mix_sd_ratio", "raw_outlet_mix_sd_ratio",
    "adjusted_outlet_branch_sd_ratio", "branch_correlation",
    "fixed_mix_variance_reduction_pct", "mean_outlet_minus_mix",
    "median_upstream_flow_share",
)


def eligible_campaigns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do not count multiple activities on a receiving date as independent dates."""
    result = frame.copy()
    result["duplicate_receiver_date"] = result.duplicated(
        ["receiver", "date_local"], keep=False)
    result["eligible"] = (result.complete_flow_campaign &
                          result.same_flow_calendar_day &
                          ~result.duplicate_receiver_date)
    return result.loc[result.eligible].copy(), result


def joint_calendar(frame: pd.DataFrame, receivers: tuple[str, ...]) -> pd.DataFrame:
    """Availability-only intersection; never select a date by a concentration."""
    if frame.duplicated(["receiver", "date_local"]).any():
        raise ValueError("Receiving dates must be unique within a configuration")
    missing = set(receivers) - set(frame.receiver)
    if missing:
        return frame.iloc[:0].copy()
    dates = set.intersection(*(set(frame.loc[frame.receiver.eq(site), "date_local"])
                               for site in receivers))
    return frame.loc[frame.receiver.isin(receivers) & frame.date_local.isin(dates)].copy()


def seasonal_design(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)
    day = dates.dt.dayofyear.to_numpy(dtype=float)
    phase = 2 * np.pi * (day - 1) / 365.25
    elapsed = (dates - dates.min()).dt.total_seconds().to_numpy() / (86400 * 365.25)
    return np.column_stack((np.ones(len(dates)), np.sin(phase), np.cos(phase),
                            elapsed - elapsed.mean()))


def project_campaigns(frame: pd.DataFrame) -> pd.DataFrame:
    """One linear projection for every series, preserving linear-mixture algebra."""
    if len(frame) < 8:
        raise ValueError("At least eight sampling dates are needed")
    result = frame.copy()
    design = seasonal_design(result.date_local)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Season/time design is not identifiable")
    w = float(result.weight_a.mean())
    result["doc_fixed_mix"] = w * result.doc_a + (1 - w) * result.doc_b
    result["fixed_weight_a"] = w
    columns = list(DOC_COLUMNS) + ["doc_fixed_mix"]
    values = result[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("DOC must be finite on every compared date")
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    for i, column in enumerate(columns):
        result[f"adjusted_{column}"] = residual[:, i]
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else np.nan


def campaign_metrics(frame: pd.DataFrame) -> dict:
    """SD measures fluctuations; CV and mean concentration are separate outputs."""
    n = len(frame)
    if n < 5:
        raise ValueError("At least five campaign dates are needed")
    w = float(frame.fixed_weight_a.iloc[0])
    a, b = (frame[f"adjusted_doc_{s}"].to_numpy() for s in ("a", "b"))
    variance_a, variance_b = np.var(a, ddof=1), np.var(b, ddof=1)
    covariance = float(np.cov(a, b, ddof=1)[0, 1])
    fixed_var = float(frame.adjusted_doc_fixed_mix.var(ddof=1))
    individual = w**2 * variance_a + (1-w)**2 * variance_b
    cross = 2*w*(1-w)*covariance
    if not np.isclose(fixed_var, individual + cross, atol=1e-9, rtol=1e-9):
        raise ValueError("Fixed-mixture variance decomposition failed")
    reference = w * variance_a + (1-w) * variance_b
    adjusted_sd = {c: float(frame[f"adjusted_{c}"].std(ddof=1)) for c in DOC_COLUMNS}
    raw_sd = {c: float(frame[c].std(ddof=1)) for c in DOC_COLUMNS}
    result = {
        "n_campaigns": n, "n_years": int(frame.date_local.dt.year.nunique()),
        "first_date": str(frame.date_local.min().date()),
        "last_date": str(frame.date_local.max().date()),
        "adjusted_outlet_mix_sd_ratio": _ratio(adjusted_sd["doc_receiver"], adjusted_sd["doc_dynamic_mix"]),
        "raw_outlet_mix_sd_ratio": _ratio(raw_sd["doc_receiver"], raw_sd["doc_dynamic_mix"]),
        "adjusted_outlet_branch_sd_ratio": _ratio(adjusted_sd["doc_receiver"], np.sqrt(reference)),
        "branch_correlation": _ratio(covariance, np.sqrt(variance_a * variance_b)),
        "fixed_mix_variance_reduction_pct": 100 * (1 - _ratio(fixed_var, reference)),
        "fixed_mix_variance": fixed_var, "individual_variance_term": individual,
        "covariance_term": cross, "mean_weight_a": w,
        "mean_outlet_minus_mix": float((frame.doc_receiver-frame.doc_dynamic_mix).mean()),
        "median_upstream_flow_share": float(frame.known_upstream_flow_share.median()),
    }
    for c in DOC_COLUMNS:
        result[c+"_mean"] = float(frame[c].mean())
        result[c+"_raw_sd"] = raw_sd[c]
        result[c+"_adjusted_sd"] = adjusted_sd[c]
        result[c+"_cv"] = _ratio(raw_sd[c], result[c+"_mean"])
        result[c+"_season_time_variance_fraction"] = 1 - _ratio(adjusted_sd[c]**2, raw_sd[c]**2)
    return result


def flow_reference(flow: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Hydro-only thresholds include unsampled days within the comparison span."""
    site = str(frame.receiver.iloc[0])
    values = flow.loc[flow.site.eq(site) & flow.date_local.between(
        frame.date_local.min(), frame.date_local.max()) & flow.value.gt(0), "value"]
    if values.empty:
        raise ValueError("No positive receiver-flow reference")
    low, high = np.quantile(values, [1/3, 2/3])
    return {"receiver": site, "q_low": float(low), "q_high": float(high),
            "n_reference_days": len(values)}


def assign_flow_states(frame: pd.DataFrame, reference: dict) -> pd.DataFrame:
    result = frame.copy()
    q = result.q_receiver_m3s
    result["flow_state"] = np.select([q.le(reference["q_low"]), q.ge(reference["q_high"])],
                                      ["low", "high"], default="middle")
    return result


def resample_years(frame: pd.DataFrame, selected_years: np.ndarray) -> pd.DataFrame:
    """Keep all configurations in a sampled calendar year together, including repeats."""
    return pd.concat([frame.loc[frame.date_local.dt.year.eq(year)]
                       for year in selected_years], ignore_index=True)
