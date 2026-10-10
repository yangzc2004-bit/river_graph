"""Measured branch mixing and partial-flow coverage at monitored confluences."""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd


def attach_daily_flows(campaigns: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    """Match each sample's fixed UTC+1 date without interpolating daily flow."""
    if flow.duplicated(["site", "date_local"]).any():
        raise ValueError("Discharge must have one record per station and local day")
    if flow.date_local.dt.tz is not None:
        raise ValueError("Discharge dates must be source-calendar dates without timezone")
    lookup = flow.set_index(["site", "date_local"]).value
    result = campaigns.copy().reset_index(drop=True)
    for role, station, timestamp in (
        ("a", "source_a", "source_a_time_utc"),
        ("b", "source_b", "source_b_time_utc"),
        ("receiver", "receiver", "receiver_time_utc"),
    ):
        if result[timestamp].dt.tz is None:
            raise ValueError("Laboratory sample timestamps must retain UTC timezone")
        dates = result[timestamp].dt.tz_convert(timezone(timedelta(hours=1))).dt.tz_localize(None).dt.normalize()
        result[f"q_date_{role}"] = dates
        index = pd.MultiIndex.from_arrays([result[station], dates], names=["site", "date_local"])
        result[f"q_{role}_m3s"] = lookup.reindex(index).to_numpy()
    qa, qb, qr = (result[f"q_{r}_m3s"] for r in ("a", "b", "receiver"))
    ca, cb, cr = (result[f"doc_{r}"] for r in ("a", "b", "receiver"))
    upstream = np.isfinite(qa) & np.isfinite(qb) & qa.ge(0) & qb.ge(0) & (qa + qb).gt(0)
    receiver = np.isfinite(qr) & qr.gt(0)
    doc = np.isfinite(ca) & np.isfinite(cb) & np.isfinite(cr) & ca.ge(0) & cb.ge(0) & cr.ge(0)
    result["valid_upstream_flow"] = upstream
    result["valid_receiver_flow"] = receiver
    result["complete_flow_campaign"] = upstream & receiver & doc
    result["same_flow_calendar_day"] = result.q_date_a.eq(result.q_date_b) & result.q_date_a.eq(result.q_date_receiver)
    result["weight_a"] = qa.div((qa + qb).where(upstream))
    result["doc_dynamic_mix"] = result.weight_a * ca + (1 - result.weight_a) * cb
    # mg C/L × m³/s = g C/s. These are partial upstream fluxes, not a complete balance.
    result["known_upstream_flux_gcs"] = (qa * ca + qb * cb).where(upstream)
    result["receiver_flux_gcs"] = (qr * cr).where(receiver)
    result["known_upstream_flow_share"] = (qa + qb).div(qr.where(receiver)).where(upstream)
    result["unaccounted_flow_m3s"] = (qr - qa - qb).where(upstream & receiver)
    result["receiver_minus_dynamic_mix"] = cr - result.doc_dynamic_mix
    return result


def concentration_stats(values: pd.Series) -> dict:
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    return {"mean": mean, "sd": sd, "cv": sd / mean if mean > 0 else np.nan}


def summarize_mixing_window(campaigns: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare all concentrations on identical complete-flow campaigns."""
    selected = campaigns.loc[campaigns.complete_flow_campaign].copy()
    if len(selected) < 5:
        raise ValueError("At least five complete-flow campaigns are needed")
    w = float(selected.weight_a.mean())
    selected["fixed_weight_a"] = w
    selected["doc_fixed_mix"] = w * selected.doc_a + (1 - w) * selected.doc_b
    result = {"n_complete_flow_campaigns": len(selected), "mean_weight_a": w,
              "weight_a_sd": float(selected.weight_a.std(ddof=1)),
              "flow_share_min": float(selected.known_upstream_flow_share.min()),
              "flow_share_median": float(selected.known_upstream_flow_share.median()),
              "flow_share_max": float(selected.known_upstream_flow_share.max()),
              "n_flow_share_gt1": int(selected.known_upstream_flow_share.gt(1).sum()),
              "n_same_flow_day": int(selected.same_flow_calendar_day.sum())}
    for name, column in (("source_a", "doc_a"), ("source_b", "doc_b"),
                         ("dynamic_mix", "doc_dynamic_mix"), ("fixed_mix", "doc_fixed_mix"),
                         ("receiver", "doc_receiver")):
        result.update({f"{name}_{metric}": value for metric, value in concentration_stats(selected[column]).items()})
    result["mean_upstream_cv"] = (result["source_a_cv"] + result["source_b_cv"]) / 2
    result["receiver_minus_mean_upstream_cv"] = result["receiver_cv"] - result["mean_upstream_cv"]
    result["dynamic_mix_minus_mean_upstream_cv"] = result["dynamic_mix_cv"] - result["mean_upstream_cv"]
    result["receiver_minus_dynamic_mix_cv"] = result["receiver_cv"] - result["dynamic_mix_cv"]
    result["receiver_minus_dynamic_mix_sd"] = result["receiver_sd"] - result["dynamic_mix_sd"]
    result["receiver_minus_dynamic_mix_mean"] = result["receiver_mean"] - result["dynamic_mix_mean"]
    result["receiver_mix_rmse"] = float(np.sqrt(np.mean(selected.receiver_minus_dynamic_mix ** 2)))
    identifiable = selected.doc_receiver.std() > 0 and selected.doc_dynamic_mix.std() > 0
    result["receiver_mix_correlation"] = selected.doc_receiver.corr(selected.doc_dynamic_mix) if identifiable else np.nan
    result["fixed_mix_variance"] = float(selected.doc_fixed_mix.var(ddof=1))
    result["individual_variance_term"] = w**2 * float(selected.doc_a.var(ddof=1)) + (1-w)**2 * float(selected.doc_b.var(ddof=1))
    result["covariance_term"] = 2 * w * (1-w) * float(selected.doc_a.cov(selected.doc_b))
    reconstructed = result["individual_variance_term"] + result["covariance_term"]
    if not np.isclose(result["fixed_mix_variance"], reconstructed, rtol=1e-10, atol=1e-10):
        raise ValueError("Fixed-weight mixture variance decomposition failed")
    return result, selected
