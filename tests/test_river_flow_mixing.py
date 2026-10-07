"""Physical mixture arithmetic, sampling clocks and non-closure diagnostics."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_flow_mixing import (
    attach_daily_flows,
    summarize_mixing_window,
)


def campaigns(n=5):
    times = pd.date_range("2020-04-01 23:30", periods=n, tz="UTC")
    return pd.DataFrame({"source_a": "A", "source_b": "B", "receiver": "R",
                         "source_a_time_utc": times, "source_b_time_utc": times,
                         "receiver_time_utc": times, "doc_a": np.arange(1., n+1),
                         "doc_b": np.arange(n, 0., -1), "doc_receiver": (n+1)/2})


def flows(qa=1., qb=1., qr=2., n=5):
    dates = pd.date_range("2020-04-02", periods=n)
    return pd.concat([pd.DataFrame({"site": site, "date_local": dates, "value": q})
                      for site, q in (("A", qa), ("B", qb), ("R", qr))], ignore_index=True)


def test_flow_weighting_and_flux_units_keep_nonclosure():
    result = attach_daily_flows(campaigns(), flows(qa=3., qb=1., qr=2.))
    assert result.q_date_a.iloc[0] == pd.Timestamp("2020-04-02")  # UTC+1, not UTC day
    assert result.weight_a.iloc[0] == .75
    assert result.doc_dynamic_mix.iloc[0] == 2
    assert result.known_upstream_flux_gcs.iloc[0] == 8
    assert result.receiver_flux_gcs.iloc[0] == 6
    assert result.known_upstream_flow_share.iloc[0] == 2  # never clip to 1
    assert result.unaccounted_flow_m3s.iloc[0] == -2


def test_antiphase_branch_concentrations_cancel_mixture_variation():
    combined = attach_daily_flows(campaigns(), flows())
    metrics, detailed = summarize_mixing_window(combined)
    assert metrics["individual_variance_term"] == 1.25
    assert metrics["covariance_term"] == -1.25
    assert metrics["fixed_mix_variance"] == 0
    assert metrics["dynamic_mix_cv"] == 0
    assert detailed.doc_fixed_mix.eq(3).all()
    assert np.isnan(metrics["receiver_mix_correlation"])


def test_missing_daily_flow_is_not_interpolated():
    flow = flows()
    flow.loc[(flow.site == "B") & (flow.date_local == pd.Timestamp("2020-04-04")), "value"] = np.nan
    result = attach_daily_flows(campaigns(), flow)
    assert len(result) == 5 and result.complete_flow_campaign.sum() == 4
    assert np.isnan(result.doc_dynamic_mix.iloc[2])
    with pytest.raises(ValueError, match="five"):
        summarize_mixing_window(result)


def test_zero_combined_flow_has_no_defined_mixture():
    result = attach_daily_flows(campaigns(), flows(qa=0., qb=0.))
    assert not result.valid_upstream_flow.any()
    assert result.doc_dynamic_mix.isna().all()


def test_duplicate_daily_gauges_are_not_many_to_many_joined():
    flow = flows()
    with pytest.raises(ValueError, match="one record"):
        attach_daily_flows(campaigns(), pd.concat([flow, flow.iloc[:1]]))


def test_cv_and_absolute_sd_can_move_in_opposite_directions():
    frame = campaigns()
    frame["doc_a"] = [1., 2., 3., 4., 5.]
    frame["doc_b"] = frame.doc_a
    frame["doc_receiver"] = 100 + 2 * frame.doc_a
    metrics, _ = summarize_mixing_window(attach_daily_flows(frame, flows()))
    assert metrics["receiver_minus_dynamic_mix_cv"] < 0
    assert metrics["receiver_minus_dynamic_mix_sd"] > 0
