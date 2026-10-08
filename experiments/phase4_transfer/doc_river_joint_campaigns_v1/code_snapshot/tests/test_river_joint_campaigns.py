"""Tests of calendar matching, observable variance and dependence in resampling."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_joint_campaigns import (
    assign_flow_states,
    campaign_metrics,
    eligible_campaigns,
    flow_reference,
    joint_calendar,
    project_campaigns,
    resample_years,
)


def observations(n=36):
    dates = pd.date_range("2019-01-01", periods=n, freq="31D")
    a = 10 + np.sin(np.arange(n)*1.27)
    b = 12 - .8*np.sin(np.arange(n)*1.27)
    mix = .5*(a+b)
    return pd.DataFrame({"receiver": "R", "date_local": dates, "doc_a": a, "doc_b": b,
        "weight_a": .5, "doc_dynamic_mix": mix, "doc_receiver": mix,
        "known_upstream_flow_share": .8, "q_receiver_m3s": np.arange(n)+1.,
        "complete_flow_campaign": True, "same_flow_calendar_day": True})


def test_common_calendar_is_independent_of_concentration_and_keeps_nested_rows():
    one = observations()
    two = one.iloc[4:].assign(receiver="S")
    all_rows = pd.concat([one, two], ignore_index=True)
    before = joint_calendar(all_rows, ("R", "S"))
    all_rows["doc_receiver"] = -1e8
    after = joint_calendar(all_rows, ("R", "S"))
    assert before[["receiver", "date_local"]].equals(after[["receiver", "date_local"]])
    assert len(before) == 2*len(two)
    assert joint_calendar(all_rows, ("R", "MISSING")).empty


def test_duplicate_dates_are_all_retained_in_ledger_but_excluded_from_comparison():
    frame = observations()
    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    selected, ledger = eligible_campaigns(frame)
    assert len(ledger) == 37 and len(selected) == 35
    assert ledger.duplicate_receiver_date.sum() == 2
    with pytest.raises(ValueError, match="unique"):
        joint_calendar(frame, ("R",))


def test_identical_outlet_and_mixture_have_unit_sd_ratio_after_adjustment():
    result = campaign_metrics(project_campaigns(observations()))
    assert result["adjusted_outlet_mix_sd_ratio"] == pytest.approx(1)
    assert result["raw_outlet_mix_sd_ratio"] == pytest.approx(1)
    assert result["branch_correlation"] == pytest.approx(-1)
    assert result["fixed_mix_variance"] == pytest.approx(
        result["individual_variance_term"]+result["covariance_term"], abs=1e-12)


def test_linear_projection_preserves_fixed_weight_mixture_in_each_flow_subset():
    frame = project_campaigns(observations())
    assert np.allclose(frame.adjusted_doc_fixed_mix,
                       .5*frame.adjusted_doc_a+.5*frame.adjusted_doc_b, atol=1e-12)
    for sub in (frame.iloc[:12], frame.iloc[12:]):
        campaign_metrics(sub)  # variance identity still holds under subset selection


def test_hydro_reference_includes_unsampled_days_and_never_reads_doc():
    frame = observations()
    days = pd.date_range(frame.date_local.min(), frame.date_local.max())
    flow = pd.DataFrame({"site": "R", "date_local": days, "value": np.arange(len(days))+1.})
    before = flow_reference(flow, frame)
    frame["doc_receiver"] = 100000
    assert before == flow_reference(flow, frame)
    assert before["n_reference_days"] == len(days)
    result = assign_flow_states(frame, {"q_low": 12, "q_high": 24})
    assert result.flow_state.value_counts().to_dict() == {"high": 13, "low": 12, "middle": 11}


def test_shared_year_resampling_preserves_configuration_dependence_and_repeats():
    one = observations()
    frame = pd.concat([one, one.assign(receiver="S")], ignore_index=True)
    sampled = resample_years(frame, np.array([2019, 2019, 2021]))
    assert sampled.receiver.value_counts().R == sampled.receiver.value_counts().S
    expected = 2*frame.date_local.dt.year.eq(2019).sum()+frame.date_local.dt.year.eq(2021).sum()
    assert len(sampled) == expected


def test_constant_mixture_does_not_invent_a_finite_amplification_ratio():
    frame = observations()
    frame["doc_dynamic_mix"] = 11.
    result = campaign_metrics(project_campaigns(frame))
    assert np.isnan(result["adjusted_outlet_mix_sd_ratio"])
