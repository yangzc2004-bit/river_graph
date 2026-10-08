"""Observational covariance and peak denominators, independent of model training."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_observed_synchrony import (
    coordination_statistics,
    excursion_block_interval,
    within_receiver_association,
)


def sample():
    rng = np.random.default_rng(93)
    a = rng.normal(size=(120, 3))+8
    y = .3*a[:, 0]+.7*a[:, 1]
    return a, y, np.array([.2, .3, .5]), pd.date_range("2000-01-01", periods=120, freq="MS")


def test_coherence_is_direct_weighted_pair_correlation():
    a, y, w, dates = sample()
    point, _ = coordination_statistics(a, y, w, dates, adjustment="raw")
    centered = a-a.mean(axis=0)
    covariance = np.cov(centered, rowvar=False, bias=True)
    sd = a.std(axis=0)
    num = den = 0.
    for i in range(3):
        for j in range(i+1, 3):
            num += w[i]*w[j]*covariance[i, j]
            den += w[i]*w[j]*sd[i]*sd[j]
    assert point["source_coherence"] == pytest.approx(num/den)
    assert point["outlet_sync_log_sd_ratio"] == pytest.approx(np.log(y.std()/(w@sd)))


def test_receiving_reference_avoids_mixture_covariance_denominator():
    u = np.tile(np.arange(20), 6).astype(float)-9.5
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    same, _ = coordination_statistics(np.c_[u, u], u, [.5, .5], dates, projected=True)
    opposite, _ = coordination_statistics(np.c_[u, -u], u, [.5, .5], dates, projected=True)
    assert same["source_coherence"] == pytest.approx(1)
    assert opposite["source_coherence"] == pytest.approx(-1)
    assert same["outlet_sync_log_sd_ratio"] == opposite["outlet_sync_log_sd_ratio"]
    assert np.isnan(opposite["outlet_mix_log_sd_ratio"])


def test_peak_groups_count_receiver_independently_and_skip_small_groups():
    a, y, w, dates = sample()
    point, trace = coordination_statistics(a, y, w, dates, adjustment="raw")
    source = a-a.mean(axis=0)
    count = (source > np.quantile(source, .75, axis=0)).sum(axis=1)
    rec = (y-y.mean()) > np.quantile(y-y.mean(), .75)
    c, s = count >= 2, count == 1
    assert point["n_coincident"] == int(c.sum())
    assert point["n_solo"] == int(s.sum())
    assert point["peak_risk_difference"] == pytest.approx(rec[c].mean()-rec[s].mean())
    np.testing.assert_array_equal(trace.receiver_high, rec)
    small, _ = coordination_statistics(a, y, w, dates, adjustment="raw", minimum_group=120)
    assert not small["peak_comparison_eligible"] and np.isnan(small["peak_risk_difference"])


def test_order_common_scale_and_source_permutation_invariance():
    a, y, w, dates = sample()
    original, _ = coordination_statistics(a, y, w, dates)
    shifted, _ = coordination_statistics(a[:, ::-1]*7, y*7, w[::-1], dates)
    reversed_, _ = coordination_statistics(a[::-1], y[::-1], w, dates[::-1])
    for name in ("source_coherence", "outlet_sync_log_sd_ratio", "outlet_mix_correlation", "peak_risk_difference"):
        assert original[name] == pytest.approx(shifted[name], abs=1e-10)
        assert original[name] == pytest.approx(reversed_[name], abs=1e-10)


def test_calendar_projection_removes_shared_seasonal_signal():
    a, y, w, dates = sample()
    signal = 5*np.sin(2*np.pi*dates.month/12)+(dates.year-2000)*.005+20
    original, _ = coordination_statistics(a, y, w, dates)
    shifted, _ = coordination_statistics(a+np.asarray(signal)[:, None], y+signal, w, dates)
    assert original["source_coherence"] == pytest.approx(shifted["source_coherence"], abs=1e-10)
    assert original["outlet_sync_log_sd_ratio"] == pytest.approx(shifted["outlet_sync_log_sd_ratio"], abs=1e-10)


def test_constant_input_is_unknown_not_zero_correlation():
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    point, _ = coordination_statistics(np.ones((24, 2)), np.ones(24), [1, 1], dates, adjustment="raw")
    assert np.isnan(point["source_coherence"])
    assert np.isnan(point["outlet_sync_log_sd_ratio"])
    assert not point["peak_comparison_eligible"]


def test_within_network_relationship_excludes_static_between_network_offset():
    f = pd.DataFrame({"target": ["a", "a", "b", "b", "c", "c"],
        "component": [0, 0, 1, 1, 2, 2], "source_coherence": [0, 1, 10, 11, 20, 21],
        "outlet_sync_log_sd_ratio": [100, 102, 80, 82, 60, 62]})
    result, _ = within_receiver_association(f, draws=100)
    assert result["correlation"] == pytest.approx(1)
    assert result["slope"] == pytest.approx(2)
    # Duplicating an entire receiver's records leaves its total weight unchanged.
    repeated = pd.concat([f, f[f.target.eq("a")]])
    again, _ = within_receiver_association(repeated, draws=100)
    assert again["correlation"] == pytest.approx(result["correlation"])


def test_time_block_interval_matches_independent_whole_block_resampling():
    a, y, w, dates = sample()
    _, trace = coordination_statistics(a, y, w, dates, adjustment="raw")
    actual = excursion_block_interval(trace, dates.year, draws=300)
    years = np.unique(dates.year)
    rng = np.random.default_rng(42)
    draws = rng.multinomial(len(years), np.full(len(years), 1/len(years)), size=300)
    values = []
    for count in draws:
        f = pd.concat([trace[dates.year == year] for year, n in zip(years, count, strict=True) for _ in range(n)])
        c, s = f.coincident_sources, f.solo_source
        if min(c.sum(), s.sum()) >= 5:
            values.append(f.loc[c, "receiver_high"].mean()-f.loc[s, "receiver_high"].mean())
    lo, hi = np.quantile(values, [.025, .975])
    assert actual["peak_ci_low"] == pytest.approx(lo)
    assert actual["peak_ci_high"] == pytest.approx(hi)


def test_duplicate_dates_and_negative_native_doc_are_rejected():
    a, y, w, dates = sample()
    with pytest.raises(ValueError, match="common dates"):
        coordination_statistics(a, y, w, dates[:1].repeat(len(dates)))
    with pytest.raises(ValueError, match="nonnegative"):
        coordination_statistics(a-20, y, w, dates)
