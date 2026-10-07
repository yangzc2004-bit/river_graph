"""Scientific contracts for observed hourly peaks and censored waveform widths."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_hourly_response import (
    optical_quality,
    peak_difference,
    shifted_flow_shape,
    waveform_profile,
)


def clock(n):
    return pd.date_range("2021-01-01", periods=n, freq="h")


def test_sampled_shift_and_width_on_known_pulse():
    t = clock(9)
    a = np.array([0, 1, 2, 4, 2, 1, 0, 0, 0.])
    b = np.r_[0, a[:-1]] * 3
    p, q = waveform_profile(t, a), waveform_profile(t, b)
    assert peak_difference(p, q)["peak_difference_lower_hours"] == 1
    assert p["width_hours"] == q["width_hours"] == 2
    fit = shifted_flow_shape(t, a, b, 1)
    assert fit["shifted_flow_scale"] == pytest.approx(3)
    assert fit["range_normalized_rmse"] == pytest.approx(0, abs=1e-14)


def test_plateau_peaks_produce_interval_not_chosen_time():
    t = clock(7)
    p = waveform_profile(t, [0, 1, 4, 4, 1, 0, 0])
    q = waveform_profile(t, [0, 0, 1, 4, 4, 1, 0])
    lag = peak_difference(p, q)
    assert lag["peak_difference_lower_hours"] == 0
    assert lag["peak_difference_upper_hours"] == 2
    assert lag["interior_peak_pair"]


def test_incomplete_limbs_are_not_extrapolated():
    p = waveform_profile(clock(5), [0, 1, 4, 3, 3])
    assert p["width_status"] == "falling_limb_censored"
    assert np.isnan(p["width_hours"])
    q = waveform_profile(clock(5), [4, 3, 2, 1, 0])
    assert q["peak_at_boundary"]
    assert not peak_difference(p, q)["interior_peak_pair"]


def test_missing_hour_is_never_bridged():
    t = clock(6).delete(3)
    with pytest.raises(ValueError, match="do not bridge gaps"):
        waveform_profile(t, [0, 1, 4, 1, 0])


def test_high_turbidity_peak_not_replaced_with_low_point():
    values = [0, 1, 4, 2, 0]
    p = waveform_profile(clock(5), values)
    quality = optical_quality(p, values, [20, 20, 700, 20, 20])
    assert p["sampled_peak"] == 4
    assert p["peak_first"] == clock(5)[2]
    assert not quality["peak_outside_reported_retrieval_regime"]
    assert not quality["optical_only_width_available"]
    assert np.isnan(quality["optical_only_width_hours"])


def test_crossing_support_quality_not_just_peak_quality():
    values = [0, 1, 4, 2, 0]
    p = waveform_profile(clock(5), values)
    quality = optical_quality(p, values, [20, 700, 20, 20, 20])
    assert quality["peak_outside_reported_retrieval_regime"]
    assert not quality["optical_only_width_available"]


def test_separated_equal_maxima_are_not_one_peak_plateau():
    p = waveform_profile(clock(7), [0, 1, 4, 0, 4, 1, 0])
    assert p["n_half_height_lobes"] == 2
    assert not p["peak_indices_contiguous"]
    assert p["width_status"] == "separated_equal_maxima"
    assert np.isnan(p["width_hours"])


def test_flat_series_and_unknown_turbidity_are_explicit():
    p = waveform_profile(clock(5), [2]*5)
    assert p["width_status"] == "flat_segment"
    assert not peak_difference(p, p)["interior_peak_pair"]
    q = waveform_profile(clock(5), [0, 1, 4, 1, 0])
    quality = optical_quality(q, [0, 1, 4, 1, 0], [20, 20, np.nan, 20, 20])
    assert quality["peak_quality_status"] == "unknown_turbidity"


@pytest.mark.skipif(
    not Path("experiments/phase4_transfer/doc_river_event_observations_v1/analysis/optical_paired_hours.parquet").exists(),
    reason="Local paired optical-DOC source product is absent",
)
def test_real_source_segments_are_unique_contiguous_and_unchanged():
    frame = pd.read_parquet("experiments/phase4_transfer/doc_river_event_observations_v1/analysis/optical_paired_hours.parquet")
    assert len(frame) == 422
    assert not frame.date_time_source_clock.duplicated().any()
    for _, sub in frame.groupby("block"):
        clock = sub.date_time_source_clock
        for site in ("upstream", "receiver"):
            values = sub[f"DOC (mg l-1)_{site}"]
            profile = waveform_profile(clock, values)
            optical_quality(profile, values, sub[f"Turbidity (FNU)_{site}"])
            assert profile["sampled_peak"] == values.max()
    excluded = pd.to_datetime(["2021-03-20 16:00", "2021-03-20 17:00"])
    assert not frame.date_time_source_clock.isin(excluded).any()
