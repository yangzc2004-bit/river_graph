"""Tests for actual-clock comparisons, normalized mixtures and survey grain."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_confluence_response import (
    channel_geometry,
    confluence_chemistry,
    maximum_clock,
    summarize_water_window,
    wave_overlap,
)


def test_wave_overlap_does_not_align_staggered_peaks():
    result = wave_overlap(np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]))
    assert result["wave_overlap"] == 0
    assert result["incoming_peak_coincidence"] == .5
    assert result["amplitude_balanced_peak_coincidence"] == .5
    same = wave_overlap(np.array([0, 1, 0, 0]), np.array([0, 5, 0, 0]))
    assert same["wave_overlap"] == 1
    assert same["incoming_peak_coincidence"] == 1


def test_flow_dominance_differs_from_wave_synchrony():
    result = wave_overlap(np.array([0, 10, 0]), np.array([0, 0, 1]))
    assert result["wave_overlap"] == 0
    assert result["incoming_peak_coincidence"] == pytest.approx(10/11)
    assert result["amplitude_balanced_peak_coincidence"] == .5


def test_tied_maxima_preserve_separate_runs():
    clock = pd.date_range("2020-01-01", periods=6, freq="30min")
    peak = maximum_clock(clock, np.array([0, 2, 2, 0, 2, 0]))
    assert peak["first"] == clock[1]
    assert peak["last"] == clock[4]
    assert peak["n_tied_samples"] == 3
    assert peak["n_tied_runs"] == 2


def test_missing_flow_is_not_filled_or_replaced():
    clock = pd.date_range("2020-01-01", periods=6, freq="30min")
    result = summarize_water_window(clock, [0, 1, 5, 2, 1, 0], [0, 1, np.nan, 3, 1, 0], [1, 3, 7, 6, 3, 1])
    assert result["n_joint_slots"] == 5
    assert not result["all_site_maxima_present_on_joint_clock"]
    assert result["peak_a_maximum"] == 5
    assert not result["primary_coverage_subset"]


def test_peak_clock_bounds_and_partial_flow_share():
    clock = pd.date_range("2020-01-01", periods=6, freq="30min")
    result = summarize_water_window(clock, [1, 3, 3, 1, 1, 1], [1, 1, 1, 4, 4, 1], [3, 5, 5, 7, 7, 3])
    assert result["b_minus_a_lower_hours"] == .5
    assert result["b_minus_a_upper_hours"] == 1.5
    assert result["branch_flow_share_median"] < 1
    assert result["full_joint_clock"]


@pytest.mark.parametrize("values", [[0, -1, 2], [0, np.inf, 2]])
def test_invalid_discharge_is_not_silently_removed(values):
    with pytest.raises(ValueError):
        summarize_water_window(pd.date_range("2020", periods=3, freq="30min"), values, [1, 2, 1], [2, 3, 2])


def chemistry_fixture():
    return pd.DataFrame({"location": ["Con-1"]*3, "season": ["fall"]*3,
                         "transect.location": ["L", "M", "R"], "Q.Ls.main": [1.]*3,
                         "Q.Ls.trib": [3.]*3, "Q.Ls": [5.]*3,
                         "pQm.mtsum": [.2]*3, "pQt.mtsum": [.6]*3,
                         "DOC.mgL.mean.main": [2.]*3, "DOC.mgL.mean.trib": [6.]*3,
                         "DOC.mgL.mean": [4., 5., 6.],
                         "SpC.main": [100.]*3, "SpC.trib": [200.]*3, "SpC": [150., 180., 190.]})


def test_mixture_uses_normalized_branch_flow_not_receiver_denominator():
    summary, points = confluence_chemistry(chemistry_fixture())
    row = summary.iloc[0]
    assert row.doc_mix_reference == 5
    assert row.doc_receiver_lateral_mean == 5
    assert row.branch_flow_share == .8
    assert row.source_weights_differ_from_normalized
    assert row.doc_mix_within_lateral_range
    assert row.doc_main_distinct_values == 1
    assert len(points) == 3
    assert points.source_row_mix.iloc[0] == pytest.approx(4)


def test_lateral_positions_cannot_be_duplicated_as_independent_sites():
    data = chemistry_fixture()
    data.loc[2, "transect.location"] = "L"
    with pytest.raises(ValueError, match="lateral"):
        confluence_chemistry(data)


def test_geometry_uses_one_width_per_transect():
    # Different counts of depth points must not weight the duplicated widths.
    rows = []
    for reach, widths in (("1.upstream", (1, 3)), ("2.downstream", (2, 6))):
        for transect, (width, n) in enumerate(zip(widths, (2, 5)), 1):
            rows.extend({"location": "1.CHC", "reach": reach, "transect": transect,
                         "wettedwidth.m": width, "depth.m": .1, "distance.to.confluence": transect*5.}
                        for _ in range(n))
    summary, transects = channel_geometry(pd.DataFrame(rows))
    assert len(transects) == 4
    row = summary.iloc[0]
    assert row.main_mean_width_m == 2
    assert row.receiver_mean_width_m == 4
    assert row.receiver_main_width_ratio == 2


def test_obstructed_depth_points_are_counted_not_filled_with_zero():
    data = pd.DataFrame([{"location": "1.CHC", "reach": reach, "transect": transect,
                          "wettedwidth.m": 2., "depth.m": depth, "distance.to.confluence": transect*5.}
                         for reach in ("1.upstream", "2.downstream")
                         for transect in (1, 2) for depth in (.1, np.nan)])
    summary, transects = channel_geometry(data)
    assert transects.n_missing_depth_points.sum() == 4
    assert summary.iloc[0].receiver_mean_depth_m == .1


RAW = Path("data/raw/river_confluence_response_v1/plont/Plontetal_WRR_database.csv")


@pytest.mark.skipif(not RAW.exists(), reason="Public confluence source archive is not available locally")
def test_public_source_normalization_and_campaign_grain():
    summary, points = confluence_chemistry(pd.read_csv(RAW))
    assert len(summary) == 10
    assert len(points) == 30
    assert summary.confluence.nunique() == 5
    # The source weight inconsistency is exposed; archived rows are not changed.
    row = summary[(summary.confluence == "Con-1") & (summary.season == "summer")].iloc[0]
    assert row.source_weights_differ_from_normalized
    assert row.source_weight_sum_min < 1
    assert row.doc_receiver_minus_mix_pct < 0
