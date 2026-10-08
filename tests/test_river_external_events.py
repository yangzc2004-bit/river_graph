"""Actual record and coverage failures must not become inferred DOC peaks."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_external_events import (
    canonical_doc,
    interval_coverage,
    longest_exact_sequence,
    normalize_watershed,
)


def test_duplicate_conflicts_are_not_averaged_or_selected():
    raw = pd.DataFrame({"Watershed": ["Kuparuk River"]*7,
        "DateTime_AKDT": ["7/1/17 00:00"]*3+["7/1/17 00:15"]*2+["7/1/17 00:30"]*2,
        "Year": [2017]*7, "DOC_mg.L": [3., 3., np.nan, 4., 8., np.nan, np.nan],
        "Notes": [None]*7, "Monitoring_Site": ["point"]*7})
    series, ledger = canonical_doc(raw)
    assert len(series) == 3 and len(ledger) == 3
    assert series.doc_mg_l.iloc[0] == 3
    assert series.doc_conflict.tolist() == [False, True, False]
    assert np.isnan(series.doc_mg_l.iloc[1:]).all()
    assert ledger.n_source_rows.sum() == 7


def test_mislabelled_year_rejected():
    raw = pd.DataFrame({"Watershed": ["Kuparuk"], "DateTime_AKDT": ["7/1/17 00:00"],
                        "Year": [2022], "DOC_mg.L": [3]})
    with pytest.raises(ValueError, match="year disagrees"):
        canonical_doc(raw)


def test_sequences_detect_time_reuse_not_shared_range_or_flat_plateau():
    a = np.array([9, 8, *range(100)])+.01345
    b = np.r_[-8, -7, a[2:82], -6]
    assert longest_exact_sequence(a, b) == (80, 2, 2)
    assert longest_exact_sequence(a, b[::-1])[0] == 0
    assert longest_exact_sequence(np.ones(30), np.ones(50))[0] == 0
    with pytest.raises(ValueError, match="finite"):
        longest_exact_sequence([np.nan], [0])


def make_series(clock, values):
    return pd.DataFrame({"timestamp": clock, "doc_mg_l": values,
                         "doc_conflict": False, "notes": ""})


def test_faster_logger_grid_cannot_inflate_coverage():
    start = pd.Timestamp("2022-07-01")
    clock = pd.date_range(start, periods=289, freq="5min")
    result = interval_coverage(make_series(clock, np.ones(289)), start, start+pd.Timedelta(days=1))
    assert result['nominal_doc_coverage'] == 1
    assert result['n_valid_doc'] == 289
    assert result['expected_nominal_bins'] == 97
    assert result['doc_coverage_qualified']


def test_boundary_gaps_exclude_truncated_event_despite_dense_interior():
    start = pd.Timestamp("2022-07-01")
    clock = pd.date_range(start+pd.Timedelta(hours=2), start+pd.Timedelta(days=1), freq="15min")
    result = interval_coverage(make_series(clock, np.ones(len(clock))), start, start+pd.Timedelta(days=1))
    assert result['nominal_doc_coverage'] > .9
    assert result['max_gap_including_boundaries_hours'] == 2
    assert not result['doc_coverage_qualified']


def test_no_measurements_is_not_zero_gap():
    start = pd.Timestamp("2022-07-01")
    clock = pd.DatetimeIndex([])
    result = interval_coverage(make_series(clock, []), start, start+pd.Timedelta(days=2))
    assert result['nominal_doc_coverage'] == 0
    assert result['max_gap_including_boundaries_hours'] == 48
    assert not result['doc_coverage_qualified']


def test_author_punctuation_does_not_change_watershed_identity():
    assert normalize_watershed('Oksrukuyik (Lake-influenced') == 'Oksrukuyik Creek'
    assert normalize_watershed('Kuparuk (River-dominated)') == 'Kuparuk River'
    with pytest.raises(ValueError, match="Unknown watershed"):
        normalize_watershed('Other')
