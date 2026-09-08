"""Tests for monthly aggregation (no network)."""

import numpy as np
import pandas as pd

from river_graph.data.aggregate import to_matrix, to_monthly


def _obs():
    return pd.DataFrame(
        {
            "site_no": ["A", "A", "A", "B"],
            "date": pd.to_datetime(
                ["2020-01-05", "2020-01-20", "2020-03-01", "2020-01-15"]
            ),
            "doc": [4.0, 6.0, 9.0, 7.0],
        }
    )


def test_to_monthly_averages_within_month():
    m = to_monthly(_obs(), "doc")
    jan_a = m[(m["site_no"] == "A") & (m["month"] == "2020-01")]["doc"].iloc[0]
    assert jan_a == 5.0
    assert len(m) == 3  # A-Jan, A-Mar, B-Jan


def test_to_monthly_empty():
    m = to_monthly(pd.DataFrame(columns=["site_no", "date", "doc"]), "doc")
    assert m.empty


def test_to_matrix_places_values_and_nan():
    m = to_monthly(_obs(), "doc")
    months = pd.date_range("2020-01", "2020-04", freq="MS")
    mat = to_matrix(m, "doc", ["A", "B"], months)
    assert mat.shape == (2, 4)
    assert mat[0, 0] == 5.0 and mat[0, 2] == 9.0
    assert mat[1, 0] == 7.0
    assert np.isnan(mat[0, 1]) and np.isnan(mat[0, 3]) and np.isnan(mat[1, 1:] ).all()
