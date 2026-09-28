"""Paired scientific comparisons must retain ecological sampling units."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'graph_compare', Path(__file__).parents[1] / 'scripts/compare_graph_mechanisms.py'
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_query_mismatch_cannot_silently_drop_cells():
    idx = pd.MultiIndex.from_tuples([('A', '1'), ('B', '2')], names=['station', 'month'])
    left = pd.DataFrame({'y_true': [1., 2.], 'y_pred': [1.1, 1.9]}, index=idx)
    with pytest.raises(ValueError, match='query station-month'):
        compare.paired_errors(left, left.iloc[:1])
    right = left.copy()
    right.loc[('B', '2'), 'y_true'] = 3.
    with pytest.raises(ValueError, match='query truth'):
        compare.paired_errors(left, right)


def test_cluster_interval_has_correct_paired_sign():
    idx = pd.MultiIndex.from_tuples([('A', '1'), ('A', '2'), ('B', '1')],
                                    names=['station', 'month'])
    errors = pd.DataFrame({'a': [1., 1., 1.], 'b': [2., 2., 2.]}, index=idx)
    assert compare.station_interval(errors) == (1., 1.)
