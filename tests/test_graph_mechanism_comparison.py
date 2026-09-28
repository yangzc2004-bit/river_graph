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


def test_tail_counts_unique_cells_instead_of_seed_replicates():
    idx = pd.MultiIndex.from_tuples([('A', '1'), ('A', '2'), ('B', '1')],
                                    names=['station', 'month'])
    left = pd.DataFrame({'y_true': [1., 10., 20.], 'y_pred': [2., 11., 21.]}, index=idx)
    right = left.assign(y_pred=[3., 12., 22.])
    cfg = {'dataset_sha256': 'dataset', 'mask_sha256': 'mask'}
    rec = {'identity': 'verified', 'q90_threshold': 10.}
    arms = {label: {('doc', 'temporal', seed): (frame, cfg, rec) for seed in (42, 43)}
            for label, frame in [('a', left), ('b', right)]}
    _, summary = compare.compare(arms)
    row = summary.iloc[0]
    assert row.paired_seeds == 2
    assert row.unique_query_cells == 3
    assert row.q90_unique_cells == 2
    assert row.q90_unstable
    assert row.q90_mae_a == 1.
    assert row.q90_mae_b == 2.
    assert row.q90_station_ci_low == row.q90_station_ci_high == 1.
