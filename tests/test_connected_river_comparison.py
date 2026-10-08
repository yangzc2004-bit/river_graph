"""Receiver folds balance upstream support and remain station disjoint."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_doc_river_connected_comparison_v1 import receiver_folds


def test_receiver_folds_preserve_station_blocks_and_support_balance():
    ids = np.arange(18)
    cells = np.arange(18*8)
    support = cells//8 < 7
    folds = receiver_folds(ids, cells, support, 8, 142)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), ids)
    counts = [len(np.intersect1d(fold, np.arange(7))) for fold in folds]
    assert max(counts)-min(counts) == 1
    for fold in folds:
        query = np.isin(cells//8, fold)
        assert not set(cells[query]//8) & set(cells[~query]//8)
    repeat = receiver_folds(ids, cells, support, 8, 142)
    for first, second in zip(folds, repeat, strict=True):
        np.testing.assert_array_equal(first, second)
