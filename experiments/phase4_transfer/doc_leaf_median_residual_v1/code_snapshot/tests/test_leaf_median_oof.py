"""A median source distribution excludes all labels of its held station fold."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_doc_leaf_median_residual_v1 import fold_distribution
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs

from river_graph.models.kgml_local_transport import fold_split, station_folds


def test_held_station_and_validation_labels_do_not_enter_nested_leaf_distribution():
    n, t = 9, 8
    rng = np.random.default_rng(42)
    data = {"y": rng.uniform(1, 8, (n, t)), "y_mask": np.ones((n, t), bool),
        "x": rng.uniform(0, 10, (n, t, 2)), "x_mask": np.ones((n, t, 2), bool),
        "regime": rng.random((n, 13)), "static": rng.random((n, 2)),
        "months": np.arange("2000-01", "2000-09", dtype="datetime64[M]"),
        "edge_index": np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])}
    split = {"train": np.arange(7*t), "val": np.arange(7*t, 8*t), "test": np.arange(8*t, 9*t),
        "context": np.array([], dtype=np.int64)}
    hidden = np.array([0, 1])
    daily = np.zeros((n, t, 8))
    outer = fold_split(split, hidden, t)
    train_x, _ = station_hidden_tree_inputs(data, outer, station_folds(outer["train"], t, 42), daily)
    forest = ExtraTreesRegressor(n_estimators=3, min_samples_leaf=2, random_state=42, n_jobs=1)
    forest.fit(train_x, np.log1p(data["y"].ravel()[outer["train"]]))
    original, held, inputs, _ = fold_distribution(data, split, hidden, daily, forest, 42)
    changed = copy.deepcopy(data)
    changed["y"][hidden] += 5000
    changed["y"][7:] += 10000
    other, other_held, other_inputs, other_outer = fold_distribution(changed, split, hidden, daily, forest, 42)
    np.testing.assert_array_equal(held, other_held)
    np.testing.assert_array_equal(inputs, other_inputs)
    np.testing.assert_array_equal(original.source_y_, other.source_y_)
    np.testing.assert_array_equal(original.predict(inputs), other.predict(other_inputs))
    assert not np.isin(other_outer["train"]//t, hidden).any()
    assert not np.isin(original.source_y_, changed["y"][hidden].ravel()).any()
