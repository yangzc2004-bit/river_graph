"""Source-tree rows must match the information at entirely unmonitored sites."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs


def test_station_fold_and_hidden_target_labels_are_absent_from_tree_covariates():
    n, t = 7, 8
    data = {"y": np.arange(n*t).reshape(n, t).astype(float), "y_mask": np.ones((n, t), bool),
            "x": np.ones((n, t, 2)), "x_mask": np.ones((n, t, 2), bool),
            "regime": np.ones((n, 13)), "static": np.ones((n, 2)),
            "months": np.arange("2000-01", "2000-09", dtype="datetime64[M]"),
            "edge_index": np.array([[0, 1, 2], [1, 2, 3]])}
    split = {"train": np.arange(5*t), "val": np.arange(5*t, 6*t), "test": np.arange(6*t, 7*t),
             "context": np.array([], dtype=np.int64)}
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
    daily = np.zeros((n, t, 8))
    training, inference = station_hidden_tree_inputs(data, split, folds, daily)
    changed = copy.deepcopy(data)
    changed["y"][5:] += 1000
    train_again, infer_again = station_hidden_tree_inputs(changed, split, folds, daily)
    np.testing.assert_array_equal(training, train_again)
    np.testing.assert_array_equal(inference, infer_again)
    changed["y"][:2] += 3000
    fold_changed, _ = station_hidden_tree_inputs(changed, split, folds, daily)
    np.testing.assert_array_equal(training[:2*t], fold_changed[:2*t])
    np.testing.assert_array_equal(training[:, [27, 28, 30, 31, 33, 34, 36, 37]], 0)
