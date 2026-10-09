"""Temporal label roles must not be treated as whole validation stations."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_doc_current_source_temporal_v3 import hidden_pair_label_probe


def test_temporal_overlap_hides_val_cells_without_corrupting_training_labels():
    data = {"y": np.arange(20.).reshape(5, 4)}
    split = {"train": np.array([0, 4, 8, 12, 16]), "context": np.array([13]),
        "val": np.array([1, 5, 9, 14, 17]), "test": np.array([2, 6, 10, 15, 18])}
    changed = hidden_pair_label_probe(data, split, np.array([0, 1]), 4)
    np.testing.assert_array_equal(changed["y"].ravel()[[8, 12, 16]], data["y"].ravel()[[8, 12, 16]])
    np.testing.assert_array_equal(changed["y"].ravel()[split["val"]], 12345.)
    np.testing.assert_array_equal(changed["y"][:2], 12345.)
    assert changed["y"][3, 1] == data["y"][3, 1]
    np.testing.assert_array_equal(data["y"], np.arange(20.).reshape(5, 4))
