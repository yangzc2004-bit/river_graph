"""The calibration development runner reads validation labels only."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("run_doc_nested_chemistry_v1", SCRIPTS / "run_doc_nested_chemistry_v1.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_development_label_view_excludes_source_and_target_doc():
    data = {"y": np.arange(12, dtype=float).reshape(3, 4)}
    split = {"train": np.arange(4), "val": np.arange(4, 8), "test": np.arange(8, 12),
             "context": np.empty(0, dtype=int)}
    original = runner.validation_label_view(data, split)
    changed = {"y": data["y"].copy()}
    changed["y"].ravel()[np.r_[split["train"], split["test"]]] = 1e6
    np.testing.assert_array_equal(original, runner.validation_label_view(changed, split))
    assert np.isnan(original[np.r_[split["train"], split["test"]]]).all()
    np.testing.assert_array_equal(original[split["val"]], data["y"].ravel()[split["val"]])
