"""Integration guard for the frame-based metrics contract and v2 freeze."""
import json
from pathlib import Path

import numpy as np
import pytest

from river_graph.experiments.doc_fusion_benchmark import metrics


def test_prediction_frame_metrics_contract(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("run_doc_interpolation_v2")
    frame = runner.frame({"y": np.array([[1., 3.]])}, np.array([0, 1]), [2., 2.],
                         [True, True], [2., 2.], "example", 42, "fixed")
    score = metrics(frame)
    np.testing.assert_allclose(score["mae"], 1.)
    np.testing.assert_allclose(score["r2"], 0.)


@pytest.mark.skipif(not Path("experiments/phase4_transfer/doc_interpolation_v2/protocol.json").exists(),
                    reason="local v2 interpolation freeze has not been generated")
def test_v2_freeze_records_unchanged_query_design():
    protocol = json.loads(Path("experiments/phase4_transfer/doc_interpolation_v2/protocol.json").read_text())
    assert len(protocol["tasks"]) == 14
    assert protocol["joint_primary"] == ["mae", "log1p_mae"]
    for seed in (42, 43, 44):
        files = [np.load(f"experiments/phase4_transfer/doc_interpolation_v2/masks/time_g{g}_s{seed}.npz") for g in (3, 6, 12)]
        for current in files[1:]:
            np.testing.assert_array_equal(files[0]["test"], current["test"])
            np.testing.assert_array_equal(files[0]["val"], current["val"])
        for file in files:
            file.close()
