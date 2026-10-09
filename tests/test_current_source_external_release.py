"""Ensure release aggregation and repeated-case comparisons keep their meaning."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from portable_current_source_ensemble_v2 import (
    PortableCurrentSourceEnsemble,
)
from run_doc_current_source_external_v2 import validate_previous_panel


def test_previous_comparison_keeps_all_observation_and_fixed_query_identities():
    cells, query, truth = np.array([0, 1, 3]), np.array([1, 3]), np.arange(5.)
    panel = pd.concat([pd.DataFrame({"population": population, "k": k, "model_name": "old",
        "cell": selected, "y_true": truth[selected]}) for population, k, selected in
        [("all_observed_k0", 0, cells), ("fixed_query_curve", 0, query), ("fixed_query_curve", 5, query)]])
    validate_previous_panel(panel, cells, query, truth)
    broken = panel.copy()
    broken.loc[broken.k.eq(5), "cell"] = [0, 3]
    with pytest.raises(AssertionError):
        validate_previous_panel(broken, cells, query, truth)
    broken = panel.copy()
    broken.iloc[-1, broken.columns.get_loc("y_true")] = 100.
    with pytest.raises(AssertionError):
        validate_previous_panel(broken, cells, query, truth)


def test_ensemble_native_mean_support_policy_and_manifest(monkeypatch, tmp_path):
    from portable_current_source_doc_v2 import PortableCurrentSourceDOC
    paths = []
    for seed in range(42, 47):
        directory = tmp_path/f"seed{seed}"
        directory.mkdir()
        (directory/"manifest.json").write_text(str(seed))
        paths.append(directory)

    class Member:
        def __init__(self, directory):
            self.metadata = {"source_seed": int(directory.name[4:])}

        def predict_components(self, inputs):
            n, t = len(inputs["site_no"]), len(inputs["months"])
            value = np.full((n, t), self.metadata["source_seed"]-40., dtype=float)
            return {"environment_pred": value, "local_temporal_correction": value/2,
                "source_transfer_correction": value/4, "final_pred": value*1.75,
                "retrieval_sources": [{"target_station": str(inputs["site_no"][0])}]}

    monkeypatch.setattr(PortableCurrentSourceDOC, "load", Member)
    calibration = {"primary_k0_log_half_width": .3, "curve": {str(k):
        {"selected": {"alpha": .5}, "log_half_width": .2} for k in (0, 1, 3, 5)}}
    model = PortableCurrentSourceEnsemble(member_paths=paths, calibration=calibration,
        metadata={"source_labels_only": True})
    inputs = {"site_no": ["newB", "newA"], "months": ["2000-01", "2000-02"]}
    parts = model.predict_components(inputs)
    np.testing.assert_allclose(parts["final_pred"], 7.)
    np.testing.assert_allclose(parts["environment_pred"]+parts["local_temporal_correction"]
        +parts["source_transfer_correction"], parts["final_pred"])
    assert [row["seed"] for row in parts["retrieval_sources"]] == list(range(42, 47))
    corrected = model.predict_with_support(inputs, k=1, support_cells=np.array([0]), support_values=np.array([31.]))
    assert corrected[0, 1] == pytest.approx(15.)
    np.testing.assert_allclose(corrected[1], 7.)
    interval = model.predict_interval(inputs)
    assert (interval["pi_lower"] < interval["y_pred"]).all()
    with pytest.raises(ValueError, match="no receiving support"):
        model.predict_interval(inputs, support_cells=np.array([0]), support_values=np.array([31.]))
    saved = tmp_path/"release.json"
    model.save(saved)
    restored = PortableCurrentSourceEnsemble.load(saved)
    np.testing.assert_array_equal(restored.predict(inputs), model.predict(inputs))
    (paths[0]/"manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="manifest changed"):
        PortableCurrentSourceEnsemble.load(saved)
