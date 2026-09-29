"""Visibility and paired-selection tests for the RF/M1 complementarity pilot."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.experiments.graph_upgrade_v2 import GraphUpgradeModel
from river_graph.experiments.temporal_h2x import build_temporal_inputs


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_observation_rf_blend")


def toy():
    torch.manual_seed(6)
    n, t = 4, 8
    data = {"y": 1+torch.rand(n, t)*4, "y_mask": torch.ones(n, t, dtype=torch.bool),
            "x": torch.rand(n, t, 2), "x_mask": torch.ones(n, t, 2),
            "static": torch.rand(n, 2), "regime": torch.rand(n, 13),
            "months": [f"2020-{m:02d}-01" for m in range(1, t+1)],
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
            "edge_attr": torch.rand(3, 6)}
    split = {"train": np.arange(16), "val": np.arange(16, 24), "test": np.arange(24, 32)}
    return data, split


def test_alpha_selection_in_native_units_and_exact_ties(monkeypatch):
    mod = module(monkeypatch)
    assert mod.select_alpha([0, 4], [0, 4], [3, 3])[0] == 1
    assert mod.select_alpha([0, 4], [3, 3], [0, 4])[0] == 0
    assert mod.select_alpha([1, 3], [1, 3], [1, 3])[0] == 1
    assert mod.select_alpha([2, 2], [0, 4], [4, 0])[0] == .5
    with pytest.raises(ValueError, match="finite"):
        mod.select_alpha([np.nan], [1], [1])


def test_test_label_perturbation_leaves_fits_and_selection_unchanged(monkeypatch):
    mod = module(monkeypatch)
    data, split = toy()
    changed = {**data, "y": data["y"].clone()}
    changed["y"].reshape(-1)[split["test"]] += 1000
    outputs = []
    for ds in (data, changed):
        gnn = GraphUpgradeModel(seed=42, hidden=8, temporal_hidden=8, max_epochs=2,
                                patience=2, dropout=0, chunk_months=8)
        gnn.fit(ds, split)
        rf_x = mod._features(ds, split, mod.FIT_VISIBILITY)
        rf = mod.RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=1)
        rf.fit(rf_x[split["train"]], np.log1p(ds["y"].numpy().ravel()[split["train"]]))
        val_pred = gnn.predict(only_visible=mod.visible_cells(split)).ravel()[split["val"]]
        rf_pred = np.expm1(rf.predict(rf_x[split["val"]]))
        alpha, _ = mod.select_alpha(ds["y"].numpy().ravel()[split["val"]], rf_pred, val_pred)
        outputs.append((gnn, val_pred, rf_pred, alpha))
    for key, tensor in outputs[0][0]._bundle.model.state_dict().items():
        assert torch.equal(tensor, outputs[1][0]._bundle.model.state_dict()[key])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    assert outputs[0][3] == outputs[1][3]


def test_validation_labels_are_not_features_and_m1_prediction_is_causal(monkeypatch):
    mod = module(monkeypatch)
    data, split = toy()
    gnn = GraphUpgradeModel(hidden=8, temporal_hidden=8, max_epochs=1,
                            patience=1, dropout=0, chunk_months=8)
    gnn.fit(data, split)
    bundle = gnn._bundle
    visible, _ = gnn._visible_input(only_visible=mod.visible_cells(split))
    before, ages = gnn._make_input(bundle.inputs, visible, bundle.model)
    changed = {**data, "y": data["y"].clone()}
    changed["y"].reshape(-1)[split["val"]] += 999
    inputs = build_temporal_inputs(changed, split, target_transform="log1p")
    after, _ = gnn._make_input(inputs, visible, bundle.model)
    assert torch.equal(before, after)
    np.testing.assert_array_equal(mod._features(data, split, mod.FIT_VISIBILITY),
                                  mod._features(changed, split, mod.FIT_VISIBILITY))
    with torch.no_grad():
        pred = gnn._forward(bundle.model, before, bundle.inputs, data, ages)
        future = before.clone()
        future[5:] += 100
        future_age = ages.clone()
        future_age[5:] += 100
        other = gnn._forward(bundle.model, future, bundle.inputs, data, future_age)
    assert torch.equal(pred[:, :5], other[:, :5])


def test_summary_uses_rf_denominator_and_not_seed_pseudoreplication(monkeypatch):
    mod = module(monkeypatch)
    frame = pd.DataFrame({"cell": [0, 1, 2, 3], "station": ["a", "a", "b", "c"],
                          "y_true": [2, 4, 6, 8], "rf_pred": [3, 5, 7, 9],
                          "gnn_pred": [1, 3, 5, 7], "blend_pred": [2.5, 4.5, 6.5, 8.5]})
    single, repeated = mod.summarize([frame]), mod.summarize([frame]*3)
    assert single["relative_improvement_pct"] == 50
    assert single["unique_cells"] == repeated["unique_cells"] == 4
    assert single["ci_low"] == repeated["ci_low"]
    assert single["ci_high"] == repeated["ci_high"]
    original = {"alpha_grid": [0, 1], "gnn": {"max_epochs": 30}, "rf_trees": 200}
    assert mod.digest(original) != mod.digest({**original, "rf_trees": 201})
    assert mod.digest(original) != mod.digest({**original, "gnn": {"max_epochs": 31}})
