"""Fixed ExtraTrees probes share visibility, parameters and causal daily slots."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.models.daily_hydro_tree import (
    build_daily_tree_features,
    daily_tree_feature_names,
    fit_daily_tree_probes,
)
from river_graph.models.kgml_local_transport import build_rf_features, target_values


@pytest.fixture
def case():
    rng = np.random.default_rng(12)
    n, t = 8, 15
    x = rng.uniform(.1, 4, (n, t, 2)).astype(np.float32)
    mask = np.ones_like(x, dtype=bool)
    mask[2, 3, 1] = False
    x[~mask] = 0
    dataset = {"y": (2 + x[..., 0] + rng.uniform(0, 2, (n, t))).astype(np.float32),
               "y_mask": np.ones((n, t), dtype=bool), "x": x, "x_mask": mask,
               "static": rng.uniform(0, 1, (n, 2)).astype(np.float32),
               "regime": rng.uniform(0, 1, (n, 13)).astype(np.float32),
               "months": np.arange("2020-01", "2021-04", dtype="datetime64[M]"),
               "edge_index": np.stack([np.arange(n-1), np.arange(1, n)])}
    split = {"train": np.arange(6*t), "val": np.arange(6*t, 7*t),
             "test": np.arange(7*t, 8*t), "context": np.array([], dtype=np.int64)}
    daily = np.ones((n, t, 8), dtype=np.float32)
    daily[..., :3] = rng.uniform(0, 1, (n, t, 3))
    daily[~mask[..., 1]] = 0
    base = build_rf_features(dataset, split, ("train",), target_transform="log1p", include_network=True)
    forest = ExtraTreesRegressor(n_estimators=5, min_samples_leaf=2, max_features=.7,
                                 max_depth=5, random_state=42, n_jobs=1).fit(
        base[split["train"]], target_values(dataset, "log1p").ravel()[split["train"]])
    return SimpleNamespace(context_forest=forest, context_name="frozen_choice"), dataset, split, daily


def test_matched_width_original_base_and_exact_window_alignment(case):
    _, data, split, daily = case
    n, t, _ = daily.shape
    current = build_daily_tree_features(data, split, daily, mode="current")
    history = build_daily_tree_features(data, split, daily, mode="history")
    assert current.shape == history.shape == (n*t, 147)
    assert current.dtype == history.dtype == np.float32
    assert len(daily_tree_feature_names()) == 147
    original = build_rf_features(data, split, ("train",), target_transform="log1p", include_network=True)
    np.testing.assert_array_equal(history[:, :39], original)
    np.testing.assert_array_equal(current[:, :39], original)
    h = history[:, 39:].reshape(n, t, 12, 9)
    c = current[:, 39:].reshape(n, t, 12, 9)
    np.testing.assert_array_equal(c[..., 8], h[..., 8])
    np.testing.assert_array_equal(c[..., :-1, :8], 0)
    np.testing.assert_array_equal(c[..., -1, :8], daily)
    np.testing.assert_array_equal(h[:, 0, :-1], 0)
    np.testing.assert_array_equal(h[:, 0, -1, :8], daily[:, 0])
    np.testing.assert_array_equal(h[:, 0, -1, 8], 1)
    np.testing.assert_array_equal(h[:, 13, :, :8], daily[:, 2:14])
    np.testing.assert_array_equal(h[:, 13, :, 8], 1)
    assert not np.shares_memory(history, daily)


def test_future_daily_values_never_change_earlier_features(case):
    _, data, split, daily = case
    changed = daily.copy()
    changed[:, 8:, :3] = 1 - changed[:, 8:, :3]
    for mode in ("current", "history"):
        before = build_daily_tree_features(data, split, daily, mode=mode).reshape(8, 15, -1)
        after = build_daily_tree_features(data, split, changed, mode=mode).reshape(8, 15, -1)
        np.testing.assert_array_equal(before[:, :8], after[:, :8])
    changed = daily.copy()
    changed[0, 4, 0] = 1 - changed[0, 4, 0]
    before_c = build_daily_tree_features(data, split, daily, mode="current")
    after_c = build_daily_tree_features(data, split, changed, mode="current")
    np.testing.assert_array_equal(before_c[8], after_c[8])
    before_h = build_daily_tree_features(data, split, daily, mode="history")
    after_h = build_daily_tree_features(data, split, changed, mode="history")
    assert not np.array_equal(before_h[8], after_h[8])


def test_fixed_selected_params_fit_and_joblib_prediction_roundtrip(case, tmp_path):
    expert, data, split, daily = case
    original_params = expert.context_forest.get_params()
    original_thresholds = [tree.tree_.threshold.copy() for tree in expert.context_forest.estimators_]
    messages = []
    out = fit_daily_tree_probes(expert, data, split, daily, n_jobs=1, progress=messages.append)
    assert set(out) == {"pred_native", "models", "records"}
    assert len(messages) == 2
    json.dumps(out["records"], allow_nan=False)
    for mode in ("current", "history"):
        model, record = out["models"][mode], out["records"][mode]
        assert model is not expert.context_forest
        assert model.get_params() == original_params
        assert model.n_features_in_ == 147
        assert record["visible_roles"] == record["inference_roles"] == ["train"]
        assert record["validation_query_cells"] == 10
        assert record["source_station_ids"] == list(range(6))
        assert record["hyperparameter_search"] is False
        assert record["replaces_neural_context"] is False
        assert record["source_oof_fitting"] is False
        features = build_daily_tree_features(data, split, daily, mode=mode)
        reference = clone(expert.context_forest).fit(
            features[split["train"]], target_values(data, "log1p").ravel()[split["train"]])
        expected = np.maximum(0, np.expm1(reference.predict(features))).reshape(8, 15)
        np.testing.assert_array_equal(out["pred_native"][mode], expected)
        assert np.isfinite(expected).all() and np.ptp(expected) > 0
        path = tmp_path / f"{mode}.joblib"
        joblib.dump(model, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.predict(features), model.predict(features))
    assert expert.context_forest.get_params() == original_params
    assert expert.context_forest.n_features_in_ == 39
    for tree, before in zip(expert.context_forest.estimators_, original_thresholds, strict=True):
        np.testing.assert_array_equal(tree.tree_.threshold, before)


def test_hidden_labels_never_enter_features_or_forest_fitting(case):
    expert, data, split, daily = case
    expected = fit_daily_tree_probes(expert, data, split, daily, n_jobs=1)
    changed = copy.deepcopy(data)
    changed["y"].ravel()[split["test"]] = np.nan
    changed["y"].ravel()[split["val"]] += 1000
    actual = fit_daily_tree_probes(expert, changed, split, daily, n_jobs=1)
    for mode in ("current", "history"):
        np.testing.assert_array_equal(build_daily_tree_features(data, split, daily, mode=mode),
                                      build_daily_tree_features(changed, split, daily, mode=mode))
        np.testing.assert_array_equal(expected["pred_native"][mode], actual["pred_native"][mode])
        for a, b in zip(expected["models"][mode].estimators_, actual["models"][mode].estimators_, strict=True):
            np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
            np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
        # Validation is reported but cannot affect either fitted model.
        assert actual["records"][mode]["validation_mae_native"] > expected["records"][mode]["validation_mae_native"]
    changed = copy.deepcopy(data)
    changed["y"].ravel()[split["test"]] += 5000
    unchanged = fit_daily_tree_probes(expert, changed, split, daily, n_jobs=1)
    assert unchanged["records"] == expected["records"]


@pytest.mark.parametrize("defect", ["nonchronological", "future_dimension", "nonfinite", "nonbinary", "masked_flow", "context"])
def test_invalid_feature_or_visibility_contracts_fail(case, defect):
    expert, data, split, daily = case
    if defect == "nonchronological":
        data["months"] = data["months"][::-1]
    elif defect == "future_dimension":
        daily = daily[:, :-1]
    elif defect == "nonfinite":
        daily[0, 0, 0] = np.nan
    elif defect == "nonbinary":
        daily[0, 0, 5] = .5
    elif defect == "masked_flow":
        daily[2, 3, 0] = .2
    else:
        split["context"] = split["test"][:1]
    with pytest.raises(ValueError):
        fit_daily_tree_probes(expert, data, split, daily, n_jobs=1)


def test_n_jobs_only_override_and_bad_mode(case):
    expert, data, split, daily = case
    out = fit_daily_tree_probes(expert, data, split, daily, n_jobs=2)
    for model in out["models"].values():
        assert model.get_params() == {**expert.context_forest.get_params(), "n_jobs": 2}
    with pytest.raises(ValueError, match="current or history"):
        build_daily_tree_features(data, split, daily, mode="future")
