"""Objective changes retain source/receiving-station information boundaries."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_doc_unmonitored_trees_v1 import station_hidden_tree_inputs

from river_graph.models.doc_reference_objectives import (
    ARMS,
    make_reference,
    predict_reference,
    reference_target,
)


def test_objective_arms_keep_matched_learner_parameters():
    old = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.7, random_state=42, n_jobs=2)
    assert make_reference("native_target_trees", old, 42).get_params() == old.get_params()
    log = make_reference("log_l1_boosting", old, 42)
    raw = make_reference("native_l1_boosting", old, 42)
    assert log.get_params() == raw.get_params()
    assert raw.loss == "absolute_error" and not raw.early_stopping and raw.max_iter == 300
    values = np.array([0., 2., 10.])
    np.testing.assert_array_equal(reference_target(values, "native_target_trees"), values)
    np.testing.assert_allclose(np.expm1(reference_target(values, "log_l1_boosting")), values)
    with pytest.raises(ValueError, match="finite nonnegative"):
        reference_target([1., np.nan], "native_l1_boosting")


@pytest.mark.parametrize("arm", ARMS)
def test_hidden_labels_cannot_change_fits_or_source_bins_and_saved_inference(arm, tmp_path):
    n, t = 7, 8
    rng = np.random.default_rng(42)
    data = {"y": rng.uniform(0, 12, (n, t)), "y_mask": np.ones((n, t), bool),
        "x": rng.uniform(0, 5, (n, t, 2)), "x_mask": np.ones((n, t, 2), bool),
        "regime": rng.random((n, 13)), "static": rng.random((n, 2)),
        "months": np.arange("2000-01", "2000-09", dtype="datetime64[M]"),
        "edge_index": np.array([[0, 1, 2], [1, 2, 3]])}
    split = {"train": np.arange(5*t), "val": np.arange(5*t, 6*t), "test": np.arange(6*t, 7*t),
        "context": np.array([], dtype=np.int64)}
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
    daily = rng.random((n, t, 8))
    changed = copy.deepcopy(data)
    changed["y"][5:] += 10000
    models, inputs = [], []
    for view in (data, changed):
        train_x, full_x = station_hidden_tree_inputs(view, split, folds, daily)
        model = make_reference(arm, ExtraTreesRegressor(n_estimators=3, random_state=42), 42)
        if arm != "native_target_trees":
            model.set_params(max_iter=3, min_samples_leaf=2)
        with threadpool_limits(limits=2):
            model.fit(train_x, reference_target(view["y"].ravel()[split["train"]], arm))
            prediction = predict_reference(model, full_x, arm)
        models.append(model)
        inputs.append((train_x, full_x, prediction))
    for first, second in zip(inputs[0], inputs[1], strict=True):
        np.testing.assert_array_equal(first, second)
    if arm != "native_target_trees":
        for first, second in zip(models[0]._bin_mapper.bin_thresholds_, models[1]._bin_mapper.bin_thresholds_, strict=True):
            np.testing.assert_array_equal(first, second)
    joblib.dump(models[0], tmp_path/"reference.joblib")
    with threadpool_limits(limits=2):
        replay = predict_reference(joblib.load(tmp_path/"reference.joblib"), inputs[0][1], arm)
    np.testing.assert_allclose(replay, inputs[0][2], atol=1e-12, rtol=1e-12)


def test_native_objective_does_not_use_transformed_targets_and_clips_negative_prediction():
    class Constant:
        def predict(self, inputs):
            return np.array([-2., 2.])
    np.testing.assert_array_equal(predict_reference(Constant(), np.zeros((2, 3)), "native_l1_boosting"), [0., 2.])
    with pytest.raises(ValueError, match="unknown reference"):
        make_reference("unspecified", ExtraTreesRegressor(), 42)
