"""Conditional leaf medians are source distributions, not averaged medians."""
from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.models.leaf_distribution_reference import LeafDistributionReference


def fixture():
    x = np.array([[0., 0.], [1., .1], [2., 1.], [3., .4], [4., .8], [5., .2]])
    y = np.array([1., 2., 7., 8., 20., 100.])
    forest = ExtraTreesRegressor(n_estimators=3, max_depth=2, min_samples_leaf=2, random_state=12, n_jobs=1)
    forest.fit(x, np.log1p(y))
    return x, y, forest


def test_leaf_mixture_normalizes_and_median_minimizes_weighted_absolute_loss():
    x, y, forest = fixture()
    model = LeafDistributionReference().fit(forest, x, y)
    query = np.array([[.4, .1], [3.5, .8], [6., .5]])
    values = model.predict_components(query, chunk_size=1)
    source_leaves, query_leaves = forest.apply(x), forest.apply(query)
    for i in range(len(query)):
        weights = np.zeros(len(y))
        for j in range(len(forest.estimators_)):
            matched = source_leaves[:, j] == query_leaves[i, j]
            weights[matched] += 1./(matched.sum()*len(forest.estimators_))
        np.testing.assert_allclose(weights.sum(), 1.)
        np.testing.assert_allclose(values["mean"][i], weights@y)
        losses = np.array([weights@np.abs(y-candidate) for candidate in y])
        loss = weights@np.abs(y-values["median"][i])
        np.testing.assert_allclose(loss, losses.min())
    np.testing.assert_array_equal(model.predict(query), values["median"])


def test_distribution_fit_preserves_tree_states_and_replays_new_node_queries(tmp_path):
    x, y, forest = fixture()
    before = forest.predict(x).copy()
    states = [(tree.tree_.threshold.copy(), tree.tree_.value.copy()) for tree in forest.estimators_]
    model = LeafDistributionReference().fit(forest, x, y)
    np.testing.assert_array_equal(forest.predict(x), before)
    for tree, (threshold, value) in zip(forest.estimators_, states, strict=True):
        np.testing.assert_array_equal(tree.tree_.threshold, threshold)
        np.testing.assert_array_equal(tree.tree_.value, value)
    joblib.dump(model, tmp_path/"distribution.joblib")
    restored = joblib.load(tmp_path/"distribution.joblib")
    query = np.repeat([[2.5, .6]], 11, axis=0)
    for name, prediction in model.predict_components(query, chunk_size=3).items():
        np.testing.assert_array_equal(prediction, restored.predict_components(query, chunk_size=4)[name])


def test_leaf_distribution_only_changes_with_supplied_source_concentrations():
    x, y, forest = fixture()
    model = LeafDistributionReference().fit(forest, x, y)
    shifted = LeafDistributionReference().fit(forest, x, y+10.)
    for name, values in model.predict_components(x).items():
        np.testing.assert_allclose(shifted.predict_components(x)[name], values+10.)
    with pytest.raises(ValueError, match="source concentrations"):
        LeafDistributionReference().fit(forest, x, [1., 2., np.nan, 4., 5., 6.])
