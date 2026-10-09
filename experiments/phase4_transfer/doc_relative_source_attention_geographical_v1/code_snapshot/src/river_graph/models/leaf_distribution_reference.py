"""Source-only empirical DOC distributions on fixed environmental tree leaves."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class LeafDistributionReference:
    """Keep forest splits fixed; average source leaf distributions, then read out.

    Source concentrations are sorted once. Sparse leaf membership supports
    chunked query CDFs without a dense full-grid by source-observation matrix.
    The median is of the mixture distribution, not a mean of leaf medians.
    """

    def fit(self, forest, source_inputs, source_y):
        source_y = np.asarray(source_y, dtype=np.float64)
        if (source_y.ndim != 1 or len(source_inputs) != len(source_y) or not len(source_y)
                or not np.isfinite(source_y).all() or (source_y < 0).any()):
            raise ValueError("aligned finite nonnegative source concentrations are required")
        if getattr(forest, "bootstrap", False):
            raise ValueError("this fixed-leaf reference requires an unbootstrapped retained forest")
        self.forest = forest
        self.source_order_ = np.argsort(source_y, kind="stable")
        self.source_y_ = source_y[self.source_order_]
        self.offsets_ = np.r_[0, np.cumsum([tree.tree_.node_count for tree in forest.estimators_])]
        leaves = forest.apply(source_inputs[self.source_order_])
        rows, columns, values = [], [], []
        self.n_trees_ = len(forest.estimators_)
        for index in range(self.n_trees_):
            tree_leaf = leaves[:, index]
            count = np.bincount(tree_leaf, minlength=self.offsets_[index+1]-self.offsets_[index])
            rows.append(tree_leaf+self.offsets_[index])
            columns.append(np.arange(len(source_y)))
            values.append(1./(self.n_trees_*count[tree_leaf]))
        self.leaf_weights_ = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(int(self.offsets_[-1]), len(source_y)))
        self.n_features_in_ = forest.n_features_in_
        return self

    def _query_weights(self, inputs):
        leaves = self.forest.apply(inputs)+self.offsets_[:-1]
        selector = csr_matrix((np.ones(leaves.size),
            (np.repeat(np.arange(len(inputs)), self.n_trees_), leaves.ravel())),
            shape=(len(inputs), int(self.offsets_[-1])))
        weights = selector@self.leaf_weights_
        mass = np.asarray(weights.sum(axis=1)).ravel()
        if not np.allclose(mass, 1., rtol=1e-12, atol=1e-12):
            raise ValueError("query leaves lack source support or distribution mass is invalid")
        return weights.multiply((1./mass)[:, None]).tocsr()

    def predict_components(self, inputs, *, chunk_size=64):
        inputs = np.asarray(inputs)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features_in_ or chunk_size < 1:
            raise ValueError("aligned two-dimensional query inputs and positive chunks are required")
        mean, median = np.empty(len(inputs)), np.empty(len(inputs))
        for start in range(0, len(inputs), chunk_size):
            stop = min(start+chunk_size, len(inputs))
            weights = self._query_weights(inputs[start:stop])
            mean[start:stop] = weights@self.source_y_
            cdf = np.cumsum(weights.toarray(), axis=1)
            median[start:stop] = self.source_y_[(cdf >= .5).argmax(axis=1)]
        if not np.isfinite(mean).all() or not np.isfinite(median).all():
            raise ValueError("nonfinite distributional reference prediction")
        return {"mean": mean, "median": median}

    def predict(self, inputs):
        return self.predict_components(inputs)["median"]
