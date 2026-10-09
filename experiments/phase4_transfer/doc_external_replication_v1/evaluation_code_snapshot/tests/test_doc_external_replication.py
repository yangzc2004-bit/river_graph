"""External support and interval choices are fixed by source information."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_doc_external_replication_v1 import (
    empirical_interval,
    log_residual_quantile,
    source_ensemble_policy,
)


def test_source_support_policy_recovers_bias_without_query_population_changes():
    cells = np.arange(24)
    prediction = np.full(24, 2.)
    truth = np.full(24, 5.)
    policy = source_ensemble_policy(prediction, cells, truth, n_months=12)
    assert policy["validation_stations"] == 2
    assert policy["curve"]["0"]["selected"]["alpha"] == 0
    assert policy["curve"]["1"]["selected"]["alpha"] == 1
    assert policy["curve"]["5"]["selected"]["validation_mae"] < 1e-12
    assert {v["query_cells"] for v in policy["curve"].values()} == {14}
    np.testing.assert_allclose(policy["primary_k0_log_half_width"], np.log(2))
    # Exact ties choose smaller support strength.
    exact = source_ensemble_policy(truth, cells, truth, n_months=12)
    assert all(v["selected"]["alpha"] == 0 for v in exact["curve"].values())


def test_log_interval_finite_rank_zero_lower_and_positive_width():
    prediction = np.full(9, 1.)
    truth = np.arange(9.)
    q = log_residual_quantile(prediction, truth)
    np.testing.assert_allclose(q, np.log(9)-np.log(2))
    lower, upper = empirical_interval(np.array([0., 1., 5.]), q)
    assert lower[0] == 0
    assert np.all(upper > lower)
    assert np.all(lower <= np.array([0., 1., 5.]))
    assert np.all(upper >= np.array([0., 1., 5.]))
    with pytest.raises(ValueError, match="validation values"):
        log_residual_quantile(np.array([1., np.nan]), np.array([1., 2.]))
    with pytest.raises(ValueError, match="width"):
        empirical_interval(np.array([1.]), -1)
