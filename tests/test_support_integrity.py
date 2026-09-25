"""Contracts for the fail-closed Stage-2 support integrity controls."""

from __future__ import annotations

import numpy as np
import pytest

from river_graph.experiments.support_integrity import (
    site_shuffle_destination,
    stable_seed,
    validate_shuffle_record,
    value_shuffle_permutation,
)


def test_value_shuffle_is_deterministic_and_preserves_support_positions():
    support = np.asarray([2, 7, 12, 17], dtype=np.int64)
    first = value_shuffle_permutation(support, seed=stable_seed("u", 5))
    second = value_shuffle_permutation(support, seed=stable_seed("u", 5))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(len(support)))


def test_site_shuffle_preserves_month_and_excludes_support_and_query():
    # Six stations x three months, all target observations visible.
    mask = np.ones(18, dtype=bool)
    record = site_shuffle_destination(
        np.asarray([1, 4]),
        np.asarray([7, 10]),
        np.arange(6),
        month_index=1,
        n_months=3,
        observed_flat=mask,
        seed=stable_seed("site"),
    )
    assert record["status"] == "identifiable"
    destinations = np.asarray(record["destination_cells"])
    assert len(destinations) == 2
    assert np.all(destinations % 3 == 1)
    assert not np.intersect1d(destinations, [1, 4, 7, 10]).size
    validate_shuffle_record(
        {
            "mode": "site_shuffle",
            "support_cells": [1, 4],
            "query_cells": [7, 10],
            "month_index": 1,
            **record,
        },
        n_months=3,
    )


def test_site_shuffle_is_explicitly_unidentifiable_when_pool_is_too_small():
    # Exactly support + query are the only observed target cells in this month.
    mask = np.zeros(12, dtype=bool)
    mask[[1, 4, 7, 10]] = True
    record = site_shuffle_destination(
        np.asarray([1, 4]),
        np.asarray([7, 10]),
        np.arange(4),
        month_index=1,
        n_months=3,
        observed_flat=mask,
        seed=stable_seed("site"),
    )
    assert record["status"] == "not_identifiable_by_design"
    assert record["destination_cells"] == []


def test_site_shuffle_rejects_month_change():
    with pytest.raises(ValueError, match="changed month"):
        validate_shuffle_record(
            {
                "mode": "site_shuffle",
                "support_cells": [1, 4],
                "query_cells": [7, 10],
                "month_index": 1,
                "status": "identifiable",
                "candidate_count": 4,
                "destination_cells": [0, 3],
            },
            n_months=3,
        )


def test_value_shuffle_record_rejects_non_permutation():
    with pytest.raises(ValueError, match="not a permutation"):
        validate_shuffle_record(
            {
                "mode": "value_shuffle",
                "support_cells": [1, 4],
                "query_cells": [7],
                "value_permutation": [0, 0],
            },
            n_months=3,
        )
