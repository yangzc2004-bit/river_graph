import numpy as np
import pytest

from river_graph.experiments.spatial_fewshot import (
    station_level_correction,
    support_schedule,
    support_view,
)


def test_support_schedule_is_nested_and_reserves_fixed_query():
    cells = np.concatenate([
        np.arange(100, 130, dtype=np.int64),
        np.arange(200, 230, dtype=np.int64),
    ])
    schedule, query = support_schedule(cells, n_months=100)
    assert len(query) == 50
    assert all(len(schedule[station]) == 5 for station in (1, 2))
    assert set(schedule[1][:1]).issubset(set(schedule[1][:3]))
    assert set(schedule[1][:3]).issubset(set(schedule[1][:5]))
    assert not np.intersect1d(query, np.concatenate(list(schedule.values()))).size


def test_support_view_keeps_query_identical_across_k():
    split = {
        "train": np.arange(0, 10, dtype=np.int64),
        "val": np.array([], dtype=np.int64),
        "context": np.array([], dtype=np.int64),
        "test": np.concatenate([
            np.arange(100, 130, dtype=np.int64),
            np.arange(200, 230, dtype=np.int64),
        ]),
    }
    queries = []
    for k in (0, 1, 3, 5):
        view, support, query = support_view(split, target_role="test", k=k, n_months=100)
        queries.append(query)
        assert len(support) == 2 * k
        assert np.intersect1d(support, query).size == 0
        assert np.intersect1d(view["train"], support).size == 0
    assert all(np.array_equal(queries[0], q) for q in queries[1:])


def test_station_level_correction_reads_support_only():
    values = np.arange(20, dtype=float).reshape(2, 10)
    support = np.array([0, 10], dtype=np.int64)
    query = np.array([1, 2, 11, 12], dtype=np.int64)
    base = np.ones(4, dtype=float)
    changed = station_level_correction(values, support, query, base, n_months=10, alpha=1.0)
    assert np.all(np.isfinite(changed))
    assert not np.allclose(changed[:2], changed[2:])
    with pytest.raises(ValueError, match="support and query"):
        station_level_correction(values, support, np.array([0, 2, 11, 12]), base,
                                 n_months=10, alpha=1.0)
