"""Protocol tests for K-shot pseudo-new-watershed splits and tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from river_graph.experiments.kshot import (
    label_tasks,
    make_region_split,
    make_support_query_tasks,
    region_station_index,
    tasks_to_frame,
    validate_split,
    validate_tasks,
)


@pytest.fixture
def toy():
    # 4 stations, 6 months; stations 2,3 form the "region"
    y_mask = np.zeros((4, 6), dtype=bool)
    # region co-observed months with n=4,3,2,1
    y_mask[2, 0] = y_mask[3, 0] = True
    y_mask[0, 0] = y_mask[1, 0] = True  # also outside for train
    y_mask[2, 1] = y_mask[3, 1] = y_mask[0, 1] = True
    y_mask[2, 2] = y_mask[3, 2] = True
    y_mask[2, 3] = True  # single region obs, no task
    y_mask[0, 4] = y_mask[1, 4] = y_mask[2, 4] = y_mask[3, 4] = True
    y_mask[1, 5] = True
    return y_mask


def test_region_split_hides_region(toy):
    split = make_region_split(toy, [2, 3], seed=0)
    assert validate_split(split, toy) == []
    t = toy.shape[1]
    region = {2, 3}
    for flat in list(split["train"]) + list(split["val"]):
        assert flat // t not in region
    for flat in split["test"]:
        assert flat // t in region
    assert set(split["region_rows"]) == {2, 3}


def test_support_query_nested_and_fixed_query(toy):
    months = [f"2000-0{i}-01" for i in range(1, 7)]
    tasks = make_support_query_tasks(toy, [2, 3], months, k_list=(0, 1, 3, 5), min_query=1, seed=0)
    tasks = label_tasks(tasks, "toy")
    assert validate_tasks(tasks, toy, k_list=(0, 1, 3, 5), min_query=1) == []
    # region co-obs months: j=0,1,2,4 (n=2) and j=3 (n=1, K=0 only)
    assert len(tasks) == 5
    by_month = {t.month_index: t for t in tasks}
    assert set(by_month[0].support_by_k) == {0, 1}
    assert set(by_month[3].support_by_k) == {0}
    for task in tasks:
        prev = ()
        for k in sorted(task.support_by_k):
            support = task.support_by_k[k]
            assert len(support) == k
            assert not set(support) & set(task.query)
            assert set(prev) <= set(support)
            prev = support
        assert len(task.query) >= 1


def test_tasks_to_frame_columns(toy):
    months = [f"2000-0{i}-01" for i in range(1, 7)]
    tasks = label_tasks(
        make_support_query_tasks(toy, [2, 3], months, k_list=(0, 1), min_query=1, seed=0),
        "toy",
    )
    df = tasks_to_frame(tasks)
    assert {"region", "k", "query_cells", "support_cells"} <= set(df.columns)
    assert set(df.k.unique()) <= {0, 1}


def test_region_station_index():
    nodes = pd.DataFrame(
        {
            "site_no": ["a", "b", "c", "d"],
            "huc_cd": ["10300101", "10300102", "70500020", "10300101"],
        }
    )
    idx = region_station_index(["a", "b", "c", "d"], nodes, "10300101")
    assert set(idx.tolist()) == {0, 3}


def test_rejects_region_without_obs():
    empty_mask = np.zeros((4, 3), dtype=bool)
    empty_mask[0, 0] = True
    with pytest.raises(ValueError, match="no observed"):
        make_region_split(empty_mask, [2, 3], seed=0)
