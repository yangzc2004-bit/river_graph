"""Contracts for the cross-basin, multi-analyte transfer route.

These tests exercise the data visibility boundary only.  They deliberately use
small in-memory arrays and do not fit a model or touch any frozen experiment
artifacts.
"""

from __future__ import annotations

import numpy as np

from river_graph.experiments.transfer import (
    K_VALUES,
    availability_tasks,
    cross_basin_view,
    make_cross_basin_split,
)


def _toy_transfer_case(*, n: int = 12, months: int = 5, target_rows=range(5, 12)):
    """Return distinct labels and a complete three-analyte observation grid."""
    rows = np.asarray(tuple(target_rows), dtype=int)
    labels = np.arange(3 * n * months, dtype=float).reshape(3, n, months)
    # Keep the perturbation checks sensitive to every analyte and cell while
    # avoiding accidental equality between source and target values.
    labels += np.arange(3, dtype=float)[:, None, None] * 10_000
    observed = np.ones_like(labels, dtype=bool)
    split = make_cross_basin_split(observed, rows, seed=42)
    return labels, rows, split


def test_cross_basin_split_hides_every_analyte_in_target_huc6():
    labels, rows, split = _toy_transfer_case()
    source_rows = np.setdiff1d(np.arange(labels.shape[1]), rows)

    assert not split["train"][:, rows, :].any()
    assert not split["val"][:, rows, :].any()
    assert split["test"][:, rows, :].all()
    assert split["train"][:, source_rows, :].any()
    assert split["val"][:, source_rows, :].any()


def test_cross_basin_view_exposes_only_target_support_labels():
    labels, rows, split = _toy_transfer_case()
    months = labels.shape[2]
    tasks = availability_tasks(
        split["observed"][1], rows, [f"2000-{m:02d}-01" for m in range(1, months + 1)], 7
    )
    task = tasks[0]
    support = np.asarray(task["support_cells_by_k"]["3"], dtype=int)
    query = np.asarray(task["query_cells"], dtype=int)

    view = cross_basin_view(
        labels,
        split["observed"],
        split["train"],
        split["val"],
        target=1,
        target_rows=rows,
        support=support,
        query=query,
    )

    # Source fit and selection views contain no target-basin labels, while the
    # target-side support view contains exactly the requested target cells.
    assert not view["fit_mask"][:, rows, :].any()
    assert not view["selection_mask"][:, rows, :].any()
    np.testing.assert_array_equal(view["fit_labels"], np.where(split["train"], labels, 0))
    np.testing.assert_array_equal(
        view["selection_labels"], np.where(split["val"], labels, 0)
    )
    expected_support = np.zeros((labels.shape[1], months), dtype=float)
    expected_support.ravel()[support] = labels[1].ravel()[support]
    np.testing.assert_array_equal(view["support_labels"], expected_support)
    assert np.all(view["support_labels"].ravel()[query] == 0)
    assert set(view["support_cells"].tolist()) == set(support.tolist())
    assert set(view["query_cells"].tolist()) == set(query.tolist())


def test_k0_keeps_target_analyte_source_labels_but_has_no_target_support():
    labels, rows, split = _toy_transfer_case()
    months = labels.shape[2]
    task = availability_tasks(
        split["observed"][1], rows, [f"2000-{m:02d}-01" for m in range(1, months + 1)], 7
    )[0]
    query = np.asarray(task["query_cells"], dtype=int)

    view = cross_basin_view(
        labels,
        split["observed"],
        split["train"],
        split["val"],
        target=1,
        target_rows=rows,
        support=np.array([], dtype=int),
        query=query,
    )

    source_rows = np.setdiff1d(np.arange(labels.shape[1]), rows)
    assert view["fit_mask"][1, source_rows, :].any()
    assert not view["fit_mask"][1, rows, :].any()
    assert "1" in view["source_statistics"]
    assert not view["support_labels"].any()


def test_cross_basin_tasks_have_fixed_queries_and_nested_support():
    labels, rows, split = _toy_transfer_case()
    months = labels.shape[2]
    tasks = availability_tasks(
        split["observed"][1], rows, [f"2000-{m:02d}-01" for m in range(1, months + 1)], 7
    )
    assert len(tasks) == months
    for task in tasks:
        queries = set(task["query_cells"])
        assert queries
        previous = set()
        for k in K_VALUES:
            support = set(task["support_cells_by_k"][str(k)])
            assert len(support) == k
            assert support.isdisjoint(queries)
            assert previous.issubset(support)
            previous = support


def test_hidden_target_label_perturbation_does_not_change_fit_visible_views():
    labels, rows, split = _toy_transfer_case()
    months = labels.shape[2]
    task = availability_tasks(
        split["observed"][1], rows, [f"2000-{m:02d}-01" for m in range(1, months + 1)], 7
    )[0]
    support = np.asarray(task["support_cells_by_k"]["3"], dtype=int)
    query = np.asarray(task["query_cells"], dtype=int)
    original = cross_basin_view(
        labels,
        split["observed"],
        split["train"],
        split["val"],
        target=1,
        target_rows=rows,
        support=support,
        query=query,
    )

    perturbed = labels.copy()
    target_cells = np.arange(labels.shape[1] * months).reshape(labels.shape[1], months)
    hidden = np.isin(target_cells // months, rows)
    hidden.ravel()[support] = False
    perturbed[:, hidden] += 1_000_000
    changed = cross_basin_view(
        perturbed,
        split["observed"],
        split["train"],
        split["val"],
        target=1,
        target_rows=rows,
        support=support,
        query=query,
    )

    for key in ("fit_labels", "fit_mask", "selection_labels", "selection_mask", "support_labels"):
        np.testing.assert_array_equal(original[key], changed[key])
    assert original["source_statistics"] == changed["source_statistics"]

