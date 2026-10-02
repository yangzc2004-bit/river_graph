from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from river_graph.experiments.spatial_fewshot import station_level_correction
from river_graph.experiments.unified_spatial_protocol import (
    SPLIT_SEEDS,
    base_inference_split,
    build_unified_spatial_split,
    support_query_cells,
    validate_unified_spatial_split,
)


def test_partitions_are_station_disjoint_complete_and_reproducible():
    observed = np.ones((40, 24), dtype=bool)
    observed[0, 3:] = False  # This station can provide source labels only.
    first, metadata = build_unified_spatial_split(observed, seed=142)
    second, _ = build_unified_spatial_split(observed, seed=142)
    assert all(np.array_equal(first[role], second[role]) for role in first)
    validate_unified_spatial_split(observed, first)
    assert 0 in metadata["source_only_ineligible_stations"]
    assert set(first["train"] // 24).isdisjoint(first["test"] // 24)
    assert set(first["val"] // 24).isdisjoint(first["test"] // 24)
    assert np.array_equal(np.sort(np.concatenate(list(first.values()))), np.flatnonzero(observed))
    assert metadata["base_input_roles"] == ["train"]


def test_new_partition_seeds_produce_different_station_tasks():
    observed = np.ones((40, 24), dtype=bool)
    held = [tuple(build_unified_spatial_split(observed, seed=seed)[0]["test"])
            for seed in SPLIT_SEEDS]
    assert len(set(held)) == 3


@pytest.mark.parametrize("target_role", ["val", "test"])
def test_support_is_nested_and_all_k_share_exact_query(target_role):
    observed = np.ones((40, 24), dtype=bool)
    split, metadata = build_unified_spatial_split(observed, seed=142)
    supports, queries = [], []
    for k in (0, 1, 3, 5):
        support, query = support_query_cells(split, target_role=target_role, k=k, n_months=24)
        supports.append(support)
        queries.append(query)
        assert len(support) == k * metadata["station_counts"][target_role]
        assert not np.intersect1d(support, query).size
        assert not np.intersect1d(support, split["train"]).size
    assert all(np.array_equal(queries[0], query) for query in queries)
    assert all(set(a).issubset(b) for a, b in pairwise(supports))
    assert np.array_equal(np.sort(np.concatenate([supports[-1], queries[0]])), split[target_role])
    station = supports[-1][0] // 24
    schedule = metadata["target_tasks"][target_role]["support_candidates"][str(station)]
    assert np.array_equal(np.asarray(schedule) % 24, [0, 11, 23, 5, 17])


def test_prediction_view_hides_every_validation_test_and_support_label():
    observed = np.ones((40, 24), dtype=bool)
    split, _ = build_unified_spatial_split(observed, seed=142)
    view = base_inference_split(split)
    visible = np.concatenate([view[role] for role in ("train", "val", "context")])
    assert np.array_equal(visible, split["train"])
    assert not np.intersect1d(visible, split["val"]).size
    assert not np.intersect1d(visible, split["test"]).size
    assert np.array_equal(view["test"], np.union1d(split["val"], split["test"]))
    # The fitting split retains validation labels only as validation targets.
    assert split["val"].size > 0


def test_calibration_reads_only_selected_support_and_preserves_query():
    observed = np.ones((40, 24), dtype=bool)
    split, _ = build_unified_spatial_split(observed, seed=142)
    support, query = support_query_cells(split, target_role="test", k=3, n_months=24)
    labels = np.linspace(1, 20, observed.size).reshape(observed.shape)
    base = np.full(len(query), 2.0)
    corrected = station_level_correction(labels, support, query, base, n_months=24, alpha=0.5)
    modified = labels.copy().ravel()
    modified[np.setdiff1d(np.arange(observed.size), support)] = 1e6
    again = station_level_correction(modified, support, query, base, n_months=24, alpha=0.5)
    assert np.array_equal(corrected, again)
    assert not np.array_equal(corrected, base)
    assert np.array_equal(station_level_correction(
        labels, np.empty(0, dtype=int), query, base, n_months=24, alpha=0.5), base)


def test_validator_rejects_hidden_role_duplicates_and_station_overlap():
    observed = np.ones((40, 24), dtype=bool)
    split, _ = build_unified_spatial_split(observed, seed=142)
    bad = {role: cells.copy() for role, cells in split.items()}
    bad["test"] = np.append(bad["test"], bad["test"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_unified_spatial_split(observed, bad)
    bad = {role: cells.copy() for role, cells in split.items()}
    moved = bad["test"][0]
    bad["test"] = bad["test"][1:]
    bad["train"] = np.append(bad["train"], moved)
    with pytest.raises(ValueError, match="stations must be disjoint"):
        validate_unified_spatial_split(observed, bad)
    bad = {role: cells.copy() for role, cells in split.items()}
    bad["context"] = bad["test"][:1]
    with pytest.raises(ValueError, match="no context"):
        validate_unified_spatial_split(observed, bad)
    with pytest.raises(ValueError, match="support/context"):
        base_inference_split(bad)


@pytest.mark.parametrize("bad_mask", [np.ones(20), np.ones((0, 12)), np.full((30, 12), 2)])
def test_invalid_observation_masks_are_rejected(bad_mask):
    with pytest.raises(ValueError):
        build_unified_spatial_split(bad_mask, seed=142)


ST357 = Path("data/processed/mississippi_graph_graphfix_st357.pt")


@pytest.mark.skipif(not ST357.exists(), reason="local ST357 DOC dataset is not available")
def test_actual_st357_protocol_has_fixed_query_population_and_all_three_splits():
    import torch

    dataset = torch.load(ST357, map_location="cpu", weights_only=False)
    observed = dataset["y_mask"].numpy()
    assert observed.shape == (357, 654)
    test_partitions = []
    for seed in SPLIT_SEEDS:
        split, metadata = build_unified_spatial_split(observed, seed=seed)
        assert metadata["station_counts"] == {"train": 232, "val": 54, "test": 71}
        assert metadata["source_only_ineligible_stations"] == []
        support, query = support_query_cells(split, target_role="test", k=5, n_months=654)
        assert len(support) == 355
        assert len(query) == len(split["test"]) - 355
        test_partitions.append(tuple(metadata["station_roles"]["test"]))
    assert len(set(test_partitions)) == 3
