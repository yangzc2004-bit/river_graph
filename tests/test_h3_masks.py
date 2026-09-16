"""T03 invariants for the h3a_v1 internal validation masks.

Two layers: synthetic masks that exercise the builders directly (fast, no data
files), and checks over the masks actually written to
experiments/h3a_v1/masks when they exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.experiments.h3_masks import (
    DEV_SCENARIOS,
    INTERNAL_VAL_SEED,
    KEY_SCENARIOS,
    MASK_SOURCES,
    assert_source_grid,
    assert_visible_roles,
    build_masks,
    component_report,
    e3_internal_split,
    e3_internal_stations,
    load_mask,
    migrate_cells,
    read_source_grid,
    split_val_context,
    station_order_digest,
    write_masks,
    write_source_grid,
)

ROOT = Path(__file__).resolve().parents[1]
H3_MASKS = ROOT / "experiments/h3a_v1r3/masks"
H3_MANIFEST = ROOT / "experiments/h3a_v1r3/split_manifest.json"
SOURCE_MASKS = ROOT / "experiments/masks"
FROZEN_MASKS = SOURCE_MASKS
SOURCE_GRID = SOURCE_MASKS / "source_grid.json"


def synthetic_world(n: int = 30, t: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_mask = rng.random((n, t)) < 0.4
    sites = [f"s{i}" for i in range(n)]
    edges = pd.DataFrame(
        {
            "source": [sites[i] for i in range(n - 1)],
            "target": [sites[i + 1] for i in range(n - 1)],
        }
    )
    return y_mask, edges, sites


def test_split_val_context_is_20_80_and_deterministic():
    cells = np.arange(1000, dtype=np.int64)
    context, target = split_val_context(cells)
    assert len(context) == 200
    assert len(target) == 800
    assert np.array_equal(np.sort(np.concatenate([context, target])), cells)
    assert not np.intersect1d(context, target).size
    again = split_val_context(cells[::-1].copy())
    np.testing.assert_array_equal(again[0], context)
    np.testing.assert_array_equal(again[1], target)


def test_e3_internal_stations_avoid_the_outer_test():
    y_mask, _, _ = synthetic_world()
    outer_test = np.flatnonzero(y_mask.ravel())[:50]
    rows = e3_internal_stations(y_mask, outer_test)
    t = y_mask.shape[1]
    test_rows = set((outer_test // t).tolist())
    assert not (set(rows.tolist()) & test_rows)
    # independent re-derivation of the 10% rule over eligible stations
    in_test = np.bincount(outer_test // t, minlength=y_mask.shape[0])
    eligible = (y_mask.sum(1) - in_test) > 0
    assert len(rows) == max(1, round(0.10 * int(eligible.sum())))
    assert 0.05 <= len(rows) / int(eligible.sum()) <= 0.15


def test_e3_internal_split_gives_val_stations_no_train_labels():
    y_mask, _, _ = synthetic_world()
    t = y_mask.shape[1]
    outer_test = np.flatnonzero(y_mask.ravel())[:50]
    split, val_rows, _ = e3_internal_split(y_mask, outer_test)
    val_rows = set(val_rows.tolist())
    assert all((cell // t) in val_rows for cell in split["val"].tolist())
    assert not any((cell // t) in val_rows for cell in split["train"].tolist())
    observed = set(np.flatnonzero(y_mask.ravel()).tolist())
    assert set(split["train"].tolist()) | set(split["val"].tolist()) | set(
        split["test"].tolist()
    ) == observed
    assert not set(split["train"].tolist()) & set(split["test"].tolist())


def test_component_report_counts_selected_stations():
    _, edges, sites = synthetic_world()
    rows = np.array([0, 5, 9], dtype=np.int64)
    report = component_report(edges, sites, rows)
    assert report["n_selected_stations"] == 3
    assert report["selected_components_nonempty"] == 1
    assert report["selected_in_largest_component"] == 3
    assert report["n_edges"] == len(sites) - 1


# ---------------------------------------------------- written masks on disk


pytestmark_data = pytest.mark.skipif(
    not H3_MASKS.is_dir(), reason="h3a_v1 masks have not been built yet"
)


def _pairs(cells, width: int) -> set:
    cells = np.asarray(cells, dtype=np.int64)
    return set(zip((cells // width).tolist(), (cells % width).tolist()))


@pytestmark_data
def test_written_masks_match_the_frozen_outer_test_in_station_month_coordinates():
    """Raw integers are grid-dependent; (station, month) pairs are not."""
    manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    old_t = manifest["source_grid_check"]["source"]["n_months"]
    new_t = manifest["source_grid_check"]["target"]["n_months"]
    for name, source in MASK_SOURCES.items():
        built = load_mask(H3_MASKS / f"{name}.npz")
        frozen = load_mask(SOURCE_MASKS / f"{source}.npz")
        frozen_pairs = _pairs(frozen["test"], old_t)
        built_pairs = _pairs(built["test"], new_t)
        assert not (built_pairs - frozen_pairs), name
        dropped = frozen_pairs - built_pairs
        assert len(dropped) == manifest["masks"][name][
            "outer_test_cells_dropped_unobserved"
        ]


@pytestmark_data
def test_written_masks_have_disjoint_roles():
    for name in KEY_SCENARIOS:
        split = load_mask(H3_MASKS / f"{name}.npz")
        seen: dict[int, str] = {}
        for role in ("train", "val", "val_context", "context", "test"):
            for cell in split.get(role, np.empty(0, dtype=np.int64)).tolist():
                assert cell not in seen, (name, role, seen.get(cell))
                seen[cell] = role


@pytestmark_data
def test_e2b_diagnostic_shares_the_scoring_target():
    """The selection-visibility diagnostic must reuse the e2b target pool."""
    a = load_mask(H3_MASKS / "e2b_partial_hiddenctx.npz")
    b = load_mask(H3_MASKS / "e2b_partial.npz")
    for role in ("train", "val", "test", "context", "val_context"):
        np.testing.assert_array_equal(a[role], b[role])
    assert "val_context" in b["visible_roles"]
    assert "val_context" not in a["visible_roles"]
    assert len(b["val_context"]) == round(
        0.2 * (len(b["val"]) + len(b["val_context"]))
    )


@pytestmark_data
def test_e2a_strict_has_no_future_context_and_never_opens_val_context():
    """R3: the strict temporal control must not see any test-period DOC."""
    split = load_mask(H3_MASKS / "e2a_strict.npz")
    assert "context" not in split
    assert "val_context" in split          # recorded, so no cell vanishes
    assert "val_context" not in split["visible_roles"]
    assert set(split["visible_roles"].tolist()) == {"train"}
    manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    old_t = manifest["source_grid_check"]["source"]["n_months"]
    new_t = manifest["source_grid_check"]["target"]["n_months"]
    frozen = load_mask(SOURCE_MASKS / "e2a_strict.npz")
    assert len(frozen["test"]) == 3184     # the original strict target pool
    # migrated to this grid; the station/month assignment is unchanged, and
    # with a complete cache nothing was dropped
    assert _pairs(frozen["test"], old_t) == _pairs(split["test"], new_t)
    assert len(split["test"]) == len(frozen["test"]) == 3184


@pytestmark_data
def test_every_written_mask_declares_valid_visible_roles():
    manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    for name in list(KEY_SCENARIOS) + list(manifest["diagnostic_scenarios"]):
        split = load_mask(H3_MASKS / f"{name}.npz")
        roles = assert_visible_roles(split)
        assert roles == tuple(manifest["masks"][name]["visible_roles"])
        assert "val" not in roles and "test" not in roles


@pytestmark_data
def test_e3_internal_validation_stations_carry_no_train_cells():
    for name in ("e3_internal_seed42", "e3_internal_seed43", "e3_internal_seed44"):
        split = load_mask(H3_MASKS / f"{name}.npz")
        t = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))[
            "source_grid_check"
        ]["target"]["n_months"]
        val_rows = set((split["val"] // t).tolist())
        assert val_rows
        assert not (set((split["train"] // t).tolist()) & val_rows)
        assert set(split["val_sites"].tolist())


@pytestmark_data
def test_manifest_matches_the_mask_files():
    manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["internal_val_seed"] == INTERNAL_VAL_SEED
    assert manifest["dev_scenarios"] == list(DEV_SCENARIOS)
    assert manifest["key_scenarios"] == list(KEY_SCENARIOS)
    import hashlib

    for name in KEY_SCENARIOS:
        digest = hashlib.sha256((H3_MASKS / f"{name}.npz").read_bytes()).hexdigest()
        assert manifest["masks"][name]["sha256"] == digest
        for role, count in manifest["masks"][name]["roles"].items():
            assert len(load_mask(H3_MASKS / f"{name}.npz")[role]) == count


def _dataset_for_masks():
    """A dataset whose (station, month) grid matches the frozen masks.

    The masks are built from y_mask, so a dataset with a different number of
    months would shift every flat cell index. The providers keep publishing
    observations, so a freshly rebuilt dataset legitimately has more months
    than the frozen masks were built on; in that case the reproduction check
    is skipped with an explicit reason instead of failing or silently passing.
    """
    import torch

    manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    wanted = manifest["dataset"]
    for candidate in sorted((ROOT / "data/processed").glob("mississippi_graph_v0*.pt")):
        if candidate.stem.endswith("_smoke50"):
            continue
        dataset = torch.load(candidate, weights_only=False)
        if (
            len(dataset["site_no"]) == wanted["n_stations"]
            and int(dataset["y_mask"].sum()) == wanted["observed_cells"]
        ):
            return dataset
    return None


@pytestmark_data
def test_rebuilding_the_masks_is_reproducible():
    """Rebuild from the same grid and compare every array."""
    dataset = _dataset_for_masks()
    if dataset is None:
        pytest.skip(
            "no local dataset matches the grid the frozen masks were built on; "
            "the providers have published more months since, which shifts every "
            "flat cell index"
        )
    y_mask = dataset["y_mask"].numpy()
    sites = list(dataset["site_no"])
    edges = pd.read_csv(ROOT / "data/processed/graph_edges.csv", dtype=str)
    masks, manifest = build_masks(SOURCE_MASKS, y_mask, edges, sites)
    for name in KEY_SCENARIOS:
        built = load_mask(H3_MASKS / f"{name}.npz")
        for role in built:
            np.testing.assert_array_equal(
                built[role], masks[name][role], err_msg=f"{name}.{role}"
            )
    assert set(manifest["masks"]) == set(masks)
    for name in KEY_SCENARIOS:
        assert manifest["masks"][name]["roles"] == json.loads(
            H3_MANIFEST.read_text(encoding="utf-8")
        )["masks"][name]["roles"]


def test_the_frozen_masks_declare_their_source_grid():
    record = json.loads(SOURCE_GRID.read_text(encoding="utf-8"))
    assert record["n_stations"] == 571
    assert record["n_months"] == 652
    assert record["observed_cells"] == 33048
    assert record["station_order_sha256"]


def test_a_missing_source_grid_record_is_refused(tmp_path):
    with pytest.raises(ValueError, match="source_grid.json"):
        read_source_grid(tmp_path)


def test_a_different_station_order_is_refused_before_import(tmp_path):
    y_mask = np.zeros((4, 6), dtype=bool)
    source = write_source_grid(tmp_path, ["a", "b", "c", "d"], y_mask=y_mask)
    assert source["n_stations"] == 4
    with pytest.raises(ValueError, match="station order differs"):
        assert_source_grid(source, y_mask, ["d", "c", "b", "a"])


def test_a_station_count_change_is_refused_before_import(tmp_path):
    y_mask = np.zeros((4, 6), dtype=bool)
    source = write_source_grid(tmp_path, ["a", "b", "c", "d"], y_mask=y_mask)
    source["station_order_sha256"] = station_order_digest(["a", "b", "c"])
    with pytest.raises(ValueError, match="station count differs"):
        assert_source_grid(source, np.zeros((3, 6), dtype=bool), ["a", "b", "c"])


def test_migration_preserves_station_and_month_pairs():
    """A flat index only means something together with its grid."""
    y_mask = np.ones((3, 13), dtype=bool)
    source = {"n_stations": 3, "n_months": 12, "observed_cells": 36}
    cells = np.array([2 * 12 + 11, 1], dtype=np.int64)
    audit: dict = {}
    moved = migrate_cells(cells, source, y_mask, "train", audit)
    assert sorted((int(c) // 13, int(c) % 13) for c in moved) == [(0, 1), (2, 11)]
    assert audit["train"] == {"before": 2, "after": 2, "dropped_unobserved": 0}


def test_migration_drops_cells_that_are_no_longer_observed():
    y_mask = np.ones((3, 13), dtype=bool)
    y_mask[2, 11] = False            # this observation disappeared
    source = {"n_stations": 3, "n_months": 12, "observed_cells": 36}
    cells = np.array([2 * 12 + 11, 1], dtype=np.int64)
    audit: dict = {}
    moved = migrate_cells(cells, source, y_mask, "val", audit)
    assert moved.tolist() == [1]
    assert audit["val"]["dropped_unobserved"] == 1


def test_migration_refuses_a_station_row_outside_the_dataset():
    y_mask = np.ones((2, 8), dtype=bool)
    source = {"n_stations": 3, "n_months": 12, "observed_cells": 36}
    with pytest.raises(ValueError, match="station row 2"):
        migrate_cells(np.array([2 * 12 + 1]), source, y_mask, "train", {})


@pytestmark_data
def test_the_written_manifest_records_the_grid_check_and_the_migration():
    manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
    check = manifest["source_grid_check"]
    assert check["source"]["n_months"] == 652
    assert check["target"]["n_months"] == 653
    assert check["requires_migration"] is True
    assert manifest["migration_audit"]
    for source, roles in manifest["migration_audit"].items():
        for role, counts in roles.items():
            assert counts["after"] <= counts["before"], (source, role)
    for name in KEY_SCENARIOS:
        assert "outer_test_cells_dropped_unobserved" in manifest["masks"][name]


def test_write_masks_records_hashes_and_leaves_no_temp_files(tmp_path):
    masks = {
        "demo": {
            "train": np.arange(4, dtype=np.int64),
            "val": np.array([8], dtype=np.int64),
            "test": np.array([9], dtype=np.int64),
        }
    }
    manifest = {"masks": {"demo": {"roles": {}}}}
    out = tmp_path / "masks"
    hashes = write_masks(masks, manifest, out)
    assert set(hashes) == {"demo"}
    assert manifest["masks"]["demo"]["sha256"] == hashes["demo"]
    assert (out / "demo.npz").is_file()
    assert (tmp_path / "split_manifest.json").is_file()
    assert not list(out.glob("*.tmp"))
    again = write_masks(masks, {"masks": {"demo": {"roles": {}}}}, out)
    assert again == hashes
