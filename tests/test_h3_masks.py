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
    build_masks,
    component_report,
    e3_internal_split,
    e3_internal_stations,
    load_mask,
    split_val_context,
    write_masks,
)

ROOT = Path(__file__).resolve().parents[1]
H3_MASKS = ROOT / "experiments/h3a_v1/masks"
H3_MANIFEST = ROOT / "experiments/h3a_v1/split_manifest.json"
SOURCE_MASKS = ROOT / "experiments/masks"


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


@pytestmark_data
def test_written_masks_match_the_frozen_outer_test():
    for name, source in MASK_SOURCES.items():
        built = load_mask(H3_MASKS / f"{name}.npz")
        frozen = load_mask(SOURCE_MASKS / f"{source}.npz")
        np.testing.assert_array_equal(built["test"], np.sort(frozen["test"]))


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
def test_e2a_and_e2b_share_the_same_scoring_target():
    a = load_mask(H3_MASKS / "e2a_partial.npz")
    b = load_mask(H3_MASKS / "e2b_partial.npz")
    for role in ("train", "val", "test", "context"):
        np.testing.assert_array_equal(a[role], b[role])
    assert "val_context" not in a
    assert len(b["val_context"]) == round(
        0.2 * (len(b["val"]) + len(b["val_context"]))
    )


@pytestmark_data
def test_e3_internal_validation_stations_carry_no_train_cells():
    for name in ("e3_internal_seed42", "e3_internal_seed43", "e3_internal_seed44"):
        split = load_mask(H3_MASKS / f"{name}.npz")
        t = 652
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


@pytestmark_data
def test_rebuilding_the_masks_is_reproducible():
    """Rebuild into a temporary directory and compare every array."""
    import torch

    dataset = torch.load(
        ROOT / "data/processed/mississippi_graph_v04.pt", weights_only=False
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
