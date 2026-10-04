"""Current auxiliary chemistry is aligned, exogenous and separately ablatable."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from river_graph.models.auxiliary_chemistry_features import (
    apply_auxiliary_mode,
    build_auxiliary_chemistry_features,
)


def datasets():
    base = {"site_no": ["b", "a"], "months": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "x": np.arange(12).reshape(2, 3, 2).astype(np.float32),
            "x_mask": np.ones((2, 3, 2), dtype=bool),
            "feature_channels": ["temperature", "discharge"],
            "edge_index": np.array([[0], [1]]), "edge_attr": np.array([[.2, .8]]),
            "y": np.ones((2, 3)), "y_mask": np.ones((2, 3), dtype=bool)}
    ph, ec = copy.deepcopy(base), copy.deepcopy(base)
    ph["y"] = np.array([[7., 8., 9.], [6., 7., 8.]])
    ec["y"] = np.array([[0., 99., 999.], [3., 8., 15.]])
    ph["y_mask"][1, 2] = False
    ec["y_mask"][1, 1:] = False
    return base, ph, ec


def test_values_flags_preserved_grid_order_and_observed_zero_conductance():
    doc, ph, ec = datasets()
    out = build_auxiliary_chemistry_features(doc, ph, ec)
    assert out["full"].shape == (2, 3, 4) and out["full"].dtype == np.float32
    np.testing.assert_allclose(out["full"][0, 0], [.5, 0, 1, 1])
    np.testing.assert_allclose(out["full"][0, 1], [8/14, np.log(100), 1, 1])
    np.testing.assert_allclose(out["full"][1, 1], [.5, 0, 1, 0])
    np.testing.assert_array_equal(out["full"][1, 2], 0)
    np.testing.assert_array_equal(out["active"], [[1, 1, 1], [1, 1, 0]])


def test_all_modes_share_actual_active_gate_and_do_not_mutate_inputs():
    data = datasets()
    chemistry = build_auxiliary_chemistry_features(*data)
    for mode in ("no_aux", "masks", "chemistry"):
        output = build_auxiliary_chemistry_features(*data, mode=mode)
        np.testing.assert_array_equal(output["active"], chemistry["active"])
        np.testing.assert_array_equal(output["full"], apply_auxiliary_mode(chemistry["full"], mode))
        if mode != "chemistry":
            assert not output["full"][..., :2].any()
        if mode == "no_aux":
            assert not output["full"].any() and output["active"].any()
    assert not np.shares_memory(chemistry["full"], apply_auxiliary_mode(chemistry["full"], "chemistry"))


def test_doc_labels_and_doc_visibility_are_never_read():
    class NoDOC(dict):
        def __getitem__(self, key):
            if key in {"y", "y_mask"}:
                raise AssertionError("DOC label access")
            return super().__getitem__(key)
    doc, ph, ec = datasets()
    expected = build_auxiliary_chemistry_features(NoDOC(doc), ph, ec)
    doc["y"] = np.full((2, 3), np.nan)
    doc["y_mask"][:] = False
    output = build_auxiliary_chemistry_features(doc, ph, ec)
    np.testing.assert_array_equal(output["full"], expected["full"])
    np.testing.assert_array_equal(output["active"], expected["active"])


def test_auxiliary_visibility_future_and_station_isolation():
    doc, ph, ec = datasets()
    baseline = build_auxiliary_chemistry_features(doc, ph, ec)["full"]
    ph["y"][0, 2] = 10
    ec["y"][0, 2] = 200
    changed = build_auxiliary_chemistry_features(doc, ph, ec)["full"]
    np.testing.assert_array_equal(changed[:, :2], baseline[:, :2])
    np.testing.assert_array_equal(changed[1], baseline[1])
    assert not np.array_equal(changed[0, 2], baseline[0, 2])
    ph["y"][1, 2], ec["y"][1, 2] = np.nan, np.inf
    invisible = build_auxiliary_chemistry_features(doc, ph, ec)["full"]
    np.testing.assert_array_equal(invisible, changed)
    ph["y"][1, 2], ph["y_mask"][1, 2] = 7, True
    observed = build_auxiliary_chemistry_features(doc, ph, ec)
    assert observed["active"][1, 2]
    np.testing.assert_array_equal(observed["full"][1, 2], [.5, 0, 1, 0])


@pytest.mark.parametrize("key", ["site_no", "months", "x", "x_mask", "edge_index", "edge_attr"])
def test_alignment_mismatch_cannot_silently_join_wrong_cells(key):
    doc, ph, ec = datasets()
    if key in {"site_no", "months"}:
        ph[key] = list(reversed(ph[key]))
    elif key == "edge_index":
        ph[key] = ph[key][::-1].copy()
    elif key == "x_mask":
        ph[key][0, 0, 0] = False
    elif key == "x":
        ph[key][0, 0, 0] += 1
    else:
        ph[key][0, 0] += .1
    with pytest.raises(ValueError, match=key):
        build_auxiliary_chemistry_features(doc, ph, ec)


def test_visible_qc_bounds_and_binary_mask_are_enforced():
    doc, ph, ec = datasets()
    ph["y"][0, 0] = 15
    with pytest.raises(ValueError, match="QC"):
        build_auxiliary_chemistry_features(doc, ph, ec)
    ph["y"][0, 0] = 7
    ec["y"][0, 0] = -1
    with pytest.raises(ValueError, match="QC"):
        build_auxiliary_chemistry_features(doc, ph, ec)
    ec["y"][0, 0] = 0
    ec["y_mask"] = ec["y_mask"].astype(float); ec["y_mask"][0, 0] = .5
    with pytest.raises(ValueError, match="binary"):
        build_auxiliary_chemistry_features(doc, ph, ec)
    with pytest.raises(ValueError, match="mode"):
        apply_auxiliary_mode(np.zeros((1, 1, 4)), "unknown")
