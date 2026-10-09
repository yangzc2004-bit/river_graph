"""Catchment joins and true-zero semantics for label-free ecological features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from river_graph.models.ecological_composition import FIELDS, composition_inputs


def tables():
    nodes = pd.DataFrame({"site_no": ["a", "b", "c"], "comid": ["101.0", "101", "999"]})
    attrs = pd.DataFrame({"comid": [101], **{key: [float(i)] for i, key in enumerate(FIELDS)}})
    return nodes, attrs


def test_shared_catchment_alignment_distinguishes_zero_from_missing():
    nodes, attrs = tables()
    result = composition_inputs(["c", "a", "b"], nodes, attrs)
    np.testing.assert_array_equal(result["matched_comid"], [False, True, True])
    np.testing.assert_array_equal(result["detailed"][1], result["detailed"][2])
    assert not result["detailed"][0].any()
    assert result["detailed"][1, 0] == 0 and result["detailed"][1, 11] == 1
    np.testing.assert_array_equal(result["aggregate_control"][:, 11:], result["detailed"][:, 11:])
    np.testing.assert_allclose(result["family_totals"][1], [.03, .07, .26, .19])


def test_reject_duplicate_keys_and_mark_sentinel_as_unavailable():
    nodes, attrs = tables()
    with pytest.raises(ValueError, match="unique"):
        composition_inputs(["a"], nodes, pd.concat([attrs, attrs]))
    attrs.loc[0, FIELDS[3]] = -9999
    result = composition_inputs(["a"], nodes, attrs)
    assert result["detailed"][0, 3] == 0 and result["detailed"][0, 14] == 0
    attrs.loc[0, FIELDS[3]] = 101
    assert not composition_inputs(["a"], nodes, attrs)["valid"][0, 3]


def test_reject_overlapping_percentages_above_land_area():
    nodes, attrs = tables()
    attrs.loc[0, list(FIELDS)] = 15.
    with pytest.raises(ValueError, match="exceed100"):
        composition_inputs(["a"], nodes, attrs)
