"""Synthetic contracts for the v3 cross-analyte visibility policies.

These tests deliberately avoid local WQP artifacts.  They pin the two pieces
that are easy to get subtly wrong when constructing a target station profile:
only same-month K-session support is exposed, and unavailable profile
dimensions are omitted from pairwise distances rather than represented by a
zero value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_spatial_cross_analyte_source_v3 import (
    masked_neighbours,
    visible_aux_mask,
)


def test_k_session_opens_only_support_months_and_is_nested() -> None:
    n_stations, n_months = 3, 654
    aux = np.ones((n_stations, n_months), dtype=bool)
    target_cells = 1 * n_months + np.array([2, 7, 10, 20, 30, 40])

    k0 = visible_aux_mask(aux, np.array([1]), target_cells, 0)
    k1 = visible_aux_mask(aux, np.array([1]), target_cells, 1)
    k3 = visible_aux_mask(aux, np.array([1]), target_cells, 3)

    # Target row is hidden by default, then exactly the first K support months
    # are reopened.  Other stations retain their independently observed data.
    assert not k0[1].any()
    assert np.flatnonzero(k1[1]).tolist() == [2]
    # support_schedule uses first/middle/last as its first three candidates.
    assert np.flatnonzero(k3[1]).tolist() == [2, 10, 40]
    assert np.array_equal(k1[1] <= k3[1], np.ones(n_months, dtype=bool))
    assert np.array_equal(k3[[0, 2]], aux[[0, 2]])


def test_k_session_respects_auxiliary_missingness() -> None:
    aux = np.ones((2, 654), dtype=bool)
    aux[1, 3] = False
    target_cells = 1 * 654 + np.array([1, 3, 5, 10, 20, 30])
    visible = visible_aux_mask(aux, np.array([1]), target_cells, 5)

    # The support schedule requests months 1, 3, 5, but an unobserved month
    # stays hidden and cannot silently become an observed auxiliary feature.
    assert np.flatnonzero(visible[1]).tolist() == [1, 5, 10, 30]


def test_masked_distance_ignores_unavailable_dimensions() -> None:
    # Source 0 is close on the dimensions shared with the target.  Source 1
    # differs only on a dimension that is unavailable for the target; changing
    # that value must not change the selected neighbour.
    descriptors = np.array(
        [
            [0.0, 0.0, 100.0],
            [0.2, 0.2, -100.0],
            [1.0, 1.0, 0.0],
            [0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    available = np.ones_like(descriptors, dtype=bool)
    available[3, 2] = False  # target's unavailable auxiliary dimension
    sources = np.array([0, 1, 2])
    selected = masked_neighbours(descriptors, available, sources, 3, 1)
    assert selected.tolist() == [0]

    perturbed = descriptors.copy()
    perturbed[0, 2] = -1e9
    perturbed[1, 2] = 1e9
    selected_after = masked_neighbours(perturbed, available, sources, 3, 1)
    assert selected_after.tolist() == selected.tolist()
