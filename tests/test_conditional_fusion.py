"""Contracts for the conditional support-fusion primitives."""

from __future__ import annotations

import numpy as np
import pytest

from river_graph.experiments.conditional_fusion import (
    blend_predictions,
    clip_gate_weight,
    fixed_weight_grid,
)


def test_blend_preserves_native_units_and_endpoints():
    h2x = np.asarray([10.0, 20.0])
    baseline = np.asarray([0.0, 4.0])
    np.testing.assert_array_equal(blend_predictions(h2x, baseline, 1.0), h2x)
    np.testing.assert_array_equal(blend_predictions(h2x, baseline, 0.0), baseline)
    np.testing.assert_allclose(blend_predictions(h2x, baseline, 0.25), [2.5, 8.0])


def test_gate_weight_is_clipped_but_nonfinite_is_rejected():
    np.testing.assert_array_equal(clip_gate_weight([-1.0, 0.5, 2.0]), [0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="finite"):
        clip_gate_weight([np.nan])


def test_weight_grid_is_deterministic():
    np.testing.assert_array_equal(fixed_weight_grid(0.25), [0.0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        fixed_weight_grid(0.0)

