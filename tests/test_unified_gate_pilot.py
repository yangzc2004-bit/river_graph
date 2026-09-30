"""Contracts for the observable-feature expert fusion pilot."""

import numpy as np
import pandas as pd

from scripts.run_unified_gate_pilot import blend, station_bootstrap_gain


def test_log_space_blend_recovers_both_experts_at_endpoints():
    frame = pd.DataFrame({"rf_context": [1.0, 4.0], "residual_nomsg": [4.0, 1.0]})
    np.testing.assert_allclose(blend(frame, np.zeros(2)), frame.rf_context)
    np.testing.assert_allclose(blend(frame, np.ones(2)), frame.residual_nomsg)


def test_station_bootstrap_is_finite_and_tracks_positive_gain():
    y = np.zeros(6)
    candidate = np.zeros(6)
    baseline = np.ones(6)
    gain, low, high = station_bootstrap_gain(
        y, candidate, baseline, np.array(["a", "a", "b", "b", "c", "c"]), seed=42
    )
    assert gain == 1.0
    assert low <= gain <= high
