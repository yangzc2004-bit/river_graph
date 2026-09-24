"""Contracts for Stage-2C arm definitions and label-free EcoRF features."""

from __future__ import annotations

import numpy as np

from river_graph.experiments.transfer_stage2c import (
    ARM_CONFIGS,
    ARMS,
    ecological_time_features,
    validate_arm_configs,
)


def _toy_dataset():
    n, t = 3, 4
    return {
        "months": ["2000-01", "2000-02", "2000-03", "2000-04"],
        "x": np.arange(n * t * 2, dtype=float).reshape(n, t, 2),
        "x_mask": np.ones((n, t, 2), dtype=bool),
        "static": np.arange(n * 2, dtype=float).reshape(n, 2),
        "regime": np.arange(n * 13, dtype=float).reshape(n, 13),
        "y": np.arange(n * t, dtype=float).reshape(n, t),
    }


def test_stage2c_arm_contract_is_fixed():
    validate_arm_configs()
    assert ARMS == ("ecorf", "h2x_full", "h2x_no_graph", "h2_no_ecology")
    assert ARM_CONFIGS["h2x_full"]["edge_set"] == "river"
    assert ARM_CONFIGS["h2x_no_graph"]["edge_set"] == "empty"
    assert ARM_CONFIGS["h2_no_ecology"]["env_groups"] == ["hydro"]


def test_ecorf_features_have_no_label_channel():
    dataset = _toy_dataset()
    first = ecological_time_features(dataset)
    changed = {**dataset, "y": dataset["y"] + 1_000_000}
    second = ecological_time_features(changed)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (12, 21)

