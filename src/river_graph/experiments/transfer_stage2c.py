"""Pure contracts and features for the cross-basin Stage-2C controls."""

from __future__ import annotations

import numpy as np

from river_graph.experiments.transfer import array

ARM_CONFIGS = {
    "ecorf": {
        "kind": "ecorf",
        "architecture": None,
        "edge_set": None,
        "edge_direction": None,
        "env_groups": None,
        "env_encoder": False,
    },
    "h2x_full": {
        "kind": "gcn",
        "architecture": "transport_enc",
        "edge_set": "river",
        "edge_direction": "both",
        "env_groups": None,
        "env_encoder": True,
    },
    "h2x_no_graph": {
        "kind": "gcn",
        "architecture": "transport_enc",
        "edge_set": "empty",
        "edge_direction": "both",
        "env_groups": None,
        "env_encoder": True,
    },
    "h2_no_ecology": {
        "kind": "gcn",
        "architecture": "transport",
        "edge_set": "river",
        "edge_direction": "both",
        "env_groups": ["hydro"],
        "env_encoder": False,
    },
}
ARMS = tuple(ARM_CONFIGS)
GNN_TRAINING = {
    "lr": 1e-3,
    "weight_decay": 0.0,
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "max_epochs": 200,
    "patience": 20,
}
ECORF_TRAINING = {"n_estimators": 200, "random_state": "unit_seed", "n_jobs": 1}


def validate_arm_configs() -> None:
    """Fail closed if one of the matched control definitions drifts."""
    expected = {"ecorf", "h2x_full", "h2x_no_graph", "h2_no_ecology"}
    if set(ARM_CONFIGS) != expected:
        raise ValueError("Stage-2C arm set changed")
    if ARM_CONFIGS["h2x_full"]["architecture"] != "transport_enc":
        raise ValueError("H2X full must use transport_enc")
    if ARM_CONFIGS["h2x_no_graph"]["edge_set"] != "empty":
        raise ValueError("no-graph arm must use an empty edge set")
    if ARM_CONFIGS["h2_no_ecology"]["architecture"] != "transport":
        raise ValueError("no-ecology arm must use transport without encoder")
    if ARM_CONFIGS["h2_no_ecology"]["env_groups"] != ["hydro"]:
        raise ValueError("no-ecology arm must retain only hydro inputs")


def ecological_time_features(dataset: dict) -> np.ndarray:
    """Return label-free EcoRF hydro/time/static/regime features."""
    x = array(dataset["x"]).astype(float)
    x_mask = array(dataset["x_mask"]).astype(float)
    static = array(dataset["static"]).astype(float)
    regime = array(dataset["regime"]).astype(float)
    n, t, channels = x.shape
    if channels != 2 or x_mask.shape != x.shape:
        raise ValueError("Stage-2C EcoRF expects two hydro channels and masks")
    if static.shape[0] != n or regime.shape[0] != n:
        raise ValueError("static/regime station dimension mismatch")
    month_num = np.asarray([int(str(value)[5:7]) for value in dataset["months"]], dtype=float)
    if month_num.shape != (t,) or np.any((month_num < 1) | (month_num > 12)):
        raise ValueError("invalid dataset month values")
    angle = 2.0 * np.pi * (month_num - 1.0) / 12.0
    season = np.column_stack([np.sin(angle), np.cos(angle)])
    return np.column_stack(
        [
            x.reshape(n * t, 2),
            x_mask.reshape(n * t, 2),
            np.tile(season, (n, 1)),
            np.repeat(static, t, axis=0),
            np.repeat(regime, t, axis=0),
        ]
    )
