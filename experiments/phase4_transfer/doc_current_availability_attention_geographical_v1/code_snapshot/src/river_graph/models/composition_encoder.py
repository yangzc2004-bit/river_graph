"""Extend the retained ecological encoder with zero-initialized input columns."""
from __future__ import annotations

import copy

import torch

from river_graph.models.hydro import TransportGCNImputer


def expand_ecological_encoder(spatial, architecture, *, added_dim=22):
    """Keep old layers and states while opening columns for physical ecology.

    The appended columns start at zero; existing ecological/spatial parameters
    are copied without modification. The ordinary TransportGCNImputer and
    EncoderNativeResidual save/load paths remain sufficient for the model.
    """
    if type(spatial) is not TransportGCNImputer or spatial.env_encoder is None:
        raise ValueError("a retained transport encoder with ecology is required")
    if not isinstance(added_dim, int) or isinstance(added_dim, bool) or added_dim < 1:
        raise ValueError("added_dim must be a positive integer")
    config = copy.deepcopy(architecture)
    old_dim = spatial.env_encoder[0].in_features
    if config["env_dim"] != old_dim:
        raise ValueError("ecological architecture and supplied weights disagree")
    config["env_dim"] += added_dim
    parameter = next(spatial.parameters())
    with torch.random.fork_rng():
        expanded = TransportGCNImputer(**config).to(dtype=parameter.dtype, device=parameter.device)
    state = copy.deepcopy(spatial.state_dict())
    weight = state["env_encoder.0.weight"]
    state["env_encoder.0.weight"] = torch.cat([weight, weight.new_zeros((len(weight), added_dim))], dim=1)
    expanded.load_state_dict(state)
    expanded.train(spatial.training)
    return expanded
