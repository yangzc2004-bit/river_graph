"""Source-only transfer training for the existing environmental/GRU/river model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from river_graph.models.unified_river_doc import UnifiedRiverDOC

RECIPES = ("reference", "episode_cells", "episode_balanced", "balanced_no_coords",
           "balanced_hydro_dropout", "balanced_gated", "combined")
GATED = ("balanced_gated", "combined")
NO_COORDS = ("balanced_no_coords", "combined")
HYDRO_DROPOUT = ("balanced_hydro_dropout", "combined")


def huc_labels(nodes, sites):
    """Canonical HUCs come from huc_cd, including CSV integer/float strings."""
    if nodes.site_no.duplicated().any():
        raise ValueError("duplicate station metadata")
    aligned = nodes.set_index("site_no").loc[np.asarray(sites, str)]
    codes = pd.to_numeric(aligned.huc_cd, errors="coerce").to_numpy(dtype=float)
    labels = []
    for code in codes:
        if not (np.isfinite(code) and 0 < code < 1_000_000_000_000 and code == int(code)):
            labels.append("unknown")
            continue
        digits = str(int(code))
        # huc_cd mixes HUC8 and HUC12 in this inventory; numeric CSV saving
        # stripped a leading zero in both. HUC10 is also a valid parent level.
        width = 8 if len(digits) <= 8 else 10 if len(digits) <= 10 else 12
        labels.append(digits.zfill(width)[:8])
    return {"huc8": np.asarray(labels), "huc4": np.asarray([s[:4] if s != "unknown" else s for s in labels]),
            "huc2": np.asarray([s[:2] if s != "unknown" else s for s in labels])}


def coordinate_policy(inputs, recipe):
    if recipe not in RECIPES:
        raise ValueError("unknown transfer recipe")
    result = dict(inputs)
    if recipe in NO_COORDS:
        result["environment"] = inputs["environment"].clone()
        result["environment"][:, 13:15] = 0
    return result


def sensor_dropout(windows, generator, temperature=.1, discharge=.5):
    """Drop whole station/channel histories per month panel; recompute ages.

    Values/masks are zeroed, ages become fully stale on real calendar tokens.
    Left-padding tokens remain zero. Seasons, targets and other channels stay.
    """
    result = windows.clone()
    dropped = torch.rand((1, windows.shape[1], 2), generator=generator) < torch.tensor([temperature, discharge])
    real_calendar = (windows[..., 6:8].square().sum(-1) > 0).to(windows.dtype)
    for channel in range(2):
        selected = dropped[:, :, channel, None]
        result[..., channel] = torch.where(selected, 0., windows[..., channel])
        result[..., channel+2] = torch.where(selected, 0., windows[..., channel+2])
        result[..., channel+4] = torch.where(selected, real_calendar, windows[..., channel+4])
    return result


def basin_weights(counts, balanced):
    values = np.asarray(counts, float)
    if np.any(values <= 0):
        raise ValueError("each source basin needs observed training DOC")
    return values.sum()/(len(values)*values) if balanced else np.ones_like(values)


class TransferRobustDOC(UnifiedRiverDOC):
    """Same MLP/GRU/three river heads; optional six reliability-gate scalars."""

    def __init__(self, recipe="reference", **kwargs):
        if recipe not in RECIPES:
            raise ValueError("unknown transfer recipe")
        super().__init__("direction_heads", **kwargs)
        self.transfer_recipe = recipe
        if recipe in GATED:
            self.cross_gate_logits = nn.Parameter(torch.full((len(self.blocks), 3), -2.))

    def gate_penalty(self):
        if self.transfer_recipe in GATED:
            return .01*self.cross_gate_logits.sigmoid().mean()
        return next(self.parameters()).new_zeros(())

    def forward(self, windows, environment, season, graph):
        if self.transfer_recipe not in GATED:
            return super().forward(windows, environment, season, graph)
        b, n, length, channels = windows.shape
        _, state = self.temporal(windows.reshape(b*n, length, channels))
        temporal = state[-1].reshape(b, n, -1)
        env = self.environment(environment).unsqueeze(0).expand(b, -1, -1)
        hidden = self.combine(torch.cat([env, temporal, season[:, None].expand(-1, n, -1)], -1))
        # Reliability is based on the two available hydro masks over the causal
        # window; never on DOC, receiving errors, or receiving statistics.
        reliability = windows[..., 2:4].mean(dim=(-1, -2))
        for layer, block in enumerate(self.blocks):
            query, key, value = block.qkv(block.norm(hidden)).reshape(b, n, 3, block.heads, block.width).unbind(2)
            logits = torch.einsum("bihd,bjhd->bhij", query, key)/block.width**.5
            bias = block.relation_bias(graph["relation"]).permute(2, 0, 1)
            weights = (logits+bias[None]).masked_fill(~graph["allowed"][None], -torch.inf).softmax(-1)
            attended = torch.einsum("bhij,bjhd->bihd", weights, value)
            evidence = torch.einsum("bhij,bj->bih", weights, reliability)
            gate = self.cross_gate_logits[layer].sigmoid()[None, None, :]*(.2+.8*evidence)
            message = value+gate[..., None]*(attended-value)
            hidden = hidden+block.drop(block.output(message.reshape(b, n, block.hidden)))
            hidden = hidden+block.drop(block.ff(block.ff_norm(hidden)))
        return self.readout(hidden).squeeze(-1)
