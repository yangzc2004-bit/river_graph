"""Spatial controls and portable preprocessing for the same MLP/GRU model."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from river_graph.models.river_relation_comparison import (
    RiverRelationModel,
    relation_graph,
)

ARMS = ("direction_heads", "generic_dense", "self_only", "permuted_river")


def controlled_graph(edges, order, rows, arm, seed=9127):
    if arm not in ARMS:
        raise ValueError("unknown unified spatial control")
    graph = relation_graph(edges, order, rows, "direction_heads")
    original = graph["relation"].clone()
    n = len(rows)
    if arm == "generic_dense":
        graph["relation"] = torch.full_like(original, 5)
        graph["relation"].fill_diagonal_(0)
        allowed = torch.ones((3, n, n), dtype=torch.bool)
    elif arm == "self_only":
        allowed = torch.eye(n, dtype=torch.bool)[None].expand(3, -1, -1)
    else:
        if arm == "permuted_river":
            # Relabel the entire permitted-node topology, preserving its exact
            # category histogram and degree distribution. This is a topology
            # permutation control, NOT a guaranteed non-connected donor control.
            code = hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).digest()
            rng = np.random.default_rng(seed + int.from_bytes(code[:4], "little"))
            connected = (original > 0) & (original < 5)
            best, overlap = None, n*n+1
            for _ in range(100):
                permutation = rng.permutation(n)
                candidate = original[permutation][:, permutation]
                score = int(((candidate > 0) & (candidate < 5) & connected).sum())
                if score < overlap:
                    best, overlap = candidate, score
            graph["relation"] = best
            graph["permutation_overlap_pairs"] = overlap
            graph["permutation_connected_pairs"] = int(connected.sum())
        relation = graph["relation"]
        allowed = torch.stack([(relation <= 2), ((relation == 0) | (relation == 3) | (relation == 4)),
                               torch.ones_like(relation, dtype=torch.bool)])
    graph["allowed"] = allowed
    return graph


class UnifiedRiverDOC(RiverRelationModel):
    """Identical tensor names, widths and initialization to direction_heads."""

    def __init__(self, arm="direction_heads", **kwargs):
        if arm not in ARMS:
            raise ValueError("unknown unified spatial control")
        super().__init__("direction_heads", **kwargs)
        self.spatial_control = arm

    def forward(self, windows, environment, season, graph):
        b, n, length, channels = windows.shape
        _, state = self.temporal(windows.reshape(b*n, length, channels))
        temporal = state[-1].reshape(b, n, -1)
        env = self.environment(environment).unsqueeze(0).expand(b, -1, -1)
        hidden = self.combine(torch.cat([env, temporal, season[:, None].expand(-1, n, -1)], -1))
        for block in self.blocks:
            query, key, value = block.qkv(block.norm(hidden)).reshape(
                b, n, 3, block.heads, block.width).unbind(2)
            logits = torch.einsum("bihd,bjhd->bhij", query, key) / block.width**.5
            bias = block.relation_bias(graph["relation"]).permute(2, 0, 1)
            logits = (logits + bias.unsqueeze(0)).masked_fill(~graph["allowed"][None], -torch.inf)
            weights = logits.softmax(-1)
            message = torch.einsum("bhij,bjhd->bihd", weights, value)
            hidden = hidden + block.drop(block.output(message.reshape(b, n, block.hidden)))
            hidden = hidden + block.drop(block.ff(block.ff_norm(hidden)))
        return self.readout(hidden).squeeze(-1)


def portable_inputs(dataset, normalization):
    """Apply source statistics without accessing any receiving-river DOC."""
    if "y" in dataset or "y_mask" in dataset:
        raise ValueError("portable input construction requires a label-free archive")
    if list(dataset["feature_channels"]) != ["temperature", "discharge"]:
        raise ValueError("unexpected covariate schema")
    lookback = normalization["lookback"]
    x = np.asarray(dataset["x"], dtype=float).copy()
    mask = np.asarray(dataset["x_mask"], dtype=bool) & np.isfinite(x)
    mask[..., 1] &= x[..., 1] >= 0
    x[..., 1] = np.log1p(np.maximum(x[..., 1], 0))
    values = np.where(mask, np.clip((x-np.asarray(normalization["hydro_mean"])) /
                      np.asarray(normalization["hydro_std"]), -8, 8), 0.)
    n, t, _ = x.shape
    ages = np.empty_like(values)
    age = np.full((n, 2), lookback, dtype=float)
    for month in range(t):
        age = np.where(mask[:, month], 0., np.minimum(age+1., lookback))
        ages[:, month] = age/lookback
    numbers = np.asarray([int(date[5:7]) for date in np.asarray(dataset["months"], str)])
    season = np.column_stack([np.sin(2*np.pi*numbers/12), np.cos(2*np.pi*numbers/12)]).astype(np.float32)
    sequence = np.concatenate([values, mask.astype(float), ages, np.broadcast_to(season, (n, t, 2))], -1).astype(np.float32)
    padded = np.pad(sequence, ((0, 0), (lookback-1, 0), (0, 0)))
    windows = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=1).transpose(1, 0, 3, 2))
    raw = np.column_stack([np.asarray(dataset["regime"], float), np.asarray(dataset["static"], float)])
    median = np.asarray(normalization["environment"]["median"])
    scale = np.asarray(normalization["environment"]["iqr"])
    env = np.clip((np.where(np.isfinite(raw), raw, median)-median)/scale, -8, 8).astype(np.float32)
    order = np.asarray(dataset["regime"])[:, 0]
    if not np.all(np.isfinite(order) & (order > 0) & (order == np.round(order))):
        raise ValueError("positive integer stream orders required; do not invent receiving orders")
    return {"windows": torch.from_numpy(windows), "environment": torch.from_numpy(env),
            "season": torch.from_numpy(season), "order": torch.tensor(order, dtype=torch.long),
            "normalization": normalization}
