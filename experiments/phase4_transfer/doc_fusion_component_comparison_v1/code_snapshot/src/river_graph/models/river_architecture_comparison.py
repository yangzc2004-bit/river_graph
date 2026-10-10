"""Matched, inductive spatial operators for a standalone DOC comparison.

All arms have the same environmental encoder, rolling hydrological GRU,
attention projections, feed-forward blocks, and prediction head. Only the
spatial attention neighbourhood and evaluation schedule change. No DOC value,
station identity embedding, or fitted station-specific parameter is an input.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ARMS = ("local", "directed_gnn", "hierarchical_gnn", "graph_transformer")


def robust_normalize(raw, source_rows):
    raw = np.asarray(raw, dtype=np.float64)
    source = raw[np.asarray(source_rows, dtype=int)]
    median = np.nanmedian(source, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    q = np.nanquantile(source, [.25, .75], axis=0)
    scale = np.maximum(np.where(np.isfinite(q[1] - q[0]), q[1] - q[0], 0.), .1)
    normalized = np.clip((np.where(np.isfinite(raw), raw, median) - median) / scale, -8, 8)
    return normalized.astype(np.float32), {"median": median.tolist(), "iqr": scale.tolist()}


def covariate_inputs(dataset, source_rows, *, lookback=12):
    """Fit scaling on source stations; assemble causal, label-free windows."""
    x = np.asarray(dataset["x"], dtype=float).copy()
    mask = np.asarray(dataset["x_mask"], dtype=bool) & np.isfinite(x)
    if x.ndim != 3 or x.shape[-1] != 2 or mask.shape != x.shape:
        raise ValueError("two hydrological channels and aligned masks are required")
    # The canonical channels are temperature and discharge, in this order.
    if list(dataset["feature_channels"]) != ["temperature", "discharge"]:
        raise ValueError("unexpected hydrological channel order")
    mask[..., 1] &= x[..., 1] >= 0
    x[..., 1] = np.log1p(np.maximum(x[..., 1], 0))
    x[~mask] = np.nan
    n, t, _ = x.shape
    source_rows = np.asarray(source_rows, dtype=int)
    means = np.nanmean(x[source_rows], axis=(0, 1))
    scales = np.nanstd(x[source_rows], axis=(0, 1))
    means = np.where(np.isfinite(means), means, 0.)
    scales = np.maximum(np.where(np.isfinite(scales), scales, 0.), .1)
    values = np.where(mask, np.clip((x - means) / scales, -8, 8), 0.)
    ages = np.empty_like(values)
    age = np.full((n, 2), lookback, dtype=float)
    for month in range(t):
        age = np.where(mask[:, month], 0., np.minimum(age + 1., lookback))
        ages[:, month] = age / lookback
    dates = np.asarray(dataset["months"], str)
    month_numbers = np.asarray([int(date[5:7]) for date in dates])
    season = np.column_stack([np.sin(2 * np.pi * month_numbers / 12),
                              np.cos(2 * np.pi * month_numbers / 12)]).astype(np.float32)
    sequence = np.concatenate([values, mask.astype(float), ages,
                               np.broadcast_to(season, (n, t, 2))], axis=-1).astype(np.float32)
    padded = np.pad(sequence, ((0, 0), (lookback - 1, 0), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=1)
    windows = np.ascontiguousarray(windows.transpose(1, 0, 3, 2))
    raw_environment = np.column_stack([np.asarray(dataset["regime"], float),
                                        np.asarray(dataset["static"], float)])
    environment, env_state = robust_normalize(raw_environment, source_rows)
    order = np.asarray(dataset["regime"])[:, 0]
    if not np.all(np.isfinite(order) & (order > 0) & (order == np.round(order))):
        raise ValueError("positive integer stream orders are required")
    return {"windows": torch.from_numpy(windows), "environment": torch.from_numpy(environment),
            "season": torch.from_numpy(season), "order": torch.tensor(order, dtype=torch.long),
            "normalization": {"environment": env_state, "hydro_mean": means.tolist(),
                              "hydro_std": scales.tolist(), "source_rows": source_rows.tolist(),
                              "discharge_transform": "log1p", "lookback": lookback}}


def graph_view(edge_index, order, rows):
    """Make an induced graph: no omitted-role node enters message passing.

    Relations are self, direct-upstream, other-ancestor, direct-downstream,
    other-descendant, and unconnected. They encode connectivity, not flux.
    """
    rows = np.asarray(rows, dtype=int)
    if len(np.unique(rows)) != len(rows) or not len(rows):
        raise ValueError("nonempty unique graph rows required")
    mapping = {int(row): i for i, row in enumerate(rows)}
    edges = [(mapping[int(a)], mapping[int(b)]) for a, b in np.asarray(edge_index).T
             if int(a) in mapping and int(b) in mapping]
    n = len(rows)
    parents, children, degree = [[i] for i in range(n)], [[] for _ in range(n)], np.zeros(n, int)
    adjacency = np.zeros((n, n), bool)
    for a, b in edges:
        if a == b or adjacency[b, a]:
            continue
        parents[b].append(a)
        children[a].append(b)
        degree[b] += 1
        adjacency[b, a] = True
    # Reachability is computed on the induced graph, including same-order chains.
    ready = list(np.flatnonzero(degree == 0))
    ancestors = np.zeros((n, n), bool)
    visited = 0
    while ready:
        a = ready.pop()
        visited += 1
        for b in children[a]:
            ancestors[b] |= ancestors[a]
            ancestors[b, a] = True
            degree[b] -= 1
            if degree[b] == 0:
                ready.append(b)
    if visited != n:
        raise ValueError("comparison requires an acyclic river graph")
    local_order = np.asarray(order)[rows]
    if any(local_order[a] > local_order[b] for a, b in edges):
        raise ValueError("order scan requires nondecreasing stream order along edges")
    relation = np.full((n, n), 5, dtype=np.int64)
    relation[ancestors] = 2
    relation[ancestors.T] = 4
    relation[adjacency] = 1
    relation[adjacency.T] = 3
    np.fill_diagonal(relation, 0)
    width = max(map(len, parents))
    neighbours = np.zeros((n, width), dtype=np.int64)
    valid = np.zeros((n, width), dtype=bool)
    for i, ids in enumerate(parents):
        neighbours[i, :len(ids)] = ids
        valid[i, :len(ids)] = True
    return {"rows": torch.tensor(rows), "neighbours": torch.tensor(neighbours),
            "valid": torch.tensor(valid), "relation": torch.tensor(relation),
            "groups": [torch.tensor(np.flatnonzero(local_order == rank))
                       for rank in sorted(np.unique(local_order))],
            "edge_count": len(edges)}


class SpatialBlock(nn.Module):
    """Same weights for sparse neighbour attention and dense graph attention."""

    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.hidden, self.heads, self.width = hidden, heads, hidden // heads
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.relation_bias = nn.Embedding(6, heads)
        nn.init.zeros_(self.relation_bias.weight)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden, graph, *, mode, query_rows=None):
        b, n, _ = hidden.shape
        rows = torch.arange(n, device=hidden.device) if query_rows is None else query_rows
        query, key, value = self.qkv(self.norm(hidden)).reshape(b, n, 3, self.heads, self.width).unbind(2)
        q = query[:, rows]
        if mode == "graph_transformer":
            logits = torch.einsum("bihd,bjhd->bhij", q, key) / self.width ** .5
            bias = self.relation_bias(graph["relation"][rows]).permute(2, 0, 1)
            weights = (logits + bias.unsqueeze(0)).softmax(-1)
            message = torch.einsum("bhij,bjhd->bihd", weights, value)
        else:
            if mode == "local":
                ids = rows[:, None]
                valid = torch.ones_like(ids, dtype=torch.bool)
            else:
                ids, valid = graph["neighbours"][rows], graph["valid"][rows]
            k, v = key[:, ids], value[:, ids]
            logits = (q.unsqueeze(2) * k).sum(-1) / self.width ** .5
            relation = graph["relation"][rows[:, None], ids]
            logits = logits + self.relation_bias(relation).unsqueeze(0)
            weights = logits.masked_fill(~valid[None, :, :, None], -torch.inf).softmax(2)
            message = (weights.unsqueeze(-1) * v).sum(2)
        result = hidden[:, rows] + self.drop(self.output(message.reshape(b, len(rows), self.hidden)))
        return result + self.drop(self.ff(self.ff_norm(result)))


class RiverArchitectureModel(nn.Module):
    """Identically initialized environmental/temporal model with four operators."""

    def __init__(self, arm, *, env_dim=15, hidden=24, temporal_hidden=12,
                 heads=3, layers=2, dropout=.1):
        super().__init__()
        if arm not in ARMS or hidden % heads:
            raise ValueError("valid arm and divisible attention width required")
        self.arm = arm
        self.environment = nn.Sequential(nn.Linear(env_dim, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden))
        self.temporal = nn.GRU(8, temporal_hidden, batch_first=True)
        self.combine = nn.Linear(hidden + temporal_hidden + 2, hidden)
        self.blocks = nn.ModuleList(SpatialBlock(hidden, heads, dropout) for _ in range(layers))
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, windows, environment, season, graph):
        b, n, length, channels = windows.shape
        _, state = self.temporal(windows.reshape(b * n, length, channels))
        temporal = state[-1].reshape(b, n, -1)
        env = self.environment(environment).unsqueeze(0).expand(b, -1, -1)
        hidden = self.combine(torch.cat([env, temporal, season[:, None].expand(-1, n, -1)], -1))
        for block in self.blocks:
            if self.arm == "hierarchical_gnn":
                # Shared operator, evaluated from low to high order. Lower-order
                # updated states can traverse cross-order skips in this pass.
                # Same-order paths remain local and span two hops over two passes.
                for rows in graph["groups"]:
                    update = block(hidden, graph, mode="directed_gnn", query_rows=rows)
                    hidden = hidden.index_copy(1, rows, update)
            else:
                hidden = block(hidden, graph, mode=self.arm)
        return self.readout(hidden).squeeze(-1)
