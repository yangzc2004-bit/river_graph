"""Equal-capacity directed river attention with label-free full-path context."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    graph_view,
)

ARMS = ("fixed_fusion", "generic_dense", "path_closure", "upstream_only", "direction_heads", "path_bias")
CANDIDATES = ARMS[2:]
DEFINITIONS = {
    "fixed_fusion": "Original induced-graph six-category dense graph Transformer",
    "generic_dense": "Original induced graph; generic self/other dense attention, no river relation categories",
    "path_closure": "Dense six-category attention; retain paths through omitted stations using known full topology only",
    "upstream_only": "Full-path closure; every attention head reads self and upstream ancestors only",
    "direction_heads": "Full-path closure; three heads specialize to upstream+self, downstream+self, and global context",
    "path_bias": "Full-path closure; six equal-capacity bias coefficients encode self/upstream/downstream/unrelated and directional log station-hop distance",
}


def station_hop_distances(edge_index, size):
    """Shortest directed station-edge counts, [receiver, upstream sender]."""
    children = [[] for _ in range(size)]
    for a, b in np.asarray(edge_index).T:
        if not (0 <= a < size and 0 <= b < size) or a == b:
            raise ValueError("valid non-self directed station edges required")
        if b not in children[a]:
            children[a].append(int(b))
    distances = np.full((size, size), -1, dtype=np.int64)
    for source in range(size):
        distances[source, source] = 0
        ready = deque([source])
        while ready:
            a = ready.popleft()
            for b in children[a]:
                if distances[b, source] < 0:
                    distances[b, source] = distances[a, source] + 1
                    ready.append(b)
    return distances


def relation_graph(edge_index, order, rows, arm):
    """Only allowed-role nodes exchange features; omitted nodes supply topology."""
    if arm not in ARMS:
        raise ValueError("unknown river relation recipe")
    result = graph_view(edge_index, order, rows)
    if arm == "fixed_fusion":
        return result
    if arm == "generic_dense":
        result["relation"] = torch.full_like(result["relation"], 5)
        result["relation"].fill_diagonal_(0)
        return result
    size = len(order)
    full = graph_view(edge_index, order, np.arange(size))
    ids = np.asarray(rows, dtype=int)
    result["relation"] = full["relation"][ids][:, ids].clone()
    hops = station_hop_distances(edge_index, size)[np.ix_(ids, ids)]
    result["station_hops"] = torch.from_numpy(hops)
    relation = result["relation"]
    upstream = (relation == 1) | (relation == 2)
    downstream = (relation == 3) | (relation == 4)
    self_mask = relation == 0
    result["upstream_allowed"] = upstream | self_mask
    result["downstream_allowed"] = downstream | self_mask
    log_hops = np.log1p(np.maximum(hops, 0)) / np.log(2.)
    log_hops = torch.from_numpy(np.minimum(log_hops, 8).astype(np.float32))
    result["path_basis"] = torch.stack([
        self_mask.float(), upstream.float(), downstream.float(), (relation == 5).float(),
        upstream.float()*log_hops, downstream.float()*log_hops.T,
    ], dim=-1)
    result["restored_ancestor_pairs"] = int(((full["relation"][ids][:, ids] == 2) &
        (graph_view(edge_index, order, rows)["relation"] == 5)).sum())
    return result


def attention_allowed(graph, arm, heads=3):
    if arm == "upstream_only":
        return graph["upstream_allowed"].unsqueeze(0).expand(heads, -1, -1)
    if arm == "direction_heads":
        if heads != 3:
            raise ValueError("direction specialization requires three heads")
        return torch.stack([graph["upstream_allowed"], graph["downstream_allowed"],
                            torch.ones_like(graph["upstream_allowed"])])
    return torch.ones((heads, *graph["relation"].shape), dtype=torch.bool)


class RiverRelationModel(RiverArchitectureModel):
    """Same tensors and seeded initialization; only masks/bias bases change."""

    def __init__(self, arm, **kwargs):
        if arm not in ARMS:
            raise ValueError("unknown river relation recipe")
        super().__init__("graph_transformer", **kwargs)
        self.recipe = arm
        if arm == "direction_heads" and self.blocks[0].heads != 3:
            raise ValueError("direction specialization requires exactly three heads")

    def forward(self, windows, environment, season, graph):
        if self.recipe in ("fixed_fusion", "generic_dense", "path_closure"):
            return super().forward(windows, environment, season, graph)
        b, n, length, channels = windows.shape
        _, state = self.temporal(windows.reshape(b*n, length, channels))
        temporal = state[-1].reshape(b, n, -1)
        env = self.environment(environment).unsqueeze(0).expand(b, -1, -1)
        hidden = self.combine(torch.cat([env, temporal, season[:, None].expand(-1, n, -1)], -1))
        for block in self.blocks:
            query, key, value = block.qkv(block.norm(hidden)).reshape(
                b, n, 3, block.heads, block.width).unbind(2)
            logits = torch.einsum("bihd,bjhd->bhij", query, key) / block.width**.5
            if self.recipe == "path_bias":
                bias = torch.einsum("ijc,ch->hij", graph["path_basis"], block.relation_bias.weight)
            else:
                bias = block.relation_bias(graph["relation"]).permute(2, 0, 1)
            logits = logits + bias.unsqueeze(0)
            if self.recipe in ("upstream_only", "direction_heads"):
                allowed = attention_allowed(graph, self.recipe, block.heads)
                logits = logits.masked_fill(~allowed[None], -torch.inf)
            weights = logits.softmax(-1)
            message = torch.einsum("bhij,bjhd->bihd", weights, value)
            hidden = hidden + block.drop(block.output(message.reshape(b, n, block.hidden)))
            hidden = hidden + block.drop(block.ff(block.ff_norm(hidden)))
        return self.readout(hidden).squeeze(-1)
