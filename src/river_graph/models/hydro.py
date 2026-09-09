"""Directed relational GCN encoder (HydroGRN H1, design_m4.md).

A plain GCN symmetrizes the river graph and loses flow direction. H1 keeps
two separate message channels per layer:

- upstream relation  (j -> i, j upstream of i): the *transport signal* —
  what the river carries into the station.
- downstream relation (j -> i, j downstream of i): the *contextual
  constraint* — a high DOC reading downstream bounds what the station
  could plausibly have had.

Everything else (inputs, loss, masks, evaluation) is identical to the GCN
baseline so H1 vs G0 isolates direction alone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


def make_directed_edges(variant: str, edge_index: torch.Tensor, n_nodes: int,
                        seed: int = 42) -> torch.Tensor:
    """Directed edge sets for H1. Unlike make_edge_index, the river graph is
    NOT symmetrized — direction is the whole point."""
    if variant == "river":
        return edge_index
    if variant == "random":
        rng = np.random.default_rng(seed)
        n_edges = edge_index.shape[1]
        pairs = rng.choice(n_nodes * (n_nodes - 1), size=n_edges, replace=False)
        src, dst = pairs // (n_nodes - 1), pairs % (n_nodes - 1)
        dst = np.where(dst >= src, dst + 1, dst)
        return torch.tensor(np.stack([src, dst]), dtype=torch.long)
    if variant == "none":
        return torch.empty((2, 0), dtype=torch.long)
    raise ValueError(f"unknown variant: {variant}")


class DirectedConv(nn.Module):
    """One layer: self transform + upstream relation + downstream relation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.self_lin = nn.Linear(in_channels, out_channels)
        self.up_conv = GCNConv(in_channels, out_channels, add_self_loops=False)
        self.down_conv = GCNConv(in_channels, out_channels, add_self_loops=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # river edges point upstream -> downstream
        ei_up = edge_index  # messages arrive from upstream neighbors
        ei_down = edge_index.flip(0)  # messages arrive from downstream neighbors
        return self.self_lin(x) + self.up_conv(x, ei_up) + self.down_conv(x, ei_down)


class DirectedGCNImputer(nn.Module):
    """Same interface as GCNImputer; forward takes the raw directed edges."""

    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_channels, *([hidden] * layers)]
        import itertools

        for a, b in itertools.pairwise(dims):
            self.convs.append(DirectedConv(a, b))
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)
