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
    """One layer: self transform + upstream relation + downstream relation.

    share_weights=True (H1.5): a single relation conv is shared across
    directions and a learnable direction embedding is added to each
    relation's aggregated message. Keeps the direction distinction while
    roughly halving relation parameters — less room to memorize regional
    neighborhood patterns, which is what hurt spatial transfer (E3).

    edge_dropout: during training, each edge is dropped independently with
    this probability per relation per forward pass, forcing the model not
    to rely on memorized reaches.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 share_weights: bool = False):
        super().__init__()
        self.self_lin = nn.Linear(in_channels, out_channels)
        self.share_weights = share_weights
        if share_weights:
            self.rel_conv = GCNConv(in_channels, out_channels, add_self_loops=False)
            # per-relation multiplicative gates: shared conv keeps the
            # parameter count near a single relation, while the gates keep
            # upstream/downstream distinguishable (an additive bias would
            # cancel out — summing both relations is swap-invariant)
            self.dir_gate = nn.Parameter(
                torch.stack([torch.ones(out_channels),
                             torch.full((out_channels,), 0.5)])
            )
        else:
            self.up_conv = GCNConv(in_channels, out_channels, add_self_loops=False)
            self.down_conv = GCNConv(in_channels, out_channels, add_self_loops=False)

    @staticmethod
    def _edge_dropout(ei: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if not training or p <= 0 or ei.numel() == 0:
            return ei
        keep = torch.rand(ei.shape[1], device=ei.device) >= p
        return ei[:, keep]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_drop_p: float = 0.0) -> torch.Tensor:
        # river edges point upstream -> downstream
        ei_up = self._edge_dropout(edge_index, edge_drop_p, self.training)
        ei_down = self._edge_dropout(edge_index.flip(0), edge_drop_p, self.training)
        if self.share_weights:
            up = self.rel_conv(x, ei_up) * self.dir_gate[0]
            down = self.rel_conv(x, ei_down) * self.dir_gate[1]
        else:
            up = self.up_conv(x, ei_up)
            down = self.down_conv(x, ei_down)
        return self.self_lin(x) + up + down


class GatedDirectedConv(nn.Module):
    """H2: directed conv with physics-informed transport gates.

    Message from j to i is  gate(e_ji) * W h_j , where the gate is a
    sigmoid over the edge's physical attributes (hop distance, reach
    length, drainage area, slope, stream order). Propagation strength is
    set by the river, not uniform averaging. Upstream and downstream
    relations keep separate weights and separate gates.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_channels, out_channels)
        self.up_lin = nn.Linear(in_channels, out_channels)
        self.down_lin = nn.Linear(in_channels, out_channels)
        self.gate_up = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(),
                                     nn.Linear(16, 1))
        self.gate_down = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(),
                                       nn.Linear(16, 1))

    @staticmethod
    def _agg(msg: torch.Tensor, ei: torch.Tensor, gate: torch.Tensor,
             n: int) -> torch.Tensor:
        """Gated mean aggregation: sum(gate * msg_src) / degree at dst."""
        if ei.numel() == 0:
            return torch.zeros(n, msg.shape[1], device=msg.device)
        src, dst = ei
        weighted = msg[src] * gate  # (E, out)
        out = torch.zeros(n, msg.shape[1], device=msg.device).index_add_(0, dst, weighted)
        deg = torch.zeros(n, 1, device=msg.device).index_add_(
            0, dst, gate.new_ones(len(src), 1))
        return out / deg.clamp(min=1.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        ei_up = edge_index              # src upstream -> dst i
        ei_down = edge_index.flip(0)    # src downstream -> dst i
        g_up = torch.sigmoid(self.gate_up(edge_attr))
        g_down = torch.sigmoid(self.gate_down(edge_attr))
        return (
            self.self_lin(x)
            + self._agg(self.up_lin(x), ei_up, g_up, n)
            + self._agg(self.down_lin(x), ei_down, g_down, n)
        )


class TransportGCNImputer(nn.Module):
    """Directed GCN with transport gates; forward(x, edge_index, edge_attr)."""

    def __init__(self, in_channels: int, edge_dim: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        import itertools

        self.convs = nn.ModuleList()
        dims = [in_channels, *([hidden] * layers)]
        for a, b in itertools.pairwise(dims):
            self.convs.append(GatedDirectedConv(a, b, edge_dim))
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


class DirectedGCNImputer(nn.Module):
    """Same interface as GCNImputer; forward takes the raw directed edges."""

    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, share_weights: bool = False,
                 edge_dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_channels, *([hidden] * layers)]
        import itertools

        for a, b in itertools.pairwise(dims):
            self.convs.append(DirectedConv(a, b, share_weights=share_weights))
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_drop_p=self.edge_dropout))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)
