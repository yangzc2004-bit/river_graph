"""Historical H2X trunk with the current label-free DOC feature contract."""
from __future__ import annotations

import torch
from torch import nn

from river_graph.models.hydro import TransportGCNImputer


class MatchedInputH2X(nn.Module):
    """Retain the released ecological encoder and two transport GCN layers.

    Historical DOC context and NHD edge attributes are unavailable under the
    current input contract. Flatten the same monthly driver windows rather
    than introducing a GRU. Static gates read the two endpoints' existing
    environmental features, with no new measurements. Edges are the original
    direct upstream connections in the common role-specific graph.
    """

    def __init__(self, *, env_dim=15, lookback=12, hidden=64, env_emb=32,
                 layers=2, dropout=.1):
        super().__init__()
        self.env_dim, self.lookback = env_dim, lookback
        self.trunk = TransportGCNImputer(
            in_channels=lookback * 8 + 2, edge_dim=env_dim * 2,
            hidden=hidden, env_dim=env_dim, env_emb=env_emb,
            layers=layers, dropout=dropout, edge_direction="both",
        )

    @staticmethod
    def directed_edges(graph):
        receiver, sender = torch.where(graph["relation"] == 1)
        return torch.stack([sender, receiver])

    def forward(self, windows, environment, season, graph):
        b, n, length, channels = windows.shape
        if length != self.lookback or channels != 16:
            raise ValueError("Expected the frozen twelve-month monthly tensor")
        if windows[..., 8:].count_nonzero():
            raise ValueError("Matched H2X permits original monthly covariates only")
        if environment.shape != (n, self.env_dim):
            raise ValueError("Aligned common environmental channels required")
        history = windows[..., :8].reshape(b, n, length * 8)
        x = torch.cat([history, season[:, None].expand(-1, n, -1)], -1)
        edges = self.directed_edges(graph)
        attributes = torch.cat([environment[edges[0]], environment[edges[1]]], -1)
        # Disjoint graph copies produce the same node-local computation for
        # each month, while using the unmodified released transport module.
        offsets = torch.arange(b, device=edges.device) * n
        batch_edges = (edges[None] + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)
        result = self.trunk(
            x.reshape(b * n, -1), batch_edges, attributes.repeat(b, 1),
            environment.repeat(b, 1),
        )
        return result.reshape(b, n)
