"""RiverDOCNet with explicit lag features and a local-to-readout path."""
from __future__ import annotations

import torch
from torch import nn

from river_graph.models.doc_dynamic_reconstruction import DynamicRiverDOC
from river_graph.models.river_relation_comparison import attention_allowed


class HybridLagRiverDOC(DynamicRiverDOC):
    """Keep environment, GRU and directional attention; enrich the time/readout.

    The lag branch reads the same eight monthly channels, not the zero-padded
    daily channels or any DOC labels. Separate local/graph normalization keeps
    the direct local route independent of graph messages when their readout
    weights are zero. All branches are trained together with the original loss.
    """

    def __init__(self, *, lag_hidden=24, lag_output=12, lookback=12, **kwargs):
        super().__init__(**kwargs)
        hidden = self.combine.out_features
        temporal_hidden = self.temporal.hidden_size
        self.lookback = lookback
        self.lag_output = lag_output
        self.lag_encoder = nn.Sequential(
            nn.Linear(lookback * 8, lag_hidden), nn.GELU(),
            nn.Linear(lag_hidden, lag_output),
        )
        self.lag_to_state = nn.Linear(lag_output, hidden, bias=False)
        self.local_width = hidden + temporal_hidden + lag_output
        self.local_norm = nn.LayerNorm(self.local_width)
        self.graph_norm = nn.LayerNorm(hidden)
        self.readout = nn.Sequential(
            nn.Linear(self.local_width + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def encode_local(self, windows, environment, season):
        b, n, length, channels = windows.shape
        if length != self.lookback or channels != 16:
            raise ValueError("Expected the frozen twelve-month, sixteen-channel tensor")
        if windows[..., 8:].count_nonzero():
            raise ValueError("Hybrid lag study permits original monthly covariates only")
        _, state = self.temporal(windows.reshape(b * n, length, channels))
        temporal = state[-1].reshape(b, n, -1)
        lag = self.lag_encoder(windows[..., :8].reshape(b, n, length * 8))
        env = self.environment(environment).unsqueeze(0).expand(b, -1, -1)
        local = torch.cat([env, temporal, lag], dim=-1)
        hidden = self.combine(torch.cat([
            env, temporal, season[:, None].expand(-1, n, -1),
        ], dim=-1)) + self.lag_to_state(lag)
        return local, hidden

    def river_context(self, hidden, graph):
        b, n, _ = hidden.shape
        for block in self.blocks:
            query, key, value = block.qkv(block.norm(hidden)).reshape(
                b, n, 3, block.heads, block.width,
            ).unbind(2)
            logits = torch.einsum("bihd,bjhd->bhij", query, key) / block.width**.5
            bias = block.relation_bias(graph["relation"]).permute(2, 0, 1)
            allowed = attention_allowed(graph, self.recipe, block.heads)
            weights = (logits + bias[None]).masked_fill(~allowed[None], -torch.inf).softmax(-1)
            message = torch.einsum("bhij,bjhd->bihd", weights, value)
            hidden = hidden + block.drop(block.output(message.reshape(b, n, block.hidden)))
            hidden = hidden + block.drop(block.ff(block.ff_norm(hidden)))
        return hidden

    def forward(self, windows, environment, season, graph):
        local, hidden = self.encode_local(windows, environment, season)
        spatial = self.river_context(hidden, graph)
        return self.readout(torch.cat([
            self.local_norm(local), self.graph_norm(spatial),
        ], dim=-1)).squeeze(-1)
