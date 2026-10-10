"""Controlled environmental, temporal and fusion alternatives for DOC.

The original graph-attention trunk is preserved. Alternatives replace one
information-processing component; the baseline reproduces its forward exactly.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from river_graph.models.river_architecture_comparison import RiverArchitectureModel

ENVIRONMENT_CHOICES = ("constant", "linear", "mlp", "residual_mlp")
TEMPORAL_CHOICES = ("current", "lag_mlp", "gru", "lstm", "transformer")
FUSION_CHOICES = ("concat", "residual_concat", "conditioned", "gated")


class ConstantEnvironment(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.value = nn.Parameter(initial.detach().clone())

    def forward(self, inputs):
        return self.value.expand(len(inputs), -1)


class ResidualEnvironment(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.layers = original

    def forward(self, inputs):
        hidden = self.layers[1](self.layers[0](inputs))
        return hidden + self.layers[2](hidden)


class TemporalAttention(nn.Module):
    """A small ordered causal encoder, returning the last window state."""

    def __init__(self, channels, hidden, heads, lookback, dropout):
        super().__init__()
        self.project = nn.Linear(channels, hidden)
        position = torch.arange(lookback).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.) / hidden))
        encoding = torch.zeros(lookback, hidden)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[:encoding[:, 1::2].shape[1]])
        self.register_buffer("position", encoding)
        self.register_buffer("causal_mask", torch.ones(lookback, lookback, dtype=torch.bool).triu(1))
        self.block = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout,
            activation="gelu", batch_first=True, norm_first=True)

    def forward(self, inputs):
        length = inputs.shape[1]
        hidden = self.project(inputs) + self.position[:length]
        return self.block(hidden, src_mask=self.causal_mask[:length, :length])[:, -1]


class DOCFusionModel(RiverArchitectureModel):
    """Common spatial backbone with one declared environmental/time/fusion tuple."""

    def __init__(self, environment="mlp", temporal="gru", fusion="concat", *,
                 env_dim=15, hidden=24, temporal_hidden=12, heads=3, layers=2,
                 dropout=.1, lookback=12):
        if environment not in ENVIRONMENT_CHOICES or temporal not in TEMPORAL_CHOICES:
            raise ValueError("unknown environmental or temporal component")
        if fusion not in FUSION_CHOICES or temporal_hidden % heads:
            raise ValueError("unknown fusion or incompatible temporal attention width")
        super().__init__("graph_transformer", env_dim=env_dim, hidden=hidden,
                         temporal_hidden=temporal_hidden, heads=heads, layers=layers, dropout=dropout)
        self.choices = {"environment": environment, "temporal": temporal, "fusion": fusion}
        self.lookback = lookback
        self.environment_width, self.temporal_width = hidden, temporal_hidden
        # Alternatives cannot perturb initial weights of the common trunk or
        # the subsequent dropout RNG. The inherited baseline needs no new draws.
        with torch.random.fork_rng(devices=[]):
            if environment == "constant":
                with torch.no_grad():
                    initial = self.environment(torch.zeros(1, env_dim))[0]
                self.environment = ConstantEnvironment(initial)
            elif environment == "linear":
                self.environment = copy.deepcopy(self.environment[0])
            elif environment == "residual_mlp":
                self.environment = ResidualEnvironment(self.environment)
            if temporal == "current":
                self.temporal = nn.Sequential(nn.Linear(8, 39), nn.GELU(), nn.Linear(39, temporal_hidden))
            elif temporal == "lag_mlp":
                self.temporal = nn.Sequential(nn.Linear(lookback * 8, 7), nn.GELU(),
                                              nn.Linear(7, temporal_hidden))
            elif temporal == "lstm":
                self.temporal = nn.LSTM(8, temporal_hidden, batch_first=True)
            elif temporal == "transformer":
                self.temporal = TemporalAttention(8, temporal_hidden, heads, lookback, dropout)
            if fusion in ("residual_concat", "conditioned"):
                width = hidden if fusion == "residual_concat" else temporal_hidden * 2
                self.fusion_extra = nn.Linear(hidden, width)
                nn.init.zeros_(self.fusion_extra.weight)
                nn.init.zeros_(self.fusion_extra.bias)
            elif fusion == "gated":
                self.fusion_extra = nn.Sequential(nn.Linear(hidden + temporal_hidden + 2, 14),
                                                  nn.GELU(), nn.Linear(14, 2))
                nn.init.zeros_(self.fusion_extra[-1].weight)
                nn.init.zeros_(self.fusion_extra[-1].bias)

    def temporal_state(self, windows):
        b, n, length, channels = windows.shape
        flat = windows.reshape(b * n, length, channels)
        choice = self.choices["temporal"]
        if choice in ("gru", "lstm"):
            _, state = self.temporal(flat)
            hidden = state[-1] if choice == "gru" else state[0][-1]
        elif choice == "current":
            hidden = self.temporal(flat[:, -1])
        elif choice == "lag_mlp":
            hidden = self.temporal(flat.flatten(1))
        else:
            hidden = self.temporal(flat)
        return hidden.reshape(b, n, -1)

    def forward(self, windows, environment, season, graph):
        if windows.shape[2:] != (self.lookback, 8):
            raise ValueError("declared historical window and eight covariate channels required")
        if self.choices == {"environment": "mlp", "temporal": "gru", "fusion": "concat"}:
            return super().forward(windows, environment, season, graph)
        b, n = windows.shape[:2]
        temporal = self.temporal_state(windows)
        env = self.environment(environment).unsqueeze(0).expand(b, -1, -1)
        current_season = season[:, None].expand(-1, n, -1)
        fusion = self.choices["fusion"]
        if fusion == "conditioned":
            scale, shift = self.fusion_extra(env).chunk(2, dim=-1)
            temporal = (1 + scale.tanh()) * temporal + shift
        joined = torch.cat([env, temporal, current_season], -1)
        if fusion == "gated":
            weights = 2 * self.fusion_extra(joined).softmax(-1)
            h, t = self.environment_width, self.temporal_width
            hidden = (weights[..., :1] * F.linear(env, self.combine.weight[:, :h])
                      + weights[..., 1:] * F.linear(temporal, self.combine.weight[:, h:h + t])
                      + F.linear(current_season, self.combine.weight[:, h + t:], self.combine.bias))
        else:
            hidden = self.combine(joined)
            if fusion == "residual_concat":
                hidden = hidden + self.fusion_extra(F.gelu(hidden))
        for block in self.blocks:
            hidden = block(hidden, graph, mode="graph_transformer")
        return self.readout(hidden).squeeze(-1)
