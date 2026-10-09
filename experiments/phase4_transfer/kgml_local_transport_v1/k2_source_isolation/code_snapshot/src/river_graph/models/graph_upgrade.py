"""Observation-driven temporal graph modules.

The released H2X/H2X-T models remain unchanged.  This module contains the
next experimental family used by ``graph_upgrade_v2``:

* observation-aware temporal memory (GRU-D style decay),
* directed lagged transport messages, and
* a small causal multi-scale temporal mixer.

The components deliberately accept the same hidden-state convention as the
released temporal model, so each mechanism can be switched on independently.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.temporal import TemporalTransportGCNImputer


class MessageResidualTransportGCNImputer(TransportGCNImputer):
    """Transport trunk with a learnable residual scale on edge messages.

    The local (self) path is evaluated once.  The two directed message paths
    are then multiplied by a per-layer scalar before they are added.  This is
    deliberately different from a conventional residual block, which adds a
    second copy of the self path and can therefore change the scale of local
    information as depth increases.  Initialising the scales at ``0.1`` makes
    the pilot a conservative test of whether a small graph correction helps;
    the parameters remain trainable and can move toward zero or one.
    """

    def __init__(self, *args, message_scale_init: float = 0.1, **kwargs):
        # The factory passes the common spatial flags to every variant.  This
        # branch owns its message-only residual path and ignores those flags
        # rather than allowing duplicate keyword errors or silently enabling
        # the conventional self-path residual.
        kwargs.pop("residual", None)
        kwargs.pop("jumping_knowledge", None)
        super().__init__(*args, residual=False, jumping_knowledge=False, **kwargs)
        self.message_scales = nn.Parameter(
            torch.full((len(self.convs),), float(message_scale_init))
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor,
                     env_raw: torch.Tensor | None = None) -> torch.Tensor:
        if self.env_encoder is not None:
            emb = self.env_encoder(env_raw)
            x = torch.cat([x, emb], dim=-1)
        h = x
        for layer, conv in enumerate(self.convs):
            if self.edge_direction not in ("both", "upstream", "downstream"):
                raise ValueError(
                    "edge_direction must be 'both', 'upstream', or 'downstream'"
                )
            n = h.shape[0]
            g_up = torch.sigmoid(conv.gate_up(edge_attr))
            g_down = torch.sigmoid(conv.gate_down(edge_attr))
            zero = torch.zeros(n, conv.up_lin.out_features, device=h.device,
                               dtype=h.dtype)
            up = conv._agg(conv.up_lin(h), edge_index, g_up, n) if self.edge_direction in (
                "both", "upstream"
            ) else zero
            down = conv._agg(
                conv.down_lin(h), edge_index.flip(0), g_down, n
            ) if self.edge_direction in ("both", "downstream") else zero
            scale = self.message_scales[layer]
            h = F.relu(conv.self_lin(h) + scale * up + scale * down)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class ObservationGatedTransportGCNImputer(TransportGCNImputer):
    """Transport trunk with a node/month adaptive message gate.

    The gate is computed from the current hidden state, which includes the
    target visibility and observation-history channels in M1. It scales the
    directed message sum while leaving the local path unchanged. Thus a
    station can suppress an unsupported graph correction without disabling
    river messages globally.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("residual", None)
        kwargs.pop("jumping_knowledge", None)
        super().__init__(*args, residual=False, jumping_knowledge=False, **kwargs)
        dims = [self.convs[0].self_lin.in_features,
                *[conv.self_lin.out_features for conv in self.convs[:-1]]]
        self.message_gate = nn.ModuleList(nn.Linear(width, 1) for width in dims)
        for gate in self.message_gate:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor,
                     env_raw: torch.Tensor | None = None) -> torch.Tensor:
        if self.env_encoder is not None:
            x = torch.cat([x, self.env_encoder(env_raw)], dim=-1)
        h = x
        for layer, conv in enumerate(self.convs):
            n = h.shape[0]
            g_up = torch.sigmoid(conv.gate_up(edge_attr))
            g_down = torch.sigmoid(conv.gate_down(edge_attr))
            zero = torch.zeros(n, conv.up_lin.out_features, device=h.device,
                               dtype=h.dtype)
            up = conv._agg(conv.up_lin(h), edge_index, g_up, n) if self.edge_direction in (
                "both", "upstream"
            ) else zero
            down = conv._agg(
                conv.down_lin(h), edge_index.flip(0), g_down, n
            ) if self.edge_direction in ("both", "downstream") else zero
            gate = torch.sigmoid(self.message_gate[layer](h))
            h = F.relu(conv.self_lin(h) + gate * (up + down))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class ObservationAwareTemporalTransportGCNImputer(
    TemporalTransportGCNImputer
):
    """Causal GRU with a learned decay driven by observation age.

    ``age_seq`` is measured in months since the most recent visible target
    value (log-scaled) and has shape ``[time, nodes]``. Decay acts on the
    recurrent state BEFORE each GRU update, including the current month.
    """

    def __init__(self, spatial, *, lookback: int = 12,
                 temporal_hidden: int = 64, chunk_months: int = 256,
                 history_ablation: str = "none"):
        super().__init__(
            spatial,
            lookback=lookback,
            temporal_hidden=temporal_hidden,
            chunk_months=chunk_months,
            history_ablation=history_ablation,
        )
        self.temporal = nn.GRUCell(temporal_hidden + 1, temporal_hidden)
        self.decay = nn.Linear(4, temporal_hidden)
        # Inputs: log age, local visibility, upstream and downstream support.
        nn.init.zeros_(self.decay.weight)
        nn.init.zeros_(self.decay.bias)
        with torch.no_grad():
            self.decay.weight[:, 0] = 0.1

    def _memory_states(self, hidden_seq: torch.Tensor,
                       age_seq: torch.Tensor | None = None,
                       support_seq: torch.Tensor | None = None) -> torch.Tensor:
        """Return the GRU-D state for every month as ``[time, nodes, hidden]``.

        Keeping the state-producing part separate lets the combination model
        reuse the same observation-aware memory without changing M1's rolling
        window semantics.
        """
        if hidden_seq.ndim != 3:
            raise ValueError("hidden_seq must have shape [time, nodes, hidden]")
        t, n, h = hidden_seq.shape
        if age_seq is None:
            age_seq = hidden_seq.new_zeros((t, n))
        if age_seq.shape != (t, n):
            raise ValueError("age_seq must have shape [time, nodes]")
        outputs: list[torch.Tensor] = []
        device = hidden_seq.device
        if support_seq is None:
            support_seq = hidden_seq.new_zeros((t, n, 3))
        decay_input = torch.cat([age_seq[..., None], support_seq], dim=-1)
        gamma = torch.exp(-torch.relu(self.decay(decay_input)))
        for start in range(0, t, self.chunk_months):
            stop = min(start + self.chunk_months, t)
            months = torch.arange(start, stop, device=device)
            offsets = torch.arange(self.lookback - 1, -1, -1, device=device)
            raw_idx = months[:, None] - offsets[None, :]
            valid = (raw_idx >= 0).float()
            idx = raw_idx.clamp(min=0)
            seq = hidden_seq[idx].permute(1, 0, 2, 3).reshape(
                self.lookback, (stop - start) * n, h
            )
            decay = gamma[idx].permute(1, 0, 2, 3).reshape(
                self.lookback, (stop - start) * n, h
            )
            valid_feature = valid.T[:, :, None].expand(
                self.lookback, stop - start, n
            ).reshape(self.lookback, (stop - start) * n, 1)
            # Padding contains no target-derived information and performs no
            # recurrent update; the first real month starts from zero state.
            seq = seq * valid_feature
            if self.history_ablation == "shuffle" and self.lookback > 1:
                seq = torch.cat([seq[:-1].flip(0), seq[-1:]], dim=0)
                valid_feature = torch.cat(
                    [valid_feature[:-1].flip(0), valid_feature[-1:]], dim=0
                )
                decay = torch.cat([decay[:-1].flip(0), decay[-1:]], dim=0)
            state = hidden_seq.new_zeros(((stop - start) * n, h))
            for step in range(self.lookback):
                updated = self.temporal(
                    torch.cat([seq[step], valid_feature[step]], dim=-1),
                    decay[step] * state,
                )
                state = torch.where(valid_feature[step].bool(), updated, state)
            last = state.reshape(stop - start, n, self.temporal_hidden)
            outputs.append(last)
        return torch.cat(outputs, dim=0)

    def _temporal_windows(self, hidden_seq: torch.Tensor,
                          age_seq: torch.Tensor | None = None,
                          support_seq: torch.Tensor | None = None) -> torch.Tensor:
        states = self._memory_states(hidden_seq, age_seq, support_seq)
        return self.spatial.predict_from_hidden(states)

    def forward_sequence(self, x_seq, edge_index, edge_attr, env_raw=None,
                         *, age_seq: torch.Tensor | None = None):
        if self.history_ablation == "hydro_only":
            x_seq = x_seq.clone()
            x_seq[:, :, 8:10] = 0.0
        hidden = self.encode_months(x_seq, edge_index, edge_attr, env_raw)
        support_seq = torch.stack([x_seq[..., 9], x_seq[..., -2], x_seq[..., -1]], -1)
        return self._temporal_windows(hidden, age_seq, support_seq)


class ObservationAwareMultiScaleTemporalTransportImputer(
    ObservationAwareTemporalTransportGCNImputer
):
    """Combine observation-aware memory with causal multi-scale paths.

    The memory path is the GRU-D style state from M1.  Short and seasonal
    causal convolutions plus a causal trend GRU provide the M3 paths.  A
    learned gate mixes the four hidden states at each station-month, so the
    model can fall back to observation memory when a long temporal path is
    poorly supported.
    """

    def __init__(self, spatial, *, lookback: int = 12,
                 temporal_hidden: int = 64, chunk_months: int = 256,
                 history_ablation: str = "none"):
        super().__init__(
            spatial,
            lookback=lookback,
            temporal_hidden=temporal_hidden,
            chunk_months=chunk_months,
            history_ablation=history_ablation,
        )
        if history_ablation != "none":
            raise ValueError("M13 history ablations require causal implementation")
        self.history_scope = "full_causal_sequence"
        h = spatial.head.in_features
        path_hidden = max(8, h // 4)
        self.path_hidden = path_hidden
        self.short_conv = nn.Conv1d(h, path_hidden, kernel_size=3)
        self.season_conv = nn.Conv1d(h, path_hidden, kernel_size=5, dilation=2)
        self.trend_gru = nn.GRU(h + 1, h, num_layers=1)
        self.short_projection = nn.Linear(path_hidden, h)
        self.season_projection = nn.Linear(path_hidden, h)
        self.scale_gate = nn.Sequential(nn.Linear(4 * h, 4), nn.Softmax(dim=-1))

    @staticmethod
    def _causal_conv(conv: nn.Conv1d, seq: torch.Tensor) -> torch.Tensor:
        pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        return conv(torch.nn.functional.pad(seq, (pad, 0)))

    def _temporal_windows(self, hidden_seq: torch.Tensor,
                          age_seq: torch.Tensor | None = None,
                          support_seq: torch.Tensor | None = None) -> torch.Tensor:
        if hidden_seq.ndim != 3:
            raise ValueError("hidden_seq must have shape [time, nodes, hidden]")
        t, n, _ = hidden_seq.shape
        memory = self._memory_states(hidden_seq, age_seq, support_seq)
        sequence = hidden_seq.permute(1, 2, 0)
        short = self.short_projection(
            self._causal_conv(self.short_conv, sequence).permute(0, 2, 1)
        )
        seasonal = self.season_projection(
            self._causal_conv(self.season_conv, sequence).permute(0, 2, 1)
        )
        valid = hidden_seq.new_ones((t, n, 1))
        trend, _ = self.trend_gru(torch.cat([hidden_seq, valid], dim=-1))
        trend = trend.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        paths = torch.cat([memory, short, seasonal, trend], dim=-1)
        gate = self.scale_gate(paths)
        fused = (
            gate[..., 0:1] * memory
            + gate[..., 1:2] * short
            + gate[..., 2:3] * seasonal
            + gate[..., 3:4] * trend
        )
        return self.spatial.predict_from_hidden(fused).permute(1, 0)


class LaggedTransportTemporalImputer(ObservationAwareTemporalTransportGCNImputer):
    """M1 memory with upstream-only transport at causal monthly lags.

    Each spatial layer combines its self path and a degree-normalized sum of
    messages from actual upstream edges. Lag weights depend on edge attributes,
    source flow at the lagged month, receiving flow now, their visibility, and
    target-observation support. Weights are normalized across available lags
    before aggregation; they are never cancelled by a per-lag normalization.
    """

    LAGS = (0, 1, 3, 6, 12)

    def __init__(self, spatial, *, lookback: int = 12,
                 temporal_hidden: int = 64, chunk_months: int = 256,
                 history_ablation: str = "none", lag_mode: str = "learned"):
        super().__init__(
            spatial,
            lookback=lookback,
            temporal_hidden=temporal_hidden,
            chunk_months=chunk_months,
            history_ablation=history_ablation,
        )
        if lag_mode not in ("static", "fixed", "learned", "none"):
            raise ValueError("invalid lag mode")
        self.lag_mode = lag_mode
        edge_dim = spatial.convs[0].gate_up[0].in_features
        self.lag_gate = nn.Sequential(nn.Linear(edge_dim + 7, 16), nn.ReLU(), nn.Linear(16, 1))

    def lag_weights(self, x_seq, edge_index, edge_attr):
        t = len(x_seq)
        src, dst = edge_index
        logits = []
        months = torch.arange(t, device=x_seq.device)
        for lag in self.LAGS:
            indices = (months - lag).clamp_min(0)
            dynamic = torch.stack([
                x_seq[indices][:, src, 2], x_seq[:, dst, 2],
                x_seq[indices][:, src, 3], x_seq[:, dst, 3],
                x_seq[indices][:, src, 9], x_seq[:, dst, 9],
                torch.full((t, len(src)), lag / 12, device=x_seq.device),
            ], -1)
            if self.lag_mode == "learned":
                features = torch.cat([edge_attr[None].expand(t, -1, -1), dynamic], -1)
                score = self.lag_gate(features).squeeze(-1)
            else:
                # Fixed comparator: equal weights on available buckets.
                score = x_seq.new_zeros((t, len(src)))
            score = score.masked_fill((months < lag)[:, None], -torch.inf)
            if self.lag_mode == "static" and lag != 0:
                score = torch.full_like(score, -torch.inf)
            logits.append(score)
        weights = torch.softmax(torch.stack(logits, -1), -1)
        return weights if self.lag_mode != "none" else torch.zeros_like(weights)

    def encode_months(self, x_seq, edge_index, edge_attr, env_raw=None):
        t, n, _ = x_seq.shape
        h = x_seq
        if self.spatial.env_encoder is not None:
            env = self.spatial.env_encoder(env_raw)[None].expand(t, -1, -1)
            h = torch.cat([h, env], -1)
        weights = self.lag_weights(x_seq, edge_index, edge_attr)
        src, dst = edge_index
        degree = torch.bincount(dst, minlength=n).to(h).clamp_min(1)[None, :, None]
        for conv in self.spatial.convs:
            message = h.new_zeros((t, n, conv.up_lin.out_features))
            if len(src) and self.lag_mode != "none":
                projected = conv.up_lin(h)
                strength = torch.sigmoid(conv.gate_up(edge_attr))
                for li, lag in enumerate(self.LAGS):
                    if lag >= t:
                        continue
                    weighted = projected[:t - lag, src] * weights[lag:, :, li, None] * strength
                    added = message.new_zeros(message.shape)
                    added[lag:].index_add_(1, dst, weighted)
                    message = message + added
            h = torch.relu(conv.self_lin(h) + message / degree)
            h = torch.nn.functional.dropout(h, self.spatial.dropout, self.training)
        return h


class MultiScaleTemporalTransportImputer(TemporalTransportGCNImputer):
    """Causal short/seasonal/trend temporal mixer over H2X states.

    The temporal paths are evaluated once on the chronological hidden
    sequence.  This is causal and avoids duplicating an ``N x lookback``
    window for every station-month, which made the first pilot needlessly
    slow.  The trend path carries state through all earlier months.
    """

    def __init__(self, spatial, *, lookback: int = 12,
                 temporal_hidden: int = 64, chunk_months: int = 256,
                 history_ablation: str = "none"):
        super().__init__(
            spatial,
            lookback=lookback,
            temporal_hidden=temporal_hidden,
            chunk_months=chunk_months,
            history_ablation=history_ablation,
        )
        h = spatial.head.in_features
        if history_ablation != 'none':
            raise ValueError('M3 history ablations require separate causal implementations')
        self.history_scope = 'full_causal_sequence'
        # The two convolutional paths are intentionally low-rank.  Applying
        # full HxH kernels to every station-month makes the pilot needlessly
        # expensive while adding no scientific degree of freedom.
        path_hidden = max(8, h // 4)
        self.path_hidden = path_hidden
        self.short_conv = nn.Conv1d(h, path_hidden, kernel_size=3, dilation=1)
        self.season_conv = nn.Conv1d(h, path_hidden, kernel_size=5, dilation=2)
        self.trend_gru = nn.GRU(h + 1, h, num_layers=1)
        self.short_projection = nn.Linear(path_hidden, h)
        self.season_projection = nn.Linear(path_hidden, h)
        self.scale_gate = nn.Sequential(nn.Linear(3 * h, 3), nn.Softmax(dim=-1))

    @staticmethod
    def _causal_conv(conv: nn.Conv1d, seq: torch.Tensor) -> torch.Tensor:
        # seq is [batch, hidden, time].  Left padding keeps output aligned to
        # the current month and never reads a future element.
        pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        return conv(torch.nn.functional.pad(seq, (pad, 0)))

    def _temporal_windows(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        if hidden_seq.ndim != 3:
            raise ValueError("hidden_seq must have shape [time, nodes, hidden]")
        t, n, _ = hidden_seq.shape
        sequence = hidden_seq.permute(1, 2, 0)  # [nodes, hidden, time]
        short = self.short_projection(
            self._causal_conv(self.short_conv, sequence).permute(0, 2, 1)
        )
        seasonal = self.season_projection(
            self._causal_conv(self.season_conv, sequence).permute(0, 2, 1)
        )
        valid = hidden_seq.new_ones((t, n, 1))
        trend, _ = self.trend_gru(torch.cat([hidden_seq, valid], dim=-1))
        trend = trend.permute(1, 0, 2)  # [nodes, time, hidden]
        gate = self.scale_gate(torch.cat([short, seasonal, trend], dim=-1))
        fused = (
            gate[..., 0:1] * short
            + gate[..., 1:2] * seasonal
            + gate[..., 2:3] * trend
        )
        # Temporal modules expose [time, nodes], matching the released H2X-T
        # wrapper and the training/evaluation code.
        return self.spatial.predict_from_hidden(fused).permute(1, 0)
