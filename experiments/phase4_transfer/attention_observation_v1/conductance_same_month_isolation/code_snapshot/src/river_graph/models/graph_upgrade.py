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
from river_graph.models.river_attention import RiverLagAttention
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


class MessageOnlyTransportGCNImputer(TransportGCNImputer):
    """Spatial encoder whose representation is created only by edge messages.

    The self path is deliberately omitted.  With an empty edge set the
    representation is exactly zero, which makes this class useful for a
    matched source-isolation control rather than a second full predictor.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("residual", None)
        kwargs.pop("jumping_knowledge", None)
        super().__init__(*args, residual=False, jumping_knowledge=False, **kwargs)
        self.message_only = True

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
            zero = torch.zeros(n, conv.up_lin.out_features, device=h.device, dtype=h.dtype)
            up = conv._agg(conv.up_lin(h), edge_index, g_up, n) if self.edge_direction in (
                "both", "upstream"
            ) else zero
            down = conv._agg(conv.down_lin(h), edge_index.flip(0), g_down, n) if self.edge_direction in (
                "both", "downstream"
            ) else zero
            # Retain the first-hop representation when adding the second
            # hop. A pure two-hop stack would discard singly linked reaches.
            h = F.relu(up + down + (h if layer else zero))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class DualMessageOnlyTransportGCNImputer(TransportGCNImputer):
    """Separate target-observation and hydro-ecology message channels.

    Both branches use the same directed graph layers, but their input
    channels are disjoint.  A two-way softmax gate then combines the hidden
    states using flow, observation age, recent support and upstream support.
    With an empty edge set both hidden states are exactly zero.
    """

    def __init__(self, *args, target_base_end: int = 10, **kwargs):
        kwargs.pop("residual", None)
        kwargs.pop("jumping_knowledge", None)
        super().__init__(*args, residual=False, jumping_knowledge=False, **kwargs)
        self.target_base_end = int(target_base_end)
        hidden = self.convs[-1].self_lin.out_features
        self.fusion_gate = nn.Linear(hidden * 2 + 4, 2)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        self.message_only = True
        self.last_fusion_weights = None

    def _branch_step(self, h, edge_index, edge_attr, conv):
        n = h.shape[0]
        g_up = torch.sigmoid(conv.gate_up(edge_attr))
        g_down = torch.sigmoid(conv.gate_down(edge_attr))
        zero = torch.zeros(n, conv.up_lin.out_features, device=h.device, dtype=h.dtype)
        up = conv._agg(conv.up_lin(h), edge_index, g_up, n) if self.edge_direction in (
            "both", "upstream"
        ) else zero
        down = conv._agg(conv.down_lin(h), edge_index.flip(0), g_down, n) if self.edge_direction in (
            "both", "downstream"
        ) else zero
        return up + down

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor,
                     env_raw: torch.Tensor | None = None) -> torch.Tensor:
        base_channels = x.shape[-1] - 9
        if self.target_base_end > base_channels:
            raise ValueError("target_base_end exceeds static input channels")
        target_mask = torch.zeros_like(x)
        target_mask[..., 8:self.target_base_end] = 1.0
        target_mask[..., base_channels:] = 1.0
        target_x = x * target_mask
        hydro_x = x * (1.0 - target_mask)
        # Ecological embeddings belong to the hydro-ecology branch, while a
        # zero embedding keeps the two branches dimensionally matched.
        if self.env_encoder is not None:
            env_emb = self.env_encoder(env_raw)
            target_x = torch.cat((target_x, torch.zeros_like(env_emb)), dim=-1)
            hydro_x = torch.cat((hydro_x, env_emb), dim=-1)
        h_target, h_hydro = target_x, hydro_x
        for layer, conv in enumerate(self.convs):
            target_msg = self._branch_step(h_target, edge_index, edge_attr, conv)
            hydro_msg = self._branch_step(h_hydro, edge_index, edge_attr, conv)
            h_target = F.relu(target_msg + (h_target if layer else 0.0))
            h_hydro = F.relu(hydro_msg + (h_hydro if layer else 0.0))
            h_target = F.dropout(h_target, p=self.dropout, training=self.training)
            h_hydro = F.dropout(h_hydro, p=self.dropout, training=self.training)
        age_idx = base_channels + 1
        support_idx = base_channels + 6
        upstream_idx = base_channels + 7
        condition = torch.cat(
            (x[..., 2:3], x[..., age_idx:age_idx + 1],
             x[..., support_idx:support_idx + 1], x[..., upstream_idx:upstream_idx + 1]),
            dim=-1,
        )
        weights = torch.softmax(self.fusion_gate(torch.cat((h_target, h_hydro, condition), dim=-1)), dim=-1)
        self.last_fusion_weights = weights.detach()
        return weights[..., :1] * h_target + weights[..., 1:2] * h_hydro


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


class CausalObservationAttention(nn.Module):
    """Causal self-attention over a node's observed temporal history.

    The input convention is ``[time, nodes, hidden]``.  For every target
    month the query is the current hidden state and keys/values are restricted
    to the previous ``lookback`` months (including the current month).  The
    module never materialises attention between different stations, so it
    cannot create an unstructured spatial shortcut.  Observation age,
    support, and the presence of a usable history enter as additive head-wise
    biases rather than as target values.

    ``history_valid`` marks whether an observation history exists for a
    station-month.  It is deliberately a bias feature, not a hard mask: a
    station with no local observation can still use hydro/ecology-derived
    hidden states.  Only left-padding before the first real month is hard
    masked.
    """

    def __init__(self, hidden_size: int, *, lookback: int = 12,
                 num_heads: int = 2, support_dim: int = 3,
                 dropout: float = 0.1, chunk_months: int = 256):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if num_heads < 1 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if support_dim < 0:
            raise ValueError("support_dim must be non-negative")
        if chunk_months < 1:
            raise ValueError("chunk_months must be positive")
        self.hidden_size = int(hidden_size)
        self.lookback = int(lookback)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.support_dim = int(support_dim)
        self.chunk_months = int(chunk_months)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.age_bias = nn.Linear(1, num_heads, bias=False)
        self.support_bias = nn.Linear(support_dim, num_heads, bias=False) \
            if support_dim else None
        self.visibility_bias = nn.Linear(1, num_heads, bias=False)
        # Relative lag is represented in the same order as the window: the
        # final position is lag zero and the first is lag lookback-1.
        self.lag_bias = nn.Parameter(torch.zeros(num_heads, lookback))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.dropout = nn.Dropout(float(dropout))

    def _forward_chunk(
        self,
        hidden_seq: torch.Tensor,
        age_seq: torch.Tensor,
        support_seq: torch.Tensor,
        history_valid: torch.Tensor,
        target_months: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate a contiguous target-month chunk against the full history."""
        _total_t, n, h = hidden_seq.shape
        chunk_t = target_months.numel()
        device = hidden_seq.device

        # [target_month, window_position] where position zero may be a
        # left-padding copy of month zero.  Padding is masked below, so the
        # copied value cannot leak information into early predictions.
        offsets = torch.arange(self.lookback - 1, -1, -1, device=device)
        raw_idx = target_months[:, None] - offsets[None, :]
        valid = raw_idx >= 0
        idx = raw_idx.clamp_min(0)

        # Build [batch=target_month*node, window, feature] tensors.  This is
        # equivalent to independent per-node temporal attention and avoids a
        # dense all-station attention matrix.
        windows = hidden_seq[idx].permute(0, 2, 1, 3).reshape(
            chunk_t * n, self.lookback, h
        )
        age_windows = age_seq[idx].permute(0, 2, 1)
        support_windows = support_seq[idx].permute(0, 2, 1, 3)
        history_windows = history_valid[idx].permute(0, 2, 1).to(
            hidden_seq.dtype
        )
        query = hidden_seq[target_months].reshape(chunk_t * n, h)
        q = self.q_proj(query).reshape(chunk_t * n, self.num_heads, self.head_dim)
        k = self.k_proj(windows).reshape(
            chunk_t * n, self.lookback, self.num_heads, self.head_dim
        )
        v = self.v_proj(windows).reshape(
            chunk_t * n, self.lookback, self.num_heads, self.head_dim
        )
        logits = torch.einsum("bhd,blhd->bhl", q, k) / (self.head_dim ** .5)
        logits = logits + self.lag_bias.unsqueeze(0)
        age_logits = self.age_bias(age_windows.unsqueeze(-1)).reshape(
            chunk_t * n, self.lookback, self.num_heads
        ).permute(0, 2, 1)
        logits = logits + age_logits
        if self.support_bias is not None:
            support_logits = self.support_bias(support_windows).reshape(
                chunk_t * n, self.lookback, self.num_heads
            ).permute(0, 2, 1)
            logits = logits + support_logits
        visibility_logits = self.visibility_bias(
            history_windows.unsqueeze(-1)
        ).reshape(chunk_t * n, self.lookback, self.num_heads).permute(0, 2, 1)
        logits = logits + visibility_logits
        logits = logits.masked_fill(
            ~valid[:, None, :].expand(chunk_t, n, self.lookback).reshape(
                chunk_t * n, 1, self.lookback
            ),
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=-1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("nonfinite causal attention weights")
        context = torch.einsum("bhl,blhd->bhd", self.dropout(weights), v)
        context = self.out_proj(context.reshape(chunk_t * n, h))
        context = self.norm1(query + self.dropout(context))
        output = self.norm2(context + self.dropout(self.ffn(context)))
        output = output.reshape(chunk_t, n, h)
        diagnostics = {
            # Keep pre-dropout weights for a stable, normalized diagnostic.
            "weights": weights.reshape(chunk_t, n, self.num_heads, self.lookback),
            "valid": valid,
            "relative_lag": offsets,
        }
        return output, diagnostics

    def forward(
        self,
        hidden_seq: torch.Tensor,
        age_seq: torch.Tensor | None = None,
        support_seq: torch.Tensor | None = None,
        history_valid: torch.Tensor | None = None,
        month_features: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del month_features  # Reserved for seasonal bias in a later variant.
        if hidden_seq.ndim != 3:
            raise ValueError("hidden_seq must have shape [time, nodes, hidden]")
        t, n, h = hidden_seq.shape
        if h != self.hidden_size:
            raise ValueError(
                f"hidden_seq width {h} does not match attention width "
                f"{self.hidden_size}"
            )
        shape = (t, n)
        if age_seq is None:
            age_seq = hidden_seq.new_zeros(shape)
        if tuple(age_seq.shape) != shape:
            raise ValueError("age_seq must have shape [time, nodes]")
        if history_valid is None:
            history_valid = torch.ones(shape, dtype=torch.bool,
                                       device=hidden_seq.device)
        if tuple(history_valid.shape) != shape:
            raise ValueError("history_valid must have shape [time, nodes]")
        if history_valid.dtype != torch.bool:
            history_valid = history_valid > 0
        if support_seq is None:
            support_seq = hidden_seq.new_zeros((t, n, self.support_dim))
        elif support_seq.ndim == 2:
            support_seq = support_seq.unsqueeze(-1)
        if support_seq.ndim != 3 or tuple(support_seq.shape[:2]) != shape:
            raise ValueError("support_seq must have shape [time, nodes, support]")
        if support_seq.shape[-1] != self.support_dim:
            raise ValueError(
                f"support_seq width {support_seq.shape[-1]} does not match "
                f"support_dim {self.support_dim}"
            )
        support_seq = support_seq.to(dtype=hidden_seq.dtype)
        age_seq = age_seq.to(dtype=hidden_seq.dtype)

        outputs = []
        weight_chunks = []
        valid_chunks = []
        for start in range(0, t, self.chunk_months):
            stop = min(start + self.chunk_months, t)
            output, diagnostics = self._forward_chunk(
                hidden_seq, age_seq, support_seq, history_valid,
                torch.arange(start, stop, device=hidden_seq.device),
            )
            outputs.append(output)
            weight_chunks.append(diagnostics["weights"])
            valid_chunks.append(diagnostics["valid"])
        output = torch.cat(outputs, dim=0) if outputs else hidden_seq.new_empty(hidden_seq.shape)
        if not return_diagnostics:
            return output
        diagnostics = {
            "weights": torch.cat(weight_chunks, dim=0),
            "valid": torch.cat(valid_chunks, dim=0),
            "relative_lag": torch.arange(
                self.lookback - 1, -1, -1, device=hidden_seq.device
            ),
        }
        return output, diagnostics


class ObservationAwareAttentionTemporalTransportGCNImputer(
    ObservationAwareTemporalTransportGCNImputer
):
    """M1 observation-aware GRU with a causal attention residual branch.

    The existing GRU remains the main temporal path.  The attention branch is
    projected through a zero-initialized residual, so a freshly constructed
    model is exactly the released M1 model.  This makes the new operator a
    genuine upgrade of the current model rather than a replacement model.
    """

    def __init__(self, spatial, *, lookback: int = 12,
                 temporal_hidden: int = 64, chunk_months: int = 256,
                 history_ablation: str = "none", attention_heads: int = 2,
                 attention_dropout: float = 0.1):
        super().__init__(
            spatial,
            lookback=lookback,
            temporal_hidden=temporal_hidden,
            chunk_months=chunk_months,
            history_ablation=history_ablation,
        )
        self.temporal_attention = CausalObservationAttention(
            temporal_hidden,
            lookback=lookback,
            num_heads=attention_heads,
            support_dim=3,
            dropout=attention_dropout,
            chunk_months=chunk_months,
        )
        self.attention_residual = nn.Linear(temporal_hidden, temporal_hidden)
        nn.init.zeros_(self.attention_residual.weight)
        nn.init.zeros_(self.attention_residual.bias)
        self.last_attention_diagnostics = None

    def _temporal_windows(self, hidden_seq: torch.Tensor,
                          age_seq: torch.Tensor | None = None,
                          support_seq: torch.Tensor | None = None) -> torch.Tensor:
        memory = self._memory_states(hidden_seq, age_seq, support_seq)
        if support_seq is None:
            support_seq = hidden_seq.new_zeros((*hidden_seq.shape[:2], 3))
        # The first support channel is the local target visibility channel in
        # build_observation_features.  It is used as a reliability bias only;
        # spatially derived hidden states remain available when it is zero.
        history_valid = support_seq[..., 0] > 0 if support_seq.shape[-1] else None
        attended, diagnostics = self.temporal_attention(
            hidden_seq,
            age_seq=age_seq,
            support_seq=support_seq,
            history_valid=history_valid,
            return_diagnostics=True,
        )
        self.last_attention_diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        }
        delta = self.attention_residual(attended)
        # The zero-initialized projection makes a fresh model exactly equal to
        # M1. Its projection weights receive gradients on the first update;
        # after they open, the attention parameters train normally.
        fused = memory + delta
        return self.spatial.predict_from_hidden(fused)


class RiverLagAttentionTemporalTransportGCNImputer(
    ObservationAwareAttentionTemporalTransportGCNImputer
):
    """M1 plus sparse upstream edge--lag attention before temporal memory."""

    def __init__(self, spatial, *, lookback: int = 13,
                 temporal_hidden: int = 64, chunk_months: int = 256,
                 history_ablation: str = "none", attention_heads: int = 2,
                 attention_dropout: float = 0.1):
        super().__init__(spatial, lookback=lookback,
                         temporal_hidden=temporal_hidden,
                         chunk_months=chunk_months,
                         history_ablation=history_ablation,
                         attention_heads=attention_heads,
                         attention_dropout=attention_dropout)
        edge_dim = spatial.convs[0].gate_up[0].in_features
        self.river_attention = RiverLagAttention(
            temporal_hidden, edge_dim, num_heads=attention_heads,
            dropout=attention_dropout,
            chunk_months=chunk_months,
        )
        self.river_residual = nn.Linear(temporal_hidden, temporal_hidden)
        nn.init.zeros_(self.river_residual.weight)
        nn.init.zeros_(self.river_residual.bias)
        self.last_river_attention_diagnostics = None

    def forward_sequence(self, x_seq, edge_index, edge_attr, env_raw=None,
                         *, age_seq=None):
        if self.history_ablation == "hydro_only":
            x_seq = x_seq.clone()
            x_seq[:, :, 8:10] = 0.0
        hidden = self.encode_months(x_seq, edge_index, edge_attr, env_raw)
        support_seq = torch.stack([x_seq[..., 9], x_seq[..., -2], x_seq[..., -1]], -1)
        river, diagnostics = self.river_attention(
            hidden, edge_index, edge_attr, x_seq=x_seq,
            support_seq=support_seq, return_diagnostics=True,
        )
        self.last_river_attention_diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        }
        hidden = hidden + self.river_residual(river)
        return self._temporal_windows(hidden, age_seq, support_seq)


class MessageOnlyTemporalTransportImputer(ObservationAwareTemporalTransportGCNImputer):
    """Temporal response to upstream messages with an exact zero null.

    The source-isolation pilot omits the history-valid input from this one
    temporal layer and removes GRU biases. A zero message sequence
    therefore remains zero, while observation age still controls decay of a
    nonzero message state. This gives one forward pass instead of subtracting
    two full temporal runs.
    """

    def __init__(self, spatial, *, lookback=12, temporal_hidden=64,
                 chunk_months=256, history_ablation="none"):
        super().__init__(spatial, lookback=lookback,
                         temporal_hidden=temporal_hidden,
                         chunk_months=chunk_months,
                         history_ablation=history_ablation)
        self.temporal = nn.GRUCell(temporal_hidden, temporal_hidden, bias=False)

    def _temporal_windows(self, hidden_seq, age_seq=None, support_seq=None):
        if age_seq is None:
            age_seq = hidden_seq.new_zeros(hidden_seq.shape[:2])
        if support_seq is None:
            support_seq = hidden_seq.new_zeros((*hidden_seq.shape[:2], 3))
        t, n, h = hidden_seq.shape
        decay_input = torch.cat([age_seq[..., None], support_seq], dim=-1)
        gamma = torch.exp(-torch.relu(self.decay(decay_input)))
        outputs = []
        for start in range(0, t, self.chunk_months):
            stop = min(start + self.chunk_months, t)
            months = torch.arange(start, stop, device=hidden_seq.device)
            offsets = torch.arange(self.lookback - 1, -1, -1, device=hidden_seq.device)
            raw_idx = months[:, None] - offsets[None, :]
            valid = (raw_idx >= 0).float()
            idx = raw_idx.clamp_min(0)
            seq = hidden_seq[idx].permute(1, 0, 2, 3).reshape(
                self.lookback, (stop - start) * n, h
            )
            decay = gamma[idx].permute(1, 0, 2, 3).reshape(
                self.lookback, (stop - start) * n, h
            )
            valid_feature = valid.T[:, :, None].expand(
                self.lookback, stop - start, n
            ).reshape(self.lookback, (stop - start) * n, 1)
            seq = seq * valid_feature
            state = hidden_seq.new_zeros(((stop - start) * n, h))
            for step in range(self.lookback):
                updated = self.temporal(seq[step], decay[step] * state)
                state = torch.where(valid_feature[step].bool(), updated, state)
            outputs.append(state.reshape(stop - start, n, h))
        states = torch.cat(outputs, dim=0)
        return F.linear(states, self.spatial.head.weight, bias=None).squeeze(-1)


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
