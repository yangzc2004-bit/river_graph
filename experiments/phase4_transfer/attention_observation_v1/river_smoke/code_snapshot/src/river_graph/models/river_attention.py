"""Sparse, causal attention over upstream river edges and time lags."""

from __future__ import annotations

import torch
from torch import nn


class RiverLagAttention(nn.Module):
    """Edge-restricted multi-head attention over upstream edge--lag pairs.

    ``hidden_seq`` is chronological ``[time, nodes, hidden]``.  For a target
    node at month ``t``, candidates are restricted to the supplied directed
    edges and source states at ``t-lag``.  The module never constructs a
    dense station-by-station attention matrix.
    """

    LAGS = (0, 1, 3, 6, 12)

    def __init__(self, hidden_size: int, edge_dim: int, *, num_heads: int = 2,
                 dropout: float = 0.1, lags: tuple[int, ...] = LAGS):
        super().__init__()
        if hidden_size < 1 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if edge_dim < 1 or num_heads < 1:
            raise ValueError("edge_dim and num_heads must be positive")
        if not lags or any(int(lag) < 0 for lag in lags):
            raise ValueError("lags must be non-negative and non-empty")
        if tuple(sorted(set(lags))) != tuple(lags):
            raise ValueError("lags must be sorted and unique")
        self.hidden_size = int(hidden_size)
        self.edge_dim = int(edge_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.lags = tuple(int(lag) for lag in lags)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # Keep the no-candidate message an exact zero. A bias here would make
        # source-isolated targets receive a learned global river signal.
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.dynamic_bias = nn.Linear(5, num_heads, bias=False)
        self.lag_bias = nn.Parameter(torch.zeros(len(self.lags), num_heads))
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _segment_softmax(logits: torch.Tensor, index: torch.Tensor,
                         size: int) -> torch.Tensor:
        """Softmax over candidates sharing a destination node."""
        heads = logits.shape[1]
        expanded = index[:, None].expand(-1, heads)
        maximum = logits.new_full((size, heads), -torch.inf)
        maximum.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
        centered = logits - maximum[index]
        values = centered.exp()
        denominator = logits.new_zeros((size, heads))
        denominator.scatter_add_(0, expanded, values)
        return values / denominator[index].clamp_min(torch.finfo(values.dtype).eps)

    def forward(self, hidden_seq: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, *, x_seq: torch.Tensor | None = None,
                support_seq: torch.Tensor | None = None,
                return_diagnostics: bool = False):
        if hidden_seq.ndim != 3:
            raise ValueError("hidden_seq must have shape [time, nodes, hidden]")
        t, n, h = hidden_seq.shape
        if h != self.hidden_size:
            raise ValueError("hidden width does not match RiverLagAttention")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must align with edge_index")
        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError("edge_attr width does not match RiverLagAttention")
        if support_seq is None:
            support_seq = hidden_seq.new_zeros((t, n, 3))
        if support_seq.shape != (t, n, 3):
            raise ValueError("support_seq must have shape [time, nodes, 3]")
        if x_seq is None:
            x_seq = hidden_seq.new_zeros((t, n, 4))
        if x_seq.ndim != 3 or x_seq.shape[:2] != (t, n):
            raise ValueError("x_seq must align with hidden_seq")

        outputs = hidden_seq.new_zeros((t, n, h))
        lag_mass = hidden_seq.new_zeros((t, n, self.num_heads, len(self.lags)))
        entropy = hidden_seq.new_zeros((t, n))
        if edge_index.numel() == 0:
            diagnostics = {
                "lag_mass": lag_mass,
                "entropy": entropy,
                "lags": torch.as_tensor(self.lags, device=hidden_seq.device),
            }
            return (outputs, diagnostics) if return_diagnostics else outputs

        src, dst = edge_index.long()
        edge_bias = self.edge_bias(edge_attr)
        q_all = self.q_proj(hidden_seq).reshape(t, n, self.num_heads, self.head_dim)
        k_all = self.k_proj(hidden_seq).reshape(t, n, self.num_heads, self.head_dim)
        v_all = self.v_proj(hidden_seq).reshape(t, n, self.num_heads, self.head_dim)

        for month in range(t):
            candidate_logits = []
            candidate_values = []
            candidate_destinations = []
            candidate_lag_ids = []
            query = q_all[month, dst]
            for lag_id, lag in enumerate(self.lags):
                if month < lag:
                    continue
                source_month = month - lag
                keys = k_all[source_month, src]
                values = v_all[source_month, src]
                # Input channels are temperature, temperature visibility,
                # discharge, discharge visibility, then target/context
                # channels. Transport bias should use discharge explicitly.
                dynamic = torch.cat((
                    x_seq[source_month, src, 2:4],
                    x_seq[month, dst, 2:4],
                    support_seq[source_month, src, :1],
                ), dim=-1)
                bias = self.dynamic_bias(dynamic) + edge_bias + self.lag_bias[lag_id]
                logits = (query * keys).sum(-1) / (self.head_dim ** 0.5) + bias
                candidate_logits.append(logits)
                candidate_values.append(values)
                candidate_destinations.append(dst)
                candidate_lag_ids.append(torch.full_like(dst, lag_id))
            logits = torch.cat(candidate_logits, dim=0)
            values = torch.cat(candidate_values, dim=0)
            destinations = torch.cat(candidate_destinations, dim=0)
            lag_ids = torch.cat(candidate_lag_ids, dim=0)
            weights = self._segment_softmax(logits, destinations, n)
            weighted = self.dropout(weights)[:, :, None] * values
            message = hidden_seq.new_zeros((n, self.num_heads, self.head_dim))
            message.index_add_(0, destinations, weighted)
            outputs[month] = self.out_proj(message.reshape(n, h))
            for lag_id in range(len(self.lags)):
                selected = lag_ids == lag_id
                if selected.any():
                    lag_mass[month, :, :, lag_id].index_add_(
                        0, destinations[selected], weights[selected]
                    )
            entropy_by_candidate = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log())
            entropy_sum = hidden_seq.new_zeros((n, self.num_heads))
            entropy_sum.index_add_(0, destinations, entropy_by_candidate)
            entropy[month] = entropy_sum.mean(-1)

        diagnostics = {"lag_mass": lag_mass, "entropy": entropy,
                       "lags": torch.as_tensor(self.lags, device=hidden_seq.device)}
        return (outputs, diagnostics) if return_diagnostics else outputs


__all__ = ["RiverLagAttention"]
