"""Support-set residual encoder for K-shot adaptation (spec v2).

Implements `experiments/kshot_protocol_v2/support_encoder_spec_v2.md`:

- set encoder over support tokens with query↔support attention;
- river-geometry bias (hop embedding, direction embedding, stream-order and
  drainage-area differences, ecological |Δregime|) added to the attention
  logits — pairwise fields in §2.1 enter here, since they depend on q;
- output is a log-space residual ``Δ_q`` so that
  ``log1p(ŷ_q) = log1p(base_q) + Δ_q``;
- K=0 returns ``Δ_q = 0`` **exactly** (skipped module, not an empty softmax).

This replaces the failed v1 blend-gate `SupportEncoder` in
`support_encoder.py`, which could only interpolate between `base` and
`local_mean` and degraded as K grew. The v1 functions stay as baselines.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Per-support content token: log1p_y, resid, streamorde, log-area, time_gap,
# regime_s(9).  Pairwise fields (hop, direction, order/area diff, |Δregime|)
# go through the attention bias and are not part of this token.
SUPPORT_DIM = 14
QUERY_DIM = 17  # log1p_base, regime_q(9), sin, cos, K, target_meta(4)
TARGET_META_DIM = 4
REGIME_DIM = 9
HOP_CAP = 10
N_DIR = 3  # downstream_of_support / upstream_of_support / other_or_multi_hop


class SupportSetEncoder(nn.Module):
    """Predict a log-space residual from a support set (spec v2, §3)."""

    def __init__(
        self,
        support_dim: int = SUPPORT_DIM,
        query_dim: int = QUERY_DIM,
        hidden: int = 64,
        hop_cap: int = HOP_CAP,
        regime_dim: int = REGIME_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hop_cap = hop_cap
        self.regime_dim = regime_dim
        self.mlp_sup = nn.Sequential(
            nn.Linear(support_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mlp_qry = nn.Sequential(
            nn.Linear(query_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.hop_emb = nn.Embedding(hop_cap + 2, 8)  # index hop_cap+1 = disconnected
        self.dir_emb = nn.Embedding(N_DIR, 4)
        # hop_emb(8) + dir_emb(4) + order_diff + area_diff + |Δregime|(regime_dim)
        self.bias_lin = nn.Linear(8 + 4 + 2 + regime_dim, 1)
        self.mlp_out = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp_out[-1].weight)
        nn.init.zeros_(self.mlp_out[-1].bias)

    def forward(
        self,
        support_tokens: torch.Tensor,   # (K, support_dim)
        support_dir: torch.Tensor,      # (K, Q) long in [0, N_DIR)
        support_hop: torch.Tensor,      # (K, Q) long; hop_cap+1 = disconnected
        support_order: torch.Tensor,    # (K,)
        support_area: torch.Tensor,     # (K,) log1p(totdasqkm)
        support_regime: torch.Tensor,   # (K, regime_dim)
        query_tokens: torch.Tensor,     # (Q, query_dim)
        query_order: torch.Tensor,      # (Q,)
        query_area: torch.Tensor,       # (Q,)
        query_regime: torch.Tensor,     # (Q, regime_dim)
    ) -> torch.Tensor:
        q = query_tokens.shape[0]
        if support_tokens.shape[0] == 0:
            return torch.zeros(q, device=query_tokens.device)

        h = self.mlp_sup(support_tokens)          # (K, d)
        hq = self.mlp_qry(query_tokens)           # (Q, d)
        hop_idx = support_hop.clamp(min=0, max=self.hop_cap + 1)
        hop_e = self.hop_emb(hop_idx)             # (K, Q, 8)
        dir_e = self.dir_emb(support_dir.clamp(min=0, max=N_DIR - 1))  # (K, Q, 4)
        d_order = (support_order.unsqueeze(1) - query_order.unsqueeze(0)).unsqueeze(-1)
        d_area = (support_area.unsqueeze(1) - query_area.unsqueeze(0)).unsqueeze(-1)
        d_regime = (
            support_regime.unsqueeze(1) - query_regime.unsqueeze(0)
        ).abs()  # (K, Q, regime_dim)
        bias_in = torch.cat([hop_e, dir_e, d_order, d_area, d_regime], dim=-1)
        bias = self.bias_lin(bias_in).squeeze(-1)                    # (K, Q)
        logits = torch.einsum("kd,qd->kq", h, hq) / math.sqrt(h.shape[-1]) + bias
        alpha = torch.softmax(logits, dim=0)                         # (K, Q)
        ctx = torch.einsum("kq,kd->qd", alpha, h)                    # (Q, d)
        return self.mlp_out(torch.cat([hq, ctx], dim=-1)).squeeze(-1)  # (Q,)


def fit_residual(
    model: SupportSetEncoder,
    support_tokens, support_dir, support_hop, support_order, support_area,
    support_regime, query_tokens, query_order, query_area, query_regime,
    base_log: torch.Tensor,
    y_log: torch.Tensor,
) -> torch.Tensor:
    """MSE of (log1p(base) + Δ) against log1p(y) over the episode's queries."""
    delta = model(
        support_tokens, support_dir, support_hop, support_order, support_area,
        support_regime, query_tokens, query_order, query_area, query_regime,
    )
    return torch.mean((base_log + delta - y_log) ** 2)
