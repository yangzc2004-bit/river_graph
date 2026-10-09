"""Output-space fusion for a frozen spatial and temporal expert pair."""

from __future__ import annotations

import math

import torch
from torch import nn


def fuse_transformed(rf_z: torch.Tensor, local_z: torch.Tensor,
                     alpha: torch.Tensor | float) -> torch.Tensor:
    """Blend two predictions in the shared transformed target space.

    ``alpha=0`` returns the RF/context expert and ``alpha=1`` returns the
    temporal/local expert.  The operation is deliberately kept outside the
    inverse target transform so DOC/EC log1p and pH standardization use the
    same gate.
    """
    if rf_z.shape != local_z.shape:
        raise ValueError("expert predictions must have identical shapes")
    weight = torch.as_tensor(alpha, dtype=rf_z.dtype, device=rf_z.device)
    if weight.ndim not in (0, rf_z.ndim):
        raise ValueError("alpha must be scalar or prediction-shaped")
    return rf_z + weight * (local_z - rf_z)


class UnifiedSpatiotemporalFusion(nn.Module):
    """A small gate over two frozen expert predictions.

    The gate is the only trainable part.  Expert predictions must already be
    in the same target space.  Initializing the gate with zero weights makes
    the first prediction a constant-alpha blend, which provides a direct
    comparison with the family-constant pilot before observable features are
    allowed to change the weight.
    """

    def __init__(self, feature_dim: int, *, hidden: int = 0,
                 init_alpha: float = 0.0, alpha_floor: float = 0.0,
                 alpha_ceiling: float = 1.0):
        super().__init__()
        if (feature_dim < 1 or hidden < 0 or not 0.0 <= alpha_floor < alpha_ceiling <= 1.0
                or not alpha_floor <= init_alpha <= alpha_ceiling):
            raise ValueError("invalid fusion gate settings")
        self.alpha_floor = alpha_floor
        self.alpha_ceiling = alpha_ceiling
        if hidden:
            self.gate = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(),
                                      nn.Linear(hidden, 1))
            output = self.gate[-1]
        else:
            self.gate = nn.Linear(feature_dim, 1)
            output = self.gate
        for parameter in self.gate.parameters():
            nn.init.zeros_(parameter)
        normalized = (init_alpha - alpha_floor) / (alpha_ceiling - alpha_floor)
        output.bias.data.fill_(self._logit(normalized))

    @staticmethod
    def _logit(alpha: float) -> float:
        eps = torch.finfo(torch.float32).eps
        alpha = min(max(alpha, float(eps)), 1.0 - float(eps))
        return math.log(alpha / (1.0 - alpha))

    def forward(self, rf_z: torch.Tensor, local_z: torch.Tensor,
                features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[0] != rf_z.shape[0]:
            raise ValueError("features must be [batch, feature_dim]")
        if rf_z.shape != local_z.shape or rf_z.ndim != 1:
            raise ValueError("expert predictions must be one-dimensional and aligned")
        raw_alpha = torch.sigmoid(self.gate(features)).squeeze(-1)
        alpha = self.alpha_floor + (self.alpha_ceiling - self.alpha_floor) * raw_alpha
        return alpha, fuse_transformed(rf_z, local_z, alpha)
