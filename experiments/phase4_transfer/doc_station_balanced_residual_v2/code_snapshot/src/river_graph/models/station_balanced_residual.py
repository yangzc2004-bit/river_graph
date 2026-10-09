"""Equal-station optimization of the existing DOC encoder/GRU residual."""
from __future__ import annotations

import torch

from river_graph.models.encoder_native_residual import EncoderNativeResidual


class StationBalancedEncoderResidual(EncoderNativeResidual):
    """Change source loss weights alone, retaining the existing predictor.

    Each station receives the same total weight. Within a station the existing
    Q90 weight still doubles a high-DOC cell's contribution. Validation remains
    the original cell-weighted MAE, allowing direct comparison to the saved
    cell-weighted control. The base checkpoint schema remains inference-compatible.
    """

    def _source_weights(self, source_cells, source_y, source_tail, months):
        weights = super()._source_weights(source_cells, source_y, source_tail, months)
        cells = torch.as_tensor(source_cells, dtype=torch.long)
        _, inverse = torch.unique(cells//months, return_inverse=True)
        totals = weights.new_zeros(int(inverse.max())+1)
        totals.index_add_(0, inverse, weights)
        # Mean one overall; exactly equal station totals, including tail weights.
        return weights/totals[inverse]*(len(cells)/len(totals))

    def to_dict(self):
        summary = super().to_dict()
        summary["training_weighting"] = "equal total station weight; Q90 weight normalized within station"
        summary["protocol"]["source_loss_weighting"] = summary["training_weighting"]
        return summary
