"""Matched DOC loss variants on the existing partially trainable encoder.

This module changes only source loss weighting/asymmetry. Features, native
prediction, uniform cell shuffling, checkpoint selection and support inference
are inherited unchanged from EncoderNativeResidual.
"""
from __future__ import annotations

import numpy as np
import torch

from river_graph.models.encoder_native_residual import EncoderNativeResidual


class SelectiveEncoderResidual(EncoderNativeResidual):
    """Select between cell weighting, a one-sided penalty and station weighting.

    ``source_weighting='cell'`` and a zero penalty exactly reproduce the
    parent model. For station weighting, the tail-weighted cell weights are
    divided by their own station's total, so each source station contributes
    one unit after tail weighting. Sampling remains uniform over cells.

    With a positive ordinary penalty, the unnormalized cell loss is
    ``w * abs(prediction-truth) + beta * (~tail) * relu(prediction-truth)``.
    The same fixed full-source mean weight divides each minibatch mean.
    The penalty applies to the final nonnegative native concentration.
    Station weighting and a positive penalty are separate experiment arms.
    """

    def __init__(self, spatial, temporal, decay, *, source_weighting="cell",
                 ordinary_overprediction_penalty=0.0, **kwargs):
        if source_weighting not in ("cell", "station"):
            raise ValueError("source_weighting must be cell or station")
        if (not np.isfinite(ordinary_overprediction_penalty)
                or ordinary_overprediction_penalty not in (0, .5)):
            raise ValueError("ordinary_overprediction_penalty must be 0 or 0.5")
        if source_weighting == "station" and ordinary_overprediction_penalty != 0:
            raise ValueError("station weighting and ordinary penalty are separate arms")
        self.source_weighting = source_weighting
        self.ordinary_overprediction_penalty = float(ordinary_overprediction_penalty)
        super().__init__(spatial, temporal, decay, **kwargs)

    def _source_weights(self, source_cells, source_y, source_tail, months):
        weights = super()._source_weights(source_cells, source_y, source_tail, months)
        if self.source_weighting == "cell":
            return weights
        stations = torch.as_tensor(source_cells, dtype=torch.long) // months
        _, inverse = torch.unique(stations, sorted=True, return_inverse=True)
        totals = weights.new_zeros(int(inverse.max()) + 1)
        totals.scatter_add_(0, inverse, weights)
        return weights / totals[inverse]

    def _training_errors(self, prediction, truth, tail, weights):
        errors = super()._training_errors(prediction, truth, tail, weights)
        if self.ordinary_overprediction_penalty == 0:
            return errors
        return errors + self.ordinary_overprediction_penalty * (~tail).to(errors.dtype) * torch.relu(prediction - truth)

    def _config(self):
        return {**super()._config(), "source_weighting": self.source_weighting,
                "ordinary_overprediction_penalty": self.ordinary_overprediction_penalty}

    def to_dict(self):
        summary = super().to_dict()
        summary["model_class"] = "SelectiveEncoderResidual"
        summary["protocol"].update({
            "source_weighting": self.source_weighting,
            "source_weights": ("tail weights normalized to unit sum within each source station"
                               if self.source_weighting == "station" else "cell weights: 1 + (tail_weight-1)*tail"),
            "sampling": "uniform source-cell shuffle once per epoch; no station resampling",
            "ordinary_overprediction_penalty": self.ordinary_overprediction_penalty,
            "training_loss": "sum(w*abs(pred-truth) + beta*ordinary*relu(pred-truth))/sum(w)",
            "normalization": "fixed full-source mean weight; never a minibatch weight mean",
            "ordinary_rule": "source_truth < source_training_Q90; training loss only",
            "penalty_prediction": "final nonnegative native concentration at residual scale 1"})
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload["model_class"] = "SelectiveEncoderResidual"
        return payload

    @classmethod
    def from_payload(cls, payload):
        """Reload new checkpoints or import old encoder checkpoints with defaults."""
        if payload.get("model_class") not in ("SelectiveEncoderResidual", "EncoderNativeResidual"):
            raise ValueError("unsupported selective encoder residual checkpoint")
        if payload["model_class"] == "SelectiveEncoderResidual":
            required = ("source_weighting", "ordinary_overprediction_penalty")
            if not all(key in payload.get("config", {}) for key in required):
                raise ValueError("selective checkpoint lacks its loss definition")
        compatible = dict(payload, model_class="EncoderNativeResidual")
        return super().from_payload(compatible)
