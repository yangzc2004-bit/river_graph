"""A concentration-relative DOC readout on the existing ecological/GRU expert."""
from __future__ import annotations

import copy

import numpy as np
import torch

from river_graph.models.encoder_native_residual import EncoderNativeResidual


class LogConcentrationEncoderResidual(EncoderNativeResidual):
    """Learn log1p correction while retaining the existing native-MAE fit loop.

    No new encoder, gate, layer or information channel is introduced. The signed
    scalar head acts in log1p concentration rather than mg/L. Its absolute effect
    therefore scales with the environmental reference. Zero head/scale gives
    the exact environmental prediction, with a nonzero training derivative.
    """

    @staticmethod
    def _combine(base, delta, scale):
        if scale == 0:
            return base.clone()
        shift = scale*delta
        raw = torch.expm1(torch.log1p(base)+shift)
        # Preserve bitwise initialization without detaching the head's gradient.
        exact_identity = raw+(base-raw).detach()
        raw = torch.where(shift == 0, exact_identity, raw)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite log-concentration residual prediction")
        return torch.where(raw >= 0, raw, torch.zeros_like(raw))

    def predict_delta(self, full_inputs, batch_size=2048):
        """Unscaled signed log1p correction; it is not a native mg/L residual."""
        return super().predict_delta(full_inputs, batch_size=batch_size)

    def predict(self, full_inputs, base_native):
        self._prepare_inputs(full_inputs)
        base = np.asarray(base_native)
        if base.shape != np.shape(full_inputs["age"]) or not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("base_native must contain finite nonnegative station/month predictions")
        if self.selected_scale_ == 0:
            return base.copy()
        shift = self.selected_scale_*self.predict_delta(full_inputs)
        prediction = np.expm1(np.log1p(base.astype(np.float64))+shift)
        prediction[shift == 0] = base[shift == 0]
        if not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite log-concentration residual prediction")
        return np.maximum(0., prediction)

    def to_dict(self):
        summary = super().to_dict()
        summary["model_class"] = type(self).__name__
        summary["protocol"].update({"residual_units": "log1p DOC",
            "output": "max(0, expm1(log1p(base_native) + selected_scale * signed_log_delta))",
            "training_loss": "existing tail-weighted native mg/L MAE; output parameterization changes only"})
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload["model_class"] = type(self).__name__
        return payload

    @classmethod
    def from_payload(cls, payload):
        if payload.get("model_class") != cls.__name__:
            raise ValueError("expected a log-concentration residual checkpoint")
        state = copy.deepcopy(payload)
        state["model_class"] = "EncoderNativeResidual"
        return super().from_payload(state)
