"""Small nonlinear readout on the existing DOC ecological/GRU residual."""
from __future__ import annotations

import copy

import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual


class ParallelResidualReadout(nn.Module):
    """Retain the linear path and add a zero-output nonlinear correction."""

    def __init__(self, linear, width, seed):
        super().__init__()
        self.linear = copy.deepcopy(linear)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.nonlinear = nn.Sequential(
                nn.LayerNorm(linear.in_features, dtype=linear.weight.dtype),
                nn.Linear(linear.in_features, width, dtype=linear.weight.dtype),
                nn.GELU(), nn.Linear(width, 1, dtype=linear.weight.dtype))
        nn.init.zeros_(self.nonlinear[-1].weight)
        nn.init.zeros_(self.nonlinear[-1].bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, features):
        return self.linear(features)+self.nonlinear(features)


class NonlinearNativeResidual(EncoderNativeResidual):
    """Native-MAE training with a small parallel nonlinear residual head.

    The encoder, GRU-D, causal windows, input visibility and signed native-unit
    correction are unchanged. LayerNorm acts only within the new head and does
    not fit dataset statistics. The default nonlinear width is32, with no
    dropout or new data channel. Zero final projection preserves the linear
    prediction at initialization, including when the supplied linear is nonzero.
    """

    def __init__(self, spatial, temporal, decay, *, readout_width=32, **kwargs):
        if not isinstance(readout_width, int) or isinstance(readout_width, bool) or readout_width < 1:
            raise ValueError("readout_width must be a positive integer")
        super().__init__(spatial, temporal, decay, **kwargs)
        self.readout_width = readout_width
        previous_count = sum(parameter.numel() for parameter in self.head.parameters())
        self.head = ParallelResidualReadout(self.head, readout_width, self.seed)
        self._initial_readout_state = copy.deepcopy(self.head.state_dict())
        self.trainable_parameter_count_ += sum(parameter.numel() for parameter in self.head.parameters())-previous_count

    def fit(self, *args, **kwargs):
        # The parent loop resets its linear head. Restore the new branch too so
        # repeated fits do not silently retain the preceding nonlinear weights.
        self.head.load_state_dict(self._initial_readout_state)
        return super().fit(*args, **kwargs)

    def _config(self):
        return {**super()._config(), "readout_width": self.readout_width}

    def to_dict(self):
        summary = super().to_dict()
        summary["model_class"] = type(self).__name__
        summary["protocol"].update({"readout": "linear + LayerNorm/Linear/GELU/zero-Linear",
            "readout_width": self.readout_width, "residual_units": "native mg/L",
            "readout_initialization": "original zero linear path and zero nonlinear output",
            "training_loss": "unchanged tail-weighted native MAE"})
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload["model_class"] = type(self).__name__
        payload["initial_readout"] = copy.deepcopy(self._initial_readout_state)
        return payload

    @classmethod
    def from_payload(cls, payload):
        if payload.get("model_class") != cls.__name__:
            raise ValueError("expected a nonlinear native residual checkpoint")
        state = copy.deepcopy(payload)
        state["model_class"] = "EncoderNativeResidual"
        obj = super().from_payload(state)
        initial = payload["initial_readout"]
        if any(not torch.isfinite(value).all() for value in initial.values()):
            raise ValueError("nonfinite initial nonlinear readout")
        if any(torch.count_nonzero(initial[key]) for key in
               ("linear.weight", "linear.bias", "nonlinear.3.weight", "nonlinear.3.bias")):
            raise ValueError("nonlinear residual initialization must have zero outputs")
        # Validate all initial shapes without replacing the fitted head.
        copy.deepcopy(obj.head).load_state_dict(initial)
        obj._initial_readout_state = copy.deepcopy(initial)
        return obj
