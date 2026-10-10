"""Use an exact inverse log transform for the source branch of a native model."""
from __future__ import annotations

import numpy as np
import torch

from river_graph.models.current_source_attention import CurrentSourceAttentionResidual
from river_graph.models.relative_source_attention import RelativeSourceAttentionResidual


class RatioSourceAttentionResidual(RelativeSourceAttentionResidual):
    """Retain native local learning and transform only the relative source output."""

    def _attention_cells(self, inputs, cells, hidden):
        return CurrentSourceAttentionResidual._attention_cells(self, inputs, cells, hidden)

    def _delta_cells(self, inputs, cells):
        cells = torch.as_tensor(cells, dtype=torch.long)
        hidden = self._hidden_cells(inputs, cells)
        t = inputs["age"].shape[1]
        station, month = cells//t, cells % t
        extra = inputs["extra"][station, month]
        pieces = [hidden, extra]
        if self.interaction_indices:
            interaction = hidden.unsqueeze(-1)*extra[:, self.interaction_indices].unsqueeze(1)
            pieces.append(interaction.flatten(1))
        state, _ = self._attention_cells(inputs, cells, hidden)
        relative_output = self.head.output(state).squeeze(-1).double()
        local_delta = self.head.linear(torch.cat(pieces, -1)).squeeze(-1).double()
        reference = inputs["attention_reference"][station, month]
        delta = local_delta + (1+reference)*torch.expm1(relative_output)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite exact-ratio source correction")
        return delta

    def diagnostics(self, inputs, cells=None):
        result = super().diagnostics(inputs, cells)
        n, t = np.shape(inputs["age"])
        selected = np.arange(n*t) if cells is None else np.asarray(cells)
        reference = np.asarray(inputs["attention_reference"]).ravel()[selected]
        weights = self.head.output.weight.detach().cpu().double().numpy().ravel()
        relative_output = result["current_source_state"] @ weights
        exact = (1+reference)*np.expm1(relative_output)
        first_order = (1+reference)*relative_output
        result.update({"source_relative_output": relative_output,
                       "source_native_correction": exact,
                       "source_first_order_correction": first_order,
                       "source_curvature_correction": exact-first_order})
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("nonfinite ratio correction diagnostics")
        return result

    def to_dict(self):
        summary = super().to_dict()
        summary["protocol"].update({
            "attention_state_units": "dimensionless seasonal log1p residual",
            "value_conversion": "(1+frozen receiving reference)*expm1(projected relative donor state)",
            "local_readout": "unchanged native residual and native training objective",
            "physical_constraint": "none; statistical source concentration-ratio transfer"})
        return summary
