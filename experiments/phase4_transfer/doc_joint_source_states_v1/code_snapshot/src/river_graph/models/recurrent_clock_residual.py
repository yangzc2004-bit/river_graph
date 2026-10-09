"""Matched clocks for the existing DOC/ecology recurrent state decay.

Only the scalar entering the decay layer changes. Raw M1 features, spatial
encoding, GRU, scalar head, losses and fitting lifecycle remain inherited.
"""

from __future__ import annotations

import numpy as np
import torch

from river_graph.models.encoder_native_residual import EncoderNativeResidual

DECAY_CLOCKS = ("legacy", "unseen_neutral", "flow_window")
FLOW_VISIBLE_INDEX = 3
LAST_OBSERVATION_VALID_INDEX = -7
FLOW_CLOCK_CAP = 12


class ClockNativeResidual(EncoderNativeResidual):
    """Change the recurrence clock while preserving the original raw inputs.

    ``legacy`` delegates to the parent exactly. ``unseen_neutral`` suppresses
    the age scalar only while the appended M1 last-observation-valid flag is
    zero. ``flow_window`` uses monthly discharge visibility within each queried
    causal window: start at 12, reset to zero on visible discharge, otherwise
    increment and cap at 12. Padding does not advance the clock.

    The fixed discharge mask is raw column 3. M1's nine appended features place
    last-observation-valid at raw column -7 (column 16 in current 23-column
    inputs); it is not an absolute column 12. No DOC values or new labels are
    read. These are shared-state decay interventions, not physical travel time.
    """

    def __init__(self, spatial, temporal, decay, *, decay_clock="legacy", **kwargs):
        if decay_clock not in DECAY_CLOCKS:
            raise ValueError(f"decay_clock must be one of {DECAY_CLOCKS}")
        self.decay_clock = decay_clock
        super().__init__(spatial, temporal, decay, **kwargs)
        if decay_clock != "legacy" and self.spatial_architecture["in_channels"] < 19:
            raise ValueError("clock variants require the base ten channels plus nine appended M1 features")
        if decay_clock == "flow_window" and self.lookback != FLOW_CLOCK_CAP:
            raise ValueError("flow_window requires the matched 12-month lookback")

    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)
        if self.decay_clock != "legacy":
            index = (LAST_OBSERVATION_VALID_INDEX if self.decay_clock == "unseen_neutral"
                     else FLOW_VISIBLE_INDEX)
            flag = prepared["raw"][..., index]
            if not ((flag == 0) | (flag == 1)).all():
                raise ValueError("the chosen clock visibility channel must contain binary zero/one flags")
        return prepared

    def _gather(self, inputs, cells):
        cells = torch.as_tensor(cells, dtype=torch.long)
        n, t = inputs["age"].shape
        if cells.ndim != 1 or not len(cells) or (cells < 0).any() or (cells >= n*t).any():
            raise ValueError("cells must be a nonempty valid flat station/month vector")
        station_ids, month_ids = cells // t, cells % t
        raw_months = month_ids[:, None]-torch.arange(self.lookback-1, -1, -1)[None, :]
        valid = raw_months >= 0
        months = raw_months.clamp_min(0)
        stations = station_ids[:, None].expand_as(months)
        return cells, stations, months, valid

    def _clock(self, inputs, stations, months, valid):
        legacy = inputs["age"][stations, months]
        if self.decay_clock == "legacy":
            return legacy
        if self.decay_clock == "unseen_neutral":
            seen = inputs["raw"][stations, months, LAST_OBSERVATION_VALID_INDEX] > 0
            return torch.where(valid & seen, legacy, torch.zeros_like(legacy))
        visible = inputs["raw"][stations, months, FLOW_VISIBLE_INDEX] > 0
        age = legacy.new_full((len(stations),), float(FLOW_CLOCK_CAP))
        sequence = []
        for step in range(self.lookback):
            updated = torch.where(visible[:, step], torch.zeros_like(age),
                                  (age+1).clamp_max(FLOW_CLOCK_CAP))
            age = torch.where(valid[:, step], updated, age)
            scaled = torch.log1p(age)/float(np.log1p(FLOW_CLOCK_CAP))
            sequence.append(torch.where(valid[:, step], scaled, torch.zeros_like(scaled)))
        return torch.stack(sequence, dim=1)

    def clock_window(self, inputs, cells, *, include_gamma=False):
        """Return label-free clock diagnostics without running hidden states.

        Arrays ``clock``, ``legacy_clock``, ``valid`` and ``month_indices`` have
        shape [cells,lookback]; ``station_indices`` has one entry per cell.
        Optional gamma is [cells,lookback,hidden], including padding positions
        that must be excluded with ``valid``. No parameters or inputs change.
        """
        prepared = self._prepare_inputs(inputs)
        _, stations, months, valid = self._gather(prepared, cells)
        with torch.no_grad():
            clock = self._clock(prepared, stations, months, valid)
            result = {"clock": clock.numpy().copy(),
                      "legacy_clock": prepared["age"][stations, months].numpy().copy(),
                      "valid": valid.numpy().copy(), "month_indices": months.numpy().copy(),
                      "station_indices": stations[:, 0].numpy().copy()}
            if include_gamma:
                values = torch.cat([clock[..., None], prepared["support"][stations, months]], -1)
                result["gamma"] = torch.exp(-torch.relu(self.decay(values))).numpy().copy()
        return result

    def _hidden_cells(self, inputs, cells):
        if self.decay_clock == "legacy":
            return super()._hidden_cells(inputs, cells)
        cells, stations, months, valid = self._gather(inputs, cells)
        self.spatial.eval()
        encoded = self.spatial.encode_nodes(
            inputs["raw"][stations[valid], months[valid]],
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, self.spatial_architecture["edge_dim"]), dtype=self.dtype),
            inputs["env"][stations[valid]])
        valid_rows = torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)
        sequence = encoded.new_zeros((len(cells)*self.lookback, self.hidden_size))
        sequence = sequence.index_copy(0, valid_rows, encoded).reshape(len(cells), self.lookback, -1)
        if self.hydro_projection is not None:
            use_history = valid.clone()
            if self.hydro_sequence_mode == "current_only":
                use_history[:, :-1] = False
            projected = self.hydro_projection(inputs["daily_history"][stations[use_history], months[use_history]])
            history_rows = torch.nonzero(use_history.reshape(-1), as_tuple=False).reshape(-1)
            hydro_sequence = sequence.new_zeros((len(cells)*self.lookback, self.hidden_size))
            hydro_sequence = hydro_sequence.index_copy(0, history_rows, projected).reshape_as(sequence)
            sequence = sequence+hydro_sequence
        clock = self._clock(inputs, stations, months, valid)
        decay_input = torch.cat([clock[..., None], inputs["support"][stations, months]], -1)
        gamma = torch.exp(-torch.relu(self.decay(decay_input)))
        state = encoded.new_zeros((len(cells), self.hidden_size))
        for step in range(self.lookback):
            indicator = valid[:, step, None]
            step_input = torch.cat([sequence[:, step]*indicator, indicator.to(self.dtype)], -1)
            updated = self.temporal(step_input, gamma[:, step]*state)
            state = torch.where(indicator, updated, state)
        return state

    def _config(self):
        return {**super()._config(), "decay_clock": self.decay_clock}

    def to_dict(self):
        summary = super().to_dict()
        summary["model_class"] = "ClockNativeResidual"
        summary["protocol"].update({
            "decay_clock": self.decay_clock, "clock_input_changed": "decay scalar only; raw M1 age unchanged",
            "flow_visibility_raw_index": FLOW_VISIBLE_INDEX,
            "last_observation_valid_raw_index": LAST_OBSERVATION_VALID_INDEX,
            "flow_window": "start12; visible discharge reset0; otherwise increment cap12; valid steps only",
            "flow_clock_scale": "log1p(months)/log1p(12)",
            "clock_scope": "queried causal window only; no external history, labels or scalar-head extras",
            "effective_capacity_note": "unseen-neutral removes age variation at never-observed stations; allocated weights unchanged",
            "clock_interpretation": "shared recurrent-state decay; not physical transport or residence time"})
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload["model_class"] = "ClockNativeResidual"
        return payload

    @classmethod
    def from_payload(cls, payload):
        """Reload a clock model, or import an old encoder as exact legacy."""
        if payload.get("model_class") not in ("ClockNativeResidual", "EncoderNativeResidual"):
            raise ValueError("unsupported recurrent-clock checkpoint")
        if payload["model_class"] == "ClockNativeResidual" and "decay_clock" not in payload.get("config", {}):
            raise ValueError("clock checkpoint lacks its decay_clock definition")
        if payload["model_class"] == "EncoderNativeResidual" and "decay_clock" in payload.get("config", {}):
            raise ValueError("old encoder checkpoint unexpectedly contains a clock override")
        return super().from_payload(dict(payload, model_class="EncoderNativeResidual"))
