"""Select permitted source DOC experience using source temperature and flow."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from river_graph.models.episodic_temporal_adapter import _state_copy
from river_graph.models.source_level_attention import SourceLevelAttentionResidual


def source_monthly_hydro_bank(raw, mode="current"):
    """Reuse frozen source preprocessing; never accept water-quality columns.

    The source-mean control retains contemporaneous visibility but replaces
    each observed value by its source station's frozen observed-period mean.
    This is a spatial transfer control, not an online forecast climatology.
    """
    raw = np.asarray(raw, float)
    if raw.ndim != 3 or raw.shape[-1] < 4 or mode not in ("current", "source_mean", "zero"):
        raise ValueError("aligned source raw inputs and a defined monthly-hydro mode required")
    bank = raw[..., :4].copy()
    if not np.isfinite(bank).all() or not np.isin(bank[..., [1, 3]], [0., 1.]).all():
        raise ValueError("finite source hydro values and binary visibility required")
    for value, visible in ((0, 1), (2, 3)):
        valid = bank[..., visible].astype(bool)
        bank[..., value] = np.where(valid, bank[..., value], 0.)
        if mode == "source_mean":
            count = valid.sum(1)
            mean = np.divide(bank[..., value].sum(1), count,
                             out=np.zeros(len(bank)), where=count > 0)
            bank[..., value] = np.where(valid, mean[:, None], 0.)
    if mode == "zero":
        bank.fill(0.)
    return bank


class SourceMonthlyHydroAttentionResidual(SourceLevelAttentionResidual):
    """Append four source-only monthly hydro columns to the existing key.

    The receiving query already contains its hydro history through the GRU.
    Native head, donor values, ecology/daily keys, query projection and first-
    order conversion are retained. New key coefficients start at zero.
    """

    def _install_head(self):
        super()._install_head()
        old = self.head.key
        new = nn.Linear(old.in_features + 4, old.out_features,
                        bias=False, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features].copy_(old.weight)
        self.head.key = new
        added = old.out_features * 4
        self.attention_parameter_count_ += added
        self.trainable_parameter_count_ += added
        self._initial_attention_head = _state_copy(self.head)

    def _prepare_inputs(self, inputs):
        arrays = super()._prepare_inputs(inputs)
        if "donor_monthly_hydro_bank" not in inputs:
            raise ValueError("source-only monthly hydro bank required")
        bank = torch.as_tensor(inputs["donor_monthly_hydro_bank"], dtype=self.dtype).detach().cpu()
        expected = (*arrays["donor_hydro_bank"].shape[:2], 4)
        if bank.shape != expected or not torch.isfinite(bank).all():
            raise ValueError("finite source-aligned monthly hydro bank required")
        if not torch.all((bank[..., [1, 3]] == 0) | (bank[..., [1, 3]] == 1)):
            raise ValueError("monthly hydro visibility must be binary")
        for value, visible in ((0, 1), (2, 3)):
            if torch.count_nonzero(bank[..., value][bank[..., visible] == 0]):
                raise ValueError("hidden monthly hydro values must be zero")
        arrays["donor_monthly_hydro_bank"] = bank
        return arrays

    def _attention_cells(self, inputs, cells, hidden):
        months = inputs["age"].shape[1]
        station, month = cells // months, cells % months
        owner = inputs["donor_owner"][station]
        daily = inputs["donor_hydro_bank"][owner.clamp_min(0), month[:, None]]
        monthly = inputs["donor_monthly_hydro_bank"][owner.clamp_min(0), month[:, None]]
        present = (owner >= 0)[..., None]
        keys = torch.cat([inputs["donor_ecology"][station], daily * present, monthly * present], -1)
        start = self.attention_config["daily_start"]
        query = torch.cat([hidden, inputs["env"][station],
                           inputs["extra"][station, month, start:start + 8]], -1)
        state, weights = self.head.allocate(query, keys, inputs["donor_values"][station, month],
            inputs["donor_valid"][station, month], inputs["donor_log_prior"][station])
        return state * (1 + inputs["attention_reference"][station, month, None]), weights

    def to_dict(self):
        summary = super().to_dict()
        summary["protocol"].update({
            "source_monthly_key": "frozen source-standardized temperature/visibility/discharge/visibility",
            "new_monthly_key_initialization": "zero coefficients; original query and key coefficients retained",
            "receiving_hydro": "existing observation-aware GRU; no new receiving water quality"})
        return summary
