"""Let source DOC state inform the existing ecological/hydro attention keys."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from river_graph.models.episodic_temporal_adapter import _state_copy
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_level_attention import source_level_input_view


def source_key_rms(library):
    """Fit once from permitted source OOF log residuals, preserving zero."""
    season = pd.PeriodIndex(library.months_, freq="M").month.to_numpy()-1
    values = library.innovations_+library.seasonal_mean_[:, season]
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("finite permitted source residuals required")
    return float(max(np.sqrt(np.mean(values[valid]**2)), 1e-6))


def source_state_key_candidates(library, receiver_names, receiver_ecology, months,
                                bank_names, bank_hydro, *, rms):
    """Use fixed source normalization and current observations, never future."""
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("positive frozen source-only key RMS required")
    arrays = current_available_candidates(library, receiver_names, receiver_ecology,
                                         months, bank_names, bank_hydro)
    arrays["donor_doc_key"] = np.stack([arrays["donor_values_real"],
                                       arrays["donor_values_seasonal"]], axis=-1)/rms
    return arrays


def source_state_key_view(inputs, candidates, mode):
    """Change only key information; actual source values stay present."""
    if mode not in ("current", "seasonal", "zero"):
        raise ValueError("current, seasonal or zero source key mode required")
    key = candidates["donor_doc_key"].copy()
    if mode == "seasonal":
        key[..., 0] = 0.
    elif mode == "zero":
        key.fill(0.)
    return {**source_level_input_view(inputs, candidates, "real"), "donor_doc_key": key}


def nested_source_state_key_candidates(truth, station_names, months, ecology, source_cells,
                                        folds, references, bank_hydro):
    """Fit each key normalizer without the query fold or receiving labels."""
    truth, cells = np.asarray(truth), np.asarray(source_cells)
    if truth.ndim != 2 or cells.ndim != 1 or cells.dtype.kind not in "iu" or not len(cells):
        raise ValueError("aligned truth and permitted source cells required")
    t = truth.shape[1]
    ids, names = np.unique(cells//t), np.asarray(station_names, str)
    folds = [np.asarray(fold, dtype=np.int64) for fold in folds]
    if len(folds) < 3 or not np.array_equal(np.sort(np.concatenate(folds)), ids):
        raise ValueError("disjoint folds must partition the permitted source stations")
    output, records = None, []
    for a, query in enumerate(folds):
        donor_cells = cells[~np.isin(cells//t, query)]
        oof = np.full(truth.shape, np.nan)
        for b, donor in enumerate(folds):
            if a == b:
                continue
            saved = references[tuple(sorted((a, b)))]
            hidden = set(query.tolist()+donor.tolist())
            if (set(saved["hidden_stations"]) != hidden
                    or set(saved["fitted_stations"]) != set(ids.tolist())-hidden):
                raise ValueError("current donor reference must exclude query AND donor folds")
            expected = cells[np.isin(cells//t, list(hidden))]
            if not np.array_equal(saved["cells"], expected) or np.shape(saved["pred_z"]) != expected.shape:
                raise ValueError("source reference population changed")
            take = np.isin(expected//t, donor)
            oof.ravel()[expected[take]] = np.asarray(saved["pred_z"])[take]
        donor_ids, residual = relative_source_residual_grid(truth, oof, donor_cells)
        library = SourceDOCInnovationLibrary().fit(names[donor_ids], months,
            np.asarray(ecology)[donor_ids], residual)
        rms = source_key_rms(library)
        arrays = source_state_key_candidates(library, names[query], np.asarray(ecology)[query],
            months, names[ids], bank_hydro, rms=rms)
        if output is None:
            output = {key: np.zeros((len(ids), *values.shape[1:]), dtype=values.dtype)
                for key, values in arrays.items() if key != "donor_hydro_bank"}
            output["donor_hydro_bank"] = arrays["donor_hydro_bank"]
        positions = np.searchsorted(ids, query)
        for key in output:
            if key != "donor_hydro_bank":
                output[key][positions] = arrays[key]
        records.append({"query_fold": a, "query_station_ids": query.tolist(),
            "library_station_ids": donor_ids.tolist(), "sigma": library.sigma_,
            "query_labels_in_library_or_reference": False,
            "source_value_units": "dimensionless OOF log1p residual including seasonal mean",
            "current_source_requires_previous_observation": False, "source_key_rms": rms})
    return output, records


class SourceDOCKeyAttentionResidual(AvailableSourceAttentionResidual):
    """Append current and seasonal source-state keys; retain the original model."""

    def _install_head(self):
        super()._install_head()
        old = self.head.key
        new = nn.Linear(old.in_features+2, old.out_features, bias=False, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features].copy_(old.weight)
        self.head.key = new
        added = old.out_features*2
        self.attention_parameter_count_ += added
        self.trainable_parameter_count_ += added
        self._initial_attention_head = _state_copy(self.head)

    def _prepare_inputs(self, inputs):
        arrays = super()._prepare_inputs(inputs)
        if "donor_doc_key" not in inputs:
            raise ValueError("source-only DOC key channels required")
        keys = torch.as_tensor(inputs["donor_doc_key"], dtype=self.dtype).detach().cpu()
        if keys.shape != (*arrays["donor_values"].shape, 2) or not torch.isfinite(keys).all():
            raise ValueError("finite candidate/calendar-aligned source DOC keys required")
        if torch.count_nonzero(keys[..., -1, :]) or torch.count_nonzero(keys[~arrays["donor_valid"]]):
            raise ValueError("invalid donors and zero prior cannot carry source-state keys")
        arrays["donor_doc_key"] = keys
        return arrays

    def _attention_cells(self, inputs, cells, hidden):
        months = inputs["age"].shape[1]
        station, month = cells//months, cells % months
        owner = inputs["donor_owner"][station]
        daily = inputs["donor_hydro_bank"][owner.clamp_min(0), month[:, None]]
        keys = torch.cat([inputs["donor_ecology"][station], daily*(owner >= 0)[..., None],
                          inputs["donor_doc_key"][station, month]], -1)
        start = self.attention_config["daily_start"]
        query = torch.cat([hidden, inputs["env"][station],
                           inputs["extra"][station, month, start:start+8]], -1)
        state, weights = self.head.allocate(query, keys, inputs["donor_values"][station, month],
            inputs["donor_valid"][station, month], inputs["donor_log_prior"][station])
        return state*(1+inputs["attention_reference"][station, month, None]), weights

    def to_dict(self):
        summary = super().to_dict()
        summary["source_state_key_norm"] = float(self.head.key.weight[:, -2:].detach().double().norm())
        summary["protocol"].update({
            "source_DOC_key": "current OOF log1p residual and source seasonal mean; frozen source-only RMS",
            "new_source_key_initialization": "zero new coefficients; original query/key retained",
            "receiving_water_quality": "none; existing observation-aware GRU and reference"})
        return summary
