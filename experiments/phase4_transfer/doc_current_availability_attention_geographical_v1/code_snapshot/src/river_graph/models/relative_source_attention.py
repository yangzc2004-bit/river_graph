"""Transfer dimensionless source departures through the existing native head."""
from __future__ import annotations

import numpy as np
import torch

from river_graph.models.current_source_attention import CurrentSourceAttentionResidual
from river_graph.models.current_source_candidates import source_attention_candidates
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary


def relative_source_residual_grid(truth, oof_log, source_cells):
    """Select allowed cells before transforming source truth to relative error."""
    truth, oof, cells = np.asarray(truth), np.asarray(oof_log), np.asarray(source_cells)
    if (truth.ndim != 2 or oof.shape != truth.shape or cells.ndim != 1
            or cells.dtype.kind not in "iu" or not len(cells)
            or (cells < 0).any() or (cells >= truth.size).any()
            or len(np.unique(cells)) != len(cells)):
        raise ValueError("aligned grids and unique permitted source cells required")
    y, base = truth.ravel()[cells], oof.ravel()[cells]
    if not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(base).all():
        raise ValueError("finite nonnegative source DOC and finite OOF log prediction required")
    ids = np.unique(cells//truth.shape[1])
    residual = np.full((len(ids), truth.shape[1]), np.nan)
    residual[np.searchsorted(ids, cells//truth.shape[1]), cells % truth.shape[1]] = np.log1p(y)-base
    return ids, residual


def nested_relative_source_candidates(truth, station_names, months, ecology, source_cells,
                                      folds, references, bank_hydro):
    """Build log-departure donors excluding both query and donor reference folds."""
    truth, source = np.asarray(truth), np.asarray(source_cells)
    if truth.ndim != 2 or source.ndim != 1 or source.dtype.kind not in "iu" or not len(source):
        raise ValueError("source grids and integer cells required")
    t = truth.shape[1]
    ids = np.unique(source//t)
    folds = [np.asarray(fold, dtype=np.int64) for fold in folds]
    if len(folds) < 3 or not np.array_equal(np.sort(np.concatenate(folds)), ids):
        raise ValueError("disjoint folds must partition all source stations")
    names = np.asarray(station_names, str)
    output, records = None, []
    for a, query in enumerate(folds):
        donor_cells = source[~np.isin(source//t, query)]
        oof = np.full(truth.shape, np.nan)
        for b, donor in enumerate(folds):
            if a == b:
                continue
            saved = references[tuple(sorted((a, b)))]
            hidden = set(query.tolist()+donor.tolist())
            if (set(saved["hidden_stations"]) != hidden
                    or set(saved["fitted_stations"]) != set(ids.tolist())-hidden):
                raise ValueError("relative donor reference must exclude query AND donor folds")
            cells = np.asarray(saved["cells"])
            expected = source[np.isin(source//t, list(hidden))]
            if not np.array_equal(cells, expected) or np.shape(saved["pred_z"]) != cells.shape:
                raise ValueError("relative reference population changed")
            selected = np.isin(cells//t, donor)
            oof.ravel()[cells[selected]] = np.asarray(saved["pred_z"])[selected]
        donor_ids, residual = relative_source_residual_grid(truth, oof, donor_cells)
        library = SourceDOCInnovationLibrary().fit(names[donor_ids], months, np.asarray(ecology)[donor_ids], residual)
        candidates = source_attention_candidates(library, names[query], np.asarray(ecology)[query],
                                                 months, names[ids], bank_hydro)
        if output is None:
            output = {key: np.zeros((len(ids), *values.shape[1:]), dtype=values.dtype)
                      for key, values in candidates.items() if key != "donor_hydro_bank"}
            output["donor_hydro_bank"] = candidates["donor_hydro_bank"]
        positions = np.searchsorted(ids, query)
        for key in output:
            if key != "donor_hydro_bank":
                output[key][positions] = candidates[key]
        records.append({"query_fold": a, "query_station_ids": query.tolist(),
                        "library_station_ids": donor_ids.tolist(), "sigma": library.sigma_,
                        "query_labels_in_library_or_reference": False,
                        "source_value_units": "dimensionless seasonal log1p residual"})
    return output, records


class RelativeSourceAttentionResidual(CurrentSourceAttentionResidual):
    """Keep native training while scaling relative donor states by a frozen base."""

    def _prepare_inputs(self, inputs):
        arrays = super()._prepare_inputs(inputs)
        if "attention_reference" not in inputs:
            raise ValueError("relative source attention needs the frozen receiving reference")
        base = torch.as_tensor(inputs["attention_reference"], dtype=self.dtype).detach().cpu()
        if base.shape != arrays["age"].shape or not torch.isfinite(base).all() or (base < 0).any():
            raise ValueError("receiving reference must be finite nonnegative and station/month aligned")
        arrays["attention_reference"] = base
        return arrays

    def _attention_cells(self, inputs, cells, hidden):
        state, weights = super()._attention_cells(inputs, cells, hidden)
        t = inputs["age"].shape[1]
        scale = 1 + inputs["attention_reference"][cells//t, cells % t]
        return state*scale[:, None], weights

    def to_dict(self):
        summary = super().to_dict()
        summary["protocol"].update({
            "source_values": "double-held-fold OOF log1p residual minus source seasonal mean",
            "source_value_units": "dimensionless",
            "attention_state_units": "native mg/L; relative state multiplied by one plus receiving reference",
            "receiving_reference": "saved station-OOF for training; source-fit for validation/inference",
            "value_conversion": "first-order log1p-to-native residual; no concentration conservation claim"})
        return summary
