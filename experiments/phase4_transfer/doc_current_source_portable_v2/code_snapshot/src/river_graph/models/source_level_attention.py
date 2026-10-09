"""Transfer full relative source residuals with the existing attention operator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.models.current_source_candidates import source_attention_candidates
from river_graph.models.relative_source_attention import (
    RelativeSourceAttentionResidual,
    relative_source_residual_grid,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary


def source_level_candidates(library, receiver_names, receiver_ecology, months, bank_names, bank_hydro):
    """Restore donor seasonal means without changing availability or candidates."""
    arrays = source_attention_candidates(library, receiver_names, receiver_ecology,
                                        months, bank_names, bank_hydro)
    seasonal = np.zeros_like(arrays["donor_values_real"])
    month_of_year = pd.PeriodIndex(months, freq="M").month.to_numpy()-1
    source_lookup = {name: i for i, name in enumerate(library.source_names_)}
    bank_names = np.asarray(bank_names, str)
    for row, owners in enumerate(arrays["donor_owner"]):
        positions = np.flatnonzero(owners >= 0)
        source_ids = np.array([source_lookup[bank_names[owners[c]]] for c in positions], int)
        means = library.seasonal_mean_[source_ids][:, month_of_year].T
        seasonal[row][:, positions] = np.where(arrays["donor_valid"][row][:, positions], means, 0.)
    arrays["donor_values_seasonal"] = seasonal
    arrays["donor_values_real"] += seasonal
    arrays["donor_values_historical"] += seasonal
    return arrays


def nested_source_level_candidates(truth, station_names, months, ecology, source_cells,
                                   folds, references, bank_hydro):
    """Fit each seasonal source bank with the query and donor reference folds out."""
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
                raise ValueError("source level reference must exclude query AND donor folds")
            cells = np.asarray(saved["cells"])
            expected = source[np.isin(source//t, list(hidden))]
            if not np.array_equal(cells, expected) or np.shape(saved["pred_z"]) != cells.shape:
                raise ValueError("source level reference population changed")
            selected = np.isin(cells//t, donor)
            oof.ravel()[cells[selected]] = np.asarray(saved["pred_z"])[selected]
        donor_ids, residual = relative_source_residual_grid(truth, oof, donor_cells)
        library = SourceDOCInnovationLibrary().fit(names[donor_ids], months,
            np.asarray(ecology)[donor_ids], residual)
        candidates = source_level_candidates(library, names[query], np.asarray(ecology)[query],
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
            "source_value_units": "dimensionless OOF log1p residual including seasonal mean"})
    return output, records


def source_level_input_view(inputs, candidates, mode):
    if mode not in ("real", "historical", "seasonal"):
        raise ValueError("real, historical or seasonal source values required")
    return {**inputs, **{key: value for key, value in candidates.items()
                        if not key.startswith("donor_values_")},
            "donor_values": candidates[f"donor_values_{mode}"]}


class SourceLevelAttentionResidual(RelativeSourceAttentionResidual):
    """Unchanged first-order head, with caller-frozen full or seasonal values."""

    def to_dict(self):
        summary = super().to_dict()
        summary["protocol"].update({
            "source_values": "double-held-fold OOF log1p residual including source seasonal mean",
            "seasonal_control": "individual donor mean only; retained aggregate inputs unchanged"})
        return summary
