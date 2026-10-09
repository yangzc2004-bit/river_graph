"""Use observed current donor DOC without requiring an older donor observation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_level_attention import (
    SourceLevelAttentionResidual,
    source_level_candidates,
)


def current_available_candidates(library, receiver_names, receiver_ecology, months, bank_names, bank_hydro):
    """Keep candidates/keys fixed; open every finite allowed current source value.

    Matched earlier-year availability remains a diagnostic. Source seasonal means
    were fitted from the permitted training library before receiving prediction.
    Current aggregate readout inputs are not changed by this helper.
    """
    arrays = source_level_candidates(library, receiver_names, receiver_ecology,
                                     months, bank_names, bank_hydro)
    arrays["donor_previous_valid"] = arrays["donor_valid"].copy()
    names = np.asarray(bank_names, str)
    source_ids = {name: i for i, name in enumerate(library.source_names_)}
    source_months = {int(month): i for i, month in enumerate(library.month_ordinals_)}
    calendar = pd.PeriodIndex(months, freq="M")
    for row, owners in enumerate(arrays["donor_owner"]):
        positions = np.flatnonzero(owners >= 0)
        ids = np.array([source_ids[names[owners[c]]] for c in positions], int)
        for month, ordinal in enumerate(calendar.asi8):
            index = source_months.get(int(ordinal))
            if index is None:
                continue
            current = library.innovations_[ids, index]
            valid = np.isfinite(current)
            mean = library.seasonal_mean_[ids, calendar[month].month-1]
            arrays["donor_valid"][row, month, positions] = valid
            arrays["donor_values_real"][row, month, positions] = np.where(valid, current+mean, 0.)
            arrays["donor_values_seasonal"][row, month, positions] = np.where(valid, mean, 0.)
    if (arrays["donor_previous_valid"] & ~arrays["donor_valid"]).any():
        raise ValueError("opening current source availability must retain all matched donors")
    return arrays


def nested_current_available_candidates(truth, station_names, months, ecology, source_cells,
                                        folds, references, bank_hydro):
    """Construct current-only source banks with the same double-held references."""
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
        arrays = current_available_candidates(library, names[query], np.asarray(ecology)[query],
            months, names[ids], bank_hydro)
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
            "current_source_requires_previous_observation": False})
    return output, records


class AvailableSourceAttentionResidual(SourceLevelAttentionResidual):
    """The same head and parameter count, receiving caller-frozen current donors."""

    def to_dict(self):
        summary = super().to_dict()
        summary["protocol"]["source_availability"] = "finite current source DOC; no earlier-year requirement"
        return summary
