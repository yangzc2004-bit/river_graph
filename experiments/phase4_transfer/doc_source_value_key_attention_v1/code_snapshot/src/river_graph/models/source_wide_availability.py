"""Wider ecological source candidates with unchanged current-value attention."""
from __future__ import annotations

import numpy as np

from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import current_available_candidates
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary


def nested_wide_available_candidates(truth, station_names, months, ecology, source_cells,
                                        folds, references, bank_hydro, *, candidate_count=60):
    """Expand only the candidate pool; preserve double-held source references."""
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
        library = SourceDOCInnovationLibrary(candidate_count=candidate_count).fit(names[donor_ids], months,
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
            "current_source_requires_previous_observation": False, "candidate_count": candidate_count})
    return output, records

