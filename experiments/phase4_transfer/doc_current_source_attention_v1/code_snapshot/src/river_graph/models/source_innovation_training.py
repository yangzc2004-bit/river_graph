"""Double-held-out source libraries for learning a DOC innovation readout."""
from __future__ import annotations

import numpy as np

from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)


def nested_innovation_inputs(truth, station_names, months, ecology, source_cells, folds, references):
    """Build query-fold information from donors and references excluding it.

    ``references[(a,b)]`` contains selected source cell predictions from a
    forest fitted on neither fold. Its fitted/hidden roles are checked before
    any values are used. Source labels at query fold A never enter its bank.
    No receiving labels are selected by this function.
    """
    truth, source = np.asarray(truth), np.asarray(source_cells)
    if truth.ndim != 2 or source.ndim != 1 or source.dtype.kind not in "iu" or not len(source):
        raise ValueError("source grids and integer cells required")
    count_months = truth.shape[1]
    source_ids = np.unique(source//count_months)
    folds = [np.asarray(fold, dtype=np.int64) for fold in folds]
    if len(folds) < 3 or not np.array_equal(np.sort(np.concatenate(folds)), source_ids):
        raise ValueError("disjoint folds must partition all source stations")
    result = {key: np.zeros((len(source_ids), count_months), dtype=np.int64 if "count" in key else float)
              for key in ("real_innovation", "historical_innovation", "support_count",
                          "all_current_support_count", "weight_mass")}
    records = []
    for a, query_ids in enumerate(folds):
        donor_cells = source[~np.isin(source//count_months, query_ids)]
        oof = np.full(truth.shape, np.nan)
        for b, donor_ids in enumerate(folds):
            if a == b:
                continue
            saved = references[tuple(sorted((a, b)))]
            hidden, fitted = set(saved["hidden_stations"]), set(saved["fitted_stations"])
            expected_hidden = set(query_ids.tolist()+donor_ids.tolist())
            if hidden != expected_hidden or fitted != set(source_ids.tolist())-hidden:
                raise ValueError("donor reference must exclude query AND donor folds")
            cells = np.asarray(saved["cells"])
            expected_cells = source[np.isin(source//count_months, list(hidden))]
            if not np.array_equal(cells, expected_cells) or np.shape(saved["pred_z"]) != cells.shape:
                raise ValueError("nested reference cell population changed")
            selected = np.isin(cells//count_months, donor_ids)
            oof.ravel()[cells[selected]] = np.asarray(saved["pred_z"])[selected]
        donor_ids, residual = source_residual_grid(truth, oof, donor_cells)
        library = SourceDOCInnovationLibrary().fit(np.asarray(station_names, str)[donor_ids],
            months, np.asarray(ecology)[donor_ids], residual)
        parts = library.predict_components(np.asarray(station_names, str)[query_ids],
            np.asarray(ecology)[query_ids], months)
        positions = np.searchsorted(source_ids, query_ids)
        for key, values in result.items():
            values[positions] = parts[key]
        records.append({"query_fold": a, "query_station_ids": query_ids.tolist(),
            "library_station_ids": donor_ids.tolist(), "library_source_cells": len(donor_cells),
            "query_labels_in_library_or_reference": False, "sigma": library.sigma_})
    return result, records


def innovation_readout_features(parts, mode, scale):
    if mode not in ("real", "historical", "availability") or not np.isfinite(scale) or scale <= 0:
        raise ValueError("real/historical/availability mode and positive source scale required")
    key = "historical_innovation" if mode == "historical" else "real_innovation"
    innovation = np.asarray(parts[key], float)
    count, mass = np.asarray(parts["support_count"]), np.asarray(parts["weight_mass"])
    if (innovation.shape != count.shape or innovation.shape != mass.shape
            or not np.isfinite(innovation).all() or not np.isfinite(mass).all()
            or (mass < 0).any() or (count < 0).any() or (count > 20).any()):
        raise ValueError("finite aligned source innovation/support required")
    value = np.zeros_like(innovation) if mode == "availability" else innovation/scale
    return np.stack([value, count/20., mass/(1.+mass)], axis=-1).astype(np.float32)
