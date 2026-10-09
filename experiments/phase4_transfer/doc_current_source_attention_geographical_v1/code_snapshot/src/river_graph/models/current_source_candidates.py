"""Matched individual donors for current-source attention, without new RF fits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)


def source_attention_candidates(library, receiver_names, receiver_ecology, months, bank_names, bank_hydro):
    """Expose the same20 matched donors and zero prior as the fixed library."""
    names = np.asarray(receiver_names, str)
    eco = np.asarray(receiver_ecology, float)
    bank_names, hydro = np.asarray(bank_names, str), np.asarray(bank_hydro, float)
    if (bank_names.ndim != 1 or len(np.unique(bank_names)) != len(bank_names)
            or not np.isin(library.source_names_, bank_names).all()
            or hydro.shape != (len(bank_names), len(months), 8) or not np.isfinite(hydro).all()):
        raise ValueError("unique aligned source names and finite monthly hydrology required")
    parts = library.predict_components(names, eco, months)
    n, t, c = len(names), len(months), library.candidate_count + 1
    arrays = {"donor_owner": np.full((n, c), -1, dtype=np.int64),
              "donor_ecology": np.zeros((n, c, eco.shape[1])),
              "donor_log_prior": np.zeros((n, c)),
              "donor_valid": np.zeros((n, t, c), dtype=bool),
              "donor_values_real": np.zeros((n, t, c)),
              "donor_values_historical": np.zeros((n, t, c)),
              "donor_hydro_bank": hydro.copy()}
    arrays["donor_valid"][..., -1] = True
    past, dates = library.historical_control()
    source_names = {name: i for i, name in enumerate(library.source_names_)}
    bank_lookup = {name: i for i, name in enumerate(bank_names)}
    source_months = {int(month): i for i, month in enumerate(library.month_ordinals_)}
    ordinals = pd.PeriodIndex(months, freq="M").asi8
    for i, record in enumerate(parts["donors"]):
        selected = np.array([source_names[name] for name in record["source_stations"]], dtype=np.int64)
        size = len(selected)
        arrays["donor_owner"][i, :size] = [bank_lookup[name] for name in record["source_stations"]]
        arrays["donor_ecology"][i, :size] = library.ecology_[selected]
        arrays["donor_log_prior"][i, :size] = -np.asarray(record["distances"])/library.sigma_
        for month, ordinal in enumerate(ordinals):
            s = source_months.get(int(ordinal))
            if s is None:
                continue
            current, historical = library.innovations_[selected, s], past[selected, s]
            matched = np.isfinite(current) & np.isfinite(historical)
            if (dates[selected[matched], s] >= ordinal).any():
                raise ValueError("historical donors must precede the receiving month")
            arrays["donor_valid"][i, month, :size] = matched
            arrays["donor_values_real"][i, month, :size] = np.where(matched, current, 0.)
            arrays["donor_values_historical"][i, month, :size] = np.where(matched, historical, 0.)
    weights = np.exp(arrays["donor_log_prior"][:, None]) * arrays["donor_valid"]
    weights /= weights.sum(-1, keepdims=True)
    for mode in ("real", "historical"):
        np.testing.assert_allclose((weights*arrays[f"donor_values_{mode}"]).sum(-1),
                                   parts[f"{mode}_innovation"], rtol=1e-12, atol=1e-12)
    return arrays


def nested_source_attention_candidates(truth, station_names, months, ecology, source_cells,
                                       folds, references, bank_hydro):
    """Build each query fold's donors using references excluding query AND donor."""
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
                raise ValueError("donor reference must exclude query AND donor folds")
            cells = np.asarray(saved["cells"])
            expected = source[np.isin(source//t, list(hidden))]
            if not np.array_equal(cells, expected) or np.shape(saved["pred_z"]) != cells.shape:
                raise ValueError("nested reference population changed")
            selected = np.isin(cells//t, donor)
            oof.ravel()[cells[selected]] = np.asarray(saved["pred_z"])[selected]
        donor_ids, residual = source_residual_grid(truth, oof, donor_cells)
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
                        "query_labels_in_library_or_reference": False})
    return output, records


def attention_input_view(inputs, candidates, mode):
    if mode not in ("real", "historical"):
        raise ValueError("real or historical source mode required")
    return {**inputs, **{key: values for key, values in candidates.items()
                        if not key.startswith("donor_values_")},
            "donor_values": candidates[f"donor_values_{mode}"]}
