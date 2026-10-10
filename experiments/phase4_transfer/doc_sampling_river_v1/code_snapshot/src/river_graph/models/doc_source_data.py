"""Nested donor-bank cross-fitting for DOC pseudo-target station episodes."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone

from river_graph.models.doc_source_retrieval import SourceResidualBank
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
)


def nested_source_episodes(forest, dataset, split, folds, *, n_jobs=2, progress=None,
                           bank_factory=SourceResidualBank, residual_callback=None):
    """Exclude pseudo-target labels from both forests and the entire donor library.

    A globally OOF donor prediction can still use the pseudo-target's labels in
    its fitted trees. Each outer episode therefore regenerates donor predictions
    inside the remaining source stations. No outer/query truth enters this bank.
    """
    months = dataset["y"].shape[1]
    train = np.asarray(split["train"], dtype=np.int64)
    truth = np.asarray(dataset["y"], dtype=float).ravel()
    records, episodes = [], []
    for outer, excluded in enumerate(folds):
        view = fold_split(split, np.asarray(excluded), months)
        donors = np.unique(view["train"]//months)
        query = train[np.isin(train//months, excluded)]
        if len(donors) < 2:
            raise ValueError("nested cross-fit requires two donor stations")
        features = build_rf_features(dataset, view, FIT_ROLES, target_transform="log1p", include_network=True)
        context = clone(forest).set_params(n_jobs=n_jobs)
        context.fit(features[view["train"]], np.log1p(truth[view["train"]]))
        query_context = np.maximum(0, np.expm1(context.predict(features[query])))
        oof = np.full(dataset["y"].shape, np.nan)
        inner_folds = np.array_split(donors, min(5, len(donors)))
        inner_records = []
        for inner, hidden in enumerate(inner_folds):
            nested = fold_split(view, hidden, months)
            held = view["train"][np.isin(view["train"]//months, hidden)]
            feature = build_rf_features(dataset, nested, FIT_ROLES, target_transform="log1p", include_network=True)
            model = clone(forest).set_params(n_jobs=n_jobs)
            model.fit(feature[nested["train"]], np.log1p(truth[nested["train"]]))
            oof.ravel()[held] = np.maximum(0, np.expm1(model.predict(feature[held])))
            inner_records.append({"inner_fold": inner, "held_stations": hidden.tolist(),
                                  "fitted_stations": np.unique(nested["train"]//months).tolist()})
            del model, feature
        if residual_callback:
            residual_callback(outer, view["train"], oof.ravel()[view["train"]], truth[view["train"]])
        bank = bank_factory().fit(np.asarray(dataset["regime"])[:, 4:13],
            np.asarray(dataset["x"]), np.asarray(dataset["x_mask"]), dataset["months"],
            view["train"], oof.ravel()[view["train"]], truth[view["train"]],
            station_names=dataset["site_no"],
            excluded_station_names=np.asarray(dataset["site_no"], str)[excluded])
        episodes.append({"bank": bank, "query_cells": query, "context": query_context})
        record = {"outer_fold": outer, "excluded_stations": np.asarray(excluded).tolist(),
                  "donor_stations": donors.tolist(), "inner_records": inner_records,
                  "n_query": len(query), "n_donor_cells": len(view["train"])}
        records.append(record)
        if progress:
            progress({"stage": "nested_source_episode", **{k: v for k, v in record.items() if k != "inner_records"}})
        del context, features, oof
    return episodes, records
