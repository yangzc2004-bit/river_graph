"""Retained seasonal values, double-held exclusion and compatible source heads."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from test_current_source_attention import _model
from test_source_innovation_training import _nested_fixture

from river_graph.models.current_source_candidates import source_attention_candidates
from river_graph.models.relative_source_attention import (
    RelativeSourceAttentionResidual,
    nested_relative_source_candidates,
    relative_source_residual_grid,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_level_attention import (
    SourceLevelAttentionResidual,
    nested_source_level_candidates,
    source_level_candidates,
    source_level_input_view,
)


def test_source_values_recover_full_residual_without_changing_donors():
    args = _nested_fixture()
    ids, residual = relative_source_residual_grid(args[0], np.zeros_like(args[0]), args[4])
    library = SourceDOCInnovationLibrary().fit(args[1][ids], args[2], args[3][ids], residual)
    hydro = np.zeros((len(ids), 36, 8))
    old = source_attention_candidates(library, args[1][6:], args[3][6:], args[2], args[1][ids], hydro)
    full = source_level_candidates(library, args[1][6:], args[3][6:], args[2], args[1][ids], hydro)
    for name in old:
        if not name.startswith("donor_values_"):
            np.testing.assert_array_equal(old[name], full[name])
    for mode in ("real", "historical"):
        np.testing.assert_allclose(full[f"donor_values_{mode}"]-full["donor_values_seasonal"],
                                   old[f"donor_values_{mode}"], atol=1e-15)
    owners = full["donor_owner"][0, :-1]
    expected = np.zeros_like(full["donor_values_real"])
    expected[0, :, :-1] = np.where(full["donor_valid"][0, :, :-1], residual[owners].T, 0.)
    np.testing.assert_allclose(full["donor_values_real"], expected, atol=1e-15)
    assert not full["donor_values_seasonal"][~full["donor_valid"]].any()
    assert not full["donor_values_seasonal"][..., -1].any()
    np.testing.assert_array_equal(source_level_input_view({}, full, "seasonal")["donor_values"],
                                  full["donor_values_seasonal"])


def test_nested_source_means_do_not_use_query_or_receiver_labels():
    args = _nested_fixture()
    hydro = np.zeros((6, 36, 8))
    candidates, records = nested_source_level_candidates(*args, hydro)
    changed = deepcopy(args)
    changed[0][:2] += 1000.
    changed[0][6] = np.nan
    altered, _ = nested_source_level_candidates(*changed, hydro)
    old, _ = nested_relative_source_candidates(*args, hydro)
    for name in candidates:
        if name != "donor_hydro_bank":
            np.testing.assert_array_equal(candidates[name][:2], altered[name][:2])
        if not name.startswith("donor_values_"):
            np.testing.assert_array_equal(candidates[name], old[name])
    assert not np.isin(candidates["donor_owner"][:2], [0, 1]).any()
    assert all(not set(row["query_station_ids"]) & set(row["library_station_ids"]) for row in records)


def test_same_operator_zero_head_reload_new_nodes_and_future_input():
    current, _, args = _model()
    for inputs in (args[0], args[4]):
        inputs["attention_reference"] = np.full(inputs["age"].shape, 3.)
    settings = {**current._config(), **current.attention_config}
    relative = RelativeSourceAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    full = SourceLevelAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    assert full.trainable_parameter_count_ == relative.trainable_parameter_count_
    np.testing.assert_array_equal(full.predict(args[4], args[5]), relative.predict(args[4], args[5]))
    full.fit(*args, tail_threshold=4., selection_role="source_validation")
    loaded = SourceLevelAttentionResidual.from_payload(full.to_payload())
    np.testing.assert_array_equal(full.predict(args[4], args[5]), loaded.predict(args[4], args[5]))
    future = deepcopy(args[4])
    for key in ("raw", "extra", "donor_values", "attention_reference"):
        future[key][:, 8:] += 100.
    future["donor_values"][..., -1] = 0.
    future["donor_values"][~future["donor_valid"]] = 0.
    future["donor_hydro_bank"][:, 8:] += 100.
    np.testing.assert_array_equal(full.predict_delta(args[4])[:, :8], loaded.predict_delta(future)[:, :8])
    extended = deepcopy(args[4])
    for key in extended:
        if key != "donor_hydro_bank":
            extended[key] = np.concatenate([extended[key], extended[key][:1]], axis=0)
    assert loaded.predict_delta(extended).shape == (3, 14)
    assert "including source seasonal mean" in loaded.to_dict()["protocol"]["source_values"]
