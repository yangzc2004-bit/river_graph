"""Current donor value identity, double-held exclusion and model compatibility."""
from copy import deepcopy

import numpy as np
from test_current_source_attention import _model
from test_source_innovation_training import _nested_fixture

from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
    nested_current_available_candidates,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_level_attention import (
    SourceLevelAttentionResidual,
    source_level_candidates,
)


def test_current_sources_do_not_require_a_previous_year_and_keep_observed_values():
    args = _nested_fixture()
    ids, residual = relative_source_residual_grid(args[0], np.zeros_like(args[0]), args[4])
    library = SourceDOCInnovationLibrary().fit(args[1][ids], args[2], args[3][ids], residual)
    hydro = np.zeros((len(ids), 36, 8))
    old = source_level_candidates(library, args[1][6:], args[3][6:], args[2], args[1][ids], hydro)
    new = current_available_candidates(library, args[1][6:], args[3][6:], args[2], args[1][ids], hydro)
    for key in ("donor_owner", "donor_ecology", "donor_log_prior", "donor_hydro_bank"):
        np.testing.assert_array_equal(old[key], new[key])
    np.testing.assert_array_equal(new["donor_previous_valid"], old["donor_valid"])
    assert new["donor_valid"][:, :12, :-1].sum() > old["donor_valid"][:, :12, :-1].sum()
    np.testing.assert_allclose(new["donor_values_real"][old["donor_valid"]],
                               old["donor_values_real"][old["donor_valid"]], atol=1e-15)
    owners = new["donor_owner"][0, :-1]
    positions = np.flatnonzero(owners >= 0)
    np.testing.assert_allclose(new["donor_values_real"][0][:, positions],
                               residual[owners[positions]].T, atol=1e-15)
    assert not new["donor_values_real"][0][:, new["donor_owner"][0] < 0].any()
    assert not new["donor_values_real"][~new["donor_valid"]].any()
    assert not new["donor_values_real"][..., -1].any()


def test_current_library_never_reads_query_or_receiver_doc():
    args = _nested_fixture()
    hydro = np.zeros((6, 36, 8))
    original, records = nested_current_available_candidates(*args, hydro)
    altered = deepcopy(args)
    altered[0][:2] += 1000.
    altered[0][6] = np.nan
    changed, _ = nested_current_available_candidates(*altered, hydro)
    for key in original:
        if key != "donor_hydro_bank":
            np.testing.assert_array_equal(original[key][:2], changed[key][:2])
    assert not np.isin(original["donor_owner"][:2], [0, 1]).any()
    assert all(not set(row["query_station_ids"]) & set(row["library_station_ids"]) for row in records)


def test_unchanged_operator_forward_training_and_reload():
    current, _, args = _model()
    for view in (args[0], args[4]):
        view["attention_reference"] = np.full(view["age"].shape, 3.)
    settings = {**current._config(), **current.attention_config}
    old = SourceLevelAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    new = AvailableSourceAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    assert old.trainable_parameter_count_ == new.trainable_parameter_count_
    np.testing.assert_array_equal(old.predict(args[4], args[5]), new.predict(args[4], args[5]))
    new.fit(*args, tail_threshold=4., selection_role="source_validation")
    restored = AvailableSourceAttentionResidual.from_payload(new.to_payload())
    np.testing.assert_array_equal(new.predict(args[4], args[5]), restored.predict(args[4], args[5]))
