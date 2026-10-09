"""Source-state key normalization, isolation, compatibility and causality."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_current_source_attention import _model
from test_source_innovation_training import _nested_fixture

from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_value_key_attention import (
    SourceDOCKeyAttentionResidual,
    nested_source_state_key_candidates,
    source_key_rms,
    source_state_key_candidates,
    source_state_key_view,
)


def test_source_rms_and_keys_never_use_future_at_inference():
    args = _nested_fixture()
    ids, residual = relative_source_residual_grid(args[0], np.zeros_like(args[0]), args[4])
    library = SourceDOCInnovationLibrary().fit(args[1][ids], args[2], args[3][ids], residual)
    rms = source_key_rms(library)
    np.testing.assert_allclose(rms, np.sqrt(np.mean(residual**2)))
    hydro = np.zeros((6, 36, 8))
    original = source_state_key_candidates(library, args[1][6:], args[3][6:], args[2],
                                          args[1][ids], hydro, rms=rms)
    altered = deepcopy(library)
    altered.innovations_[:, 16:] += 1000.
    future = source_state_key_candidates(altered, args[1][6:], args[3][6:], args[2],
                                        args[1][ids], hydro, rms=rms)
    np.testing.assert_array_equal(original["donor_doc_key"][:, :16], future["donor_doc_key"][:, :16])
    with pytest.raises(ValueError, match="positive frozen"):
        source_state_key_candidates(library, args[1][6:], args[3][6:], args[2], args[1][ids], hydro, rms=0.)
    for mode in ("current", "seasonal", "zero"):
        view = source_state_key_view({}, original, mode)
        np.testing.assert_array_equal(view["donor_values"], original["donor_values_real"])
        np.testing.assert_array_equal(view["donor_valid"], original["donor_valid"])
        assert not view["donor_doc_key"][~view["donor_valid"]].any()
    assert not source_state_key_view({}, original, "seasonal")["donor_doc_key"][..., 0].any()
    assert not source_state_key_view({}, original, "zero")["donor_doc_key"].any()


def test_query_and_receiving_labels_cannot_change_source_keys_or_normalizers():
    args = _nested_fixture()
    hydro = np.zeros((6, 36, 8))
    original, records = nested_source_state_key_candidates(*args, hydro)
    changed = deepcopy(args)
    changed[0][:2] += 1000.
    changed[0][6] = np.nan
    altered, changed_records = nested_source_state_key_candidates(*changed, hydro)
    for key in original:
        if key != "donor_hydro_bank":
            np.testing.assert_array_equal(original[key][:2], altered[key][:2])
    assert records[0] == changed_records[0]
    assert not np.isin(original["donor_owner"][:2], [0, 1]).any()
    assert all(not set(r["query_station_ids"]) & set(r["library_station_ids"]) for r in records)


def _key_model():
    current, _, args = _model()
    settings = {**current._config(), **current.attention_config}
    old = AvailableSourceAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    new = SourceDOCKeyAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    rng = np.random.default_rng(501)
    for position in (0, 4):
        view = args[position]
        view["attention_reference"] = np.full(view["age"].shape, 3.)
        key = rng.normal(size=(*view["donor_values"].shape, 2))
        key[~view["donor_valid"]] = 0.
        key[..., -1, :] = 0.
        view["donor_doc_key"] = key
    return old, new, args


def test_zero_coefficients_preserve_original_attention_and_finite_gradient():
    old, new, args = _key_model()
    assert new.trainable_parameter_count_ == old.trainable_parameter_count_+128
    with torch.no_grad():
        old.head.output.weight.fill_(.1)
        new.head.output.weight.copy_(old.head.output.weight)
    np.testing.assert_allclose(old.predict(args[4], args[5]), new.predict(args[4], args[5]), rtol=0, atol=1e-14)
    arrays = new._prepare_inputs(args[4])
    new._delta_cells(arrays, torch.tensor([4, 15])).sum().backward()
    assert torch.isfinite(new.head.key.weight.grad).all()
    assert torch.count_nonzero(new.head.key.weight.grad[:, -2:]) > 0


def test_training_reload_new_receivers_and_future_source_keys():
    _, model, args = _key_model()
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    restored = SourceDOCKeyAttentionResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(model.predict(args[4], args[5]), restored.predict(args[4], args[5]))
    future = deepcopy(args[4])
    future["donor_doc_key"][:, 8:, :-1] += 100.
    future["donor_doc_key"][~future["donor_valid"]] = 0.
    np.testing.assert_array_equal(restored.predict_delta(args[4])[:, :8], restored.predict_delta(future)[:, :8])
    more = deepcopy(args[4])
    for key, values in more.items():
        if key != "donor_hydro_bank":
            more[key] = np.concatenate([values, values[:1]], axis=0)
    assert restored.predict_delta(more).shape == (3, 14)
