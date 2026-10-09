"""Current donor exclusion, causal gathering and matched attention controls."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_encoder_native_residual import fixture
from test_source_innovation_training import _nested_fixture

from river_graph.models.current_source_attention import CurrentSourceAttentionResidual
from river_graph.models.current_source_candidates import (
    attention_input_view,
    nested_source_attention_candidates,
    source_attention_candidates,
)
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary


def _model(mode="learned", epochs=2):
    old, args, *_ = fixture("last_self_ecology", epochs=epochs)
    settings = {**old._config(), "extra_dim": 38}
    old = EncoderNativeResidual(old.spatial, old.temporal, old.decay, **settings)
    model = CurrentSourceAttentionResidual(old.spatial, old.temporal, old.decay,
                                          attention_mode=mode, **settings)
    args = list(deepcopy(args))
    rng = np.random.default_rng(110)
    for i in (0, 4):
        inputs = args[i]
        n, t = inputs["age"].shape
        inputs["extra"] = np.concatenate([inputs["extra"], rng.uniform(0, 1, (n, t, 8))], -1)
        inputs.update({"donor_owner": np.tile([0, 1, -1], (n, 1)),
            "donor_ecology": np.tile([[.1, .3], [.7, .2], [0., 0.]], (n, 1, 1)),
            "donor_log_prior": np.tile([-1., -.3, 0.], (n, 1)),
            "donor_hydro_bank": rng.uniform(0, 1, (2, t, 8)),
            "donor_valid": np.ones((n, t, 3), bool),
            "donor_values": np.tile([2., .5, 0.], (n, t, 1))})
        inputs["donor_valid"][:, 0, :2] = False
        inputs["donor_values"][:, 0, :2] = 0
    return model, old, args


def test_zero_projection_exact_old_head_and_prior_without_donors():
    model, old, args = _model()
    with torch.no_grad():
        old.head.weight.fill_(.07)
        old.head.bias.fill_(.1)
        model.head.linear.load_state_dict(old.head.state_dict())
    np.testing.assert_array_equal(model.predict_delta(args[4]), old.predict_delta(args[4]))
    diagnostics = model.diagnostics(args[4], np.array([0, 14, 4]))
    np.testing.assert_array_equal(diagnostics["prior_mass"][:2], np.ones((2, 2)))
    np.testing.assert_array_equal(diagnostics["current_source_state"][:2], np.zeros((2, 2)))
    assert model.trainable_parameter_count_ == old.trainable_parameter_count_ + (4+2+8)*64+(2+8)*64+2


@pytest.mark.parametrize("mode", ["learned", "fixed_prior"])
def test_training_save_reload_new_node_count_and_future_inputs(mode):
    model, old, args = _model(mode)
    np.testing.assert_array_equal(model.head.linear.state_dict()["weight"], old.head.weight.detach())
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    assert model.optimizer_steps_ > 0 and model.to_dict()["attention_output_norm"] > 0
    assert (model.to_dict()["query_parameter_change"] > 0) == (mode == "learned")
    assert (model.to_dict()["key_parameter_change"] > 0) == (mode == "learned")
    restored = CurrentSourceAttentionResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(model.predict(args[4], args[5]), restored.predict(args[4], args[5]))
    extended = deepcopy(args[4])
    for key in extended:
        if key != "donor_hydro_bank":
            extended[key] = np.concatenate([extended[key], extended[key][:1]], axis=0)
    assert restored.predict_delta(extended).shape == (3, 14)
    future = deepcopy(args[4])
    for key in ("raw", "extra", "donor_values"):
        future[key][:, 8:] += 100.
    future["donor_values"][..., -1] = 0.
    future["donor_values"][~future["donor_valid"]] = 0.
    future["donor_hydro_bank"][:, 8:] += 100.
    np.testing.assert_array_equal(model.predict_delta(args[4])[:, :8], model.predict_delta(future)[:, :8])


def test_attention_normalization_fixed_ecological_weights_and_gradients():
    model, _, args = _model("fixed_prior")
    arrays = model._prepare_inputs(args[4])
    cells = torch.tensor([0, 4, 15])
    hidden = model._hidden_cells(arrays, cells)
    state, weights = model._attention_cells(arrays, cells, hidden)
    torch.testing.assert_close(weights.sum(-1), torch.ones((3, 2), dtype=model.dtype), rtol=0, atol=1e-15)
    expected = np.exp([-1., -.3, 0.])
    expected /= expected.sum()
    np.testing.assert_allclose(weights.detach().numpy()[1:], np.tile(expected, (2, 2, 1)), atol=1e-15)
    with torch.no_grad():
        model.head.output.weight.fill_(.1)
    model._delta_cells(arrays, cells).sum().backward()
    for parameter in model.head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(model.head.query.weight.grad) == 0
    assert torch.count_nonzero(model.head.key.weight.grad) == 0
    np.testing.assert_array_equal(state.detach().numpy()[0], np.zeros(2))


def test_nested_donor_library_excludes_query_fold_and_receiving_truth():
    args = _nested_fixture()
    hydro = np.arange(6*36*8.).reshape(6, 36, 8)/100
    candidates, records = nested_source_attention_candidates(*args, hydro)
    changed = deepcopy(args)
    changed[0][6] = np.nan
    changed[0][:2] += 1000.
    altered, _ = nested_source_attention_candidates(*changed, hydro)
    for key in candidates:
        if key == "donor_hydro_bank":
            np.testing.assert_array_equal(candidates[key], altered[key])
        else:
            np.testing.assert_array_equal(candidates[key][:2], altered[key][:2])
    assert not np.isin(candidates["donor_owner"][:2], [0, 1]).any()
    assert all(not set(row["query_station_ids"]) & set(row["library_station_ids"]) for row in records)
    leaked = deepcopy(args)
    leaked[-1][(0, 1)]["fitted_stations"].append(0)
    with pytest.raises(ValueError, match="query AND donor"):
        nested_source_attention_candidates(*leaked, hydro)


def test_candidate_values_match_fixed_library_and_hide_missing_values():
    args = _nested_fixture()
    library = SourceDOCInnovationLibrary().fit(args[1][:6], args[2], args[3][:6], args[0][:6])
    bank = np.zeros((6, 36, 8))
    candidates = source_attention_candidates(library, args[1][6:], args[3][6:], args[2], args[1][:6], bank)
    real = attention_input_view({}, candidates, "real")
    past = attention_input_view({}, candidates, "historical")
    np.testing.assert_array_equal(real["donor_valid"], past["donor_valid"])
    assert not np.count_nonzero(real["donor_values"][~real["donor_valid"]])
    assert real["donor_valid"][..., -1].all()
    model, _, fit_args = _model()
    broken = deepcopy(fit_args[4])
    broken["donor_values"][0, 0, 0] = 999.
    with pytest.raises(ValueError, match="invalid donors"):
        model.predict_delta(broken)
