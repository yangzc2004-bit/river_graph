"""Chemical support coordinates use source variation and row-local inference."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from river_graph.models.chemical_support_basis import ChemicalSupportBasis
from river_graph.models.nonlinear_chemistry_head import NonlinearChemistryHead


@pytest.fixture
def problem():
    rng = np.random.default_rng(29)
    aux = np.zeros((4, 8, 4), dtype=np.float32)
    aux[..., 0] = rng.uniform(.3, .8, (4, 8))
    aux[..., 1] = rng.uniform(2, 7, (4, 8))
    aux[..., 2:] = 1
    aux[0, :2, (0, 2)] = 0
    aux[1, 3:5, (1, 3)] = 0
    aux[2, 5] = 0
    aux[3, 1] = 0
    features = np.zeros((16, 554), dtype=np.float32)
    features[:, -4:] = aux[:2].reshape(-1, 4)
    base = np.full(16, 5.)
    active = np.any(features[:, -2:] != 0, axis=1)
    head = NonlinearChemistryHead(epochs=0, seed=42).fit(
        features, base, base, features, base, base,
        tail_threshold=9., source_active=active, validation_active=active)
    return aux, head


def independent_phi(aux, head, mode="chemistry"):
    values = aux.astype(np.float64).copy()
    values[..., :2] = np.where(values[..., 2:] != 0, values[..., :2], 0)
    if mode == "masks":
        values[..., :2] = 0
    x = torch.as_tensor(values.reshape(-1, 4), dtype=torch.float64)
    x = (x - head.feature_mean_[-4:]) / head.feature_scale_[-4:]
    with torch.no_grad():
        phi = F.silu(head.head.phi(x)).numpy().reshape(*aux.shape[:2], 8)
    return phi, np.any(aux[..., 2:] != 0, axis=-1)


def test_source_covariance_and_global_projection_are_distinct(problem):
    aux, head = problem
    model = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training")
    phi, active = independent_phi(aux, head)
    centered = np.concatenate([phi[i, active[i]] - phi[i, active[i]].mean(0) for i in [0, 1]])
    source_rows = np.concatenate([phi[i, active[i]] for i in [0, 1]])
    np.testing.assert_allclose(model.covariance_, centered.T @ centered / len(centered), atol=1e-15)
    np.testing.assert_allclose(model.source_global_mean_, source_rows.mean(0), atol=1e-15)
    assert model.to_dict()["n_source_active_rows"] == len(centered)
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(2), atol=1e-14)
    assert np.all(np.diff(model.eigenvalues_) <= 0)
    assert all(vector[np.argmax(np.abs(vector))] > 0 for vector in model.components_)
    assert model.identified_.all()
    expected = (phi - source_rows.mean(0)) @ model.components_.T / np.sqrt(model.eigenvalues_)
    expected[~active] = 0
    np.testing.assert_allclose(model.transform(aux), expected, atol=1e-14)
    # Target-station coordinates do not get centered on their own timeline.
    assert not np.allclose(model.transform(aux)[2, active[2]].mean(0), 0)


def test_source_fit_ignores_all_target_measurements_and_source_order(problem):
    aux, head = problem
    original = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training")
    changed = aux.copy()
    changed[2:] = np.nan
    other = ChemicalSupportBasis(head).fit(changed, [1, 0], source_role="source_training")
    assert original.to_dict() == other.to_dict()
    with pytest.raises(ValueError, match="source_training"):
        ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="test")
    with pytest.raises(TypeError, match="source_role"):
        ChemicalSupportBasis(head).fit(aux, [0, 1])
    for stations in ([0, 0], [0, 4], [-1, 0], [0., 1.]):
        with pytest.raises(ValueError, match="indices"):
            ChemicalSupportBasis(head).fit(aux, stations, source_role="source_training")


def test_fixed_projection_has_no_future_or_other_station_dependence(problem):
    aux, head = problem
    model = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training")
    original = model.transform(aux)
    changed = aux.copy()
    changed[2, 6:, :2] += 3
    changed[3, :, :2] += 2
    result = model.transform(changed)
    np.testing.assert_array_equal(result[:2], original[:2])
    np.testing.assert_array_equal(result[2, :6], original[2, :6])
    assert not np.array_equal(result[2, 6:], original[2, 6:])
    # A single target month can be evaluated without the station's other months.
    np.testing.assert_allclose(model.transform(aux[2:3, 3:4]), original[2:3, 3:4], atol=1e-14)


def test_frozen_head_is_copied_without_mutating_it_or_rng(problem):
    aux, head = problem
    state = {key: value.clone() for key, value in head.head.state_dict().items()}
    rng_state = torch.random.get_rng_state().clone()
    model = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training")
    expected = model.transform(aux)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    for key, value in head.head.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in head.head.parameters())
    with torch.no_grad():
        head.head.phi.weight.add_(10)
        head.feature_mean_[-4:] += 4
    np.testing.assert_array_equal(model.transform(aux), expected)


def test_masks_mode_and_missing_placeholder_invariance(problem):
    aux, head = problem
    model = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training", mode="masks")
    expected = model.transform(aux)
    changed = aux.copy()
    changed[..., :2] += 300
    np.testing.assert_array_equal(model.transform(changed), expected)
    with pytest.raises(ValueError, match="match"):
        model.transform(aux, mode="chemistry")
    chemistry = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training")
    only_missing = aux.copy()
    only_missing[..., :2] = np.where(aux[..., 2:] != 0, aux[..., :2], 999)
    np.testing.assert_array_equal(chemistry.transform(only_missing), chemistry.transform(aux))
    assert np.all(chemistry.transform(aux)[~np.any(aux[..., 2:] != 0, axis=-1)] == 0)
    invalid = aux.copy(); invalid[0, 0, 2] = .5
    with pytest.raises(ValueError, match="binary"):
        chemistry.transform(invalid)


def test_degenerate_source_variation_is_not_declared_identified(problem):
    aux, head = problem
    constant = aux.copy()
    constant[0] = [.5, 5, 1, 1]
    constant[1] = [.7, 6, 1, 1]
    model = ChemicalSupportBasis(head).fit(constant, [0, 1], source_role="source_training")
    assert model.to_dict()["identified_components"] == 0
    np.testing.assert_array_equal(model.transform(aux), np.zeros((4, 8, 2)))
    assert np.isfinite(model.whitening_scale_).all()
    absent = aux.copy(); absent[:2] = 0
    empty = ChemicalSupportBasis(head).fit(absent, [0, 1], source_role="source_training")
    assert empty.to_dict()["n_source_active_rows"] == 0
    assert empty.to_dict()["n_source_active_stations"] == 0
    assert empty.to_dict()["identified_components"] == 0
    np.testing.assert_array_equal(empty.transform(aux), np.zeros((4, 8, 2)))


def test_json_roundtrip_and_flat_legacy_augmentation(problem):
    aux, head = problem
    model = ChemicalSupportBasis(head).fit(aux, [0, 1], source_role="source_training")
    state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = ChemicalSupportBasis.from_dict(state)
    assert restored.to_dict() == model.to_dict()
    np.testing.assert_array_equal(restored.transform(aux), model.transform(aux))
    legacy = np.arange(4*8*2, dtype=np.float64).reshape(4, 8, 2) / 100
    result = model.augment(aux, legacy.reshape(-1, 2))
    np.testing.assert_array_equal(result["augmented_basis"][..., :2], legacy)
    np.testing.assert_array_equal(result["augmented_basis"][..., 2:], result["chemical_basis"])
    np.testing.assert_array_equal(result["augmented_basis"], model.augment(aux, legacy)["augmented_basis"])
    np.testing.assert_array_equal(result["active"], np.any(aux[..., 2:] != 0, axis=-1))
    assert result["source_fit"] == state
    with pytest.raises(ValueError, match="aligned"):
        model.augment(aux, legacy[:2])
    invalid = copy.deepcopy(state); invalid["auxiliary_scale"][0] = 0
    with pytest.raises(ValueError, match="whitening"):
        ChemicalSupportBasis.from_dict(invalid)
    invalid = copy.deepcopy(state); invalid["identified"][0] = False
    with pytest.raises(ValueError, match="identification"):
        ChemicalSupportBasis.from_dict(invalid)


def test_requires_fitted_head_and_basis(problem):
    aux, head = problem
    with pytest.raises(ValueError, match="fitted"):
        ChemicalSupportBasis(NonlinearChemistryHead())
    with pytest.raises(ValueError, match="positive"):
        ChemicalSupportBasis(head, eigenvalue_floor=0)
    with pytest.raises(RuntimeError, match="fit"):
        ChemicalSupportBasis(head).transform(aux)
