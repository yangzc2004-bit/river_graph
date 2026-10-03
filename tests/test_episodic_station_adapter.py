"""Scientific contracts for frozen-feature supervised episodic projection."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.models.episodic_station_adapter import (
    EpisodicStationProjector,
    _episode_queries,
    _tensor_episode_loss,
    stratified_support_months,
)
from river_graph.models.support_shape_adapter import (
    StationTemporalBasis,
    SupportShapeAdapter,
    SupportShapeEpisode,
)


def fixture_arrays():
    rng = np.random.default_rng(17)
    source = rng.normal(size=(6, 18, 4)) * np.array([5., 4., 2., 1.])
    validation = rng.normal(size=(3, 18, 4)) * np.array([5., 4., 2., 1.])
    source_base = np.full(source.shape[:2], np.log(5.))
    validation_base = np.full(validation.shape[:2], 4.)
    source_truth = np.expm1(source_base + .12 * source[..., 2] + .10 * source[..., 3])
    validation_truth = np.expm1(np.log1p(validation_base)
                                + .12 * validation[..., 2] + .10 * validation[..., 3])
    return (source, source_base, source_truth, np.ones(source.shape[:2], dtype=bool),
            validation, validation_base, validation_truth,
            np.ones(validation.shape[:2], dtype=bool))


def fitted(*, epochs=2, arrays=None):
    return EpisodicStationProjector(n_components=4, epochs=epochs, batch_size=3).fit(
        *(arrays if arrays is not None else fixture_arrays()), selection_role="source_validation")


def test_support_sampling_is_label_free_stratified_nested_and_deterministic():
    mask = np.ones((3, 21), dtype=bool)
    mask[:, [1, 4, 8]] = False
    support = stratified_support_months(mask, seed=42, epoch=1)
    np.testing.assert_array_equal(support, stratified_support_months(mask, seed=42, epoch=1))
    assert not np.array_equal(support, stratified_support_months(mask, seed=42, epoch=2))
    chronological = support[:, [0, 3, 1, 4, 2]]
    for station, observed in enumerate(mask):
        for month, group in zip(chronological[station], np.array_split(np.flatnonzero(observed), 5)):
            assert month in group
    query = _episode_queries(mask, support)
    assert not query[np.arange(3)[:, None], support].any()
    np.testing.assert_array_equal(query.sum(1), mask.sum(1) - 5)
    # The same query explicitly excludes the two supports unused by K=3.
    assert not query[np.arange(3)[:, None], support[:, 3:]].any()


def test_differentiable_episode_matches_existing_support_ridge_and_pooled_weighting():
    rng = np.random.default_rng(32)
    basis = rng.normal(size=(2, 10, 2))
    prediction = np.full((2, 10), 3.)
    truth = np.expm1(np.log1p(prediction) + .2 * basis[..., 0])
    mask = np.ones((2, 10), dtype=bool)
    mask[0, 8:] = False
    schedule, query = support_schedule(np.flatnonzero(mask), 10)
    support = np.stack([schedule[i] % 10 for i in range(2)])
    query_mask = _episode_queries(mask, support)
    learned = torch.tensor(basis, requires_grad=True)
    loss, components = _tensor_episode_loss(
        learned, torch.tensor(np.log1p(prediction)), torch.tensor(truth),
        torch.tensor(support), torch.tensor(query_mask), pooled=True)
    expected = []
    for k in (3, 5):
        chosen = np.concatenate([cells[:k] for cells in schedule.values()])
        for ridge in (1., 10.):
            episode = SupportShapeEpisode(
                k, query, truth.ravel()[query], prediction.ravel()[query], chosen,
                truth.ravel()[chosen], prediction.ravel()[chosen],
                basis.reshape(-1, 2)[query], basis.reshape(-1, 2)[chosen])
            adapter = SupportShapeAdapter(n_months=10, alpha_values=(1.,), ridge_strengths=(ridge,))
            adapter.fit([episode], selection_role="source_validation")
            output = adapter.adapt(
                episode.query_prediction, query, episode.support_prediction, chosen,
                episode.support_values, query_basis=episode.query_basis,
                support_basis=episode.support_basis, k=k)
            expected.append(np.abs(output - truth.ravel()[query]).mean())
    np.testing.assert_allclose(components.detach().numpy(), expected, atol=1e-12)
    assert float(loss.detach()) == pytest.approx(np.mean(expected))
    loss.backward()
    assert torch.isfinite(learned.grad).all()
    assert learned.grad.abs().sum() > 0


def test_training_orthonormal_checkpoint_selection_serialization_and_callback():
    progress = []
    arrays = list(fixture_arrays())
    arrays[2] = np.expm1(arrays[1] + .08 * arrays[0][..., 0] + .12 * arrays[0][..., 2])
    arrays[6] = np.expm1(np.log1p(arrays[5])
                        + .08 * arrays[4][..., 0] + .12 * arrays[4][..., 2])
    model = EpisodicStationProjector(n_components=4, epochs=3, batch_size=3).fit(
        *arrays, selection_role="source_validation", progress=progress.append)
    scores = [row["validation_mae"] for row in model.trace_]
    assert model.best_epoch_ == int(np.argmin(scores))
    assert model.validation_metrics_["validation_mae"] <= scores[0]
    assert model.trace_[0]["training_mae"] is None
    assert model.epochs_run_ == 3
    assert progress == model.trace_
    assert model.best_epoch_ > 0
    np.testing.assert_allclose(model.projection_ @ model.projection_.T, np.eye(2), atol=1e-12)
    state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    assert state["projection_subspace_displacement"] > 0
    restored = EpisodicStationProjector.from_dict(state)
    np.testing.assert_array_equal(restored.transform(fixture_arrays()[4]),
                                  model.transform(fixture_arrays()[4]))
    assert restored.to_dict() == model.to_dict()


def test_epoch_zero_is_exact_common_pca_control_and_matches_previous_basis():
    source, _, _, _, target, *_ = fixture_arrays()
    model = fitted(epochs=0)
    np.testing.assert_array_equal(model.transform(target), model.transform_pca(target))
    previous = StationTemporalBasis().fit(source)
    np.testing.assert_allclose(model.transform_pca(target), previous.transform(target), atol=1e-12)
    assert model.best_epoch_ == 0
    assert model.to_dict()["projection_subspace_displacement"] == 0


def test_epoch_zero_retained_and_patience_stops_when_validation_does_not_improve():
    model = EpisodicStationProjector(n_components=4, epochs=20, patience=2, batch_size=3).fit(
        *fixture_arrays(), selection_role="source_validation")
    assert model.best_epoch_ == 0
    assert model.epochs_run_ == 2
    assert all(row["validation_mae"] >= model.trace_[0]["validation_mae"] for row in model.trace_)
    np.testing.assert_array_equal(model.transform(fixture_arrays()[4]),
                                  model.transform_pca(fixture_arrays()[4]))


def test_pca_source_only_and_unselected_labels_never_affect_training():
    original = list(fixture_arrays())
    original[3][:, 2] = False
    original[7][:, 2] = False
    changed = [array.copy() for array in original]
    original[1][:, 2], original[2][:, 2], original[6][:, 2] = np.nan, np.nan, np.nan
    changed[1][:, 2], changed[2][:, 2], changed[6][:, 2] = 1e300, -17., np.inf
    a = fitted(arrays=original)
    b = fitted(arrays=changed)
    assert a.to_dict() == b.to_dict()
    val_shift = [array.copy() for array in original]
    val_shift[4] *= 2
    val_shift[6][val_shift[7]] *= 3
    c = fitted(epochs=0, arrays=val_shift)
    np.testing.assert_array_equal(a.basis_.components_, c.basis_.components_)
    np.testing.assert_array_equal(a.basis_.eigenvalues_, c.basis_.eigenvalues_)


def test_transform_has_no_labels_and_preserves_other_station_projection():
    model = fitted()
    target = fixture_arrays()[4]
    expected = model.transform(target)
    target[1] *= 20
    np.testing.assert_array_equal(model.transform(target)[0], expected[0])
    np.testing.assert_allclose(model.transform(target + 100), model.transform(target), atol=1e-12)
    assert list(inspect.signature(model.transform).parameters) == ["features"]
    with pytest.raises(ValueError, match="source_validation"):
        model.fit(*fixture_arrays(), selection_role="target_test")


def test_rank_floor_zeros_near_constant_modes_and_rejects_rank_one():
    arrays = list(fixture_arrays())
    arrays[0][..., 2:] *= 1e-9
    model = fitted(arrays=arrays)
    assert model.effective_rank_ == 2
    np.testing.assert_array_equal(model.projection_[:, ~model.active_modes_], 0.)
    np.testing.assert_array_equal(model._whiten(arrays[4])[..., ~model.active_modes_], 0.)
    assert model.eigenvalue_floor_ == pytest.approx(1e-5 * model.basis_.eigenvalues_[0])
    arrays[0][..., 1] *= 1e-9
    with pytest.raises(ValueError, match="effective rank"):
        fitted(arrays=arrays)


def test_nonfinite_episode_fails_without_upper_clipping():
    mask = torch.tensor([[False] * 5 + [True] * 2])
    z = torch.tensor([[0.] * 5 + [1000.] * 2], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="no upper clipping"):
        _tensor_episode_loss(torch.zeros((1, 7, 2), dtype=torch.float64), z,
                             torch.ones((1, 7), dtype=torch.float64),
                             torch.tensor([[0, 1, 2, 3, 4]]), mask, pooled=True)


def test_invalid_roles_masks_and_projection_rejected():
    model = fitted(epochs=0)
    arrays = list(fixture_arrays())
    arrays[3] = arrays[3].astype(int)
    with pytest.raises(ValueError, match="boolean mask"):
        fitted(arrays=arrays)
    state = model.to_dict()
    state["projection"][0][0] = 20
    with pytest.raises(ValueError, match="invalid serialized"):
        EpisodicStationProjector.from_dict(state)
    with pytest.raises(RuntimeError, match="fit"):
        EpisodicStationProjector().transform(np.ones((2, 3, 16)))
