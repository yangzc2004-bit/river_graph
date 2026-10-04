"""Frozen-state supervised readout: exact control, honest episodes and replay."""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import torch

from river_graph.models.episodic_support_readout import EpisodicSupportReadout
from river_graph.models.episodic_temporal_adapter import anchor_normalize


def problem(hidden=4, *, source_count=12, validation_count=5, months=20):
    rng = np.random.default_rng(181)

    def side(n):
        x = rng.normal(0, .5, (n, months, hidden))
        time = np.linspace(-1, 1, months)
        # A real transferable direction lies outside the initial two axes.
        x[..., 2] = np.sin(time[None, :]*3 + rng.uniform(-.3, .3, (n, 1)))
        base = np.full((n, months), 4.)
        truth = np.expm1(np.log1p(base)+.5*x[..., 2]+rng.uniform(-.1, .1, (n, 1)))
        mask = np.ones((n, months), bool)
        mask[:, 2] = False
        return x, base, truth, mask

    initial = np.eye(hidden, 2)
    return initial, (*side(source_count), *side(validation_count))


def make(initial, **kwargs):
    return EpisodicSupportReadout(initial, anchor_count=8, max_epochs=8, patience=5,
                                  batch_size=4, **kwargs)


def test_initial_projection_and_anchor_formula_exact_across_projection_batches():
    rng = np.random.default_rng(22)
    hidden = rng.normal(size=(5, 500, 64)).astype(np.float32)
    initial = rng.normal(size=(64, 2))
    model = EpisodicSupportReadout(initial)
    flat = torch.as_tensor(hidden).double().reshape(-1, 64)
    readout = torch.as_tensor(initial)
    projected = torch.cat([flat[start:start+2048] @ readout
                           for start in range(0, len(flat), 2048)]).reshape(5, 500, 2)
    anchors = torch.as_tensor(np.linspace(0, 499, 32, dtype=np.int64))
    expected = anchor_normalize(projected, anchors, scale_floor=1e-4).reshape(-1, 2).numpy()
    np.testing.assert_array_equal(model.transform_initial(hidden), expected)
    np.testing.assert_array_equal(model.initial_readout, initial)


def test_actual_readout_learning_on_frozen_hidden_states_and_four_losses():
    initial, data = problem()
    model = EpisodicSupportReadout(initial, anchor_count=8, max_epochs=30,
                                    patience=5, batch_size=4).fit(*data)
    assert model.best_epoch_ > 0
    assert model.validation_metrics_["validation_mae"] < model.trace_[0]["validation_mae"]-1e-6
    assert torch.count_nonzero(model.delta_) > 0
    assert any(row["gradient_norm_max_before_clip"] > 0 for row in model.trace_[1:])
    assert model.transform(data[4]).shape == (5*20, 2)
    assert not np.array_equal(model.transform(data[4]), model.transform_initial(data[4]))
    assert [(v["k"], v["ridge_strength"]) for v in model.trace_[0]["validation_by_k_ridge"]] == [
        (3, 1.), (3, 10.), (5, 1.), (5, 10.)]


def test_only_128_parameters_and_no_input_or_readout_gradients():
    initial, data = problem(hidden=64, source_count=4, validation_count=2)
    source = torch.tensor(data[0], requires_grad=True)
    validation = torch.tensor(data[4], requires_grad=True)
    readout = torch.tensor(initial, requires_grad=True)
    values = (source, *data[1:4], validation, *data[5:])
    original = source.detach().clone()
    model = EpisodicSupportReadout(readout, max_epochs=1, batch_size=4).fit(*values)
    assert model.trainable_parameter_count_ == 128 and model.delta_.shape == (64, 2)
    assert source.grad is None and validation.grad is None and readout.grad is None
    torch.testing.assert_close(source.detach(), original, rtol=0, atol=0)
    torch.testing.assert_close(readout.detach(), torch.as_tensor(initial), rtol=0, atol=0)


def test_masked_labels_and_baselines_do_not_change_fit_or_checkpoint():
    initial, data = problem(source_count=4, validation_count=2)
    changed = copy.deepcopy(data)
    for offset in (0, 4):
        for array in (changed[offset+1], changed[offset+2]):
            array[~changed[offset+3]] = np.nan
    a, b = make(initial).fit(*data), make(initial).fit(*changed)
    assert a.to_dict() == b.to_dict()
    torch.testing.assert_close(a.delta_, b.delta_, rtol=0, atol=0)
    np.testing.assert_array_equal(a.transform(data[4]), b.transform(data[4]))
    with pytest.raises(ValueError, match="source_validation"):
        make(initial).fit(*data, selection_role="test")


def test_support_schedules_match_observed_cells_and_queries_exclude_all_five():
    initial, data = problem(source_count=4, validation_count=2)
    model = make(initial).fit(*data)
    for record in model.source_episode_schedules_:
        months = np.asarray(record["support_months"])
        assert months.shape == (4, 5)
        assert all(len(np.unique(row)) == 5 for row in months)
        assert data[3][np.arange(4)[:, None], months].all()
    support = model.validation_support_months_ + np.arange(2)[:, None]*20
    assert not np.isin(model.validation_query_cells_, support).any()
    np.testing.assert_array_equal(
        np.sort(np.concatenate((model.validation_query_cells_, support.ravel()))),
        np.flatnonzero(data[7]))
    assert len(model.source_episode_schedules_) == model.epochs_run_


def test_epoch_zero_and_constant_hidden_are_finite_with_rms_floor():
    initial, data = problem(source_count=4, validation_count=2)
    data[0][:] = 0
    data[4][:] = 0
    model = make(initial).fit(*data)
    assert model.best_epoch_ == 0 and model.epochs_run_ == model.patience
    assert not torch.count_nonzero(model.delta_)
    assert all(row["validation_anchor_floor_hits"] == 2 for row in model.trace_)
    assert not np.count_nonzero(model.transform(data[4]))
    assert all(np.isfinite(row["validation_mae"]) for row in model.trace_)


def test_checkpoint_roundtrip_refit_and_config_are_exact():
    initial, data = problem(source_count=4, validation_count=2)
    model = make(initial).fit(*data)
    stream = io.BytesIO()
    torch.save(model.to_payload(), stream)
    stream.seek(0)
    restored = EpisodicSupportReadout.from_payload(torch.load(stream, weights_only=True))
    assert restored.to_dict() == model.to_dict()
    np.testing.assert_array_equal(restored.transform(data[4]), model.transform(data[4]))
    np.testing.assert_array_equal(restored.transform_initial(data[4]), model.transform_initial(data[4]))
    restored.fit(*data)
    assert restored.to_dict() == model.to_dict()
    bad = copy.deepcopy(model.to_payload()); bad["delta"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="delta"):
        EpisodicSupportReadout.from_payload(bad)


def test_transform_station_isolation_and_source_validation_dimension_rules():
    initial, data = problem(source_count=4, validation_count=2)
    model = make(initial).fit(*data)
    expected = model.transform(data[4]).reshape(2, 20, 2)
    revised = data[4].copy(); revised[1] += 1
    np.testing.assert_array_equal(model.transform(revised).reshape(2, 20, 2)[0], expected[0])
    with pytest.raises(ValueError, match="month grids"):
        make(initial).fit(*data[:4], *(x[:, :-1] for x in data[4:]))
    bad = copy.deepcopy(data); bad[7][0, :] = False; bad[7][0, :5] = True
    with pytest.raises(ValueError, match="validation station"):
        make(initial).fit(*bad)
    with pytest.raises(ValueError, match="hidden"):
        model.transform(np.full((2, 20, 4), np.nan))
