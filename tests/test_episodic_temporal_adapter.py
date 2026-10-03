"""Rolling-memory, adaptation and persistence checks on tiny synthetic inputs."""
from __future__ import annotations

import copy
import inspect
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from river_graph.models.episodic_temporal_adapter import (
    EpisodicTemporalAdapter,
    anchor_normalize,
    gathered_rolling_states,
)
from river_graph.models.graph_upgrade import ObservationAwareTemporalTransportGCNImputer


def setup_fixture(*, epochs=2, dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(57)
        spatial = nn.Module()
        spatial.head = nn.Linear(4, 1)
        original = ObservationAwareTemporalTransportGCNImputer(
            spatial, lookback=12, temporal_hidden=4, chunk_months=5).to(dtype=dtype)
        with torch.no_grad():
            original.decay.bias.fill_(.1)
            original.decay.weight.fill_(.03)
    rng = np.random.default_rng(56)

    def inputs(n):
        return {"encoded": rng.normal(size=(n, 18, 4)),
                "age": rng.uniform(0, 2, size=(n, 18)),
                "support": rng.uniform(0, 1, size=(n, 18, 3))}

    source, validation = inputs(4), inputs(3)
    readout = rng.normal(size=(4, 2))
    adapter = EpisodicTemporalAdapter(
        original.temporal, original.decay, readout, epochs=epochs, patience=2,
        batch_size=2, anchor_count=5, learning_rate=1e-3)
    source_base = np.full((4, 18), np.log(5.))
    source_truth = np.expm1(source_base + .1 * source["encoded"][..., 0])
    val_base = np.full((3, 18), 4.)
    val_truth = np.expm1(np.log1p(val_base) + .1 * validation["encoded"][..., 0])
    source_mask, val_mask = np.ones((4, 18), dtype=bool), np.ones((3, 18), dtype=bool)
    source_mask[:, 3], val_mask[:, 3] = False, False
    args = (source, source_base, source_truth, source_mask,
            validation, val_base, val_truth, val_mask)
    return original, adapter, args


def test_gathered_states_match_existing_rolling_memory_at_padding_and_arbitrary_months():
    original, adapter, args = setup_fixture(epochs=0)
    inputs = adapter._prepare_inputs(args[0])
    station = np.array([0, 1, 2, 3, 1, 0, 2])
    month = np.array([0, 1, 11, 12, 17, 8, 4])
    expected = original._memory_states(
        inputs["encoded"].permute(1, 0, 2), inputs["age"].T,
        inputs["support"].permute(1, 0, 2))
    result = gathered_rolling_states(adapter.temporal, adapter.decay, inputs, station, month,
                                    lookback=12)
    torch.testing.assert_close(result, expected[month, station], rtol=1e-13, atol=1e-14)
    # Month zero executes just its real update, despite eleven padding months.
    row = torch.cat([inputs["encoded"][0, 0], torch.ones(1, dtype=adapter.dtype)])
    single = adapter.temporal(row, torch.zeros(4, dtype=adapter.dtype))
    torch.testing.assert_close(result[0], single, rtol=1e-13, atol=1e-14)


def test_future_inputs_do_not_change_recurrent_states_before_anchor_normalization():
    _, adapter, args = setup_fixture(epochs=0)
    initial = adapter._prepare_inputs(args[0])
    changed = {key: value.clone() for key, value in initial.items()}
    for values in changed.values():
        values[:, 9:] += 100
    stations, months = np.repeat(np.arange(4), 9), np.tile(np.arange(9), 4)
    expected = gathered_rolling_states(adapter.temporal, adapter.decay, initial,
                                      stations, months, lookback=12)
    actual = gathered_rolling_states(adapter.temporal, adapter.decay, changed,
                                    stations, months, lookback=12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_observed_plus_anchor_evaluation_matches_full_timeline_on_observed_rows():
    _, adapter, args = setup_fixture(epochs=0)
    inputs = adapter._prepare_inputs(args[0])
    selected = np.zeros((4, 18), dtype=bool)
    selected[:, [0, 2, 7, 12, 17]] = True
    sparse = adapter._basis_for_rows(inputs, np.arange(4), selected_months=selected)
    complete = adapter.transform(args[0])
    np.testing.assert_allclose(sparse.detach().numpy()[selected], complete[selected], atol=1e-13)
    np.testing.assert_array_equal(adapter.anchor_months(18), [0, 4, 8, 12, 17])
    np.testing.assert_array_equal(adapter.anchor_months(3), [0, 1, 2])


def test_anchor_normalization_uses_one_scalar_and_is_gain_translation_invariant():
    raw = torch.tensor(np.random.default_rng(6).normal(size=(2, 13, 2)), dtype=torch.float64)
    anchors = torch.tensor([0, 3, 6, 9, 12])
    result = anchor_normalize(raw, anchors, scale_floor=1e-4)
    shifted = anchor_normalize(raw * 7 + torch.tensor([5., -7.]), anchors, scale_floor=1e-4)
    torch.testing.assert_close(result, shifted, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(result[:, anchors].mean(dim=1), torch.zeros((2, 2), dtype=torch.float64),
                               rtol=0, atol=1e-14)
    torch.testing.assert_close(result[:, anchors].square().mean(dim=(1, 2)),
                               torch.ones(2, dtype=torch.float64), rtol=1e-13, atol=1e-13)
    assert not torch.allclose(result[:, anchors].square().mean(dim=1), torch.ones((2, 2), dtype=torch.float64))
    constant = torch.ones((1, 9, 2), dtype=torch.float64, requires_grad=True)
    zero, stats = anchor_normalize(constant, torch.tensor([0, 4, 8]), scale_floor=1e-4,
                                   return_stats=True)
    assert stats["floor_hit"].tolist() == [True]
    zero.sum().backward()
    assert torch.isfinite(constant.grad).all()


def test_training_updates_both_copied_modules_without_mutating_original_or_readout():
    original, adapter, args = setup_fixture(dtype=torch.float32)
    original_state = copy.deepcopy(original.state_dict())
    fixed = adapter.readout.clone()
    reports = []
    adapter.fit(*args, selection_role="source_validation", progress=reports.append)
    assert adapter.trace_[1]["temporal_parameter_distance"] > 0
    assert adapter.trace_[1]["decay_parameter_distance"] > 0
    assert adapter.trace_[0]["temporal_parameter_distance"] == 0
    assert adapter.trace_[0]["decay_parameter_distance"] == 0
    assert adapter.best_epoch_ == np.argmin([row["validation_mae"] for row in adapter.trace_])
    assert reports == adapter.trace_
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    torch.testing.assert_close(adapter.readout, fixed, rtol=0, atol=0)
    assert not adapter.readout.requires_grad
    assert adapter.trainable_parameter_count_ == sum(
        p.numel() for module in (original.temporal, original.decay) for p in module.parameters())
    assert adapter.trace_[0]["source_anchor_floor_hits"] is None
    assert adapter.trace_[1]["source_anchor_floor_hits"] is not None
    assert adapter.to_dict()["selected_validation_anchor_floor_hits"] >= 0


def test_unobserved_label_perturbations_leave_training_and_selection_unchanged():
    _, a, arrays = setup_fixture(epochs=1)
    _, b, altered = setup_fixture(epochs=1)
    arrays[1][~arrays[3]] = np.nan
    arrays[2][~arrays[3]] = np.nan
    arrays[6][~arrays[7]] = np.nan
    altered[1][~altered[3]] = 1e300
    altered[2][~altered[3]] = -1e300
    altered[6][~altered[7]] = np.inf
    a.fit(*arrays, selection_role="source_validation")
    b.fit(*altered, selection_role="source_validation")
    assert a.to_dict() == b.to_dict()
    np.testing.assert_array_equal(a.transform(arrays[4]), b.transform(arrays[4]))
    assert list(inspect.signature(a.transform).parameters) == ["inputs"]
    with pytest.raises(ValueError, match="source_validation"):
        a.fit(*arrays, selection_role="target_test")


def test_epoch_zero_and_torch_checkpoint_restore_initial_and_selected_basis():
    _, adapter, args = setup_fixture(epochs=0)
    adapter.fit(*args, selection_role="source_validation")
    np.testing.assert_array_equal(adapter.transform(args[4]), adapter.transform_initial(args[4]))
    assert adapter.best_epoch_ == adapter.epochs_run_ == 0
    expected = adapter.transform(args[4])
    buffer = io.BytesIO()
    torch.save(adapter.to_payload(), buffer)
    buffer.seek(0)
    restored = EpisodicTemporalAdapter.from_payload(torch.load(buffer, weights_only=True))
    np.testing.assert_array_equal(restored.transform(args[4]), expected)
    np.testing.assert_array_equal(restored.transform_initial(args[4]), expected)
    assert restored.to_dict() == adapter.to_dict()
    json.dumps(restored.to_dict(), allow_nan=False)


def test_normalization_diagnostics_use_only_anchors_and_report_floor_hits():
    _, adapter, args = setup_fixture(epochs=0)
    initial = adapter.normalization_stats(args[0], initial=True)
    assert len(initial["station_rms"]) == 4
    assert initial["anchor_months"] == adapter.anchor_months(18).tolist()
    assert initial["floor_hits"] == sum(initial["floor_hit"])
    adapter.readout.zero_()
    zero = adapter.normalization_stats(args[0])
    assert zero["floor_hits"] == 4
    assert zero["station_rms"] == [0.] * 4
    assert np.isfinite(adapter.transform(args[0])).all()


def test_bad_inputs_and_checkpoint_nonfinite_values_rejected():
    _, adapter, args = setup_fixture(epochs=0)
    invalid = {key: value.copy() for key, value in args[0].items()}
    invalid["support"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite aligned"):
        adapter.transform(invalid)
    adapter.fit(*args, selection_role="source_validation")
    payload = adapter.to_payload()
    payload["temporal"]["weight_hh"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite temporal"):
        EpisodicTemporalAdapter.from_payload(payload)
