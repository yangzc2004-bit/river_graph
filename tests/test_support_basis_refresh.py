"""Inference-only refresh with the original readout and anchor normalization."""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import torch
from test_daily_hydro_memory import fit_model, fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.support_basis_refresh import refresh_support_basis


def make(mode="off", *, dtype=torch.float64, epochs=0):
    model, args = fixture(mode, dtype=dtype, epochs=epochs)
    readout = torch.tensor([[.7, -.2], [.2, .4], [-.3, .6], [.1, .5]], dtype=torch.float64)
    adapter = EpisodicTemporalAdapter(
        model.temporal, model.decay, readout, lookback=model.lookback,
        anchor_count=4, scale_floor=1e-4)
    return model, args, adapter


@pytest.mark.parametrize("mode", ["off", "current_only", "full_history"])
def test_projection_and_anchor_statistics_exact_independent_replay(mode):
    model, args, adapter = make(mode)
    if model.hydro_projection is not None:
        with torch.no_grad():
            model.hydro_projection.weight.fill_(.13)
    inputs = args[0]
    result = refresh_support_basis(model, inputs, adapter, batch_size=7)
    prepared = model._prepare_inputs(inputs)
    with torch.no_grad():
        # Same bounded cells, independently assembled projection and formula.
        hidden = torch.cat([model._hidden_cells(prepared, torch.arange(start, min(start+7, 24)))
                            for start in range(0, 24, 7)])
        raw = (hidden.double() @ adapter.readout).reshape(3, 8, 2)
        anchors = np.linspace(0, 7, 4, dtype=np.int64)
        mean = raw[:, anchors].mean(1, keepdim=True)
        variance = (raw[:, anchors]-mean).square().mean((1, 2), keepdim=True)
        scale = variance.clamp_min(1e-8).sqrt()
    np.testing.assert_array_equal(result["raw_basis"], raw.numpy())
    np.testing.assert_array_equal(result["basis"], ((raw-mean)/scale).numpy())
    np.testing.assert_array_equal(result["station_anchor_mean"], mean[:, 0].numpy())
    np.testing.assert_array_equal(result["station_rms"], variance.sqrt().reshape(-1).numpy())
    np.testing.assert_array_equal(result["station_scale"], scale.reshape(-1).numpy())
    np.testing.assert_array_equal(result["floor_hit"], result["station_rms"] < 1e-4)
    np.testing.assert_array_equal(result["anchor_months"], anchors)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_hydro_projection_keeps_off_current_full_basis_exact(dtype):
    baseline, args, adapter = make(dtype=dtype)
    expected = refresh_support_basis(baseline, args[0], adapter, batch_size=5)
    for mode in ("current_only", "full_history"):
        model, _, same_adapter = make(mode, dtype=dtype)
        actual = refresh_support_basis(model, args[0], same_adapter, batch_size=5)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_source_full_alignment_station_isolation_and_finite_outputs():
    model, args, adapter = make("full_history")
    with torch.no_grad():
        model.hydro_projection.weight.fill_(.1)
    source, validation = args[0], args[4]
    combined = {key: np.concatenate((source[key], validation[key])) for key in source}
    source_result = refresh_support_basis(model, source, adapter, batch_size=8)
    full_result = refresh_support_basis(model, combined, adapter, batch_size=8)
    for key in ("basis", "raw_basis", "station_anchor_mean", "station_rms", "station_scale", "floor_hit"):
        np.testing.assert_array_equal(source_result[key], full_result[key][:3])
    assert full_result["basis"].shape == (5, 8, 2)
    assert all(np.isfinite(value).all() for value in full_result.values())
    changed = copy.deepcopy(combined)
    changed["raw"][3:] += 100
    changed_result = refresh_support_basis(model, changed, adapter, batch_size=8)
    np.testing.assert_array_equal(changed_result["basis"][:3], full_result["basis"][:3])


def test_raw_causal_prefix_but_anchor_normalization_is_retrospective():
    model, args, adapter = make("full_history")
    with torch.no_grad():
        model.hydro_projection.weight.fill_(.3)
    baseline = refresh_support_basis(model, args[0], adapter, batch_size=8)
    future = copy.deepcopy(args[0])
    future["daily_history"][:, 6:] += 3
    revised = refresh_support_basis(model, future, adapter, batch_size=8)
    np.testing.assert_array_equal(revised["raw_basis"][:, :6], baseline["raw_basis"][:, :6])
    assert not np.array_equal(revised["raw_basis"][:, 6:], baseline["raw_basis"][:, 6:])
    # Future calendar anchors deliberately affect even early normalized months.
    assert not np.array_equal(revised["basis"][:, :6], baseline["basis"][:, :6])


def test_no_label_reads_no_mutation_and_checkpoint_replay():
    model, args, adapter = make("full_history", epochs=1)
    fit_model(model, args)
    payload_before = copy.deepcopy(model.to_payload())
    readout = adapter.readout.clone()
    model.spatial.train()
    model.spatial.env_encoder.eval()
    roots = (model.spatial, model.temporal, model.decay, model.head, model.hydro_projection)
    modules = [module for root in roots for module in root.modules()]
    flags = [module.training for module in modules]
    parameters = [parameter for root in roots for parameter in root.parameters()]
    gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
    rng_before = torch.get_rng_state().clone()
    unknown = copy.deepcopy(args[4])
    unknown["y"] = np.full((2, 8), np.nan)
    unknown["query_truth"] = "must never be used"
    actual = refresh_support_basis(model, unknown, adapter, batch_size=8)
    assert [module.training for module in modules] == flags
    assert torch.equal(torch.get_rng_state(), rng_before)
    torch.testing.assert_close(adapter.readout, readout, rtol=0, atol=0)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    after = model.to_payload()
    assert after["summary"] == payload_before["summary"]
    for name in ("spatial", "temporal", "decay", "head", "hydro_projection"):
        for key in after[name]:
            torch.testing.assert_close(after[name][key], payload_before[name][key], rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(after, stream)
    stream.seek(0)
    restored = EncoderNativeResidual.from_payload(torch.load(stream, weights_only=True))
    replay = refresh_support_basis(restored, args[4], adapter, batch_size=8)
    for key in actual:
        np.testing.assert_array_equal(actual[key], replay[key])


def test_constant_readout_and_single_month_have_finite_floor_outputs():
    model, args, adapter = make()
    adapter.readout.zero_()
    result = refresh_support_basis(model, args[0], adapter, batch_size=8)
    assert np.count_nonzero(result["basis"]) == 0
    assert result["floor_hit"].all()
    np.testing.assert_array_equal(result["station_scale"], np.full(3, 1e-4))
    # One calendar month has exactly zero anchor variance even with a nonzero readout.
    _, _, adapter = make()
    single = {key: value if key == "env" else value[:, :1] for key, value in args[0].items()}
    result = refresh_support_basis(model, single, adapter, batch_size=3)
    assert result["basis"].shape == (3, 1, 2)
    assert np.count_nonzero(result["basis"]) == 0 and result["floor_hit"].all()


def test_bad_input_rejected_and_exception_restores_mode():
    model, args, adapter = make("full_history")
    for batch in (0, -2, 1.2, True):
        with pytest.raises(ValueError, match="batch_size"):
            refresh_support_basis(model, args[0], adapter, batch_size=batch)
    adapter.readout = torch.zeros((5, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="readout"):
        refresh_support_basis(model, args[0], adapter)
    _, _, adapter = make()
    bad = copy.deepcopy(args[0]); bad["daily_history"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="daily_history"):
        refresh_support_basis(model, bad, adapter)
    model.spatial.train()
    with torch.no_grad():
        model.temporal.weight_hh.fill_(torch.nan)
    with pytest.raises(FloatingPointError, match="hidden"):
        refresh_support_basis(model, args[0], adapter)
    assert model.spatial.training
