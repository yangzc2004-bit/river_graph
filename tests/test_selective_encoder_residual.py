"""Loss estimands, matched controls and visibility for selective DOC learning."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.selective_encoder_residual import SelectiveEncoderResidual

ARMS = ((2, "cell", 0.), (1, "cell", 0.), (2, "cell", .5), (2, "station", 0.))


def fixture(*, cls=SelectiveEncoderResidual, tail_weight=2, source_weighting="cell",
            penalty=0., epochs=2, dtype=torch.float64):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        spatial = TransportGCNImputer(5, 2, hidden=4, layers=2, dropout=.5,
                                     env_dim=2, env_emb=3, edge_direction="upstream").to(dtype=dtype)
        temporal, decay = nn.GRUCell(5, 4).to(dtype=dtype), nn.Linear(4, 4).to(dtype=dtype)
        with torch.no_grad():
            for conv in spatial.convs:
                conv.self_lin.weight.fill_(.1)
                conv.self_lin.bias.fill_(.15)
            for parameter in spatial.env_encoder.parameters():
                parameter.fill_(.2)
    options = {"encoder_mode": "last_self_ecology", "encoder_learning_rate": 1e-3,
               "epochs": epochs, "patience": 2, "batch_size": 9, "learning_rate": 1e-3,
               "head_learning_rate": .04, "tail_weight": tail_weight}
    if cls is SelectiveEncoderResidual:
        options.update(source_weighting=source_weighting, ordinary_overprediction_penalty=penalty)
    model = cls(spatial, temporal, decay, **options)
    rng = np.random.default_rng(19)

    def inputs(n):
        return {"raw": rng.uniform(.1, 1, (n, 14, 5)), "env": rng.uniform(.2, 1, (n, 2)),
                "age": rng.uniform(0, 2, (n, 14)), "support": rng.uniform(0, 1, (n, 14, 3)),
                "extra": rng.uniform(-.2, .2, (n, 14, 30))}

    source, validation = inputs(3), inputs(2)
    source_mask = np.arange(14)[None, :] < np.array([4, 9, 13])[:, None]
    val_mask = np.ones((2, 14), bool)
    val_mask[:, [1, 7]] = False
    base, val_base = np.full((3, 14), 3.), np.full((2, 14), 3.)
    truth = np.broadcast_to(np.where(np.arange(14) % 3 == 0, 2., 4.5), base.shape).copy()
    val_truth = truth[:2].copy()
    return model, (source, base, truth, source_mask, validation, val_base, val_truth, val_mask)


def fit_model(model, args):
    return model.fit(*args, tail_threshold=4., selection_role="source_validation")


def test_station_weights_have_equal_total_after_tail_weighting_and_exact_estimand():
    model, _ = fixture(source_weighting="station", epochs=0)
    cells = np.array([0, 1, 2, 14, 15, 28, 29, 30, 31])
    truth = torch.tensor([2., 5., 3., 6., 8., 1., 2., 3., 7.], dtype=torch.float64)
    tail = truth >= 4
    weights = model._source_weights(cells, truth, tail, 14)
    raw_weights = 1. + tail.double()
    errors = torch.tensor([1., 2., 4., 3., 1., 2., 4., 8., 6.], dtype=torch.float64)
    station_losses = []
    for station in (0, 1, 2):
        use = torch.as_tensor(cells // 14 == station)
        torch.testing.assert_close(weights[use].sum(), torch.tensor(1., dtype=torch.float64), rtol=1e-15, atol=0)
        torch.testing.assert_close(weights[use], raw_weights[use] / raw_weights[use].sum(), rtol=0, atol=0)
        station_losses.append((raw_weights[use] * errors[use]).sum() / raw_weights[use].sum())
    expected = torch.stack(station_losses).mean()
    actual = model._training_errors(truth + errors, truth, tail, weights).sum() / weights.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-15, atol=0)
    # A fixed full-source denominator retains the complete objective across batches.
    batches = torch.tensor_split(torch.arange(len(cells)), 4)
    recovered = sum(model._training_errors(truth[rows] + errors[rows], truth[rows], tail[rows], weights[rows]).mean()
                    / weights.mean() * len(rows) for rows in batches) / len(cells)
    torch.testing.assert_close(recovered, expected, rtol=1e-15, atol=0)
    # Changing another station's labels cannot change this station's weights.
    changed = truth.clone()
    changed[cells // 14 == 1] = 0
    changed_weights = model._source_weights(cells, changed, changed >= 4, 14)
    torch.testing.assert_close(weights[cells // 14 != 1], changed_weights[cells // 14 != 1], rtol=0, atol=0)


def test_selective_penalty_gradient_and_final_prediction_semantics():
    model, _ = fixture(penalty=.5, epochs=0)
    truth = torch.tensor([2., 2., 5., 5., 4.], dtype=torch.float64)
    prediction = torch.tensor([3., 1., 6., 4., 5.], dtype=torch.float64, requires_grad=True)
    tail = truth >= 4
    weights = 1. + tail.double()
    errors = model._training_errors(prediction, truth, tail, weights)
    torch.testing.assert_close(errors, torch.tensor([1.5, 1., 2., 2., 2.], dtype=torch.float64), rtol=0, atol=0)
    errors.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.5, -1., 2., -2., 2.], dtype=torch.float64), rtol=0, atol=0)
    # A positive residual that still underpredicts incurs no overprediction penalty.
    pred = model._combine(torch.tensor([1.], dtype=torch.float64), torch.tensor([.5], dtype=torch.float64), 1.)
    torch.testing.assert_close(model._training_errors(pred, truth[:1], tail[:1], weights[:1]),
                               torch.tensor([.5], dtype=torch.float64), rtol=0, atol=0)
    # A negative correction can still leave an overprediction and is penalized.
    pred = model._combine(torch.tensor([4.], dtype=torch.float64), torch.tensor([-.5], dtype=torch.float64), 1.)
    torch.testing.assert_close(model._training_errors(pred, truth[:1], tail[:1], weights[:1]),
                               torch.tensor([2.25], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("tail_weight", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_cell_control_matches_parent_training_and_checkpoint_exactly(tail_weight, dtype):
    parent, args = fixture(cls=EncoderNativeResidual, tail_weight=tail_weight, dtype=dtype)
    control, same = fixture(tail_weight=tail_weight, dtype=dtype)
    fit_model(parent, args)
    fit_model(control, same)
    assert parent.trace_ == control.trace_
    assert parent.best_epoch_ == control.best_epoch_ and parent.selected_scale_ == control.selected_scale_
    for name in ("spatial", "temporal", "decay", "head"):
        for key, value in getattr(parent, name).state_dict().items():
            torch.testing.assert_close(value, getattr(control, name).state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(parent.predict_delta(args[4]), control.predict_delta(args[4]))
    np.testing.assert_array_equal(parent.predict(args[4], args[5]), control.predict(args[4], args[5]))


@pytest.mark.parametrize("tail_weight,weighting,penalty", ARMS)
def test_all_objectives_checkpoint_reload_and_refit(tail_weight, weighting, penalty):
    model, args = fixture(tail_weight=tail_weight, source_weighting=weighting, penalty=penalty, epochs=1)
    fit_model(model, args)
    summary = model.to_dict()
    assert summary["model_class"] == "SelectiveEncoderResidual"
    assert summary["config"]["source_weighting"] == weighting
    assert summary["protocol"]["validation_loss"] == "unweighted pooled fixed-query raw MAE"
    buffer = io.BytesIO()
    torch.save(model.to_payload(), buffer)
    buffer.seek(0)
    restored = SelectiveEncoderResidual.from_payload(torch.load(buffer, weights_only=True))
    assert restored.to_dict() == summary
    json.dumps(summary, allow_nan=False)
    np.testing.assert_array_equal(restored.predict(args[4], args[5]), model.predict(args[4], args[5]))
    np.testing.assert_array_equal(restored.predict_delta(args[4]), model.predict_delta(args[4]))
    fit_model(restored, args)
    assert restored.to_dict() == summary


@pytest.mark.parametrize("weighting,penalty", [("cell", .5), ("station", 0.)])
def test_hidden_labels_and_bases_do_not_enter_weights_losses_or_selection(weighting, penalty):
    a, original = fixture(source_weighting=weighting, penalty=penalty, epochs=1)
    b, changed = fixture(source_weighting=weighting, penalty=penalty, epochs=1)
    for args, fill in ((original, np.nan), (changed, -1e200)):
        for values, mask in ((args[1], args[3]), (args[2], args[3]), (args[5], args[7]), (args[6], args[7])):
            values[~mask] = fill
    fit_model(a, original)
    fit_model(b, changed)
    assert a.to_dict() == b.to_dict()
    np.testing.assert_array_equal(a.predict_delta(original[4]), b.predict_delta(original[4]))
    with pytest.raises(ValueError, match="source_validation"):
        a.fit(*original, tail_threshold=4., selection_role="outer_test")


@pytest.mark.parametrize("weighting,penalty", [("cell", .5), ("station", 0.)])
def test_future_inputs_do_not_change_earlier_nonzero_predictions(weighting, penalty):
    model, args = fixture(source_weighting=weighting, penalty=penalty, epochs=0)
    with torch.no_grad():
        model.head.weight.fill_(.05)
    changed = copy.deepcopy(args[0])
    for name in ("raw", "age", "support", "extra"):
        changed[name][:, 8:] += 10
    before, after = model.predict_delta(args[0]), model.predict_delta(changed)
    np.testing.assert_array_equal(before[:, :8], after[:, :8])
    assert not np.allclose(before[:, 8:], after[:, 8:])


def test_old_encoder_payload_import_and_invalid_combinations():
    parent, args = fixture(cls=EncoderNativeResidual, epochs=1)
    fit_model(parent, args)
    restored = SelectiveEncoderResidual.from_payload(parent.to_payload())
    assert restored.source_weighting == "cell" and restored.ordinary_overprediction_penalty == 0
    np.testing.assert_array_equal(restored.predict_delta(args[4]), parent.predict_delta(args[4]))
    for settings in ({"source_weighting": "unknown"}, {"penalty": -1}, {"penalty": np.nan},
                     {"penalty": .25}, {"source_weighting": "station", "penalty": .5}):
        with pytest.raises(ValueError):
            fixture(**settings)
    payload = restored.to_payload()
    del payload["config"]["source_weighting"]
    with pytest.raises(ValueError, match="loss definition"):
        SelectiveEncoderResidual.from_payload(payload)
