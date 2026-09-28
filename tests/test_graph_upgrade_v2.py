"""Scientific contracts for observation-driven graph upgrades."""

import numpy as np
import torch

from river_graph.experiments.graph_upgrade_v2 import (
    GraphUpgradeModel,
    build_observation_features,
    observation_statistics,
    sample_training_view,
)
from river_graph.experiments.temporal_h2x import build_temporal_inputs


def _toy_dataset():
    n, t = 5, 8
    y = torch.arange(n * t, dtype=torch.float32).reshape(n, t) / 10
    return {
        "y": y, "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.randn(n, t, 2), "x_mask": torch.ones(n, t, 2),
        "months": [f"2020-{j:02d}-01" for j in range(1, t + 1)],
        "site_no": [str(i) for i in range(n)], "static": torch.randn(n, 2),
        "regime": torch.randn(n, 13),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        "edge_attr": torch.randn(4, 6),
    }


def _split(n=5, t=8):
    cells = np.arange(n * t)
    return {"train": cells[:20], "val": cells[20:28], "test": cells[28:]}


def test_observation_statistics_are_causal_and_hidden_safe():
    y = torch.tensor([[1., 2., 3., 4.], [10., 20., 30., 40.]])
    visible = torch.tensor([[True, False, False, True], [False, True, False, False]])
    edge = torch.tensor([[0], [1]])
    first = observation_statistics(y, visible, edge)
    changed = y.clone()
    changed[:, 2:] = 10000
    second = observation_statistics(changed, visible, edge)
    for a, b in zip(first, second):
        assert torch.equal(a[:2], b[:2])
    # Station 0 at month 3 uses its month 0 value, not the future month 3 value.
    assert first[0][3, 0].item() == 4.0
    assert first[1][2, 0].item() > first[1][0, 0].item()


def test_observation_features_ignore_hidden_labels():
    data, split = _toy_dataset(), _split()
    inputs = build_temporal_inputs(data, split, target_transform="log1p")
    visible = torch.zeros_like(inputs.y_model, dtype=torch.bool)
    visible.reshape(-1)[split["train"]] = True
    edge = data["edge_index"]
    x1, age1 = build_observation_features(inputs, visible, edge)
    changed = data["y"].clone()
    changed.reshape(-1)[np.setdiff1d(np.arange(40), split["train"])] = 9999
    inputs_changed = build_temporal_inputs({**data, "y": changed}, split, target_transform="log1p")
    x2, age2 = build_observation_features(inputs_changed, visible, edge)
    # Train-derived target scaling can differ in the changed fixture; the
    # causal observation channels and age must remain identical.
    assert torch.equal(x1[..., -9:], x2[..., -9:])
    assert torch.equal(age1, age2)


def test_mask_sampler_produces_point_temporal_and_station_views():
    train = torch.arange(5 * 12)
    for mode in ("point", "temporal_block", "station_block"):
        context, target, used = sample_training_view(
            train, 5, 12, np.random.default_rng(42), mode=mode
        )
        assert used == mode
        assert len(context) and len(target)
        assert not np.intersect1d(context.numpy(), target.numpy()).size
        assert set(context.numpy()) | set(target.numpy()) == set(train.numpy())


def test_graph_upgrade_m1_smoke_fit_predict():
    model = GraphUpgradeModel(
        mechanism="m1", seed=42, hidden=8, temporal_hidden=8,
        max_epochs=1, patience=1, chunk_months=4,
    )
    pred = model.fit_predict(_toy_dataset(), _split())
    assert pred.shape == (5, 8)
    assert np.isfinite(pred).all()
    assert np.std(pred) > 0


def test_graph_upgrade_m2_and_m3_forward_shapes():
    data, split = _toy_dataset(), _split()
    for mechanism in ("m2", "m3"):
        model = GraphUpgradeModel(
            mechanism=mechanism, seed=42, hidden=8, temporal_hidden=8,
            max_epochs=1, patience=1, chunk_months=4,
        )
        pred = model.fit_predict(data, split)
        assert pred.shape == (5, 8)
        assert np.isfinite(pred).all()


def test_upgrade_temporal_predictions_do_not_read_future_months():
    data, split = _toy_dataset(), _split()
    model = GraphUpgradeModel(
        mechanism="m2", seed=42, hidden=8, temporal_hidden=8,
        max_epochs=1, patience=1, chunk_months=8,
    )
    model.fit(data, split)
    bundle = model._bundle
    visible, y_feed = model._visible_input()
    x, age = model._make_input(bundle.inputs, visible, bundle.model, y_feed)
    bundle.model.eval()
    with torch.no_grad():
        first = model._forward(bundle.model, x, bundle.inputs, data, age)
        changed = x.clone()
        changed[5:] += 100.0
        second = model._forward(bundle.model, changed, bundle.inputs, data, age)
    assert torch.equal(first[:, :5], second[:, :5])


def test_multiscale_temporal_path_is_causal():
    data, split = _toy_dataset(), _split()
    model = GraphUpgradeModel(
        mechanism="m3", seed=42, hidden=8, temporal_hidden=8,
        max_epochs=1, patience=1, chunk_months=8, lookback=24,
    )
    model.fit(data, split)
    bundle = model._bundle
    visible, y_feed = model._visible_input()
    x, _ = model._make_input(bundle.inputs, visible, bundle.model, y_feed)
    bundle.model.eval()
    with torch.no_grad():
        first = model._forward(bundle.model, x, bundle.inputs, data)
        changed = x.clone()
        changed[5:] += 100.0
        second = model._forward(bundle.model, changed, bundle.inputs, data)
    assert torch.equal(first[:, :5], second[:, :5])
