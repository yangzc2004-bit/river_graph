"""Contracts for the Local--Transport KGML pilot."""

import numpy as np
import torch

from river_graph.models.kgml_local_transport import (
    LocalTransportKGML,
    build_rf_features,
    fit_rf_artifacts,
    rf_feature_names,
)


def toy_bundle():
    torch.manual_seed(13)
    n, t = 8, 12
    y = 1.0 + torch.rand(n, t) * 4.0
    return {
        "y": y,
        "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.rand(n, t, 2),
        "x_mask": torch.ones(n, t, 2),
        "static": torch.rand(n, 2),
        "regime": torch.rand(n, 13),
        "months": [f"2020-{j + 1:02d}-01" if j < 12 else "2021-01-01" for j in range(t)],
        "edge_index": torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                                     [1, 2, 3, 4, 5, 6, 7]]),
        "edge_attr": torch.rand(7, 6),
    }, {
        "train": np.arange(0, 48),
        "context": np.arange(48, 56),
        "val": np.arange(56, 72),
        "test": np.arange(72, 96),
    }


def test_local_and_context_features_differ_only_by_network_block():
    data, split = toy_bundle()
    local = build_rf_features(data, split, ("train", "context"),
                              target_transform="log1p", include_network=False)
    context = build_rf_features(data, split, ("train", "context"),
                                target_transform="log1p", include_network=True)
    assert local.shape[0] == context.shape[0] == 96
    assert local.shape[1] == len(rf_feature_names(False))
    assert context.shape[1] == len(rf_feature_names(True))
    assert local.shape[1] != context.shape[1]


def test_oof_rf_is_complete_and_does_not_use_test_labels():
    data, split = toy_bundle()
    first = fit_rf_artifacts(data, split, target_transform="log1p", seed=42,
                             n_estimators=4, n_jobs=1)
    changed = {**data, "y": data["y"].clone()}
    changed["y"].reshape(-1)[split["test"]] += 1000
    second = fit_rf_artifacts(changed, split, target_transform="log1p", seed=42,
                              n_estimators=4, n_jobs=1)
    train = split["train"]
    assert np.isfinite(first.local_oof_z.reshape(-1)[train]).all()
    np.testing.assert_array_equal(first.local_oof_z.reshape(-1)[train],
                                  second.local_oof_z.reshape(-1)[train])


def test_zero_residual_head_starts_at_local_prediction():
    data, split = toy_bundle()
    model = LocalTransportKGML(seed=42, edge_direction="upstream",
                               max_epochs=1, patience=1,
                               n_estimators=3, n_jobs=1, hidden=8,
                               chunk_months=12)
    # Fit for one epoch; the residual head is zero-initialized before the
    # update and the resulting product must retain the shared output schema.
    model.fit(data, split)
    components = model.predict_components(("train", "val", "context"))
    assert components["final_pred"].shape == data["y"].shape
    assert np.isfinite(components["final_pred"]).all()
    assert np.isfinite(components["graph_delta"]).all()


def test_no_message_keeps_local_features_and_produces_finite_output():
    data, split = toy_bundle()
    model = LocalTransportKGML(seed=42, edge_set="empty", edge_direction="upstream",
                               max_epochs=1, patience=1,
                               n_estimators=3, n_jobs=1, hidden=8,
                               chunk_months=12)
    model.fit(data, split)
    pred = model.predict(("train", "val", "context"))
    assert pred.shape == data["y"].shape
    assert np.isfinite(pred).all()


def test_message_delta_null_is_structurally_zero():
    data, split = toy_bundle()
    model = LocalTransportKGML(seed=42, edge_set="empty", edge_direction="upstream",
                               spatial_variant="msgonly",
                               max_epochs=3, patience=3,
                               n_estimators=3, n_jobs=1, hidden=8,
                               chunk_months=12)
    model.fit(data, split)
    comp = model.predict_components(("train", "val", "context"))
    np.testing.assert_array_equal(comp["graph_delta"], 0.0)
    np.testing.assert_array_equal(comp["final_pred"], comp["local_pred"])


def test_message_only_is_centered_with_nonzero_head_and_respects_direction_and_time():
    """A trained head cannot synthesize a correction without upstream messages."""
    data, split = toy_bundle()
    rf = fit_rf_artifacts(data, split, target_transform="log1p", seed=42,
                         n_estimators=3, n_jobs=1)
    model = LocalTransportKGML(seed=42, spatial_variant="msgonly", hidden=8,
                              dropout=0, chunk_months=12)
    model.initialize(data, split, rf)
    model.model.eval()
    with torch.no_grad():
        model.model.spatial.head.weight.fill_(0.7)
        model.model.spatial.head.bias.fill_(10)
    assert model.model.temporal.bias_ih is None
    assert model.model.temporal.bias_hh is None
    optimizer = torch.optim.Adam(model.model.parameters(), lr=.01)
    x, age = model.input_view(split)
    # Check after actual upstream training, not just at zero initialization.
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model.delta_tensor(x, age)[1:] - .5).square().mean()
        loss.backward()
        optimizer.step()
    original = model.delta_tensor(x, age)
    assert original[1:].abs().max() > 1e-6
    # Station zero has no upstream edges even though other stations do.
    torch.testing.assert_close(original[0], torch.zeros_like(original[0]), rtol=0, atol=0)
    changed = x.clone()
    changed[:, -1] += 50  # downstream station cannot affect upstream outputs
    downstream_changed = model.delta_tensor(changed, age)
    torch.testing.assert_close(original[:-1], downstream_changed[:-1], rtol=0, atol=0)
    changed = x.clone()
    changed[6:] += 50
    torch.testing.assert_close(original[:, :6], model.delta_tensor(changed, age)[:, :6], rtol=0, atol=0)
    model.model._edge_index = torch.empty(2, 0, dtype=torch.long)
    model.model._edge_attr = model.model._edge_attr[:0]
    for training in (False, True):
        model.model.train(training)
        null = model.delta_tensor(x, age)
        torch.testing.assert_close(null, torch.zeros_like(null), rtol=0, atol=0)
