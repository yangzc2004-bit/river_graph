"""New-site inference preserves fitted preprocessing and label isolation."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.ensemble import ExtraTreesRegressor
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from portable_doc_reconstructor_v1 import (
    PortableDOCReconstructor,
    fitted_preprocessing,
    fitted_source_bank,
)

from river_graph.experiments.graph_upgrade_v2 import build_observation_features
from river_graph.experiments.temporal_h2x import build_temporal_inputs
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    role_visible,
)
from river_graph.models.regime_head_features import build_regime_head_features


@pytest.fixture
def sample():
    rng = np.random.default_rng(44)
    n, months = 6, 16
    dataset = {"site_no": np.array([f"site{i}" for i in range(n)]),
        "months": np.arange("2000-01", "2001-05", dtype="datetime64[M]"),
        "y": rng.uniform(2, 6, (n, months)), "y_mask": np.ones((n, months), bool),
        "x": rng.uniform(1, 20, (n, months, 2)), "x_mask": np.ones((n, months, 2), bool),
        "static": rng.uniform(1, 10, (n, 2)), "regime": rng.uniform(0, 1, (n, 13)),
        "edge_index": np.array([[0, 1, 4], [4, 5, 2]])}
    split = {"train": np.arange(4*months), "val": np.arange(4*months, 5*months),
             "test": np.arange(5*months, 6*months), "context": np.array([], dtype=int)}
    daily = np.zeros((n, months, 8), np.float32)
    x = np.column_stack([build_rf_features(dataset, split, FIT_ROLES,
        target_transform="log1p", include_network=True), daily.reshape(-1, 8)])
    forest = ExtraTreesRegressor(n_estimators=3, max_depth=3, n_jobs=1, random_state=42)
    forest.fit(x[split["train"]], np.log1p(dataset["y"].ravel()[split["train"]]))
    context = np.expm1(forest.predict(x)).reshape(n, months)
    oof = np.full((n, months), np.log1p(3.))
    features = build_regime_head_features(dataset["regime"], split["train"],
        np.expm1(oof.ravel()[split["train"]]), context,
        build_causal_flow_features(dataset)["full"], n_months=months)
    with torch.random.fork_rng():
        torch.manual_seed(10)
        spatial = TransportGCNImputer(23, 2, hidden=4, layers=2, dropout=.1, env_dim=9, env_emb=3)
        model = EncoderNativeResidual(spatial, nn.GRUCell(5, 4), nn.Linear(4, 4),
            epochs=0, extra_dim=38, interaction_indices=(0, 2, 4, 28, 30, 31, 32), batch_size=128)
    memory = EcologicalResidualTransfer(k_grid=(2,), ridge_grid=(.1,), gamma_grid=(0., .5)).fit(
        dataset["regime"], split["train"], np.expm1(oof.ravel()[split["train"]]),
        dataset["y"].ravel()[split["train"]], n_months=months,
        validation_cells=split["val"], validation_y=dataset["y"].ravel()[split["val"]],
        validation_context=context.ravel()[split["val"]], validation_temporal=context.ravel()[split["val"]])
    portable = PortableDOCReconstructor(forest=forest, native=model, memory_state=memory.to_dict(),
        preprocessing=fitted_preprocessing(dataset, split), source_bank=fitted_source_bank(dataset, split, oof),
        readout_normalization=features["normalization"])
    inputs = {k: np.asarray(dataset[k])[4:] for k in ("site_no", "x", "x_mask", "static", "regime")}
    inputs.update(months=dataset["months"], daily_features=daily[4:])
    _, _, neural = portable.prepare_inputs(inputs)
    valid = np.ones((2, months), bool)
    model.fit(neural, context[4:], context[4:]+.2, valid,
        neural, context[4:], context[4:]+.2, valid, tail_threshold=5., selection_role="source_validation")
    # Exercise the neural path and nonzero ecological fusion, not only a forest.
    with torch.no_grad():
        model.head.weight.fill_(.001)
        model.head.bias.fill_(.1)
    model.selected_scale_ = .5
    portable.memory_state["selected"]["gamma"] = .5
    return portable, inputs, dataset, split, context


def test_prepared_features_replay_the_existing_hidden_m1_view(sample):
    portable, inputs, dataset, split, context = sample
    forest, prediction, neural = portable.prepare_inputs(inputs)
    expected = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    np.testing.assert_array_equal(forest[..., :39], expected.reshape(6, 16, 39)[4:])
    np.testing.assert_allclose(prediction, context[4:], rtol=1e-12, atol=1e-12)
    static = build_temporal_inputs(dataset, split, target_transform="log1p")
    visible = torch.as_tensor(role_visible(split, FIT_ROLES, dataset["y"].shape))
    raw, age = build_observation_features(static, visible, torch.as_tensor(dataset["edge_index"]))
    raw[..., -1] = 0  # Same upstream-only KGML feature view in all arms.
    np.testing.assert_allclose(neural["raw"], raw.permute(1, 0, 2)[4:], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(neural["age"], age.T[4:])
    np.testing.assert_array_equal(neural["env"], static.env_raw[4:])


def test_new_water_labels_never_enter_features_or_saved_source_statistics(sample):
    portable, inputs, dataset, split, _ = sample
    first = portable.predict(inputs)
    altered = copy.deepcopy(inputs)
    altered.update(y=np.full((2, 16), np.nan), ph=np.full((2, 16), 1e7),
                   spec_conductance=np.full((2, 16), -1e9))
    np.testing.assert_array_equal(first, portable.predict(altered))
    data_changed = copy.deepcopy(dataset)
    data_changed["y"][4:] = np.nan
    a, b = fitted_preprocessing(dataset, split), fitted_preprocessing(data_changed, split)
    for key in ("hydro", "coordinates", "regime", "target"):
        for statistic in ("mean", "sd"):
            np.testing.assert_array_equal(a[key][statistic], b[key][statistic])
    bank = portable.source_bank
    assert not np.intersect1d(bank["station"], inputs["site_no"]).size
    assert len(bank["profile_truth"]) == len(split["train"])


def test_causal_future_features_and_arbitrary_new_station_count(sample):
    portable, inputs, *_ = sample
    before = portable.predict(inputs)
    changed = copy.deepcopy(inputs)
    changed["x"][:, 10:] *= 80
    changed["daily_features"][:, 10:] = .75
    after = portable.predict(changed)
    np.testing.assert_array_equal(before[:, :10], after[:, :10])
    assert not np.allclose(before[:, 10:], after[:, 10:])
    # Apply one fixed model to a different N; results cannot depend on the
    # surrounding target cohort or its order/normalization.
    one = {k: (v[:1] if k != "months" else v) for k, v in inputs.items()}
    np.testing.assert_allclose(portable.predict(one), before[:1], rtol=1e-6, atol=1e-6)
    reordered = {k: (v[::-1] if k != "months" else v) for k, v in inputs.items()}
    np.testing.assert_allclose(portable.predict(reordered), before[::-1], rtol=1e-6, atol=1e-6)
    extended = copy.deepcopy(one)
    extended["site_no"] = np.array(["another-new-site"])
    assert portable.predict(extended).shape == (1, 16)


def test_component_closure_save_load_and_no_unknown_calendar_extrapolation(sample, tmp_path):
    portable, inputs, *_ = sample
    first = portable.predict_components(inputs)
    np.testing.assert_allclose(first["final_pred"], first["environment_pred"]+
        first["local_temporal_correction"]+first["source_transfer_correction"]+first["river_correction"])
    portable.save(tmp_path/"fitted")
    restored = PortableDOCReconstructor.load(tmp_path/"fitted")
    np.testing.assert_array_equal(restored.predict(inputs), first["final_pred"])
    with pytest.raises(FileExistsError):
        portable.save(tmp_path/"fitted")
    outside = copy.deepcopy(inputs)
    outside["months"] = np.arange("2100-01", "2101-05", dtype="datetime64[M]")
    forest, _, _ = portable.prepare_inputs(outside)
    np.testing.assert_array_equal(forest[..., 21:27], 0.)
    assert np.isfinite(portable.predict(outside)).all()
    overlapping = copy.deepcopy(inputs)
    overlapping["site_no"][0] = portable.source_bank["station"][0]
    with pytest.raises(ValueError, match="exclude the source library"):
        portable.predict(overlapping)


def test_support_calibration_uses_only_explicit_support_and_saved_source_policy(sample):
    portable, inputs, *_ = sample
    portable.metadata["support_adapters"] = {str(k): {"selected": {"alpha": .5}}
                                           for k in (1, 3, 5)}
    cells, values = np.array([0, 16]), np.array([8., 7.])
    base = portable.predict(inputs)
    np.testing.assert_array_equal(portable.predict_with_support(inputs, k=0,
        support_cells=np.array([], dtype=int), support_values=np.array([])), base)
    corrected = portable.predict_with_support(inputs, k=1, support_cells=cells, support_values=values)
    changed = copy.deepcopy(inputs)
    changed["y"] = np.full((2, 16), 900000.)
    np.testing.assert_array_equal(portable.predict_with_support(changed, k=1,
        support_cells=cells, support_values=values), corrected)
    assert not np.allclose(corrected, base)
    # No future support is implicitly discovered in the supplied label table.
    with pytest.raises(ValueError, match="exceeds"):
        portable.predict_with_support(inputs, k=1, support_cells=np.array([0, 1]), support_values=values)
