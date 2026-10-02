"""End-to-end contracts for the fitted DOC reconstruction and adaptation model."""

import copy

import numpy as np
import pytest
import torch

from river_graph.experiments.unified_spatial_protocol import (
    build_unified_spatial_split,
    support_query_cells,
)
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.unified_doc import UnifiedDOCReconstructor


def _small_station_dataset():
    """Station shifts make support useful while preserving a temporal signal."""
    rng = np.random.default_rng(71)
    n, t = 14, 18
    x = rng.uniform(0.2, 2.0, size=(n, t, 2)).astype(np.float32)
    static = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
    seasonal = np.sin(np.arange(t) * 2 * np.pi / 12)
    y = (1.0 + x[..., 0] + 0.3 * seasonal + 0.2 * static[:, :1]
         + rng.uniform(0, 0.1, size=(n, t))).astype(np.float32)
    observed = np.ones((n, t), dtype=bool)
    split, metadata = build_unified_spatial_split(observed, seed=142)
    # A deliberately measurable target-station offset exercises calibration.
    for role in ("val", "test"):
        for index, station in enumerate(metadata["station_roles"][role]):
            y[station] += 3.0 + 4.0 * index
    months = np.arange("2020-01", "2021-07", dtype="datetime64[M]")
    data = {
        "y": torch.as_tensor(y),
        "y_mask": torch.as_tensor(observed),
        "x": torch.as_tensor(x),
        "x_mask": torch.ones(n, t, 2),
        "static": torch.as_tensor(static),
        "regime": torch.as_tensor(rng.uniform(0, 1, size=(n, 13)).astype(np.float32)),
        "months": months.astype(str).tolist(),
        "site_no": [f"station_{i}" for i in range(n)],
        "edge_index": torch.tensor([np.arange(n - 1).tolist(), np.arange(1, n).tolist()]),
        "edge_attr": torch.as_tensor(rng.uniform(0.1, 1, size=(n - 1, 6)).astype(np.float32)),
    }
    return data, split


def _new_model():
    return UnifiedDOCReconstructor(seed=42, n_estimators=3, n_jobs=1,
                                   max_epochs=1, patience=1, hidden=8,
                                   chunk_months=12)


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory):
    data, split = _small_station_dataset()
    model = _new_model().fit(data, split)
    directory = tmp_path_factory.mktemp("unified_doc_model")
    model.save(directory)
    return model, data, split, directory


def _predictions(model, data, split, *, arm, k):
    support, query = support_query_cells(
        split, target_role="test", k=k, n_months=data["y"].shape[1],
    )
    values = data["y"].numpy().ravel()[support]
    prediction = model.predict(query, support_cells=support, support_values=values,
                               k=k, arm=arm)
    return support, query, prediction


def test_fit_and_predict_four_k_values_with_exact_zero_shot_identity(fitted_model):
    model, data, split, _ = fitted_model
    components = model.predict_components()
    assert model.residual.epochs_run == 1
    assert len(model.rf.folds) == 5
    assert all(pred.shape == data["y"].shape for pred in components.values())
    assert all(np.isfinite(pred).all() for pred in components.values())
    assert model.adapter.fusion_selection_role_ == "source_validation"
    assert model.adapter.calibration_selection_role_ == "source_validation"
    for arm in ("context", "hybrid"):
        queries = []
        for k in (0, 1, 3, 5):
            support, query, prediction = _predictions(model, data, split, arm=arm, k=k)
            queries.append(query)
            assert len(support) == 3 * k
            assert prediction.shape == query.shape
            assert np.isfinite(prediction).all()
            assert (prediction >= 0).all()
            if k == 0:
                np.testing.assert_array_equal(prediction, components[f"{arm}_pred"].ravel()[query])
                np.testing.assert_array_equal(prediction, model.predict(query, arm=arm, calibrated=False))
        assert all(np.array_equal(queries[0], query) for query in queries)


def test_serialized_full_model_reproduces_components_and_adaptation(fitted_model):
    model, data, split, directory = fitted_model
    restored = UnifiedDOCReconstructor.load(directory, data, split)
    assert restored.context_selection == model.context_selection
    assert restored.adapter.to_dict() == model.adapter.to_dict()
    for key, values in model.predict_components().items():
        np.testing.assert_array_equal(restored.predict_components()[key], values)
    for arm in ("context", "hybrid"):
        for k in (0, 1, 3, 5):
            expected = _predictions(model, data, split, arm=arm, k=k)[2]
            actual = _predictions(restored, data, split, arm=arm, k=k)[2]
            np.testing.assert_array_equal(actual, expected)


def test_retraining_cannot_read_outer_test_queries(fitted_model):
    model, data, split, _ = fitted_model
    altered = copy.deepcopy(data)
    _, query = support_query_cells(split, target_role="test", k=0, n_months=18)
    altered["y"].reshape(-1)[query] += 1000
    refit = _new_model().fit(altered, split)
    assert refit.context_name == model.context_name
    assert refit.context_selection == model.context_selection
    assert refit.rf.target_mu == model.rf.target_mu
    assert refit.rf.target_sd == model.rf.target_sd
    assert refit.adapter.to_dict() == model.adapter.to_dict()
    assert refit.residual.trace == model.residual.trace
    for first, second in ((refit.rf.local_oof_z, model.rf.local_oof_z),
                          (refit.rf.context_oof_z, model.rf.context_oof_z)):
        np.testing.assert_array_equal(first, second)
    for name, parameter in model.residual.model.state_dict().items():
        torch.testing.assert_close(refit.residual.model.state_dict()[name], parameter,
                                   rtol=0, atol=0)
    for key, expected in model.predict_components().items():
        np.testing.assert_array_equal(refit.predict_components()[key], expected)
    for arm in ("context", "hybrid"):
        for k in (0, 1, 3, 5):
            # The perturbation leaves all five reserved support labels intact.
            expected = _predictions(model, data, split, arm=arm, k=k)[2]
            actual = _predictions(refit, altered, split, arm=arm, k=k)[2]
            np.testing.assert_array_equal(actual, expected)


def test_inference_rebuild_hides_val_and_test_labels_but_support_adapter_uses_values(fitted_model):
    model, data, split, directory = fitted_model
    changed = copy.deepcopy(data)
    for role in ("val", "test"):
        changed["y"].reshape(-1)[split[role]] += 500
    # Reload rebuilds normalized temporal inputs from altered labels, so this
    # check cannot pass simply by returning the original component cache.
    loaded = UnifiedDOCReconstructor.load(directory, changed, split)
    for network in (False, True):
        before = build_rf_features(data, split, FIT_ROLES,
                                   target_transform="log1p", include_network=network)
        after = build_rf_features(changed, split, FIT_ROLES,
                                  target_transform="log1p", include_network=network)
        np.testing.assert_array_equal(after, before)
    original_input, original_age = model.residual.input_view(split, FIT_ROLES)
    changed_input, changed_age = loaded.residual.input_view(split, FIT_ROLES)
    torch.testing.assert_close(changed_input, original_input, rtol=0, atol=0)
    torch.testing.assert_close(changed_age, original_age, rtol=0, atol=0)
    for key, expected in model.predict_components().items():
        np.testing.assert_array_equal(loaded.predict_components(recompute=True)[key], expected)
    support, query, expected = _predictions(model, data, split, arm="context", k=5)
    labels = data["y"].numpy().ravel()[support]
    unchanged = loaded.predict(query, support_cells=support, support_values=labels,
                               k=5, arm="context")
    np.testing.assert_array_equal(unchanged, expected)
    # This synthetic station shift makes nonzero source-selected shrinkage
    # useful; an explicit support update must therefore affect calibration.
    assert loaded.adapter.alpha_by_arm_["context"][5] > 0
    updated = loaded.predict(query, support_cells=support, support_values=labels + 2,
                             k=5, arm="context")
    assert np.all(updated > unchanged)
    np.testing.assert_array_equal(loaded.predict_components()["context_pred"],
                                  model.predict_components()["context_pred"])
