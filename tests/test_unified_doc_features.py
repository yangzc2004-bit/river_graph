"""Frozen representation checks; no neural training or production data required."""

import copy

import numpy as np
import pytest
import torch
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    RFArtifacts,
    build_rf_features,
    fold_split,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor
from river_graph.models.unified_doc_features import (
    extract_basis_inputs,
    extract_gru_hidden,
    tree_prediction_features,
)


@pytest.fixture(scope="module")
def frozen_model():
    """Initialize a GRU without optimization; fit three tiny synthetic trees."""
    rng = np.random.default_rng(421)
    n, months = 12, 9
    data = {
        "y": torch.tensor(rng.uniform(0.2, 6, (n, months)), dtype=torch.float32),
        "y_mask": torch.ones(n, months, dtype=torch.bool),
        "x": torch.tensor(rng.uniform(0.2, 4, (n, months, 2)), dtype=torch.float32),
        "x_mask": torch.ones(n, months, 2),
        "static": torch.tensor(rng.uniform(0.1, 1, (n, 2)), dtype=torch.float32),
        "regime": torch.tensor(rng.uniform(0.1, 1, (n, 13)), dtype=torch.float32),
        "months": np.arange("2020-01", "2020-10", dtype="datetime64[M]").astype(str).tolist(),
        "site_no": [f"site{i}" for i in range(n)],
        "edge_index": torch.tensor([np.arange(n - 1).tolist(), np.arange(1, n).tolist()]),
        "edge_attr": torch.tensor(rng.uniform(0.1, 1, (n - 1, 6)), dtype=torch.float32),
    }
    split = {"train": np.arange(8 * months), "val": np.arange(8 * months, 10 * months),
             "test": np.arange(10 * months, n * months), "context": np.array([], dtype=np.int64)}
    features = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    target = np.log1p(data["y"].numpy()).ravel()[split["train"]]
    forest = ExtraTreesRegressor(n_estimators=3, min_samples_leaf=4, random_state=42, n_jobs=1)
    forest.fit(features[split["train"]], target)
    model = UnifiedDOCReconstructor(seed=42, n_estimators=3, n_jobs=1, hidden=8, chunk_months=4)
    model.dataset, model.split = data, split
    model.context_forest = forest
    model.rf = RFArtifacts(
        forest, forest, np.full((n, months), np.nan), list(np.array_split(np.arange(8), 5)),
        float(target.mean()), float(target.std()), "log1p", forest_backend="extra_trees",
        forest_min_samples_leaf=4, forest_max_features=1.0,
    )
    model.residual = model._new_residual()
    model.residual.initialize(data, split, model.rf)
    with torch.no_grad():
        # Nonzero head ensures equivalence is not a vacuous zero-output check.
        model.residual.model.spatial.head.weight.copy_(torch.linspace(-0.5, 0.5, 8).reshape(1, 8))
        model.residual.model.spatial.head.bias.fill_(0.07)
    model.residual.model.eval()
    return model


def test_hidden_states_reproduce_nonzero_existing_head_and_preserve_training_flag(frozen_model):
    residual = frozen_model.residual
    residual.model.train()
    hidden = extract_gru_hidden(residual, frozen_model.split, verify_head=True)
    assert residual.model.training
    residual.model.eval()
    assert hidden.shape == (12, 9, 8)
    assert hidden.dtype == np.float32
    assert np.isfinite(hidden).all()
    with torch.inference_mode():
        reconstructed = residual.model.spatial.predict_from_hidden(torch.as_tensor(hidden)).numpy()
        x, age = residual.input_view(frozen_model.split, FIT_ROLES)
        expected = residual.delta_tensor(x, age).numpy()
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-6, atol=1e-7)
    assert float(expected.std()) > 1e-4


def test_per_tree_mean_matches_forest_prediction_and_batch_size_is_irrelevant(frozen_model):
    x = build_rf_features(frozen_model.dataset, frozen_model.split, FIT_ROLES,
                          target_transform="log1p", include_network=True)
    individual = tree_prediction_features(frozen_model.context_forest, x, row_batch_size=7)
    np.testing.assert_array_equal(individual, tree_prediction_features(frozen_model.context_forest, x))
    np.testing.assert_allclose(individual.mean(axis=1), frozen_model.context_forest.predict(x), rtol=2e-7)
    assert individual.dtype == np.float32
    assert individual.shape == (108, 3)


def test_val_and_test_label_perturbation_does_not_change_frozen_features(frozen_model):
    baseline = extract_basis_inputs(frozen_model, frozen_model.dataset, frozen_model.split)
    modified = copy.deepcopy(frozen_model)
    hidden_cells = np.concatenate([modified.split["val"], modified.split["test"]])
    modified.dataset["y"].reshape(-1)[hidden_cells] += 1000
    modified.residual.inputs.y_model.reshape(-1)[hidden_cells] += 1000
    changed = extract_basis_inputs(modified, modified.dataset, modified.split)
    for name in baseline:
        np.testing.assert_array_equal(changed[name], baseline[name])


def test_source_views_hide_held_station_labels_with_weights_and_statistics_frozen(frozen_model):
    baseline = extract_basis_inputs(frozen_model, frozen_model.dataset, frozen_model.split)
    modified = copy.deepcopy(frozen_model)
    held = modified.rf.folds[0]
    modified.dataset["y"][held] += 1000
    modified.residual.inputs.y_model[held] += 1000
    changed = extract_basis_inputs(modified, modified.dataset, modified.split)
    positions = np.searchsorted(baseline["source_station_ids"], held)
    for name in ("source_gru", "source_tree"):
        np.testing.assert_array_equal(changed[name][positions], baseline[name][positions])
    # Verify the exported rows really come from the fold-hidden representation.
    view = fold_split(frozen_model.split, held, 9)
    expected = extract_gru_hidden(frozen_model.residual, view)
    np.testing.assert_array_equal(baseline["source_gru"][positions], expected[held])
    x, _ = frozen_model.residual.input_view(view, FIT_ROLES)
    assert torch.count_nonzero(x[:, held, 8:10]) == 0


def test_basis_output_alignment_progress_and_model_weights_unchanged(frozen_model):
    prior = {key: value.clone() for key, value in frozen_model.residual.model.state_dict().items()}
    reports = []
    result = extract_basis_inputs(frozen_model, frozen_model.dataset, frozen_model.split,
                                  progress=reports.append, verify_head=True)
    np.testing.assert_array_equal(result["source_station_ids"], np.arange(8))
    assert result["source_gru"].shape == (8, 9, 8)
    assert result["source_tree"].shape == (8, 9, 3)
    assert result["full_tree"].shape == (12, 9, 3)
    assert [row["stage"] for row in reports] == ["full_gru", "full_tree"] + ["source_fold"] * 5
    for key, expected in prior.items():
        torch.testing.assert_close(frozen_model.residual.model.state_dict()[key], expected, rtol=0, atol=0)


def test_unsupported_temporal_paths_and_malformed_folds_are_rejected(frozen_model):
    modified = copy.deepcopy(frozen_model)
    modified.residual.temporal_operator = "gru_attention"
    with pytest.raises(ValueError, match="standard M1"):
        extract_gru_hidden(modified.residual, modified.split)
    modified.residual.temporal_operator = "gru"
    modified.residual.model.history_ablation = "hydro_only"
    with pytest.raises(ValueError, match="hydro-only"):
        extract_gru_hidden(modified.residual, modified.split)
    modified.residual.model.history_ablation = "none"
    modified.rf.folds[0] = np.array([0, 0])
    with pytest.raises(ValueError, match="exactly once"):
        extract_basis_inputs(modified, modified.dataset, modified.split)
