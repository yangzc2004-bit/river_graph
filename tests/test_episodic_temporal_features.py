"""Temporal-cache visibility and exact frozen-GRU replay without training."""

import copy

import numpy as np
import pytest
import torch

from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.kgml_local_transport import FIT_ROLES, RFArtifacts, fold_split
from river_graph.models.unified_doc import UnifiedDOCReconstructor
from river_graph.models.unified_doc_features import extract_gru_hidden


@pytest.fixture(scope="module")
def frozen_model():
    rng = np.random.default_rng(621)
    n, months = 12, 15
    data = {
        "y": torch.tensor(rng.uniform(0.2, 6, (n, months)), dtype=torch.float32),
        "y_mask": torch.ones(n, months, dtype=torch.bool),
        "x": torch.tensor(rng.uniform(0.2, 4, (n, months, 2)), dtype=torch.float32),
        "x_mask": torch.ones(n, months, 2),
        "static": torch.tensor(rng.uniform(0.1, 1, (n, 2)), dtype=torch.float32),
        "regime": torch.tensor(rng.uniform(0.1, 1, (n, 13)), dtype=torch.float32),
        "months": np.arange("2020-01", "2021-04", dtype="datetime64[M]").astype(str).tolist(),
        "site_no": [f"site{i}" for i in range(n)],
        "edge_index": torch.tensor([np.arange(n - 1).tolist(), np.arange(1, n).tolist()]),
        "edge_attr": torch.tensor(rng.uniform(0.1, 1, (n - 1, 6)), dtype=torch.float32),
    }
    # Nonconsecutive source stations and unsorted fold members exercise
    # assembly by global station identity rather than concatenation order.
    source_ids = np.array([0, 2, 3, 5, 6, 8, 9, 11])
    cells = np.arange(n * months).reshape(n, months)
    split = {"train": cells[source_ids].ravel(), "val": cells[[1, 7]].ravel(),
             "test": cells[[4, 10]].ravel(), "context": np.array([], dtype=np.int64)}
    target = np.log1p(data["y"].numpy()).ravel()[split["train"]]
    model = UnifiedDOCReconstructor(seed=42, n_estimators=3, n_jobs=1, hidden=8, chunk_months=4)
    model.dataset, model.split = data, split
    # No forests are needed to extract spatial states; only their saved fold
    # identities and target normalization are required by model initialization.
    model.rf = RFArtifacts(
        None, None, np.full((n, months), np.nan),
        [np.array([8, 0]), np.array([11, 3]), np.array([2, 9]), np.array([5]), np.array([6])],
        float(target.mean()), float(target.std()), "log1p", forest_backend="extra_trees",
        forest_min_samples_leaf=4, forest_max_features=1.0,
    )
    model.residual = model._new_residual()
    model.residual.initialize(data, split, model.rf)
    with torch.no_grad():
        model.residual.model.spatial.head.weight.copy_(torch.linspace(-0.5, 0.5, 8).reshape(1, 8))
        model.residual.model.spatial.head.bias.fill_(0.07)
    model.residual.model.eval()
    return model


def _replay(residual, cache, prefix):
    encoded = torch.from_numpy(cache[f"{prefix}_encoded"]).transpose(0, 1).contiguous()
    age = torch.from_numpy(cache[f"{prefix}_age"]).T.contiguous()
    support = torch.from_numpy(cache[f"{prefix}_support"]).transpose(0, 1).contiguous()
    with torch.inference_mode():
        return residual.model._memory_states(encoded, age, support).permute(1, 0, 2).numpy()


def test_full_cache_replays_frozen_gru_exactly_and_preserves_weights_and_modes(frozen_model):
    model = copy.deepcopy(frozen_model)
    residual = model.residual
    residual.model.train()
    residual.model.spatial.eval()
    prior_modes = [module.training for module in residual.model.modules()]
    prior_weights = {key: value.clone() for key, value in residual.model.state_dict().items()}
    reports = []
    cached = extract_temporal_inputs(model, model.dataset, model.split, reports.append)
    assert [module.training for module in residual.model.modules()] == prior_modes
    assert cached["full_encoded"].shape == (12, 15, 8)
    assert cached["full_age"].shape == (12, 15)
    assert cached["full_support"].shape == (12, 15, 3)
    assert cached["full_encoded"].dtype == np.float32
    residual.model.eval()
    expected = extract_gru_hidden(residual, model.split, verify_head=True)
    actual = _replay(residual, cached, "full")
    np.testing.assert_array_equal(actual, expected)
    assert np.ptp(actual) > 1e-4
    assert [row["stage"] for row in reports] == ["full_temporal_inputs"] + ["source_fold"] * 5
    for key, value in residual.model.state_dict().items():
        torch.testing.assert_close(value, prior_weights[key], rtol=0, atol=0)


def test_source_cache_matches_saved_fold_visibility_and_station_alignment(frozen_model):
    cache = extract_temporal_inputs(frozen_model, frozen_model.dataset, frozen_model.split)
    source_ids = cache["source_station_ids"]
    np.testing.assert_array_equal(source_ids, np.array([0, 2, 3, 5, 6, 8, 9, 11]))
    assert cache["source_encoded"].shape == (8, 15, 8)
    assert cache["source_age"].shape == (8, 15)
    assert cache["source_support"].shape == (8, 15, 3)
    residual = frozen_model.residual
    for stations in frozen_model.rf.folds:
        view = fold_split(frozen_model.split, stations, 15)
        positions = np.searchsorted(source_ids, stations)
        with torch.inference_mode():
            x, age = residual.input_view(view, FIT_ROLES)
            encoded = residual.model.encode_months(
                x, residual.model._edge_index, residual.model._edge_attr, residual.inputs.env_raw,
            ).permute(1, 0, 2).numpy()
            support = torch.stack([x[..., 9], x[..., -2], x[..., -1]], dim=-1).permute(1, 0, 2).numpy()
        np.testing.assert_array_equal(cache["source_encoded"][positions], encoded[stations])
        np.testing.assert_array_equal(cache["source_age"][positions], age.T.numpy()[stations])
        np.testing.assert_array_equal(cache["source_support"][positions], support[stations])
        assert torch.count_nonzero(x[:, stations, 8:10]) == 0
    assert not np.array_equal(cache["source_encoded"], cache["full_encoded"][source_ids])


def test_hidden_val_and_test_labels_cannot_change_temporal_cache(frozen_model):
    baseline = extract_temporal_inputs(frozen_model, frozen_model.dataset, frozen_model.split)
    changed = copy.deepcopy(frozen_model)
    hidden = np.concatenate([changed.split["val"], changed.split["test"]])
    changed.dataset["y"].reshape(-1)[hidden] += 10000
    changed.residual.inputs.y_model.reshape(-1)[hidden] += 10000
    actual = extract_temporal_inputs(changed, changed.dataset, changed.split)
    for name, expected in baseline.items():
        np.testing.assert_array_equal(actual[name], expected)


def test_held_source_fold_labels_cannot_change_its_temporal_cache(frozen_model):
    baseline = extract_temporal_inputs(frozen_model, frozen_model.dataset, frozen_model.split)
    changed = copy.deepcopy(frozen_model)
    stations = changed.rf.folds[0]
    changed.dataset["y"][stations] += 10000
    changed.residual.inputs.y_model[stations] += 10000
    actual = extract_temporal_inputs(changed, changed.dataset, changed.split)
    positions = np.searchsorted(baseline["source_station_ids"], stations)
    for name in ("source_encoded", "source_age", "source_support"):
        np.testing.assert_array_equal(actual[name][positions], baseline[name][positions])


@pytest.mark.parametrize("defect", ["mechanism", "operator", "hydro_only", "shuffle", "messages", "folds"])
def test_unsupported_paths_and_invalid_folds_rejected(frozen_model, defect):
    model = copy.deepcopy(frozen_model)
    if defect == "mechanism":
        model.residual.builder.mechanism = "m2"
    elif defect == "operator":
        model.residual.temporal_operator = "gru_attention"
    elif defect in ("hydro_only", "shuffle"):
        model.residual.model.history_ablation = defect
    elif defect == "messages":
        model.residual.model._edge_index = torch.tensor([[0], [1]])
    else:
        model.rf.folds[0] = np.array([0, 0])
    with pytest.raises(ValueError, match="standard M1|history ablation|no-message|exactly once"):
        extract_temporal_inputs(model, model.dataset, model.split)


def test_training_modes_restore_after_extraction_failure(frozen_model):
    model = copy.deepcopy(frozen_model)
    model.residual.model.train()
    model.residual.model.spatial.eval()
    before = [module.training for module in model.residual.model.modules()]

    def fail(_):
        raise RuntimeError("progress interruption")

    with pytest.raises(RuntimeError, match="progress interruption"):
        extract_temporal_inputs(model, model.dataset, model.split, fail)
    assert [module.training for module in model.residual.model.modules()] == before
