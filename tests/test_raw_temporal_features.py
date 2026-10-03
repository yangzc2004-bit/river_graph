"""Raw cache alignment, label visibility and frozen-encoder replay."""

import copy

import numpy as np
import pytest
import torch

from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.kgml_local_transport import FIT_ROLES, RFArtifacts, fold_split
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.unified_doc import UnifiedDOCReconstructor


@pytest.fixture(scope="module")
def expert():
    rng = np.random.default_rng(623)
    n, months = 10, 15
    data = {
        "y": torch.tensor(rng.uniform(.2, 6, (n, months)), dtype=torch.float32),
        "y_mask": torch.ones(n, months, dtype=torch.bool),
        "x": torch.tensor(rng.uniform(.2, 4, (n, months, 2)), dtype=torch.float32),
        "x_mask": torch.ones(n, months, 2),
        "static": torch.tensor(rng.uniform(.1, 1, (n, 2)), dtype=torch.float32),
        "regime": torch.tensor(rng.uniform(.1, 1, (n, 13)), dtype=torch.float32),
        "months": np.arange("2020-01", "2021-04", dtype="datetime64[M]").astype(str).tolist(),
        "site_no": [f"site{i}" for i in range(n)],
        "edge_index": torch.tensor([np.arange(n-1).tolist(), np.arange(1, n).tolist()]),
        "edge_attr": torch.tensor(rng.uniform(.1, 1, (n-1, 6)), dtype=torch.float32),
    }
    cells = np.arange(n*months).reshape(n, months)
    split = {"train": cells[[0, 2, 4, 5, 7, 9]].ravel(), "val": cells[[1, 6]].ravel(),
             "test": cells[[3, 8]].ravel(), "context": np.array([], dtype=np.int64)}
    target = np.log1p(data["y"].numpy()).ravel()[split["train"]]
    model = UnifiedDOCReconstructor(seed=42, n_estimators=3, n_jobs=1, hidden=8, chunk_months=4)
    model.dataset, model.split = data, split
    # Extraction needs saved fold identities, not trained forests.
    model.rf = RFArtifacts(None, None, np.full((n, months), np.nan),
        [np.array([7, 0]), np.array([9]), np.array([2]), np.array([5]), np.array([4])],
        float(target.mean()), float(target.std()), "log1p", forest_backend="extra_trees",
        forest_min_samples_leaf=4, forest_max_features=1.0)
    model.residual = model._new_residual()
    model.residual.initialize(data, split, model.rf)
    model.residual.model.eval()
    return model


def _encode(expert, raw, env):
    model = expert.residual.model
    with torch.inference_mode():
        return model.encode_months(torch.from_numpy(raw).transpose(0, 1).contiguous(),
            model._edge_index, model._edge_attr, torch.from_numpy(env)).permute(1, 0, 2).numpy()


def test_raw_cache_exactly_replays_frozen_encoder_and_preserves_state(expert):
    model = copy.deepcopy(expert)
    model.residual.model.train()
    model.residual.model.spatial.eval()
    modes = [module.training for module in model.residual.model.modules()]
    weights = {name: value.clone() for name, value in model.residual.model.state_dict().items()}
    reports = []
    raw = extract_raw_temporal_inputs(model, model.dataset, model.split, reports.append)
    assert [module.training for module in model.residual.model.modules()] == modes
    assert raw["full_raw"].shape[:2] == (10, 15)
    assert raw["source_raw"].shape[:2] == (6, 15)
    assert all(np.isfinite(array).all() for array in raw.values())
    assert raw["full_raw"].dtype == np.float32
    np.testing.assert_array_equal(raw["env"], model.residual.inputs.env_raw.numpy())
    np.testing.assert_array_equal(raw["source_env"], raw["env"][raw["source_station_ids"]])
    assert [row["stage"] for row in reports] == ["full_raw_temporal_inputs"] + ["source_fold"]*5
    model.residual.model.eval()
    encoded = extract_temporal_inputs(model, model.dataset, model.split)
    np.testing.assert_array_equal(_encode(model, raw["full_raw"], raw["env"]), encoded["full_encoded"])
    # Keep the original graph-batch dimensions, replacing each source row by
    # its held-fold input. No-message encoding has no cross-station dependency.
    source_view = raw["full_raw"].copy()
    source_view[raw["source_station_ids"]] = raw["source_raw"]
    np.testing.assert_array_equal(_encode(model, source_view, raw["env"])[raw["source_station_ids"]],
                                  encoded["source_encoded"])
    for prefix in ("full", "source"):
        for name in ("age", "support"):
            np.testing.assert_array_equal(raw[f"{prefix}_{name}"], encoded[f"{prefix}_{name}"])
    for name, value in model.residual.model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_source_station_alignment_and_support_channels(expert):
    cache = extract_raw_temporal_inputs(expert, expert.dataset, expert.split)
    np.testing.assert_array_equal(cache["source_station_ids"], [0, 2, 4, 5, 7, 9])
    for stations in expert.rf.folds:
        x, age = expert.residual.input_view(fold_split(expert.split, stations, 15), FIT_ROLES)
        positions = np.searchsorted(cache["source_station_ids"], stations)
        expected = x.permute(1, 0, 2).numpy()[stations]
        np.testing.assert_array_equal(cache["source_raw"][positions], expected)
        np.testing.assert_array_equal(cache["source_age"][positions], age.T.numpy()[stations])
        np.testing.assert_array_equal(cache["source_support"][positions], expected[..., [9, -2, -1]])
        assert np.count_nonzero(expected[..., 8:10]) == 0
    assert not np.array_equal(cache["source_raw"], cache["full_raw"][cache["source_station_ids"]])


def test_hidden_validation_and_test_labels_do_not_change_any_raw_view(expert):
    baseline = extract_raw_temporal_inputs(expert, expert.dataset, expert.split)
    changed = copy.deepcopy(expert)
    cells = np.concatenate([changed.split["val"], changed.split["test"]])
    changed.dataset["y"].reshape(-1)[cells] += 10000
    changed.residual.inputs.y_model.reshape(-1)[cells] += 10000
    actual = extract_raw_temporal_inputs(changed, changed.dataset, changed.split)
    for name, value in baseline.items():
        np.testing.assert_array_equal(actual[name], value)


def test_held_source_fold_labels_do_not_change_its_raw_view(expert):
    baseline = extract_raw_temporal_inputs(expert, expert.dataset, expert.split)
    changed = copy.deepcopy(expert)
    stations = changed.rf.folds[0]
    changed.dataset["y"][stations] += 10000
    changed.residual.inputs.y_model[stations] += 10000
    actual = extract_raw_temporal_inputs(changed, changed.dataset, changed.split)
    positions = np.searchsorted(baseline["source_station_ids"], stations)
    for name in ("source_raw", "source_age", "source_support", "source_env"):
        np.testing.assert_array_equal(actual[name][positions], baseline[name][positions])


def test_future_visible_labels_cannot_change_current_or_earlier_inputs(expert):
    baseline = extract_raw_temporal_inputs(expert, expert.dataset, expert.split)
    changed = copy.deepcopy(expert)
    changed.dataset["y"][:, 8:] += 10000
    changed.residual.inputs.y_model[:, 8:] += 10000
    actual = extract_raw_temporal_inputs(changed, changed.dataset, changed.split)
    for prefix in ("full", "source"):
        for name in ("raw", "age", "support"):
            np.testing.assert_array_equal(actual[f"{prefix}_{name}"][:, :8], baseline[f"{prefix}_{name}"][:, :8])
    np.testing.assert_array_equal(actual["env"], baseline["env"])
    assert not np.array_equal(actual["full_raw"][:, 8:], baseline["full_raw"][:, 8:])


@pytest.mark.parametrize("defect", ["mechanism", "operator", "history", "messages", "folds", "env"])
def test_incompatible_models_or_inputs_rejected(expert, defect):
    model = copy.deepcopy(expert)
    if defect == "mechanism":
        model.residual.builder.mechanism = "m2"
    elif defect == "operator":
        model.residual.temporal_operator = "gru_attention"
    elif defect == "history":
        model.residual.model.history_ablation = "shuffle"
    elif defect == "messages":
        model.residual.model._edge_index = torch.tensor([[0], [1]])
    elif defect == "env":
        model.residual.inputs.env_raw[0, 0] = float("nan")
    else:
        model.rf.folds[0] = np.array([0, 0])
    with pytest.raises(ValueError, match="standard M1|history ablation|no-message|exactly once|ecological input"):
        extract_raw_temporal_inputs(model, model.dataset, model.split)


def test_modes_restored_after_progress_failure(expert):
    model = copy.deepcopy(expert)
    model.residual.model.train()
    model.residual.model.spatial.eval()
    modes = [module.training for module in model.residual.model.modules()]

    def fail(_):
        raise RuntimeError("interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        extract_raw_temporal_inputs(model, model.dataset, model.split, fail)
    assert [module.training for module in model.residual.model.modules()] == modes
