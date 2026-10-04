"""Fresh native stage roles and cache contracts without neural optimization."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.kgml_local_transport import RFArtifacts
from river_graph.models.unified_doc import UnifiedDOCReconstructor


def helper():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("confirmation_native", scripts / "doc_chemistry_confirmation_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_inputs():
    """Initialize real input processors/weights, but fit no forest or neural model."""
    rng = np.random.default_rng(71)
    n, months = 7, 9
    data = {
        "y": torch.tensor(rng.uniform(.2, 6, (n, months)), dtype=torch.float32),
        "y_mask": torch.ones(n, months, dtype=torch.bool),
        "x": torch.tensor(rng.uniform(.2, 4, (n, months, 2)), dtype=torch.float32),
        "x_mask": torch.ones(n, months, 2),
        "static": torch.tensor(rng.uniform(.1, 1, (n, 2)), dtype=torch.float32),
        "regime": torch.tensor(rng.uniform(.1, 1, (n, 13)), dtype=torch.float32),
        "months": np.arange("2020-01", "2020-10", dtype="datetime64[M]").astype(str).tolist(),
        "site_no": [f"site{i}" for i in range(n)],
        "edge_index": torch.tensor([np.arange(n - 1).tolist(), np.arange(1, n).tolist()]),
        "edge_attr": torch.tensor(rng.uniform(.1, 1, (n - 1, 6)), dtype=torch.float32),
    }
    data["y_mask"][1, 2] = False
    cells = np.arange(n * months).reshape(n, months)
    train = cells[:5][data["y_mask"][:5].numpy()]
    split = {"train": train, "val": cells[5], "test": cells[6],
             "context": np.empty(0, dtype=np.int64)}
    target = np.log1p(data["y"].numpy()).ravel()[train]
    expert = UnifiedDOCReconstructor(seed=42, hidden=64, n_estimators=3, n_jobs=1)
    expert.dataset, expert.split = data, split
    expert.rf = RFArtifacts(None, None, np.full((n, months), np.nan),
        [np.array([i]) for i in range(5)], float(target.mean()), float(target.std()),
        "log1p", forest_backend="extra_trees", forest_min_samples_leaf=4, forest_max_features=1.)
    expert.residual = expert._new_residual()
    expert.residual.initialize(data, split, expert.rf)
    expert.residual.model.eval()
    oof = np.full((n, months), np.nan)
    oof.ravel()[train] = np.log1p(1.5)
    initial = {"expert": expert, "context": np.full(n * months, 2.), "oof_z": oof,
        "features": extract_temporal_inputs(expert, data, split),
        "source_station_ids": np.arange(5), "legacy_basis": np.zeros((n * months, 2))}
    daily = np.zeros((n, months, 8), dtype=np.float32)
    metadata = {"value_feature_indices": [0, 1, 2], "availability_feature_indices": [3, 4, 5, 6, 7],
                "feature_names": [f"daily{i}" for i in range(8)], "policy": {"example": "fixed"}}
    return data, split, initial, daily, metadata


class _BeforeFit(Exception):
    pass


def capture_fit(module, monkeypatch, root, arguments, *, smoke=False):
    data, split, initial, daily, metadata = arguments
    capture = {}

    def intercept(directory, name, model, arrays, threshold, config, progress):
        capture.update(name=name, model=model, arrays=arrays, threshold=threshold, config=config)
        raise _BeforeFit

    monkeypatch.setattr(module, "_fit_cached", intercept)
    with pytest.raises(_BeforeFit):
        module.fit_native_and_ecology(root, data, split, 42, initial, daily=daily,
                                      daily_metadata=metadata, smoke=smoke)
    return capture


def test_native_prep_uses_oof_source_and_fixed_val_queries_only(tiny_inputs, tmp_path, monkeypatch):
    module = helper()
    captured = capture_fit(module, monkeypatch, tmp_path, tiny_inputs)
    data, split, initial, _, _ = tiny_inputs
    source, base, truth, mask, validation, val_base, val_truth, val_mask = captured["arrays"]
    _, query = support_query_cells(split, target_role="val", k=0, n_months=9)
    support, _ = support_query_cells(split, target_role="val", k=5, n_months=9)
    assert captured["name"] == "interaction_tuned"
    np.testing.assert_array_equal(base[mask], np.full(mask.sum(), 1.5))
    assert np.isnan(base[~mask]).all()
    np.testing.assert_array_equal(truth[mask], np.asarray(data["y"])[:5][mask])
    np.testing.assert_array_equal(np.flatnonzero(val_mask) + 5 * 9, query)
    assert not val_mask.ravel()[support % 9].any()
    assert (val_truth.ravel()[support % 9] == 0).all()
    np.testing.assert_array_equal(val_base, np.full((1, 9), 2.))
    np.testing.assert_array_equal(source["encoded"], initial["features"]["source_encoded"])
    np.testing.assert_array_equal(validation["encoded"], initial["features"]["full_encoded"][[5]])
    config = captured["config"]
    assert config["interaction_tuned"]["epochs"] == 30 and config["off"]["epochs"] == 120
    assert config["off"]["encoder_mode"] == "last_self_ecology"
    assert config["off"]["interaction_indices"] == [0, 2, 4, 28, 30, 31, 32]
    assert config["profile_initializer"] == "interaction_tuned"
    assert captured["threshold"] == np.quantile(np.asarray(data["y"]).ravel()[split["train"]], .9)


def test_hidden_doc_changes_neither_prepared_inputs_nor_stage_identity(tiny_inputs, tmp_path, monkeypatch):
    module = helper()
    baseline = capture_fit(module, monkeypatch, tmp_path / "first", tiny_inputs)
    changed = copy.deepcopy(tiny_inputs)
    data, split, initial, _, _ = changed
    reserved, _ = support_query_cells(split, target_role="val", k=5, n_months=9)
    hidden = np.r_[split["test"], reserved]
    data["y"].reshape(-1)[hidden] += 10000
    initial["expert"].residual.inputs.y_model.reshape(-1)[hidden] += 10000
    initial["features"] = extract_temporal_inputs(initial["expert"], data, split)
    actual = capture_fit(module, monkeypatch, tmp_path / "second", changed)
    assert actual["config"] == baseline["config"]
    for index in (0, 4):
        for name, value in baseline["arrays"][index].items():
            np.testing.assert_array_equal(actual["arrays"][index][name], value)
    for index in (1, 2, 3, 5, 6, 7):
        np.testing.assert_array_equal(actual["arrays"][index], baseline["arrays"][index])


def test_real_source_raw_view_hides_receiving_fold_and_context_marks_only_oof(tiny_inputs, tmp_path, monkeypatch):
    module = helper()
    original = module.extract_raw_temporal_inputs
    recorded = {}

    def inspect(expert, data, split):
        raw = original(expert, data, split)
        recorded.update(raw)
        assert np.count_nonzero(raw["source_raw"][..., 8:10]) == 0
        assert np.count_nonzero(np.asarray(data["y"])[5:]) == 0
        return raw

    monkeypatch.setattr(module, "extract_raw_temporal_inputs", inspect)
    captured = capture_fit(module, monkeypatch, tmp_path, tiny_inputs, smoke=True)
    assert recorded["source_raw"].shape[:2] == (5, 9)
    assert captured["config"]["interaction_tuned"]["epochs"] == 1
    assert captured["config"]["off"]["epochs"] == 1
    definition = json.loads((tmp_path / "native" / "feature_definition.json").read_text())
    assert definition["normalization"]["context"]["source_cells"] == len(tiny_inputs[1]["train"])
    assert definition["normalization"]["context"]["source_station_ids"] == list(range(5))
    assert definition["extra_dim"] == 38


def test_invalid_oof_or_daily_definition_fails_before_fitting(tiny_inputs, tmp_path, monkeypatch):
    module = helper()
    changed = copy.deepcopy(tiny_inputs)
    changed[2]["oof_z"][6, 1] = 0.
    with pytest.raises(ValueError, match="OOF cover exactly train"):
        capture_fit(module, monkeypatch, tmp_path / "oof", changed)
    changed = copy.deepcopy(tiny_inputs)
    changed[4]["value_feature_indices"] = [0, 1, 3]
    with pytest.raises(ValueError, match="eight-channel"):
        capture_fit(module, monkeypatch, tmp_path / "daily", changed)


class _TinyCheckpoint:
    """Exercise serialization/cache orchestration without torch optimization."""
    fits = 0

    def fit(self, *args, **kwargs):
        type(self).fits += 1
        self.summary = {"selected_scale": .25, "trace": [{"epoch": 0, "mae": 1.}]}
        return self

    def to_payload(self):
        return {"summary": self.summary}

    def to_dict(self):
        return self.summary

    @classmethod
    def from_payload(cls, value):
        obj = cls()
        obj.summary = value["summary"]
        return obj


def test_component_cache_reloads_and_rejects_changed_settings_or_contents(tmp_path):
    module = helper()
    _TinyCheckpoint.fits = 0
    config = {"seed": 42, "source": "fixed-role-hash"}
    module._fixed_config(tmp_path / "config.json", config)
    first = module._fit_cached(tmp_path, "toy", _TinyCheckpoint(), (), 3., config, None)
    second = module._fit_cached(tmp_path, "toy", _TinyCheckpoint(), (), 3., config, None)
    assert _TinyCheckpoint.fits == 1 and first.to_dict() == second.to_dict()
    with pytest.raises(ValueError, match="inputs/settings changed"):
        module._fixed_config(tmp_path / "config.json", {**config, "seed": 43})
    with pytest.raises(ValueError, match="Changed configuration"):
        module._fit_cached(tmp_path, "toy", _TinyCheckpoint(), (), 3., {**config, "seed": 43}, None)
    (tmp_path / "toy.json").write_text('{}')
    with pytest.raises(ValueError, match="Changed saved product"):
        module._fit_cached(tmp_path, "toy", _TinyCheckpoint(), (), 3., config, None)
    assert _TinyCheckpoint.fits == 1


def test_zero_scale_retains_native_base_without_a_floating_point_roundtrip():
    module = helper()
    base = np.array([0., .1, 1e9])
    delta = np.array([5., -.5, -1e9])
    result = module._combine(base, delta, 0.)
    np.testing.assert_array_equal(result, base)
    assert not np.shares_memory(result, base)
    np.testing.assert_array_equal(module._combine(base, delta, 1.), [5., 0., 0.])
