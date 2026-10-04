"""Retained branch settings, label roles and resumable confirmation stages."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def helper():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("confirmation_initial", scripts / "doc_chemistry_confirmation_initial.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_recipe_budgets_and_smoke_change_only_tree_count_and_epochs():
    module = helper()
    production, smoke = module.retained_initial_recipe(), module.retained_initial_recipe(smoke=True)
    assert production["initial"]["n_estimators"] == 300
    assert production["initial"]["max_epochs"] == 20
    assert production["initial"]["patience"] == 5
    assert production["projector"]["epochs"] == 100 and production["projector"]["patience"] == 15
    assert production["memory"]["epochs"] == 30 and production["memory"]["patience"] == 5
    assert production["memory"]["anchor_count"] == 32 and production["memory"]["scale_floor"] == 1e-4
    expected = copy.deepcopy(production)
    expected["initial"].update(n_estimators=20, max_epochs=1)
    expected["projector"]["epochs"] = expected["memory"]["epochs"] = 1
    assert smoke == expected


def test_basis_truth_excludes_test_and_keeps_all_validation_support_candidates():
    module = helper()
    data = {"y": torch.arange(4*9, dtype=torch.float32).reshape(4, 9)}
    split = {"train": np.arange(9), "val": np.arange(9, 18), "test": np.arange(18, 36),
             "context": np.array([], dtype=np.int64)}
    truth, masks = module._safe_role_arrays(data, split)
    assert masks["train"].sum() == 9 and masks["val"].sum() == 9
    assert masks["val"][1].all()  # Five future support slots are not removed twice.
    assert np.isnan(truth[2:]).all()
    data["y"][2:] += 10000
    alternate, _ = module._safe_role_arrays(data, split)
    np.testing.assert_array_equal(truth, alternate)
    broken = copy.deepcopy(split); broken["val"] = np.r_[broken["val"], 0]
    with pytest.raises(ValueError, match="disjoint"):
        module._safe_role_arrays(data, broken)


def test_gru_source_features_keep_original_full_cohort_fold_views(monkeypatch):
    module = helper()
    n, months = 7, 4
    cells = np.arange(n*months).reshape(n, months)
    split = {"train": cells[:5].ravel(), "val": cells[5], "test": cells[6],
             "context": np.array([], dtype=np.int64)}
    calls = []

    def fake_hidden(residual, view, *, verify_head):
        assert verify_head
        calls.append(view)
        # Simulated observation-sensitive representations reveal accidental
        # self-label exposure in any represented source fold.
        hidden = np.ones((n, months, 3), dtype=np.float32)
        hidden[..., 0] = 0
        hidden.reshape(n*months, 3)[view["train"], 0] = 10
        return hidden

    monkeypatch.setattr(module, "extract_gru_hidden", fake_hidden)
    expert = SimpleNamespace(residual=object(), rf=SimpleNamespace(folds=[np.array([i]) for i in range(5)]))
    source, full, ids = module._gru_projector_features(expert, {"y": np.zeros((n, months))}, split, lambda *args, **kwargs: None)
    assert len(calls) == 6 and source.shape == (5, months, 3) and full.shape == (n, months, 3)
    np.testing.assert_array_equal(ids, np.arange(5))
    np.testing.assert_array_equal(source[..., 0], np.zeros((5, months)))
    np.testing.assert_array_equal(full[:5, :, 0], np.full((5, months), 10))
    for fold, view in enumerate(calls[1:]):
        assert not np.isin(view["train"]//months, [fold, 5, 6]).any()


def test_readout_is_retained_whitened_source_projection():
    module = helper()
    rng = np.random.default_rng(92)
    components = rng.normal(size=(4, 6))
    projection = rng.normal(size=(2, 4))
    values = np.array([4., 1., .01, 0.])
    active = np.array([True, True, True, False])
    model = SimpleNamespace(basis_=SimpleNamespace(components_=components, eigenvalues_=values,
                                                   eigenvalue_floor=1e-8),
                            projection_=projection, active_modes_=active)
    features = rng.normal(size=(2, 8, 6))
    centered = features-features.mean(axis=1, keepdims=True)
    whitened = centered @ components.T / np.sqrt(np.maximum(values, 1e-8))
    whitened[..., ~active] = 0
    expected = whitened @ projection.T
    np.testing.assert_allclose(centered @ module.projector_readout(model), expected, atol=1e-12)


def test_stage_cache_verifies_content_and_refuses_changed_roles_or_settings(tmp_path):
    module = helper()
    identity = {"seed": 42, "split_content_hash": module._content_hash({"train": np.array([0, 1])})}
    config, complete = module._stage(tmp_path, "example", identity)
    assert not complete
    product = tmp_path / "state.json"; product.write_text('{"fitted":true}')
    module.bind_files(tmp_path, "complete.json", [tmp_path / "config.json", product], config)
    assert module._stage(tmp_path, "example", identity)[1]
    with pytest.raises(ValueError, match="different"):
        module._stage(tmp_path, "example", {**identity, "seed": 43})
    product.write_text('{"fitted":false}')
    with pytest.raises(ValueError, match="Changed saved"):
        module._stage(tmp_path, "example", identity)
    a = {"x": torch.zeros((2, 2)), "sites": ["A", "B"]}
    assert module._content_hash(a) == module._content_hash(copy.deepcopy(a))
    a["x"][0, 0] = 1
    assert module._content_hash(a) != module._content_hash({"x": torch.zeros((2, 2)), "sites": ["A", "B"]})
