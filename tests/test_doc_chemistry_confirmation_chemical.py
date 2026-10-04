"""Fresh chemistry role boundaries and resumable stages without model fitting."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def helper():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("confirmation_chemical", scripts / "doc_chemistry_confirmation_chemical.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_fixture(module, tmp_path):
    output = tmp_path / "chemical"
    config = {"seed": 42, "smoke": True, "input_hash": "source", "recipe": module.retained_chemical_recipe(smoke=True)}
    assert module._prepare_chemical_stage(output, config) is None
    for name in module.EXPECTED_FILES - {"config.json", "full_grid.parquet", "predictions.parquet"}:
        (output / name).write_bytes(b"fixed-checkpoint-fixture")
    module.write_json(output / "stage.json", {"models": list(module.MODELS), "epochs": 1})
    module.write_json(output / "basis_selection.json", {"selection_role": "source_validation", "choices": {}})
    pd.DataFrame({"model_name": ["point_integrated_legacy"], "k": [0], "mae": [1.25]}).to_csv(
        output / "source_validation.csv", index=False)
    full = pd.DataFrame({"cell": [0, 1], "point_pred": [1., 2.]})
    predictions = pd.DataFrame({"cell": [1], "model_name": ["point_integrated_legacy"], "k": [0], "y_pred": [2.]})
    result = module._finish_chemical_stage(output, config, full, predictions)
    return output, config, result


def test_recipe_preserves_matched_heads_and_changes_only_smoke_epoch_cap():
    module = helper()
    production, smoke = module.retained_chemical_recipe(), module.retained_chemical_recipe(smoke=True)
    expected = copy.deepcopy(production)
    expected["epochs"] = 1
    assert smoke == expected
    assert production["epochs"] == 120 and production["patience"] == 10
    assert production["head_features"] == 554 and production["head_parameters"] == 1111
    assert production["modes"] == ["no_aux", "masks", "chemistry"]
    assert len(module.MODELS) == len(set(module.MODELS)) == 20


def test_fit_and_support_label_views_are_isolated_from_query_truth():
    module = helper()
    data = {"y": np.arange(36, dtype=float).reshape(4, 9)}
    split = {"train": np.arange(9), "val": np.arange(9, 18), "test": np.arange(18, 36),
             "context": np.array([], dtype=np.int64)}
    fit = module._fit_label_view(data, split)
    supports = module._target_support_view(data, split)
    assert np.isfinite(fit).sum() == 18 and np.isnan(fit[split["test"]]).all()
    assert np.isfinite(supports).sum() == 10 and np.isnan(supports[split["train"]]).all()
    support, query = module.support_query_cells(split, target_role="test", k=5, n_months=9)
    changed = {"y": data["y"].copy()}
    changed["y"].ravel()[query] += 10000
    np.testing.assert_array_equal(module._fit_label_view(changed, split), fit)
    np.testing.assert_array_equal(module._target_support_view(changed, split), supports)
    changed["y"].ravel()[support] += 1
    np.testing.assert_array_equal(module._fit_label_view(changed, split), fit)
    np.testing.assert_array_equal(module._target_support_view(changed, split)[support], supports[support] + 1)
    broken = copy.deepcopy(split)
    broken["val"] = np.r_[broken["val"], split["test"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        module._fit_label_view(data, broken)


def test_completed_cache_returns_products_without_refitting(tmp_path, monkeypatch):
    module = helper()
    _, config, expected = completed_fixture(module, tmp_path)
    monkeypatch.setattr(module, "_chemical_config", lambda *args, **kwargs: config)

    def forbidden(*args, **kwargs):
        raise AssertionError("A verified completed stage must not refit a head")

    monkeypatch.setattr(module.NonlinearChemistryHead, "fit", forbidden)
    restored = module.fit_chemistry_and_calibration(tmp_path, None, None, 42, None, None, smoke=True)
    for name in ("full_grid", "predictions", "source_validation"):
        pd.testing.assert_frame_equal(restored[name], expected[name])
    assert restored["metadata"] == expected["metadata"]
    assert restored["selection"] == expected["selection"]
    assert "chemical/complete.json" in restored["model_files"]
    assert "y_true" not in restored["predictions"]


def test_completed_cache_rejects_changed_recipe_and_changed_file(tmp_path):
    module = helper()
    output, config, _ = completed_fixture(module, tmp_path)
    with pytest.raises(ValueError, match="recipe changed"):
        module._prepare_chemical_stage(output, {**config, "seed": 43})
    (output / "neural_chemistry.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed saved product"):
        module._prepare_chemical_stage(output, config)
    assert not list(tmp_path.glob("chemical_partial_*"))


def test_incomplete_stage_is_preserved_and_only_chemical_stage_restarts(tmp_path):
    module = helper()
    output = tmp_path / "chemical"
    output.mkdir()
    (output / "neural_no_aux.pt").write_bytes(b"interrupted-checkpoint")
    (output / "config.json").write_text('{"old": true}')
    sibling = tmp_path / "native"
    sibling.mkdir()
    (sibling / "off.pt").write_bytes(b"retained-native")
    config = {"new": True}
    assert module._prepare_chemical_stage(output, config) is None
    archives = list(tmp_path.glob("chemical_partial_*"))
    assert len(archives) == 1
    assert (archives[0] / "neural_no_aux.pt").read_bytes() == b"interrupted-checkpoint"
    assert json.loads((output / "config.json").read_text()) == config
    assert list(output.iterdir()) == [output / "config.json"]
    assert (sibling / "off.pt").read_bytes() == b"retained-native"


def test_completed_cache_requires_full_declared_artifact_set(tmp_path):
    module = helper()
    output, config, _ = completed_fixture(module, tmp_path)
    complete = json.loads((output / "complete.json").read_text())
    del complete["files"]["neural_chemistry.pt"]
    module.write_json(output / "complete.json", complete)
    with pytest.raises(ValueError, match="missing required products"):
        module._prepare_chemical_stage(output, config)
