"""Input, source-only selection and frozen DOC literature comparison contracts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.dummy import DummyRegressor

from river_graph.experiments.doc_domain_work import (
    CANDIDATES,
    FAMILIES,
    features,
    make_estimator,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_domain_work_v1")
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DEPS = Path("cache/doc_domain_work_v1/python_deps")


def input_example():
    return {"environment": torch.arange(3 * 15).reshape(3, 15).float(),
        "windows": torch.arange(12 * 3 * 12 * 8).reshape(12, 3, 12, 8).float(),
        "season": torch.zeros(12, 2), "normalization": {"source_rows": [0]}}


def test_current_ecological_and_full_history_features_are_declared():
    inputs = input_example()
    cells = np.array([2, 18, 29])
    for family in FAMILIES:
        x = features(inputs, cells, 12, family)
        assert x.shape == (3, 23 if family in FAMILIES[:2] else 113)
        np.testing.assert_array_equal(x[:, :15], inputs["environment"].numpy()[cells // 12])
    compact = features(inputs, cells, 12, FAMILIES[0])
    np.testing.assert_array_equal(compact[:, 15:], inputs["windows"].numpy()[cells % 12, cells // 12, -1])
    with pytest.raises(ValueError, match="undeclared"):
        features(inputs, cells, 12, "unknown")


@pytest.mark.skipif(not DEPS.exists(), reason="isolated pyGAM/XGBoost dependencies are unavailable")
@pytest.mark.parametrize("family", FAMILIES)
def test_official_estimators_fit_and_predict_finite_values(family):
    rng = np.random.default_rng(7)
    columns = 23 if family in FAMILIES[:2] else 113
    x = rng.normal(size=(60, columns))
    y = np.maximum(1 + .3 * x[:, 1] + .1 * x[:, 15], 0)
    recipe = next(iter(CANDIDATES[family].values()))
    model = make_estimator(family, recipe, 42, 1).fit(x, y)
    prediction = model.predict(x[:4])
    assert prediction.shape == (4,) and np.isfinite(prediction).all()


def runner_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("doc_domain_runner", scripts / "run_doc_domain_work_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_target_doc_poison_cannot_change_fit_or_source_selection(tmp_path, monkeypatch):
    runner = runner_module()
    inputs = input_example()
    ctx = {"months": 12, "inputs": inputs, "split": {"train": np.array([0, 1, 2, 3]),
        "val": np.array([12, 13]), "test": np.array([24, 25])}}
    data = {"y": np.ones((3, 12)), "site_no": ["a", "b", "c"]}
    data["y"][0, :4] = [1, 2, 3, 4]
    protocol = {"runtime_snapshot": {}, "dataset_sha256": "data", "mask_sha256": {"1013": "mask"}, "threads": 1}
    monkeypatch.setattr(runner, "context", lambda *_: ctx)
    monkeypatch.setattr(runner, "make_estimator", lambda *_: DummyRegressor())
    runner.fit_region(tmp_path / "clean", "1013", protocol, data)
    poisoned = {**data, "y": data["y"].copy()}
    poisoned["y"][2] = np.nan
    runner.fit_region(tmp_path / "poisoned", "1013", protocol, poisoned)
    directories = [tmp_path / name / "regions/huc4_1013" for name in ("clean", "poisoned")]
    selections = [json.loads((path / "selection.json").read_text())["selections"] for path in directories]
    assert selections[0] == selections[1]
    assert len(list(directories[0].glob("fits/*/*/seed*/fit_complete.json"))) == 24
    changed = {**protocol, "dataset_sha256": "changed"}
    with pytest.raises(ValueError, match="config changed"):
        runner.fit_region(tmp_path / "clean", "1013", changed, data)
    next(directories[0].glob("fits/*/*/seed*/model.joblib")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed package artifact"):
        runner.fit_region(tmp_path / "clean", "1013", protocol, data)


@pytest.mark.skipif(not DATA.exists() or not (ROOT / "protocol.json").exists(),
    reason="local ST357 dataset or frozen DOC literature protocol is unavailable")
def test_frozen_protocol_preserves_original_dispatcher_and_literature_status():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["candidates"] == CANDIDATES
    assert protocol["joint_primary"] == ["native_mae", "log1p_mae"]
    assert protocol["budgets"]["unique_candidate_fits"] == 120
    assert not protocol["prior_paper_endpoints_modified"]
    assert "not complete paper reproductions" in protocol["adaptation_status"]
    assert sha256_file("scripts/run_ladder.py") == protocol["original_repository_dispatcher_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT / "code_snapshot" / name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected
